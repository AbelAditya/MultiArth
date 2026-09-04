"""
workers/_detector.py
---------------------
YOLO11n-seg person detection + instance segmentation, run through
`onnxruntime`. This is the *detection* stage of the detector-first pose
pipeline: it finds people and produces a per-person mask, and everything
downstream consumes those detections rather than asking MediaPipe to find
people itself.

## Why this exists

MediaPipe's `PoseLandmarker` bundles its own person detector (BlazePose),
trained overwhelmingly on close-range, single-dominant-person imagery. On
this project's footage — a small speaker on a wide stage, cluttered
background, seated audience — that detector is being used well outside its
design envelope, with two consequences that motivated this module:

  1. The speaker is sometimes not returned at all. `num_poses` is a hard
     cap on candidates, so if the audience fills those slots the speaker
     is absent from the results entirely and no downstream selection —
     heuristic or gallery-based — can recover them.
  2. An over-large or mis-centred ROI yields a skeleton fitted across the
     speaker *and* background structure (typically legs on the speaker,
     arms thrown onto background). This needs no human-looking background
     at all: BlazePose's landmark model is a single-person regressor that
     always emits all 33 landmarks over whatever region it is given, with
     no part-association step that could decline to attach a limb.

A COCO-trained detector is a much better fit: full-body people across wide
scale variation, heavy clutter and crowding, with hard-negative mining
against background. Feeding MediaPipe a tight per-person crop also puts
BlazePose back *inside* its own training distribution (one dominant,
close-range person) rather than replacing it.

## Why ONNX rather than the `ultralytics` runtime

`onnxruntime` is already a dependency of this project, so running the
exported graph costs no new runtime package. `ultralytics` itself is heavy
(torchvision, polars, matplotlib) and AGPL-3.0; it is needed only to
*produce* the .onnx and is absent from the runtime image. See
scripts/export_yolo11n_seg.py.

The cost of that choice is that the post-processing below (anchor decode,
NMS, prototype-mask assembly, un-letterboxing) is written here by hand
rather than inherited from a well-tested library — which is exactly the
kind of code that fails silently and plausibly. It is therefore validated
against ultralytics' own output on real frames; see
tests/test_detector_parity.py.

## Licensing

YOLO11 weights and architecture are AGPL-3.0. This was a deliberate,
explicit project decision, not an oversight. Relevant because the
dashboard is network-served, which is what AGPL's network clause turns on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DETECTOR_MODEL_PATH = _MODELS_DIR / "yolo11n-seg.onnx"

# Must match scripts/export_yolo11n_seg.py's _IMGSZ — the exported graph
# has this resolution baked into its input shape, so a mismatch here is a
# hard shape error at session.run, not a silent degradation.
_INPUT_SIZE = 640

_PERSON_CLASS_ID = 0    # COCO class 0 is "person"; every other class is
# discarded before NMS, so a chair or a potted plant never competes for a
# detection slot with a real person.

_NUM_MASK_COEFFS = 32   # matches the prototype-mask count in the second
# output tensor, (1, 32, 160, 160)
_PROTO_SIZE = 160       # prototype masks are at input/4 resolution

# Detection confidence floor. Deliberately lower than a typical demo's 0.5:
# the failure this module exists to fix is the *speaker being missed*, and
# a small, partly-occluded speaker on a wide stage is exactly the sort of
# detection that scores modestly. Downstream selection (gallery matching /
# centrality) is what decides which detection is the subject, so admitting
# a few extra weak candidates is far cheaper here than dropping the true
# one. Not empirically tuned against this project's footage yet.
_CONF_THRESHOLD = 0.25

_IOU_THRESHOLD = 0.45   # NMS overlap threshold, Ultralytics' own default
_MAX_DETECTIONS = 20    # bound on candidates handed downstream per frame —
# a crowd shot shouldn't make one frame cost 100 re-ID embeddings.

_MASK_BINARY_THRESHOLD = 0.5


@dataclass
class PersonDetection:
    """One detected person. `mask` is full-frame-sized so it can be handed
    straight to workers/_reid.py's crop_via_mask, which is shape-agnostic
    between this and MediaPipe's own (H, W, 1) masks (it squeezes)."""
    box: tuple[int, int, int, int]   # (x0, y0, x1, y1), frame pixels
    score: float
    mask: np.ndarray                 # (H, W) bool, frame-sized


def ensure_detector_weights() -> None:
    """Unlike this project's other models, there is nothing to download:
    Ultralytics publishes only `.pt` weights (their HuggingFace `.onnx`
    path 404s — confirmed, not assumed), and pre-exported community ONNX
    files were rejected on provenance grounds. The .onnx is produced from
    the official release by scripts/export_yolo11n_seg.py, which the
    Dockerfile's builder stage also runs. So this only reports a clear,
    actionable error rather than silently fetching something."""
    if DETECTOR_MODEL_PATH.exists():
        return
    raise FileNotFoundError(
        f"{DETECTOR_MODEL_PATH} is missing. It is generated, not downloaded:\n"
        "    uv pip install ultralytics onnx onnxslim\n"
        "    python scripts/export_yolo11n_seg.py\n"
        "See workers/_detector.py's module docstring for why."
    )


def load_detector():
    """Builds an onnxruntime session. Callers own their own lazily-cached
    singleton — same division of responsibility as workers/_reid.py's
    load_reid_model, and for the same reason: GestureWorker and the
    dashboard's gallery builder want different lifetimes for it."""
    import onnxruntime as ort

    ensure_detector_weights()
    logger.info("[detector] Loading YOLO11n-seg person detector...")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(DETECTOR_MODEL_PATH), opts, providers=["CPUExecutionProvider"],
    )


def _letterbox(rgb: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Resizes into a square _INPUT_SIZE canvas preserving aspect ratio,
    padding the remainder. Aspect is preserved rather than stretched
    because a stretched person is off-distribution for a detector trained
    on real photographs, and because the crops this ultimately produces
    feed MediaPipe, whose world-landmark (and hence yaw) estimation
    assumes undistorted geometry.

    Returns (canvas, scale, pad_x, pad_y) — the three values needed to map
    coordinates back to the original frame."""
    h, w = rgb.shape[:2]
    scale = min(_INPUT_SIZE / w, _INPUT_SIZE / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((_INPUT_SIZE, _INPUT_SIZE, 3), 114, dtype=np.uint8)  # 114
    # is Ultralytics' own padding value; matching it keeps this consistent
    # with how the network saw padding during training.
    pad_x, pad_y = (_INPUT_SIZE - nw) // 2, (_INPUT_SIZE - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def _build_masks(
    coeffs: np.ndarray, protos: np.ndarray, boxes_in: np.ndarray,
    frame_wh: tuple[int, int], scale: float, pad_x: int, pad_y: int,
) -> list[np.ndarray]:
    """Assembles per-instance masks from the prototype basis.

    YOLO-seg does not emit a mask per detection directly; it emits 32
    prototype masks for the whole image plus, per detection, 32
    coefficients that linearly combine them. So a mask is
    sigmoid(coeffs @ protos), then cropped to its own box (the linear
    combination is global and spills outside the instance otherwise), then
    un-letterboxed back to frame resolution.
    """
    w, h = frame_wh
    flat = protos.reshape(_NUM_MASK_COEFFS, -1)          # (32, 160*160)
    m = (coeffs @ flat).reshape(-1, _PROTO_SIZE, _PROTO_SIZE)   # logits, NOT
    # probabilities — no sigmoid is applied anywhere here. Binarising at
    # logit > 0 is exactly equivalent to sigmoid(x) > 0.5 and skips the
    # transcendental, which is what Ultralytics does too (`masks.gt_(0.0)`).

    nw, nh = int(round(w * scale)), int(round(h * scale))
    # Half-open pixel-index grids for the box crop, at *input* resolution.
    gx = np.arange(_INPUT_SIZE, dtype=np.float32)[None, :]
    gy = np.arange(_INPUT_SIZE, dtype=np.float32)[:, None]

    out: list[np.ndarray] = []
    for i in range(m.shape[0]):
        # Order matters, and not in the intuitive direction: upsample to
        # input resolution FIRST, then crop to the box. Cropping at
        # prototype resolution and upsampling afterwards lets bilinear
        # interpolation smear the mask edge back outside the box — this is
        # Ultralytics' own upstream issue #24272, and getting it backwards
        # here measurably degraded agreement with their output (worst-case
        # mask IoU 0.66 vs 0.99) with no error raised anywhere. Cropping
        # against float box bounds rather than rounded integer slice
        # indices matters for the same reason, and disproportionately so on
        # physically small detections — precisely the small-distant-speaker
        # case this detector exists to get right.
        full = cv2.resize(m[i], (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        x0, y0, x1, y1 = boxes_in[i]
        binary = (full > 0.0) & (gx >= x0) & (gx < x1) & (gy >= y0) & (gy < y1)
        if not binary.any():
            out.append(np.zeros((h, w), dtype=bool))
            continue
        # letterboxed input -> strip padding -> original frame resolution.
        unpadded = binary[pad_y:pad_y + nh, pad_x:pad_x + nw].astype(np.float32)
        resized = cv2.resize(unpadded, (w, h), interpolation=cv2.INTER_LINEAR)
        out.append(resized > _MASK_BINARY_THRESHOLD)
    return out


def detect_people(
    session,
    rgb: np.ndarray,
    conf_threshold: float = _CONF_THRESHOLD,
    iou_threshold: float = _IOU_THRESHOLD,
    max_detections: int = _MAX_DETECTIONS,
) -> list[PersonDetection]:
    """
    Returns person detections for one RGB frame, highest-scoring first.

    An empty list means "nobody detected here" and is a normal outcome
    (a cutaway, a slide, an empty stage) — callers treat it the same way
    they previously treated MediaPipe returning no poses.
    """
    h, w = rgb.shape[:2]
    canvas, scale, pad_x, pad_y = _letterbox(rgb)

    blob = canvas.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None, ...]      # (1, 3, 640, 640)

    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    preds, protos = outputs[0], outputs[1]

    preds = preds[0].T                                   # (8400, 116)
    scores = preds[:, 4 + _PERSON_CLASS_ID]
    keep = scores >= conf_threshold
    if not np.any(keep):
        return []
    preds, scores = preds[keep], scores[keep]

    # Boxes arrive as (cx, cy, w, h) in letterboxed-input pixels.
    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes_in = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

    # cv2's NMS wants (x, y, w, h) ints; using OpenCV's implementation
    # rather than a hand-rolled one keeps one more piece of fiddly,
    # easy-to-get-subtly-wrong logic out of this file.
    nms_boxes = [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes_in]
    idxs = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), conf_threshold, iou_threshold)
    if len(idxs) == 0:
        return []
    idxs = np.array(idxs).flatten()
    order = idxs[np.argsort(-scores[idxs])][:max_detections]

    masks = _build_masks(
        preds[order, 4 + 80:4 + 80 + _NUM_MASK_COEFFS], protos[0],
        boxes_in[order], (w, h), scale, pad_x, pad_y,
    )

    detections: list[PersonDetection] = []
    for i, det_i in enumerate(order):
        x0, y0, x1, y1 = boxes_in[det_i]
        # Undo the letterbox, then clamp into the frame — a box may
        # legitimately extend past the edge for a partially-visible person.
        bx0 = int(round(max(0, (x0 - pad_x) / scale)))
        by0 = int(round(max(0, (y0 - pad_y) / scale)))
        bx1 = int(round(min(w, (x1 - pad_x) / scale)))
        by1 = int(round(min(h, (y1 - pad_y) / scale)))
        if bx1 <= bx0 or by1 <= by0:
            continue
        detections.append(PersonDetection(
            box=(bx0, by0, bx1, by1), score=float(scores[det_i]), mask=masks[i],
        ))
    return detections
