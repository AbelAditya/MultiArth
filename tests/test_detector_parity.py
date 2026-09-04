"""
tests/test_detector_parity.py
------------------------------
Validates workers/_detector.py's hand-written YOLO11n-seg post-processing
against Ultralytics' own implementation on real frames.

Why this test earns its keep: to keep `ultralytics` (heavy, AGPL-3.0) out
of the runtime image, _detector.py re-implements anchor decode, NMS,
prototype-mask assembly and un-letterboxing directly on the ONNX outputs.
That is precisely the class of code that fails *silently and plausibly* —
a mask offset by a few pixels, or an edge smeared outside its box, raises
nothing and still looks broadly like a person. It cost two real bugs
during development, neither of which produced an error:

  1. Cropping masks at prototype resolution and upsampling afterwards,
     instead of upsampling first and cropping at input resolution. This is
     Ultralytics' own upstream issue #24272 ("cropping first smears the
     bilinear edge outside the bbox"), and it degraded worst-case mask IoU
     to 0.66.
  2. Slicing the box crop with rounded integer indices rather than
     comparing against float bounds, which shifted mask edges by up to a
     prototype pixel (4 input pixels) — worst on physically small
     detections, i.e. the small-distant-speaker case the detector exists
     to get right.

With both fixed, agreement is exact (mask IoU 1.0000).

`ultralytics` is not a project dependency, so this test skips unless it is
explicitly installed:

    uv pip install ultralytics onnx onnxslim
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from workers._detector import _INPUT_SIZE, DETECTOR_MODEL_PATH, detect_people, load_detector

ultralytics = pytest.importorskip(
    "ultralytics", reason="export-time-only dependency; see module docstring"
)

_ROOT = Path(__file__).resolve().parent.parent
_VIDEOS = [
    (_ROOT / "vids" / "test_vid_1.mp4", [5.0, 20.0, 40.0]),
    (_ROOT / "vids" / "TED TEST_2.mp4", [3.0, 10.0]),
    (_ROOT / "vids" / "test_vid_3.mp4", [5.0, 12.0, 18.0]),
]

pytestmark = pytest.mark.skipif(
    not DETECTOR_MODEL_PATH.exists(),
    reason=f"{DETECTOR_MODEL_PATH} not built — run scripts/export_yolo11n_seg.py",
)


def _ref_mask_to_frame(mask_640: np.ndarray, w: int, h: int) -> np.ndarray:
    """Ultralytics returns masks at letterboxed *input* resolution
    (confirmed directly: `Results.masks.data` is (N, 640, 640) for a
    576x1024 frame), not frame resolution. Undoing the letterbox the same
    way _detector.py does is what makes the two comparable at all — an
    earlier version of this comparison skipped it and produced a spurious
    ~0.5 IoU that looked like a detector bug but was a test bug."""
    scale = min(_INPUT_SIZE / w, _INPUT_SIZE / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    pad_x, pad_y = (_INPUT_SIZE - nw) // 2, (_INPUT_SIZE - nh) // 2
    unpadded = mask_640[pad_y:pad_y + nh, pad_x:pad_x + nw].astype(np.float32)
    return cv2.resize(unpadded, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5


def _box_iou(a, b) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _frames():
    for path, timestamps in _VIDEOS:
        if not path.exists():
            continue
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
            ok, bgr = cap.read()
            if ok:
                yield path.name, ts, bgr
        cap.release()


def test_matches_ultralytics():
    from ultralytics import YOLO

    session = load_detector()
    reference = YOLO(str(DETECTOR_MODEL_PATH), task="segment")

    compared = 0
    for name, ts, bgr in _frames():
        h, w = bgr.shape[:2]
        mine = detect_people(session, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ref = reference.predict(bgr, conf=0.25, iou=0.45, classes=[0], verbose=False)[0]

        if ref.boxes is None or len(ref.boxes) == 0:
            assert not mine, f"{name}@{ts}: found {len(mine)} where reference found none"
            continue

        ref_boxes = ref.boxes.xyxy.cpu().numpy()
        ref_masks = ref.masks.data.cpu().numpy()
        assert len(mine) == len(ref_boxes), (
            f"{name}@{ts}: detection count {len(mine)} != reference {len(ref_boxes)}"
        )

        for det in mine:
            j = int(np.argmax([_box_iou(det.box, b) for b in ref_boxes]))
            # Boxes are deliberately rounded to integer pixels here (they
            # index crops), so exact equality isn't expected — sub-pixel
            # rounding is the only permitted source of disagreement.
            assert _box_iou(det.box, ref_boxes[j]) > 0.95, f"{name}@{ts}: box drift"

            ref_mask = _ref_mask_to_frame(ref_masks[j], w, h)
            union = (det.mask | ref_mask).sum()
            mask_iou = (det.mask & ref_mask).sum() / union if union else 0.0
            assert mask_iou > 0.99, (
                f"{name}@{ts}: mask IoU {mask_iou:.4f} — post-processing has "
                "diverged from Ultralytics (check crop/upsample order)"
            )
            compared += 1

    assert compared > 0, "no frames compared — are vids/ present?"
