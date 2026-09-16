"""
workers/_reid.py
-----------------
Shared OSNet loading/embedding helpers — used by both
workers/gesture_worker.py (runtime speaker matching during analysis) and
core/gallery_builder.py (the dashboard's interactive gallery-confirmation
flow, bulk upload only). Kept in exactly one place so the embedding math
— preprocessing, normalisation, crop extraction — can't silently drift
between "how a gallery exemplar was embedded" and "how a live candidate is
embedded", which would quietly break every similarity comparison between
the two without either side raising an error.

See workers/gesture_worker.py's module docstring ("Speaker
re-identification") for the full design this supports, and
workers/_osnet.py for the vendored model architecture itself.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REID_MODEL_PATH = _MODELS_DIR / "osnet_x0_25_msmt17.pth"
REID_MODEL_URL = (
    "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
    "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
    "b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)
REID_NUM_CLASSES = 4101  # MSMT17's real identity count — needed to build the
# matching classifier-head shape before load_state_dict; unused past
# loading, since eval-mode forward() returns the 512-d embedding directly
# (confirmed against _osnet.py's own forward(), not assumed).
REID_INPUT_WH = (128, 256)  # torchreid's own input convention (W, H)
REID_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet
REID_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)   # normalisation,
# matching what OSNet was trained with.

# Minimum foreground pixels for a segmentation-mask crop to be worth
# embedding at all — filters out a mask that's mostly noise/too small to
# be a real detection. Not empirically tuned.
MIN_MASK_PIXELS = 100


def ensure_reid_weights() -> None:
    if REID_MODEL_PATH.exists():
        return
    REID_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[reid] Downloading osnet_x0_25 model to {REID_MODEL_PATH}...")
    urllib.request.urlretrieve(REID_MODEL_URL, str(REID_MODEL_PATH))


def load_reid_model():
    """Builds and returns a ready-to-use (eval mode) OSNet instance.
    Callers each own their own lazily-cached singleton — this function
    just does the actual construction + weight-loading work; it doesn't
    cache anything itself, since GestureWorker and the dashboard's gallery
    builder have different lifetimes for how long a loaded model should
    stick around (see gesture_worker.py's module docstring)."""
    import torch
    from workers._osnet import osnet_x0_25

    ensure_reid_weights()
    logger.info("[reid] Loading OSNet speaker re-identification model...")
    model = osnet_x0_25(pretrained=False, num_classes=REID_NUM_CLASSES)
    state = torch.load(str(REID_MODEL_PATH), map_location="cpu")
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(
            f"[reid] OSNet state_dict mismatch — missing={len(missing)} "
            f"unexpected={len(unexpected)} (expected 0/0; check REID_NUM_CLASSES "
            "matches the checkpoint's real identity count)"
        )
    model.eval()  # eval mode -> forward() returns the 512-d embedding
    # directly, not classifier logits.
    return model


def limit_torch_threads(n: int = 1) -> None:
    """Caps PyTorch's intra-op thread pool, for callers that run OSNet
    interleaved with other models on the same cores.

    Deliberately *not* called from load_reid_model: `torch.set_num_threads`
    is process-global, so making it a side effect of loading a model would
    silently reconfigure every other torch user in the process. It is an
    explicit opt-in instead, and only workers/gesture_worker.py takes it.

    ## Why one thread is not a sacrifice

    OSNet x0_25 is tiny (~0.2M params on a 128x256 input), and measured in
    isolation it is no slower single-threaded than it is on six: 13.16ms vs
    12.98ms, inside run-to-run noise. At twelve it is catastrophically worse
    (81.9ms) — logical-core oversubscription, not real parallelism.

    What this buys is contention, not throughput. In the real per-frame
    sequence (detector -> OSNet -> MediaPipe, three libraries each sizing a
    thread pool for a machine it assumes it owns) the same embedding costs
    55ms. Disabling the detector's spin-wait brings that to ~48ms; capping
    torch here takes it to ~26ms. Combined, the two are worth ~28% of the
    whole per-frame budget.

    ## The one conflict, and why it is accepted

    This is process-global, and Orchestrator._run_parallel runs the four
    workers as concurrent threads in one process, so it also caps
    VerbalWorker's SenseVoice (funasr) — which is torch-backed and loaded
    locally for Chinese audio unless SENSEVOICE_REMOTE_URL routes it to
    colab/sensevoice_server.ipynb instead. That is a real slowdown for that
    one path, and it is taken knowingly: gesture is the job's critical path
    by orders of magnitude (hours against minutes), so verbal finishing
    later still finishes long before the job does, and the cores SenseVoice
    stops monopolising are cores gesture gets back.
    """
    import torch

    torch.set_num_threads(n)


# How far past its own detection box a mask may be searched for. Masks are
# built by cropping at the detector's 640x640 input resolution and then
# upsampling to frame resolution (~3x at 1080p), while the box is rounded to
# integer frame pixels separately — so the two disagree slightly at the
# edges and a mask can spill a little outside the box it belongs to.
# Measured across 68 detections on real footage the worst spill was 3px,
# always right/bottom; 8 leaves room without meaningfully enlarging the
# search. crop_via_mask does not rely on this being sufficient — see its
# containment guard.
_BOX_SEARCH_MARGIN = 8


def crop_via_mask(
    rgb: np.ndarray, mask: np.ndarray,
    box: Optional[tuple[int, int, int, int]] = None,
) -> Optional[np.ndarray]:
    """Tight bbox crop with background zeroed out via the person's own
    segmentation mask (not a raw bounding box, which would bleed in
    background/neighbours) — verified directly against real footage, see
    wikis/Gesture-Worker.md's re-ID section. mask comes back as (H, W, 1)
    from MediaPipe (confirmed directly, not assumed).

    `box` is the detection's own bounding box, and is an optimisation only:
    the mask's extent is found by searching inside it rather than scanning
    the whole frame. Passing it does not change the result — only how long
    finding it takes. Omitting it falls back to the full-frame scan, so
    callers that have no box (or a mask not derived from one) stay correct.

    ## Why the box is worth passing

    The mask is frame-sized, so on 1080p footage the unguided `np.where`
    scans 2.07M booleans to locate a person occupying perhaps 50k of them —
    and the detector has already said where they are. Measured, that scan is
    8.00ms per candidate against 0.13ms for the box-local one, with the
    returned crop byte-identical across 208 detections on four videos.

    It matters most on the Searching path, where *every* candidate is
    cropped: a crowded auditorium frame at _MAX_DETECTIONS goes from ~160ms
    of scanning to ~2.6ms. Locked state crops one candidate, so it saves the
    one 8ms there.
    """
    mask_bin = mask.squeeze() > 0.5
    h, w = mask_bin.shape

    ys = xs = None
    if box is not None:
        bx0, by0 = max(0, box[0] - _BOX_SEARCH_MARGIN), max(0, box[1] - _BOX_SEARCH_MARGIN)
        bx1, by1 = min(w, box[2] + _BOX_SEARCH_MARGIN), min(h, box[3] + _BOX_SEARCH_MARGIN)
        wys, wxs = np.where(mask_bin[by0:by1, bx0:bx1])
        # Containment guard: the shortcut is only valid if the whole mask
        # lies inside the window. If any mask pixel sits on a window edge
        # that isn't also a frame edge, the window may have clipped it, and
        # the extent found here would be wrong rather than merely slower —
        # so fall through to the full scan. Measured, this never fires on
        # real footage (0/208), but the failure it guards against is a
        # silently-shifted crop, which is precisely the kind that would
        # degrade every similarity comparison downstream without raising.
        if len(wxs) and not (
            (wxs.min() == 0 and bx0 > 0)
            or (wys.min() == 0 and by0 > 0)
            or (wxs.max() == bx1 - bx0 - 1 and bx1 < w)
            or (wys.max() == by1 - by0 - 1 and by1 < h)
        ):
            ys, xs = wys + by0, wxs + bx0

    if xs is None:
        ys, xs = np.where(mask_bin)

    if len(xs) < MIN_MASK_PIXELS:
        return None
    x0, x1 = max(0, xs.min() - 5), min(w, xs.max() + 5)
    y0, y1 = max(0, ys.min() - 5), min(h, ys.max() + 5)
    if (x1 - x0) < 20 or (y1 - y0) < 20:
        return None
    region = rgb[y0:y1, x0:x1].copy()
    region_mask = mask_bin[y0:y1, x0:x1]
    region[~region_mask] = 0
    return region


def embed_crop(model, rgb_crop: np.ndarray) -> np.ndarray:
    """Returns a 512-d, L2-normalised OSNet embedding for an RGB crop
    (background already zeroed out by crop_via_mask)."""
    import torch

    resized = cv2.resize(rgb_crop, REID_INPUT_WH, interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - REID_MEAN) / REID_STD
    x = np.transpose(x, (2, 0, 1))[None, ...]
    with torch.no_grad():
        v = model(torch.from_numpy(x).float())
    v = v.numpy()[0]
    return v / (np.linalg.norm(v) + 1e-8)


def top_k_similarity(query: np.ndarray, gallery: np.ndarray, k: int) -> float:
    """Mean of the top-k per-exemplar cosine similarities against the
    gallery. gallery rows and query are both already L2-normalised, so a
    plain dot product is cosine similarity.

    Used by gallery-*building*'s redundancy check
    (core/gallery_builder.py's record_confirmation): "is this confirmed
    frame a duplicate of several looks we already hold". Deliberately not
    what runtime matching uses — see max_similarity below.

    The property that makes it right there and wrong for runtime matching
    is the same one: because the mean pulls in the gallery's other,
    legitimately different looks, it systematically under-scores a look the
    gallery holds only once or twice. For a *duplicate* test that is the
    desired conservatism; for a *match* test it rejected genuine speakers
    in thinly-sampled scenes (measured 0.66-0.68 where max gave 0.90-0.97).
    k is clamped to the gallery size, so a 1- or 2-entry gallery degrades
    to max / mean-of-2 rather than erroring."""
    sims = gallery @ query
    k = min(k, len(sims))
    top = np.sort(sims)[-k:]
    return float(np.mean(top))


def max_similarity(query: np.ndarray, gallery: np.ndarray) -> float:
    """Single nearest-neighbour similarity — "does this look like *any*
    single exemplar we hold".

    Used by runtime matching (workers/gesture_worker.py's
    _gallery_match): "is this candidate one of our confirmed looks at
    all". One strong match against a single exemplar is enough, which is
    what makes a thinly-sampled look still matchable at runtime — the
    failure mode that top_k_similarity caused here before.

    Gallery *building* deliberately uses top_k_similarity instead, for the
    different question of whether a new confirmation is redundant. See
    core/gallery_builder.py's GALLERY_REDUNDANCY_TOP_K for why the two
    sides differ on purpose."""
    if len(gallery) == 0:
        return 0.0
    return float((gallery @ query).max())
