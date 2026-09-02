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
MIN_MASK_PIXELS = 200


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


def crop_via_mask(rgb: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """Tight bbox crop with background zeroed out via the person's own
    segmentation mask (not a raw bounding box, which would bleed in
    background/neighbours) — verified directly against real footage, see
    wikis/Gesture-Worker.md's re-ID section. mask comes back as (H, W, 1)
    from MediaPipe (confirmed directly, not assumed)."""
    mask = mask.squeeze()
    h, w = mask.shape
    mask_bin = mask > 0.5
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
    gallery — not a plain mean over the whole gallery, which would dilute
    a genuine match against one look by averaging in the gallery's other,
    legitimately different-looking exemplars. gallery rows and query are
    both already L2-normalised, so a plain dot product is cosine
    similarity."""
    sims = gallery @ query
    k = min(k, len(sims))
    top = np.sort(sims)[-k:]
    return float(np.mean(top))


def max_similarity(query: np.ndarray, gallery: np.ndarray) -> float:
    """Single nearest-neighbour similarity — used for gallery-*building*'s
    own redundancy check (core/gallery_builder.py), a deliberately
    different question from top_k_similarity's runtime-matching use: "is
    this a near-duplicate of any single existing look" rather than "does
    this match the gallery well enough overall". See
    workers/gesture_worker.py's module docstring for why these two use
    different pooling."""
    if len(gallery) == 0:
        return 0.0
    return float((gallery @ query).max())
