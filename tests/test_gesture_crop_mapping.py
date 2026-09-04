"""
tests/test_gesture_crop_mapping.py
-----------------------------------
Unit tests for the detector-first pipeline's coordinate transform:
_letterbox_crop packs a detection box into a square canvas, MediaPipe
returns landmarks normalised to that square, and _crop_norm_to_frame_norm
maps them back to frame-normalised coordinates.

This arithmetic gets its own tests because it is the one part of the
detector-first change that fails *silently*. A forgotten padding offset or
an off-by-one produces landmarks that are wrong but entirely
plausible-looking — a skeleton slightly beside the person, with nothing
raised anywhere and no visibly broken output to notice. Every other
failure mode in that pipeline announces itself (a missing model raises, a
shape mismatch raises, a bad mask is visible on screen); this one does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from workers.gesture_worker import _box_center, _crop_norm_to_frame_norm, _letterbox_crop

FRAME_W, FRAME_H = 640, 480


def _frame():
    return np.random.default_rng(0).integers(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)


@pytest.mark.parametrize("box", [
    (100, 50, 200, 300),    # tall (portrait person)
    (10, 10, 400, 60),      # wide
    (0, 0, 64, 64),         # already square, at the origin
    (600, 440, 640, 480),   # flush against the bottom-right frame edge
])
def test_letterbox_is_square_and_lossless(box):
    """The crop must be padded, never scaled or cropped further — a
    distant speaker's few pixels are exactly what we're trying to keep."""
    rgb = _frame()
    square, side, off_x, off_y = _letterbox_crop(rgb, box)
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0

    assert square.shape == (side, side, 3), "canvas must be square"
    assert side == max(bw, bh), "side must be the longer box edge, unscaled"
    # The original pixels survive at their native resolution, at the offset.
    assert np.array_equal(square[off_y:off_y + bh, off_x:off_x + bw], rgb[y0:y1, x0:x1])


@pytest.mark.parametrize("box", [
    (100, 50, 200, 300),
    (10, 10, 400, 60),
    (0, 0, 64, 64),
    (317, 201, 355, 296),   # odd dimensions -> asymmetric integer padding
])
def test_roundtrip_box_corners(box):
    """A landmark at a known point in the crop must map back to that same
    point in the frame. Checking the box's own corners pins down both the
    padding offset and the scale in one go."""
    rgb = _frame()
    _, side, off_x, off_y = _letterbox_crop(rgb, box)
    x0, y0, x1, y1 = box

    for want_px, want_py in [(x0, y0), (x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2)]:
        # Forward: frame pixel -> crop-normalised, as MediaPipe would report it.
        nx = (want_px - x0 + off_x) / side
        ny = (want_py - y0 + off_y) / side
        fx, fy = _crop_norm_to_frame_norm(nx, ny, box, side, off_x, off_y, FRAME_W, FRAME_H)
        assert fx * FRAME_W == pytest.approx(want_px, abs=1e-6)
        assert fy * FRAME_H == pytest.approx(want_py, abs=1e-6)


def test_crop_centre_maps_to_box_centre():
    """The single most likely place a padding bug hides: crop-space (0.5,
    0.5) is the centre of the *padded square*, which is the centre of the
    box only because padding is symmetric."""
    box = (100, 50, 200, 300)
    rgb = _frame()
    _, side, off_x, off_y = _letterbox_crop(rgb, box)
    fx, fy = _crop_norm_to_frame_norm(0.5, 0.5, box, side, off_x, off_y, FRAME_W, FRAME_H)
    cx, cy = _box_center(box, FRAME_W, FRAME_H)
    # Integer padding can be off by half a pixel on an odd-sized box.
    assert fx == pytest.approx(cx, abs=1.0 / FRAME_W)
    assert fy == pytest.approx(cy, abs=1.0 / FRAME_H)


def test_extrapolated_landmarks_are_not_clamped():
    """MediaPipe legitimately places landmarks outside its input (a
    speaker whose legs fall below the crop). Those must map to
    out-of-range frame coordinates rather than being folded onto the
    border, which would silently assert they were observed at the edge."""
    box = (100, 50, 200, 300)
    rgb = _frame()
    _, side, off_x, off_y = _letterbox_crop(rgb, box)

    x0, y0, x1, y1 = box
    # "Out of range" means outside the *box*, not necessarily outside the
    # frame — a point below a mid-frame crop can still land inside the
    # image, which is exactly the legs-below-the-crop case.
    fx, fy = _crop_norm_to_frame_norm(1.4, 1.4, box, side, off_x, off_y, FRAME_W, FRAME_H)
    assert fy * FRAME_H > y1, "below the crop must map below the box"
    assert fx * FRAME_W > x1, "right of the crop must map right of the box"

    fx, fy = _crop_norm_to_frame_norm(-0.3, -0.3, box, side, off_x, off_y, FRAME_W, FRAME_H)
    assert fy * FRAME_H < y0, "above the crop must map above the box"
    assert fx * FRAME_W < x0, "left of the crop must map left of the box"

    # And a landmark far enough below genuinely leaves the frame, unclamped.
    _, fy = _crop_norm_to_frame_norm(0.5, 2.0, box, side, off_x, off_y, FRAME_W, FRAME_H)
    assert fy > 1.0, "far below the crop must exceed the frame, not clamp to 1.0"


def test_box_center_is_frame_normalised():
    assert _box_center((0, 0, FRAME_W, FRAME_H), FRAME_W, FRAME_H) == (0.5, 0.5)
    assert _box_center((0, 0, 320, 240), FRAME_W, FRAME_H) == (0.25, 0.25)
