"""
tests/test_gesture_crop_mapping.py
-----------------------------------
Unit tests for the detector-first pipeline's coordinate transform:
_square_crop takes a square region of the frame around a detection box,
MediaPipe returns landmarks normalised to that region, and
_crop_norm_to_frame_norm maps them back to frame-normalised coordinates.

This arithmetic gets its own tests because it is the one part of the
detector-first change that fails *silently*. A wrong origin or a wrong
crop size produces landmarks that are wrong but entirely
plausible-looking — a skeleton slightly beside the person, with nothing
raised anywhere and no visibly broken output to notice. Every other
failure mode in that pipeline announces itself (a missing model raises, a
shape mismatch raises, a bad mask is visible on screen); this one does not.

These replaced tests written against an earlier *letterbox* crop, which
cut the box out exactly and padded it to square with black. That was
found to starve MediaPipe on tall, thin subjects — 75% of its input was
black for a 76x302 box — and returned no pose at all on 25 consecutive
frames that every other stage had passed cleanly. See _square_crop.
"""

from __future__ import annotations

import numpy as np
import pytest

from workers.gesture_worker import (
    _POSE_CROP_SCALE,
    _box_center,
    _crop_norm_to_frame_norm,
    _square_crop,
)

FRAME_W, FRAME_H = 640, 480


def _frame():
    return np.random.default_rng(0).integers(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)


@pytest.mark.parametrize("box", [
    (100, 50, 200, 300),    # tall (a standing person)
    (10, 10, 400, 60),      # wide
    (0, 0, 64, 64),         # already square, at the origin
    (600, 440, 640, 480),   # flush against the bottom-right frame edge
    (0, 200, 30, 460),      # very tall and thin, against the left edge
])
def test_crop_is_square_and_inside_the_frame(box):
    """The region must stay square and lie wholly within the frame — it is
    slid inward near an edge, never clipped, so it never contains padding."""
    rgb = _frame()
    crop, ox, oy = _square_crop(rgb, box)
    ch, cw = crop.shape[:2]

    assert cw == ch, "crop must be square"
    assert 0 <= ox and 0 <= oy
    assert ox + cw <= FRAME_W and oy + ch <= FRAME_H, "crop must lie inside the frame"


@pytest.mark.parametrize("box", [
    (100, 50, 200, 300),
    (10, 10, 400, 60),
    (0, 0, 64, 64),
    (317, 201, 355, 296),   # odd dimensions
])
def test_crop_contains_real_pixels_not_padding(box):
    """The whole point of the change: every pixel comes from the frame, so
    the crop is a genuine sub-image. Checked by matching it against the
    source region rather than trusting the offsets."""
    rgb = _frame()
    crop, ox, oy = _square_crop(rgb, box)
    ch, cw = crop.shape[:2]
    assert np.array_equal(crop, rgb[oy:oy + ch, ox:ox + cw])


@pytest.mark.parametrize("box", [
    (100, 50, 200, 300),
    (10, 10, 400, 60),
    (0, 0, 64, 64),
    (317, 201, 355, 296),
    (600, 440, 640, 480),
])
def test_roundtrip_known_points(box):
    """A landmark at a known frame pixel must map back to that same pixel.
    Pins down the origin and the scale together."""
    rgb = _frame()
    crop, ox, oy = _square_crop(rgb, box)
    ch, cw = crop.shape[:2]

    for want_px, want_py in [(ox, oy), (ox + cw, oy + ch),
                             (ox + cw / 2, oy + ch / 2)]:
        nx = (want_px - ox) / cw          # forward: frame px -> crop-normalised
        ny = (want_py - oy) / ch
        fx, fy = _crop_norm_to_frame_norm(nx, ny, ox, oy, cw, ch, FRAME_W, FRAME_H)
        assert fx * FRAME_W == pytest.approx(want_px, abs=1e-6)
        assert fy * FRAME_H == pytest.approx(want_py, abs=1e-6)


def test_crop_expands_beyond_the_box():
    """The crop must be meaningfully larger than the detection box — that
    surrounding context is what MediaPipe needs. A 1.0x region of the image
    is exactly as large as the old padded square and still recovered 0/25
    failing frames, so this margin is load-bearing, not cosmetic."""
    box = (100, 50, 200, 300)               # 100 x 250
    crop, _, _ = _square_crop(_frame(), box)
    longest = max(box[2] - box[0], box[3] - box[1])
    assert crop.shape[0] >= longest * (_POSE_CROP_SCALE - 0.01)


def test_box_fits_inside_the_crop():
    """The detection must actually be contained in what MediaPipe sees —
    otherwise the crop is centred somewhere useless."""
    for box in [(100, 50, 200, 300), (317, 201, 355, 296), (0, 0, 64, 64)]:
        crop, ox, oy = _square_crop(_frame(), box)
        ch, cw = crop.shape[:2]
        assert ox <= box[0] and oy <= box[1]
        assert ox + cw >= box[2] and oy + ch >= box[3]


def test_extrapolated_landmarks_are_not_clamped():
    """MediaPipe legitimately places landmarks outside its input (a speaker
    whose legs fall below the crop). Those must map to out-of-range
    coordinates rather than being folded onto the border, which would
    silently assert they were observed at the edge."""
    box = (100, 50, 200, 300)
    crop, ox, oy = _square_crop(_frame(), box)
    ch, cw = crop.shape[:2]

    fx, fy = _crop_norm_to_frame_norm(1.4, 1.4, ox, oy, cw, ch, FRAME_W, FRAME_H)
    assert fx * FRAME_W > ox + cw, "right of the crop must map right of it"
    assert fy * FRAME_H > oy + ch, "below the crop must map below it"

    fx, fy = _crop_norm_to_frame_norm(-0.3, -0.3, ox, oy, cw, ch, FRAME_W, FRAME_H)
    assert fx * FRAME_W < ox, "left of the crop must map left of it"
    assert fy * FRAME_H < oy, "above the crop must map above it"


def test_crop_larger_than_frame_is_truncated_not_padded():
    """A box whose expanded square exceeds the frame gets truncated, so the
    caller must use the crop's real shape. Guards the assumption
    _pose_on_crop relies on."""
    rgb = _frame()
    box = (10, 10, 630, 470)                # expanded, this exceeds 480
    crop, ox, oy = _square_crop(rgb, box)
    ch, cw = crop.shape[:2]
    assert max(ch, cw) <= min(FRAME_W, FRAME_H)
    assert np.array_equal(crop, rgb[oy:oy + ch, ox:ox + cw])


def test_box_center_is_frame_normalised():
    assert _box_center((0, 0, FRAME_W, FRAME_H), FRAME_W, FRAME_H) == (0.5, 0.5)
    assert _box_center((0, 0, 320, 240), FRAME_W, FRAME_H) == (0.25, 0.25)
