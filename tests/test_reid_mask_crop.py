"""
tests/test_reid_mask_crop.py
-----------------------------
Unit tests for crop_via_mask's box-local search.

`box` is an optimisation: it lets the mask's extent be found by scanning
inside the detection box instead of the whole frame (8.00ms -> 0.13ms per
candidate at 1080p). The contract is that it changes *nothing* about the
returned crop, so these tests are almost entirely equivalence tests —
box-local against full-frame, on the same mask.

This gets its own tests because getting it wrong fails silently. A crop
shifted by a few pixels, or clipped along one edge, is still a perfectly
valid-looking image; it just embeds to a slightly different 512-d vector,
and every downstream similarity comparison — gallery matching, the Locked
continuity check, gallery-building's redundancy test — quietly moves with
it. Nothing raises, and the symptom would surface as "re-ID got worse",
which is a long way from here.

The specific hazard the guard exists for: masks are built by cropping at
the detector's 640x640 input resolution and then upsampling to frame
resolution, while boxes are rounded to integer frame pixels separately, so
a mask genuinely can spill a little outside its own box (measured: up to
3px, always right/bottom). The margin covers the observed case and the
containment guard covers the unobserved one.
"""

from __future__ import annotations

import numpy as np
import pytest

from workers._reid import MIN_MASK_PIXELS, crop_via_mask

FRAME_W, FRAME_H = 640, 480


def _rgb():
    return np.random.default_rng(0).integers(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)


def _mask_rect(x0, y0, x1, y1):
    m = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _assert_same(rgb, mask, box):
    """The whole contract in one assertion: passing a box changes speed,
    not output."""
    without = crop_via_mask(rgb, mask)
    with_box = crop_via_mask(rgb, mask, box)
    assert (without is None) == (with_box is None)
    if without is not None:
        assert np.array_equal(without, with_box)
    return with_box


@pytest.mark.parametrize("rect", [
    (100, 50, 200, 300),      # an ordinary standing person
    (0, 0, 60, 200),          # flush against the top-left frame edge
    (580, 280, 640, 480),     # flush against the bottom-right frame edge
    (317, 201, 355, 296),     # odd dimensions
])
def test_box_local_matches_full_scan(rect):
    rgb = _rgb()
    mask = _mask_rect(*rect)
    assert _assert_same(rgb, mask, rect) is not None


@pytest.mark.parametrize("spill", [1, 3, 8])
def test_mask_spilling_outside_its_box_is_still_exact(spill):
    """The real-footage case: the mask extends past the box it came with,
    because mask and box are rounded to frame pixels by different paths.
    Within the margin this must still agree with a full scan."""
    rgb = _rgb()
    box = (100, 50, 200, 300)
    mask = _mask_rect(box[0], box[1], box[2] + spill, box[3] + spill)
    assert _assert_same(rgb, mask, box) is not None


def test_mask_far_outside_the_box_falls_back_rather_than_clipping():
    """Past the margin the box-local window would clip the mask, so the
    containment guard must reject the shortcut and rescan. Deliberately
    exercised well beyond the 3px worst case ever measured — the point is
    that exceeding the margin degrades to slow, never to wrong."""
    rgb = _rgb()
    box = (100, 50, 200, 300)
    mask = _mask_rect(box[0], box[1], box[2] + 40, box[3] + 40)
    crop = _assert_same(rgb, mask, box)
    # The full extent was found, not the window-clipped one.
    assert crop.shape[0] >= (box[3] + 40) - box[1]
    assert crop.shape[1] >= (box[2] + 40) - box[0]


def test_mask_disjoint_from_its_box():
    """A pathological pairing (mask nowhere near the box) must still return
    the mask's own crop, not None and not an empty region."""
    rgb = _rgb()
    mask = _mask_rect(400, 300, 500, 420)
    assert _assert_same(rgb, mask, (10, 10, 80, 90)) is not None


def test_too_few_pixels_rejected_identically():
    """The MIN_MASK_PIXELS rejection must not depend on which search ran."""
    rgb = _rgb()
    side = int(np.sqrt(MIN_MASK_PIXELS)) - 2      # comfortably under the floor
    box = (100, 50, 100 + side, 50 + side)
    mask = _mask_rect(*box)
    assert mask.sum() < MIN_MASK_PIXELS
    assert _assert_same(rgb, mask, box) is None


def test_empty_mask_rejected_identically():
    rgb = _rgb()
    mask = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    assert _assert_same(rgb, mask, (100, 50, 200, 300)) is None


def test_thin_mask_rejected_identically():
    """Under the 20px side floor — again, the same verdict either way."""
    rgb = _rgb()
    box = (100, 50, 108, 300)
    assert _assert_same(rgb, mask := _mask_rect(*box), box) is None
    assert mask.sum() >= MIN_MASK_PIXELS      # rejected on shape, not count


def test_background_is_zeroed_and_foreground_untouched():
    """Guards the actual purpose of the function, which the box-local path
    must not disturb: background pixels zeroed, subject pixels verbatim."""
    rgb = _rgb()
    box = (100, 50, 200, 300)
    mask = _mask_rect(*box)
    crop = crop_via_mask(rgb, mask, box)
    # The bbox is padded by 5px on each side, but xs.max()/ys.max() are
    # inclusive indices while the slice end is exclusive — so the trailing
    # pad is 4, not 5. Border either way is background.
    assert not crop[0].any() and not crop[:, 0].any()
    assert not crop[-1].any() and not crop[:, -1].any()
    # Interior matches the source frame exactly.
    assert np.array_equal(crop[5:-4, 5:-4], rgb[box[1]:box[3], box[0]:box[2]])


def test_mediapipe_shaped_mask_still_accepted():
    """MediaPipe hands back (H, W, 1); the detector hands back (H, W). The
    squeeze must keep working now the shape feeds a windowed index too."""
    rgb = _rgb()
    box = (100, 50, 200, 300)
    mask = _mask_rect(*box).astype(np.float32)[:, :, None]
    assert _assert_same(rgb, mask, box) is not None
