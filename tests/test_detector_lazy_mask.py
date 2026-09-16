"""
tests/test_detector_lazy_mask.py
---------------------------------
Unit tests for PersonDetection's lazy mask.

`mask` is assembled on first access instead of up front, because the
runtime consumer usually wants one of them: workers/gesture_worker.py in
its Locked state embeds a single candidate and discards the rest, so on a
crowded frame the eager version built up to twenty masks (~3.9ms each) to
use one.

The interesting failure here is not "the mask is missing" — that would
raise immediately and loudly. It is that a closure over a loop variable
binds *late*, so every detection ends up sharing the last one's mask: still
a full set of plausible masks, one per detection, simply attached to the
wrong people. Nothing raises, the detector's own parity test still passes
(it matches each detection to its nearest reference box, and the boxes are
untouched), and the visible symptom is re-ID quietly scoring the wrong
crops. Hence test_each_detection_gets_its_own_mask, which is the whole
reason detect_people uses functools.partial rather than a lambda.

Real-frame agreement against Ultralytics lives in test_detector_parity.py,
which needs the export-time-only dependency; these tests are synthetic so
they run everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from workers._detector import PersonDetection


def test_builder_is_not_called_until_mask_is_accessed():
    calls = []

    def build():
        calls.append(1)
        return np.zeros((4, 4), dtype=bool)

    det = PersonDetection(box=(0, 0, 4, 4), score=0.9, mask_builder=build)
    assert calls == [], "constructing a detection must not assemble its mask"
    det.mask
    assert calls == [1]


def test_mask_is_cached_after_first_access():
    """Callers legitimately read .mask more than once (the parity test does
    it twice per detection). It must not be rebuilt each time."""
    calls = []

    def build():
        calls.append(1)
        return np.ones((4, 4), dtype=bool)

    det = PersonDetection(box=(0, 0, 4, 4), score=0.9, mask_builder=build)
    first = det.mask
    second = det.mask
    assert calls == [1], "mask was reassembled on second access"
    assert first is second


def test_each_detection_gets_its_own_mask():
    """The late-binding trap. Built the way detect_people builds them —
    one detection per loop iteration — each must keep *its* mask."""
    from functools import partial

    def build(i):
        m = np.zeros((4, 4), dtype=bool)
        m[i, i] = True
        return m

    dets = [
        PersonDetection(box=(i, i, i + 4, i + 4), score=0.5, mask_builder=partial(build, i))
        for i in range(4)
    ]
    # Access in reverse, so an order-dependent cache bug shows up too.
    for i in reversed(range(4)):
        assert dets[i].mask[i, i], f"detection {i} got another detection's mask"
        assert dets[i].mask.sum() == 1


def test_box_and_score_are_plain_attributes():
    det = PersonDetection(box=(1, 2, 3, 4), score=0.75, mask_builder=lambda: None)
    assert det.box == (1, 2, 3, 4)
    assert det.score == 0.75


def test_repr_does_not_build_the_mask():
    """Logging or debugging a detection must not silently pay for a mask —
    that would defeat the point on exactly the crowded frames it exists
    for."""
    calls = []
    det = PersonDetection(
        box=(0, 0, 4, 4), score=0.9,
        mask_builder=lambda: (calls.append(1), np.zeros((4, 4), dtype=bool))[1],
    )
    assert "unbuilt" in repr(det)
    assert calls == []
    det.mask
    assert "built" in repr(det)
