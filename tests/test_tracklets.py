"""
tests/test_tracklets.py
------------------------
Unit tests for tracklet association and selection.

These exist because the mechanism's two most important parameters are the
ones whose failure is silent and *inverted*:

  - Gap tolerance. The speaker's detectability swings with her pose
    (measured 0.64 with arms raised, 0.162 with them down). If association
    breaks her track at every missed detection she fragments into stubs
    while the static projection stays one clean track — and any criterion
    that prefers longer tracks then confidently selects the screen.
  - Minimum support. Same inversion: raise it above the length of her
    genuine track and the projection is the only thing left standing.

Both would produce a complete, plausible gesture track of the wrong
subject, with nothing raised anywhere. Hence
test_speaker_with_gaps_stays_one_tracklet and
test_short_speaker_track_still_beats_long_projection_track.

Boxes here are synthetic, so these run without models or video.
"""

from __future__ import annotations

import numpy as np
import pytest

from workers.gesture_worker import (
    _TRACK_MAX_GAP_FRAMES,
    _TRACK_MIN_SUPPORT,
    _GALLERY_MATCH_THRESHOLD,
    _associate,
    _iou,
    _select_tracklet,
)

W, H = 1000, 500
PASS = _GALLERY_MATCH_THRESHOLD + 0.05
PASS_HIGH = _GALLERY_MATCH_THRESHOLD + 0.15
FAIL = _GALLERY_MATCH_THRESHOLD - 0.20


def _box(cx, cy, half=30):
    """Box centred on frame-normalised (cx, cy)."""
    x, y = int(cx * W), int(cy * H)
    return (x - half, y - half, x + half, y + half)


def _scored(track, score):
    track.score = score
    return track


# ── association ──────────────────────────────────────────────────────────

def test_iou_basics():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert 0 < _iou((0, 0, 10, 10), (5, 0, 15, 10)) < 1


def test_a_stationary_subject_forms_one_tracklet():
    frames = [(i, [_box(0.3, 0.3)]) for i in range(20)]
    tracks = _associate(frames)
    assert len(tracks) == 1
    assert tracks[0].support == 20


def test_two_separated_subjects_form_two_tracklets():
    frames = [(i, [_box(0.2, 0.2), _box(0.8, 0.8)]) for i in range(15)]
    tracks = _associate(frames)
    assert len(tracks) == 2
    assert all(t.support == 15 for t in tracks)


def test_speaker_with_gaps_stays_one_tracklet():
    """The flicker case. She is detected, drops out for a run of frames,
    and reappears in the same place. That must remain ONE tracklet — if it
    fragments, her track loses to the projection's."""
    present = list(range(0, 5)) + list(range(20, 25)) + list(range(45, 60))
    frames = [(i, [_box(0.5, 0.5)] if i in present else []) for i in range(60)]
    tracks = _associate(frames)
    assert len(tracks) == 1, f"fragmented into {len(tracks)} tracklets"
    assert tracks[0].support == len(present)


def test_gap_longer_than_tolerance_starts_a_new_tracklet():
    gap = _TRACK_MAX_GAP_FRAMES + 5
    present = list(range(0, 5)) + list(range(5 + gap, 10 + gap))
    frames = [(i, [_box(0.5, 0.5)] if i in present else [])
              for i in range(10 + gap)]
    assert len(_associate(frames)) == 2


def test_a_moving_subject_is_followed():
    """She walks across the stage; consecutive boxes overlap, so it stays
    one track even though start and end do not overlap at all."""
    frames = [(i, [_box(0.2 + 0.01 * i, 0.5)]) for i in range(40)]
    tracks = _associate(frames)
    assert len(tracks) == 1
    assert tracks[0].support == 40


def test_crossing_subjects_do_not_merge_into_one():
    frames = []
    for i in range(30):
        frames.append((i, [_box(0.2, 0.2), _box(0.8, 0.8)]))
    tracks = _associate(frames)
    assert len(tracks) == 2


# ── selection ────────────────────────────────────────────────────────────

def _speaker_and_projection(speaker_frames, projection_frames):
    """Projection high in frame, speaker central and low — the measured
    geometry (speaker 0.05 from centre, projection 0.36)."""
    frames = []
    for i in range(max(speaker_frames + projection_frames) + 1):
        boxes = []
        if i in projection_frames:
            boxes.append(_box(0.28, 0.17))
        if i in speaker_frames:
            boxes.append(_box(0.45, 0.52))
        frames.append((i, boxes))
    return _associate(frames)


def test_central_tracklet_wins_over_higher_scoring_projection():
    tracks = _speaker_and_projection(list(range(60)), list(range(60)))
    assert len(tracks) == 2
    for t in tracks:
        # the higher score goes to the projection, as measured
        t.score = PASS_HIGH if t.median_centre_distance(W, H) > 0.25 else PASS
    chosen = _select_tracklet(tracks, W, H)
    assert chosen is not None
    assert chosen.median_centre_distance(W, H) < 0.25, "picked the projection"


def test_short_speaker_track_still_beats_long_projection_track():
    """She is detected for a fraction of the scene; the screen for all of
    it. Selection must not reward length."""
    speaker = list(range(20, 20 + _TRACK_MIN_SUPPORT + 2))
    tracks = _speaker_and_projection(speaker, list(range(200)))
    for t in tracks:
        t.score = PASS_HIGH if t.median_centre_distance(W, H) > 0.25 else PASS
    chosen = _select_tracklet(tracks, W, H)
    assert chosen is not None and chosen.support == len(speaker)


def test_single_passing_tracklet_returns_none():
    """One passer is not the ambiguous case — the caller should fall back
    to the ordinary per-frame path rather than this mechanism."""
    tracks = _speaker_and_projection(list(range(60)), list(range(60)))
    for t in tracks:
        t.score = PASS if t.median_centre_distance(W, H) < 0.25 else FAIL
    assert _select_tracklet(tracks, W, H) is None


def test_no_passing_tracklets_returns_none():
    tracks = _speaker_and_projection(list(range(60)), list(range(60)))
    for t in tracks:
        t.score = FAIL
    assert _select_tracklet(tracks, W, H) is None


def test_unscored_tracklets_are_ignored():
    tracks = _speaker_and_projection(list(range(60)), list(range(60)))
    assert _select_tracklet(tracks, W, H) is None


def test_tracklet_below_support_floor_cannot_be_selected():
    speaker = list(range(20, 20 + _TRACK_MIN_SUPPORT - 1))   # one short
    tracks = _speaker_and_projection(speaker, list(range(200)))
    for t in tracks:
        t.score = PASS
    chosen = _select_tracklet(tracks, W, H)
    # only the projection clears the floor -> not two passers -> fall back
    assert chosen is None


def test_median_not_mean_for_centrality():
    """A few frames at the frame edge must not decide a track that is
    otherwise central. Built directly rather than through _associate,
    because boxes that far apart would (correctly) fail to associate into
    one tracklet at all."""
    from workers.gesture_worker import _Tracklet
    t = _Tracklet(0, _box(0.95, 0.5))
    for i in range(1, 6):
        t.add(i, _box(0.95, 0.5))        # 6 outliers at the frame edge
    for i in range(6, 40):
        t.add(i, _box(0.5, 0.5))         # 34 central
    assert t.median_centre_distance(W, H) < 0.1
    mean = float(np.mean([
        abs(0.95 - 0.5) if i < 6 else 0.0 for i in range(40)
    ]))
    assert mean > 0.06, "the mean would have been dragged by the outliers"
