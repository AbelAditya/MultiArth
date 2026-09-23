"""
tests/test_gesture_params.py
-----------------------------
GestureParams and the pluggable tie-break.

The tests that matter here are the ones covering failures that would be
*silent*: a parameter that looks set but isn't read, a tie-break that picks
the wrong axis sign, and pool children falling back to defaults while the
parent uses overrides — that last one would process some of a job's windows
with different settings from the rest, and nothing in the output would say
so.
"""

from __future__ import annotations

import inspect

import pytest

from workers._gesture_params import DEFAULT_PARAMS, TIE_BREAKS, GestureParams
from workers.gesture_worker import (
    _Tracklet,
    _associate,
    _select_tracklet,
)

W, H = 1000, 500


def _box(cx, cy, half=30):
    x, y = int(cx * W), int(cy * H)
    return (x - half, y - half, x + half, y + half)


# ── the dataclass ────────────────────────────────────────────────────────

def test_defaults_match_the_values_the_pipeline_shipped_with():
    """These are the measured/tuned values documented in the worker; a
    change here silently changes every job that passes no params."""
    p = DEFAULT_PARAMS
    assert (p.conf_threshold, p.gallery_match_threshold, p.continuity_threshold) == (0.1, 0.80, 0.80)
    assert (p.lock_lease_frames, p.max_track_jump) == (30, 0.3)
    assert (p.track_iou_min, p.track_max_gap_frames, p.track_min_support) == (0.3, 30, 10)
    assert (p.track_id_stride, p.track_id_samples) == (10, 5)
    assert p.tie_break == "centrality"


def test_module_constants_still_mirror_the_defaults():
    """The worker keeps the old constant names for its docstrings; they
    must not drift from the dataclass."""
    import workers.gesture_worker as g
    assert g._GALLERY_MATCH_THRESHOLD == DEFAULT_PARAMS.gallery_match_threshold
    assert g._TRACK_MIN_SUPPORT == DEFAULT_PARAMS.track_min_support
    assert g._MAX_TRACK_JUMP == DEFAULT_PARAMS.max_track_jump


def test_params_are_frozen():
    with pytest.raises(Exception):
        DEFAULT_PARAMS.conf_threshold = 0.5


def test_unknown_tie_break_is_rejected():
    with pytest.raises(ValueError, match="expected one of"):
        GestureParams(tie_break="lowest")


def test_out_of_range_thresholds_are_rejected():
    with pytest.raises(ValueError, match="must be in"):
        GestureParams(gallery_match_threshold=1.4)
    with pytest.raises(ValueError, match="must be in"):
        GestureParams(conf_threshold=0.0)


def test_from_dict_rejects_a_typo_rather_than_ignoring_it():
    """A misspelled key that was silently dropped would leave you tuning a
    parameter that never took effect."""
    with pytest.raises(ValueError, match="gallery_threshold"):
        GestureParams.from_dict({"gallery_threshold": 0.9})


def test_from_dict_overrides_only_what_it_names():
    p = GestureParams.from_dict({"conf_threshold": 0.25, "tie_break": "vertical"})
    assert (p.conf_threshold, p.tie_break) == (0.25, "vertical")
    assert p.gallery_match_threshold == DEFAULT_PARAMS.gallery_match_threshold


def test_round_trips_through_dict_for_provenance():
    p = GestureParams(conf_threshold=0.25, tie_break="vertical")
    assert GestureParams.from_dict(p.to_dict()) == p


# ── tie-breaks ───────────────────────────────────────────────────────────

def test_centrality_prefers_the_central_candidate():
    key = TIE_BREAKS["centrality"]
    assert key((0.5, 0.5)) < key((0.5, 0.9)) < key((0.1, 0.95))


def test_vertical_prefers_the_candidate_lowest_in_frame():
    """Box coords are image coords (y down), so "lower in frame" is larger
    cy. Getting this sign backwards would select the projection every
    time — the exact failure the rule exists to prevent."""
    key = TIE_BREAKS["vertical"]
    speaker, projection = (0.5, 0.75), (0.5, 0.25)
    assert key(speaker) < key(projection)


def test_vertical_ignores_horizontal_position():
    """The point of the rule: it survives a panning shot, where x sweeps
    across the frame but the speaker's feet stay on the stage floor."""
    key = TIE_BREAKS["vertical"]
    assert key((0.05, 0.8)) == key((0.95, 0.8))


# ── selection through the tie-break ──────────────────────────────────────

def _two_tracks(speaker_centre, projection_centre, n=40):
    frames = [(i, [_box(*projection_centre), _box(*speaker_centre)]) for i in range(n)]
    tracks = _associate(frames)
    assert len(tracks) == 2
    for t in tracks:
        t.score = DEFAULT_PARAMS.gallery_match_threshold + 0.05
    return tracks


def test_centred_screen_defeats_centrality_but_not_vertical():
    """The motivating video: the projection is dead centre, the speaker
    below and to the side. Centrality picks the screen; vertical picks
    her."""
    speaker, projection = (0.35, 0.78), (0.5, 0.42)
    tracks = _two_tracks(speaker, projection)

    by_centrality = _select_tracklet(tracks, W, H, GestureParams(tie_break="centrality"))
    by_vertical = _select_tracklet(tracks, W, H, GestureParams(tie_break="vertical"))

    def centre_y(t):
        return t.median_key(W, H, lambda c: c[1])

    assert centre_y(by_centrality) == pytest.approx(projection[1], abs=0.02)
    assert centre_y(by_vertical) == pytest.approx(speaker[1], abs=0.02)


def test_vertical_still_needs_two_gallery_passers():
    """The tie-break is only ever consulted among candidates that already
    cleared the gallery — it cannot rescue a scene where nobody did."""
    tracks = _two_tracks((0.35, 0.78), (0.5, 0.42))
    for t in tracks:
        t.score = DEFAULT_PARAMS.gallery_match_threshold - 0.2
    assert _select_tracklet(tracks, W, H, GestureParams(tie_break="vertical")) is None


def test_params_change_association_behaviour():
    """A parameter is only real if the code reads it: with the default gap
    tolerance the two halves join, with a tiny one they don't."""
    present = list(range(0, 5)) + list(range(20, 25))
    frames = [(i, [_box(0.5, 0.5)] if i in present else []) for i in range(25)]
    assert len(_associate(frames, GestureParams(track_max_gap_frames=30))) == 1
    assert len(_associate(frames, GestureParams(track_max_gap_frames=2))) == 2


def test_support_floor_is_read_from_params():
    tracks = _two_tracks((0.35, 0.78), (0.5, 0.42), n=12)
    assert _select_tracklet(tracks, W, H, GestureParams(track_min_support=10)) is not None
    assert _select_tracklet(tracks, W, H, GestureParams(track_min_support=20)) is None


# ── plumbing ─────────────────────────────────────────────────────────────

def test_worker_defaults_to_the_shared_defaults():
    from workers.gesture_worker import GestureWorker
    assert GestureWorker(store=None).params is DEFAULT_PARAMS
    p = GestureParams(tie_break="vertical")
    assert GestureWorker(store=None, params=p).params is p


def test_pool_init_accepts_params_so_children_cannot_fall_back():
    """A spawned child imports the module fresh. If params did not travel
    through _pool_init, children would use defaults while the parent used
    overrides — different settings across one job's windows, with nothing
    in the output to show it."""
    from workers.gesture_worker import _pool_init
    sig = inspect.signature(_pool_init)
    assert "params" in sig.parameters


def test_params_survive_pickling():
    """initargs are pickled to reach a spawned child."""
    import pickle
    p = GestureParams(conf_threshold=0.25, tie_break="vertical")
    assert pickle.loads(pickle.dumps(p)) == p


# ── MediaPipe pose thresholds ────────────────────────────────────────────

def test_pose_thresholds_default_to_the_shipped_values():
    assert DEFAULT_PARAMS.pose_detection_confidence == 0.60   # not MediaPipe's 0.5
    assert DEFAULT_PARAMS.pose_presence_confidence == 0.5


def test_pose_thresholds_are_range_checked():
    with pytest.raises(ValueError, match="must be in"):
        GestureParams(pose_detection_confidence=1.5)


def test_no_tracking_confidence_field():
    """min_tracking_confidence applies only in VIDEO/LIVE_STREAM mode and
    this landmarker runs in IMAGE mode, so a field for it would be a knob
    that does nothing."""
    assert "min_tracking_confidence" not in GestureParams.__dataclass_fields__
    assert "tracking_confidence" not in GestureParams.__dataclass_fields__


def test_landmarker_is_built_from_params(monkeypatch):
    """The value must actually reach PoseLandmarkerOptions — a param read
    from nowhere is the failure this class exists to prevent."""
    from workers.gesture_worker import GestureWorker
    import workers.gesture_worker as g

    captured = {}

    class FakeOptions:
        def __init__(self, **kw):
            captured.update(kw)

    class FakeLandmarker:
        @staticmethod
        def create_from_options(opts):
            return "landmarker"

    class FakeVision:
        PoseLandmarkerOptions = FakeOptions
        PoseLandmarker = FakeLandmarker
        class RunningMode:
            IMAGE = "IMAGE"

    import sys, types
    fake_mp = types.ModuleType("mediapipe")
    fake_tasks = types.ModuleType("mediapipe.tasks")
    fake_python = types.ModuleType("mediapipe.tasks.python")
    fake_python.BaseOptions = lambda **kw: kw
    fake_vision = types.ModuleType("mediapipe.tasks.python.vision")
    for name in ("PoseLandmarkerOptions", "PoseLandmarker", "RunningMode"):
        setattr(fake_vision, name, getattr(FakeVision, name))
    fake_tasks.python = fake_python
    fake_python.vision = fake_vision
    fake_mp.tasks = fake_tasks
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks", fake_tasks)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks.python", fake_python)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks.python.vision", fake_vision)
    monkeypatch.setattr(g, "_ensure_model", lambda *a, **k: None)

    params = GestureParams(pose_detection_confidence=0.9, pose_presence_confidence=0.7)
    GestureWorker(store=None, params=params)._open_landmarker()

    assert captured["min_pose_detection_confidence"] == 0.9
    assert captured["min_pose_presence_confidence"] == 0.7
