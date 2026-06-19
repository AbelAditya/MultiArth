"""
tests/test_models.py
--------------------
Unit tests for core data models and feature store serialisation.
Run with: uv run pytest tests/
"""

import pytest
from core.models import (
    FusedWindow,
    GestureFeatures,
    JobStatus,
    Landmark,
    ProsodyFeatures,
    TimeWindow,
    VerbalFeatures,
    CameraFeatures,
    ShotType,
)


def make_window(start=0.0, end=5.0) -> TimeWindow:
    return TimeWindow(start_s=start, end_s=end)


class TestTimeWindow:
    def test_duration(self):
        w = make_window(0.0, 5.0)
        assert w.duration == pytest.approx(5.0)

    def test_midpoint(self):
        w = make_window(0.0, 10.0)
        assert w.midpoint == pytest.approx(5.0)


class TestGestureFeatures:
    def test_serialise_roundtrip(self):
        gf = GestureFeatures(
            window=make_window(),
            mean_wrist_velocity=12.5,
            max_wrist_displacement=80.0,
            gesture_amplitude=120.0,
            bilateral_symmetry_score=0.75,
            hands_above_shoulder_ratio=0.3,
            gesture_rate=1.2,
            pose_present_ratio=0.95,
        )
        raw = gf.model_dump_json()
        restored = GestureFeatures.model_validate_json(raw)
        assert restored.mean_wrist_velocity == pytest.approx(12.5)
        assert restored.pose_present_ratio == pytest.approx(0.95)


class TestProsodyFeatures:
    def test_none_fields_allowed(self):
        pf = ProsodyFeatures(
            window=make_window(),
            mean_f0=None,
            f0_range=None,
            f0_std=None,
            voiced_fraction=0.0,
            mean_intensity_db=-60.0,
            intensity_range_db=0.0,
        )
        assert pf.mean_f0 is None
        assert pf.voiced_fraction == 0.0

    def test_serialise_roundtrip(self):
        pf = ProsodyFeatures(
            window=make_window(),
            mean_f0=180.5,
            f0_range=40.0,
            f0_std=15.2,
            voiced_fraction=0.8,
            mean_intensity_db=65.0,
            intensity_range_db=20.0,
            speech_rate_syl_per_s=4.5,
            jitter_local=0.02,
            shimmer_local=0.05,
            hnr_db=18.0,
        )
        raw = pf.model_dump_json()
        restored = ProsodyFeatures.model_validate_json(raw)
        assert restored.mean_f0 == pytest.approx(180.5)
        assert restored.hnr_db == pytest.approx(18.0)


class TestFusedWindow:
    def test_is_complete_false_when_missing(self):
        fw = FusedWindow(window=make_window())
        assert not fw.is_complete()

    def test_is_complete_true(self):
        fw = FusedWindow(
            window=make_window(),
            gesture=GestureFeatures(
                window=make_window(),
                mean_wrist_velocity=0,
                max_wrist_displacement=0,
                gesture_amplitude=0,
                bilateral_symmetry_score=0,
                hands_above_shoulder_ratio=0,
                gesture_rate=0,
                pose_present_ratio=0,
            ),
            prosody=ProsodyFeatures(
                window=make_window(),
                voiced_fraction=0,
                mean_intensity_db=0,
                intensity_range_db=0,
            ),
            verbal=VerbalFeatures(
                window=make_window(),
                transcript="hello",
                tokens=[],
                word_count=1,
                filler_word_count=0,
                filler_word_rate=0,
                mean_word_confidence=1.0,
                sentence_count=1,
                mean_sentence_length=1.0,
                type_token_ratio=1.0,
                hedge_count=0,
                pause_count=0,
                mean_pause_duration_s=0.0,
            ),
            camera=CameraFeatures(
                window=make_window(),
                scene_cuts=[],
                cut_count=0,
                cut_rate=0,
                dominant_shot_type=ShotType.UNKNOWN,
            ),
        )
        assert fw.is_complete()
