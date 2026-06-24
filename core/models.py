"""
core/models.py
--------------
Shared Pydantic data models for inter-worker communication.
All workers serialize their output into these structures before
pushing onto the Redis feature store.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

class TimeWindow(BaseModel):
    """Half-open interval [start_s, end_s) in seconds."""
    start_s: float
    end_s: float

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s

    @property
    def midpoint(self) -> float:
        return (self.start_s + self.end_s) / 2.0


# ---------------------------------------------------------------------------
# Gesture features  (MediaPipe)
# ---------------------------------------------------------------------------

class Landmark(BaseModel):
    x: float
    y: float
    z: float
    visibility: float = 1.0


class GestureFrame(BaseModel):
    """Pose + hand keypoints for a single frame."""
    frame_idx: int
    timestamp_s: float
    pose: list[Landmark]          # 33 MediaPipe pose landmarks
    left_hand: list[Landmark]     # 21 landmarks, empty list if not detected
    right_hand: list[Landmark]    # 21 landmarks, empty list if not detected


class PoseKeyframe(BaseModel):
    """Lightweight normalised pose snapshot for the timeline animation viewer."""
    ts: float                  # timestamp in seconds
    pose_x: list[float]        # 33 normalised x-coords [0, 1]
    pose_y: list[float]        # 33 normalised y-coords, already flipped (1 − raw_y) so up = high
    pose_vis: list[float]      # 33 visibility scores


class GestureFeatures(BaseModel):
    """Aggregated gesture features over a time window."""
    window: TimeWindow
    mean_wrist_velocity: float             # pixels/s, averaged L+R
    max_wrist_displacement: float          # peak displacement in window
    pose_present_ratio: float              # fraction of frames pose was detected
    # Handedness: −1 = fully left-dominant, 0 = bilateral, +1 = fully right-dominant
    handedness_ratio: float = 0.0
    # Per-frame pose data (subsampled) for the animated pose viewer
    pose_keyframes: list[PoseKeyframe] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pitch / auditory tone features  (parselmouth)
# ---------------------------------------------------------------------------

class PitchFrame(BaseModel):
    """F0 sample at a single time point."""
    timestamp_s: float
    f0_hz: Optional[float]   # None = unvoiced
    intensity_db: float


class ProsodyFeatures(BaseModel):
    """Aggregated prosodic features over a time window."""
    window: TimeWindow
    mean_f0: Optional[float]
    f0_range: Optional[float]              # max – min over voiced frames
    f0_std: Optional[float]
    voiced_fraction: float                 # fraction of frames that are voiced
    mean_intensity_db: float
    intensity_range_db: float
    speech_rate_syl_per_s: Optional[float] # populated after alignment with ASR
    jitter_local: Optional[float]          # cycle-to-cycle F0 variation
    shimmer_local: Optional[float]         # cycle-to-cycle amplitude variation
    hnr_db: Optional[float]               # harmonics-to-noise ratio


# ---------------------------------------------------------------------------
# Verbal language features  (faster-whisper + spaCy)
# ---------------------------------------------------------------------------

class WordToken(BaseModel):
    word: str
    start_s: float
    end_s: float
    confidence: float


class VerbalFeatures(BaseModel):
    """Aggregated verbal features over a time window."""
    window: TimeWindow
    transcript: str
    tokens: list[WordToken]
    word_count: int
    filler_word_count: int                 # uh, um, like, you know, etc.
    filler_word_rate: float                # fillers per minute
    mean_word_confidence: float
    sentence_count: int
    mean_sentence_length: float            # words per sentence
    type_token_ratio: float                # vocabulary richness proxy
    hedge_count: int                       # maybe, perhaps, kind of, etc.
    pause_count: int                       # inter-word gaps > 250 ms
    mean_pause_duration_s: float


# ---------------------------------------------------------------------------
# Camera / scene features  (PySceneDetect)
# ---------------------------------------------------------------------------

class ShotType(str, Enum):
    UNKNOWN          = "unknown"
    EXTREME_CLOSE_UP = "extreme_close_up"  # partial face / isolated detail
    CLOSE_UP         = "close_up"           # head and shoulders
    MEDIUM_CLOSE     = "medium_close"       # cut at waist
    MEDIUM           = "medium"             # cut at knees
    MEDIUM_LONG      = "medium_long"        # full figure, ankles visible
    LONG             = "long"               # full figure, person ≥40% frame height
    VERY_LONG        = "very_long"          # full figure, person <40% frame height


class SceneCut(BaseModel):
    timestamp_s: float
    frame_idx: int
    cut_score: float               # PySceneDetect content score


class CameraFeatures(BaseModel):
    """Camera/editorial features over a time window."""
    window: TimeWindow
    scene_cuts: list[SceneCut]
    cut_count: int
    cut_rate: float                        # cuts per minute
    dominant_shot_type: ShotType
    mean_face_bbox_area: Optional[float]  # proxy for zoom level; None if undetected
    face_bbox_trend: Optional[float]      # positive = zooming in


# ---------------------------------------------------------------------------
# Fused per-window record
# ---------------------------------------------------------------------------

class FusedWindow(BaseModel):
    """All four modalities merged for a single time window."""
    window: TimeWindow
    gesture: Optional[GestureFeatures] = None
    prosody: Optional[ProsodyFeatures] = None
    verbal: Optional[VerbalFeatures] = None
    camera: Optional[CameraFeatures] = None

    def is_complete(self) -> bool:
        return all([
            self.gesture is not None,
            self.prosody is not None,
            self.verbal is not None,
            self.camera is not None,
        ])


# ---------------------------------------------------------------------------
# Job / status models
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class AnalysisJob(BaseModel):
    job_id: str
    video_path: str
    window_size_s: float = Field(default=5.0, description="Aggregation window in seconds")
    status: JobStatus = JobStatus.PENDING
    error: Optional[str] = None
    created_at: float = Field(default=0.0)
    completed_at: Optional[float] = None
