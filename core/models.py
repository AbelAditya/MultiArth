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
# Gesture features  (MediaPipe BlazePose — 33 landmarks; see
# workers/gesture_worker.py. Main branch uses MeTRAbs's 19-landmark
# coco_19 instead — this shape is `light-gesture`-branch specific.)
# ---------------------------------------------------------------------------

class Landmark(BaseModel):
    x: float
    y: float
    z: float
    visibility: float = 1.0


class GestureFrame(BaseModel):
    """Pose keypoints for a single frame."""
    frame_idx: int
    timestamp_s: float
    pose: list[Landmark]          # 33 MediaPipe BlazePose landmarks (image-space pixels)
    left_hand: list[Landmark]     # unused (HandLandmarker was tried and removed again —
    right_hand: list[Landmark]    # see gesture_worker.py's module docstring; always empty)
    pose_world: list[Landmark] = Field(default_factory=list)  # 33 metric-ish, hip-relative world-space coords, meters — see gesture_worker.py's "World coordinates"


class PoseKeyframe(BaseModel):
    """Lightweight normalised pose snapshot for the timeline animation viewer."""
    ts: float                  # timestamp in seconds
    pose_x: list[float]        # 33 normalised x-coords [0, 1] (main branch: 19, MeTRAbs coco_19)
    pose_y: list[float]        # 33 normalised y-coords, already flipped (1 − raw_y) so up = high
    pose_vis: list[float]      # 33 MediaPipe visibility scores (real per-landmark confidence,
                                # not the 0.0/1.0 in-frame pseudo-visibility MeTRAbs needed —
                                # see gesture_worker.py's "Per-landmark confidence")
    world_x: Optional[list[float]] = None  # 33 metric world x-coords, meters (right positive)
    world_y: Optional[list[float]] = None  # 33 metric world y-coords, meters (down positive)
    world_z: Optional[list[float]] = None  # 33 metric world z-coords, meters (away from camera positive)


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
    mean_intensity_db: float
    intensity_range_db: float
    speech_rate_syl_per_s: Optional[float] # populated after alignment with ASR


# ---------------------------------------------------------------------------
# Verbal language features  (faster-whisper + spaCy)
# ---------------------------------------------------------------------------

class WordToken(BaseModel):
    word: str
    start_s: float
    end_s: float
    confidence: float


class VerbalFeatures(BaseModel):
    """Transcription and token data for a time window."""
    window: TimeWindow
    transcript: str
    tokens: list[WordToken]
    word_count: int


# ---------------------------------------------------------------------------
# Camera / scene features  (PySceneDetect)
# ---------------------------------------------------------------------------

class HorizontalAngle(str, Enum):
    UNKNOWN  = "unknown"
    FRONTAL  = "frontal"   # |yaw| < 30° — subject faces roughly toward camera
    OBLIQUE  = "oblique"   # |yaw| ≥ 30° — body turned away from camera


class VerticalAngle(str, Enum):
    UNKNOWN   = "unknown"
    HIGH      = "high"       # pitch > 10°  — camera above, subject looks up
    EYE_LEVEL = "eye_level"  # |pitch| ≤ 10° — camera roughly level with face
    LOW       = "low"        # pitch < -10° — camera below, subject looks down


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
    mean_shoulder_yaw_deg: Optional[float] = None   # 0° = frontal, ±90° = profile
    horizontal_angle: HorizontalAngle = HorizontalAngle.UNKNOWN
    mean_face_pitch_deg: Optional[float] = None     # positive = HIGH, negative = LOW
    vertical_angle: VerticalAngle = VerticalAngle.UNKNOWN


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
    total_windows: Optional[int] = None
    status: JobStatus = JobStatus.PENDING
    error: Optional[str] = None
    created_at: float = Field(default=0.0)
    completed_at: Optional[float] = None
