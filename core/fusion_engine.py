"""
core/fusion_engine.py
---------------------
Merges per-window outputs from all four workers into FusedWindow
records, then enriches them with cross-modal derived features
(e.g. speech rate backfilled into prosody, emphasis detection).
"""

from __future__ import annotations

import math

import numpy as np
from loguru import logger

from core.feature_store import FeatureStore
from core.models import (
    FusedWindow, HorizontalAngle, PoseKeyframe, ShotType, TimeWindow, VerticalAngle,
)


def _classify_shot_from_pose(keyframes: list[PoseKeyframe]) -> ShotType:
    """
    Determine shot type from which MediaPipe body landmarks are consistently
    visible across keyframes.

    Landmark y-coords in PoseKeyframe are stored pre-flipped as (1 - raw_y),
    so pose_y=1.0 is the top of the frame and pose_y=0.0 is the bottom.

    Classification ladder (most inclusive wins):
      feet (31,32) visible + tall person  → LONG
      feet visible + small person         → VERY_LONG
      ankles (27,28) visible              → MEDIUM_LONG
      knees (25,26) visible               → MEDIUM
      hips  (23,24) visible               → MEDIUM_CLOSE
      shoulders (11,12) visible           → CLOSE_UP
      nose  (0) only                      → EXTREME_CLOSE_UP
      nothing detected                    → UNKNOWN
    """
    n = len(keyframes)
    if n == 0:
        return ShotType.UNKNOWN

    VIS_MIN    = 0.5   # MediaPipe visibility threshold
    IN_FRAME_Y = 0.05  # landmark must be >5% above bottom edge (pose_y > 0)
    THRESH     = 0.4   # fraction of frames the landmark must be visible

    def ratio(indices: list[int]) -> float:
        count = 0
        for kf in keyframes:
            if all(
                i < len(kf.pose_vis)
                and kf.pose_vis[i] > VIS_MIN
                and kf.pose_y[i] > IN_FRAME_Y
                for i in indices
            ):
                count += 1
        return count / n

    feet_r     = ratio([31, 32])
    ankle_r    = ratio([27, 28])
    knee_r     = ratio([25, 26])
    hip_r      = ratio([23, 24])
    shoulder_r = ratio([11, 12])
    nose_r     = ratio([0])

    if shoulder_r < THRESH and nose_r < THRESH:
        return ShotType.UNKNOWN

    if feet_r >= THRESH:
        heights = []
        for kf in keyframes:
            if kf.pose_vis[0] > VIS_MIN:
                foot_ys = [
                    kf.pose_y[i] for i in [31, 32]
                    if i < len(kf.pose_vis) and kf.pose_vis[i] > VIS_MIN
                ]
                if foot_ys:
                    heights.append(kf.pose_y[0] - min(foot_ys))
        mean_height = float(np.mean(heights)) if heights else 0.5
        return ShotType.LONG if mean_height >= 0.4 else ShotType.VERY_LONG

    if ankle_r   >= THRESH: return ShotType.MEDIUM_LONG
    if knee_r    >= THRESH: return ShotType.MEDIUM
    if hip_r     >= THRESH: return ShotType.MEDIUM_CLOSE
    if shoulder_r >= THRESH: return ShotType.CLOSE_UP
    return ShotType.EXTREME_CLOSE_UP


def _compute_angles_from_pose(
    keyframes: list[PoseKeyframe],
) -> tuple[float | None, float | None]:
    """
    Returns (mean_shoulder_yaw_deg, mean_face_pitch_deg) across keyframes.

    Shoulder yaw — angle of the shoulder vector in the XZ plane:
      0° = subject faces camera directly, ±90° = pure profile.
    Face pitch — elevation of nose relative to mid-ear anchor:
      positive = subject looking up (HIGH angle), negative = looking down (LOW).

    MediaPipe world_y is downward-positive, so the pitch sign is negated to
    give an intuitive result.
    """
    yaws: list[float] = []
    pitches: list[float] = []

    for kf in keyframes:
        if kf.world_x is None or len(kf.world_x) < 33:
            continue

        wx, wy, wz = kf.world_x, kf.world_y, kf.world_z

        # Shoulder yaw (landmarks 11 = left, 12 = right)
        dx = wx[12] - wx[11]
        dz = wz[12] - wz[11]
        if abs(dx) + abs(dz) > 1e-4:
            yaws.append(math.degrees(math.atan2(dz, dx)))

        # Face pitch (nose=0, left_ear=7, right_ear=8)
        ear_mid_x = (wx[7] + wx[8]) / 2
        ear_mid_y = (wy[7] + wy[8]) / 2
        ear_mid_z = (wz[7] + wz[8]) / 2
        dy = wy[0] - ear_mid_y
        xz_dist = math.sqrt((wx[0] - ear_mid_x) ** 2 + (wz[0] - ear_mid_z) ** 2)
        if xz_dist > 1e-4:
            # Negate because world y is down-positive; positive result = looking up
            pitches.append(-math.degrees(math.atan2(dy, xz_dist)))

    mean_yaw = float(np.mean(yaws)) if yaws else None
    mean_pitch = float(np.mean(pitches)) if pitches else None
    return mean_yaw, mean_pitch


class FusionEngine:
    def __init__(self, store: FeatureStore):
        self.store = store

    def fuse_job(self, job_id: str, num_windows: int) -> list[FusedWindow]:
        logger.info(f"[fusion] Fusing {num_windows} windows for job {job_id}")
        results: list[FusedWindow] = []

        for idx in range(num_windows):
            gesture = self.store.get_gesture(job_id, idx)
            prosody = self.store.get_prosody(job_id, idx)
            verbal = self.store.get_verbal(job_id, idx)
            camera = self.store.get_camera(job_id, idx)

            # Derive a window from whichever modality is available
            window = self._resolve_window(gesture, prosody, verbal, camera, idx)

            fused = FusedWindow(
                window=window,
                gesture=gesture,
                prosody=prosody,
                verbal=verbal,
                camera=camera,
            )

            # Cross-modal enrichment
            fused = self._enrich(fused)

            self.store.put_fused(job_id, idx, fused)
            results.append(fused)

        logger.info(f"[fusion] Done — {len(results)} fused windows")
        return results

    # ------------------------------------------------------------------
    # Cross-modal enrichment
    # ------------------------------------------------------------------

    def _enrich(self, fused: FusedWindow) -> FusedWindow:
        # 1. Speech rate: word count / window duration as a proxy
        if fused.verbal and fused.prosody:
            duration = fused.window.duration
            if duration > 0:
                words_per_s = fused.verbal.word_count / duration
                fused.prosody.speech_rate_syl_per_s = words_per_s * 1.5

        # 2. Pose-based shot classification + camera angle (both from world landmarks)
        if fused.gesture and fused.camera and fused.gesture.pose_keyframes:
            kfs = fused.gesture.pose_keyframes
            fused.camera.dominant_shot_type = _classify_shot_from_pose(kfs)

            yaw, pitch = _compute_angles_from_pose(kfs)
            if yaw is not None:
                fused.camera.mean_shoulder_yaw_deg = yaw
                fused.camera.horizontal_angle = (
                    HorizontalAngle.FRONTAL if abs(yaw) < 30 else HorizontalAngle.OBLIQUE
                )
            if pitch is not None:
                fused.camera.mean_face_pitch_deg = pitch
                if pitch > 10:
                    fused.camera.vertical_angle = VerticalAngle.HIGH
                elif pitch < -10:
                    fused.camera.vertical_angle = VerticalAngle.LOW
                else:
                    fused.camera.vertical_angle = VerticalAngle.EYE_LEVEL

        return fused

    # ------------------------------------------------------------------

    def _resolve_window(self, gesture, prosody, verbal, camera, idx: int) -> TimeWindow:
        for obj in (verbal, prosody, gesture, camera):
            if obj is not None:
                return obj.window
        # Fallback: construct a dummy window
        return TimeWindow(start_s=float(idx * 5), end_s=float((idx + 1) * 5))
