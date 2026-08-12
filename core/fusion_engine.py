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


# MeTRAbs "coco_19" skeleton indices — see workers/gesture_worker.py for how
# these were established (verified directly against the model, no
# authoritative public spec exists). Neither MediaPipe's old 33-point
# topology nor RTMPose's COCO-17 apply anymore.
_NOSE = 1
_L_HIP, _R_HIP = 6, 12
_L_KNEE, _R_KNEE = 7, 13
_L_ANKLE, _R_ANKLE = 8, 14
_L_SHOULDER, _R_SHOULDER = 3, 9
_L_EAR, _R_EAR = 16, 18


def _classify_shot_from_pose(keyframes: list[PoseKeyframe]) -> ShotType:
    """
    Determine shot type from which body landmarks are consistently visible
    across keyframes. Indices are MeTRAbs's "coco_19" skeleton (see
    workers/gesture_worker.py), not MediaPipe's old 33-point topology or
    RTMPose's COCO-17.

    Landmark y-coords in PoseKeyframe are stored pre-flipped as (1 - raw_y),
    so pose_y=1.0 is the top of the frame and pose_y=0.0 is the bottom.

    Classification ladder (most inclusive wins):
      ankles visible + tall person  → LONG
      ankles visible + small person → VERY_LONG
      knees  visible                → MEDIUM
      hips   visible                → MEDIUM_CLOSE
      shoulders visible             → CLOSE_UP
      nose   only                   → EXTREME_CLOSE_UP
      nothing detected              → UNKNOWN

    Like COCO-17, coco_19 has no foot/toe landmark distinct from ankle
    (MediaPipe had both, at 27/28 and 31/32), so the old feet-vs-ankle
    split — LONG/VERY_LONG vs. a separate MEDIUM_LONG tier — stays
    collapsed into one ankle-based tier here. MEDIUM_LONG is not emitted.

    "Visible" here is a pseudo-visibility MeTRAbs itself doesn't provide
    (see workers/gesture_worker.py's module docstring) — a landmark counts
    if its 2D projection actually lands inside the frame, not extrapolated
    off-screen from occlusion. VIS_MIN below effectively just distinguishes
    that 0.0/1.0 flag; it isn't a real confidence threshold anymore.
    """
    n = len(keyframes)
    if n == 0:
        return ShotType.UNKNOWN

    VIS_MIN    = 0.5   # in-frame flag threshold (see docstring)
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

    ankle_r    = ratio([_L_ANKLE, _R_ANKLE])
    knee_r     = ratio([_L_KNEE, _R_KNEE])
    hip_r      = ratio([_L_HIP, _R_HIP])
    shoulder_r = ratio([_L_SHOULDER, _R_SHOULDER])
    nose_r     = ratio([_NOSE])

    if shoulder_r < THRESH and nose_r < THRESH:
        return ShotType.UNKNOWN

    if ankle_r >= THRESH:
        heights = []
        for kf in keyframes:
            if kf.pose_vis[_NOSE] > VIS_MIN:
                ankle_ys = [
                    kf.pose_y[i] for i in [_L_ANKLE, _R_ANKLE]
                    if i < len(kf.pose_vis) and kf.pose_vis[i] > VIS_MIN
                ]
                if ankle_ys:
                    heights.append(kf.pose_y[_NOSE] - min(ankle_ys))
        mean_height = float(np.mean(heights)) if heights else 0.5
        return ShotType.LONG if mean_height >= 0.4 else ShotType.VERY_LONG

    if knee_r     >= THRESH: return ShotType.MEDIUM
    if hip_r      >= THRESH: return ShotType.MEDIUM_CLOSE
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

    Needs metric 3D world-space landmarks (kf.world_x/world_y/world_z).
    MediaPipe provided these; RTMPose (2D-only) didn't. MeTRAbs (the current
    model — see workers/gesture_worker.py) provides them again, in
    millimetres, camera-relative: verified directly (not just from docs)
    that x increases rightward and y increases downward — same convention
    MediaPipe used, so the formulas below are unchanged from the
    MediaPipe-era version — and z increases *away* from the camera (the
    opposite of MediaPipe's "toward camera positive" — confirmed from a
    real detection where a facing-camera subject's nose had a smaller z
    than their neck, i.e. closer to camera, which only makes sense if
    smaller-z means nearer). The shoulder-yaw formula doesn't care about
    the z sign (atan2(dz, dx) still gives ~0° for a frontal pose either
    way), but the docstring here is corrected since the underlying model
    changed.

    The `len(kf.world_x) < 19` guard reflects MeTRAbs's coco_19 skeleton
    (19 landmarks), not MediaPipe's old 33.
    """
    yaws: list[float] = []
    pitches: list[float] = []

    for kf in keyframes:
        if kf.world_x is None or len(kf.world_x) < 19:
            continue

        wx, wy, wz = kf.world_x, kf.world_y, kf.world_z

        # Shoulder yaw. L-R (not R-L) so a frontal pose lands near 0°: verified
        # against a real detection that dx=wx[R]-wx[L] is consistently
        # negative for a camera-facing subject under MeTRAbs's x convention
        # (left shoulder appears at larger x, i.e. screen-right, than right
        # shoulder — same handedness MediaPipe had, but atan2 needs the
        # operand order that puts frontal poses at 0 given this model's
        # actual sign convention, confirmed by hand-checking real output).
        dx = wx[_L_SHOULDER] - wx[_R_SHOULDER]
        dz = wz[_L_SHOULDER] - wz[_R_SHOULDER]
        if abs(dx) + abs(dz) > 1e-4:
            yaws.append(math.degrees(math.atan2(dz, dx)))

        # Face pitch (nose vs. mid-ear anchor)
        ear_mid_x = (wx[_L_EAR] + wx[_R_EAR]) / 2
        ear_mid_y = (wy[_L_EAR] + wy[_R_EAR]) / 2
        ear_mid_z = (wz[_L_EAR] + wz[_R_EAR]) / 2
        dy = wy[_NOSE] - ear_mid_y
        xz_dist = math.sqrt((wx[_NOSE] - ear_mid_x) ** 2 + (wz[_NOSE] - ear_mid_z) ** 2)
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
