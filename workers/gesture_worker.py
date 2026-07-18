"""
workers/gesture_worker.py
--------------------------
MediaPipe Holistic worker.

For each time window it:
  1. Reads frames from the video
  2. Runs MediaPipe Holistic to extract pose + hand landmarks
  3. Computes kinematic features (velocity, amplitude, symmetry, etc.)
  4. Stores a GestureFeatures record in the FeatureStore
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from loguru import logger

from core.feature_store import FeatureStore
from core.models import GestureFeatures, GestureFrame, Landmark, PoseKeyframe, TimeWindow
from core.preprocessing import VideoMeta, frames_for_window

# MediaPipe landmark indices
_LEFT_WRIST = 15
_RIGHT_WRIST = 16


class GestureWorker:
    def __init__(self, store: FeatureStore):
        self.store = store
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        self._holistic.close()

    def process_job(
        self,
        job_id: str,
        meta: VideoMeta,
        windows: list[tuple[float, float]],
    ) -> None:
        logger.info(f"[gesture] Starting job {job_id} — {len(windows)} windows")
        for idx, (start, end) in enumerate(windows):
            try:
                features = self._process_window(meta, start, end)
                self.store.put_gesture(job_id, idx, features)
                logger.debug(f"[gesture] window {idx} done")
            except Exception as exc:
                logger.error(f"[gesture] Window {idx} failed: {exc}")
        logger.info(f"[gesture] Job {job_id} complete")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_window(
        self, meta: VideoMeta, start_s: float, end_s: float
    ) -> GestureFeatures:
        raw_frames = frames_for_window(meta.path, start_s, end_s, meta.fps)
        gesture_frames: list[GestureFrame] = []

        for frame_idx, (ts, bgr) in enumerate(raw_frames):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = self._holistic.process(rgb)
            gf = self._parse_result(frame_idx, ts, result, meta.width, meta.height)
            gesture_frames.append(gf)

        return self._aggregate(start_s, end_s, gesture_frames, meta.width, meta.height)

    def _parse_result(
        self,
        frame_idx: int,
        ts: float,
        result,
        width: int,
        height: int,
    ) -> GestureFrame:
        def to_landmarks(lm_list) -> list[Landmark]:
            if lm_list is None:
                return []
            return [
                Landmark(
                    x=lm.x * width,
                    y=lm.y * height,
                    z=lm.z,
                    visibility=getattr(lm, "visibility", 1.0),
                )
                for lm in lm_list.landmark
            ]

        def to_world_landmarks(lm_list) -> list[Landmark]:
            if lm_list is None:
                return []
            return [
                Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=getattr(lm, "visibility", 1.0))
                for lm in lm_list.landmark
            ]

        return GestureFrame(
            frame_idx=frame_idx,
            timestamp_s=ts,
            pose=to_landmarks(result.pose_landmarks),
            left_hand=to_landmarks(result.left_hand_landmarks),
            right_hand=to_landmarks(result.right_hand_landmarks),
            pose_world=to_world_landmarks(result.pose_world_landmarks),
        )

    def _aggregate(
        self,
        start_s: float,
        end_s: float,
        frames: list[GestureFrame],
        width: int,
        height: int,
    ) -> GestureFeatures:
        window = TimeWindow(start_s=start_s, end_s=end_s)

        # Require at least 33 landmarks (full MediaPipe pose) so all
        # landmark index accesses are safe
        pose_present = [f for f in frames if len(f.pose) >= 33]
        pose_present_ratio = len(pose_present) / max(len(frames), 1)

        if not pose_present:
            return GestureFeatures(
                window=window,
                mean_wrist_velocity=0.0,
                max_wrist_displacement=0.0,
                pose_present_ratio=0.0,
                pose_keyframes=[],
            )

        # Wrist positions over time
        left_wrists = self._wrist_positions(pose_present, _LEFT_WRIST)
        right_wrists = self._wrist_positions(pose_present, _RIGHT_WRIST)

        mean_vel = self._mean_velocity(left_wrists + right_wrists, pose_present)
        max_disp = self._max_displacement(left_wrists + right_wrists)
        handedness = self._compute_handedness(left_wrists, right_wrists)
        keyframes = self._extract_keyframes(pose_present, width, height)

        return GestureFeatures(
            window=window,
            mean_wrist_velocity=mean_vel,
            max_wrist_displacement=max_disp,
            pose_present_ratio=pose_present_ratio,
            handedness_ratio=handedness,
            pose_keyframes=keyframes,
        )

    # ------------------------------------------------------------------
    # Handedness + representative-frame helpers
    # ------------------------------------------------------------------

    def _compute_handedness(
        self,
        left_wrists: list,
        right_wrists: list,
    ) -> float:
        """
        Ratio of right-hand motion to total wrist motion.
        0.0 = fully left-dominant, 0.5 = bilateral, 1.0 = fully right-dominant.
        """
        def _total(positions):
            total = 0.0
            for i in range(1, len(positions)):
                p0, p1 = positions[i - 1], positions[i]
                if p0 is not None and p1 is not None:
                    total += math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
            return total

        lm = _total(left_wrists)
        rm = _total(right_wrists)
        total = lm + rm
        return (rm - lm) / total if total > 1e-6 else 0.0

    def _extract_keyframes(
        self,
        frames: list[GestureFrame],
        width: int,
        height: int,
        step: int = 3,
    ) -> list[PoseKeyframe]:
        """
        Return every `step`-th frame as a PoseKeyframe with normalised coords.
        y is pre-flipped (stored as 1 − raw_y) so the JS viewer needs no flip.
        """
        keyframes = []
        for i in range(0, len(frames), step):
            f = frames[i]
            if len(f.pose) < 33:
                continue
            has_world = len(f.pose_world) == 33
            keyframes.append(PoseKeyframe(
                ts=f.timestamp_s,
                pose_x=[lm.x / width for lm in f.pose],
                pose_y=[1.0 - lm.y / height for lm in f.pose],
                pose_vis=[lm.visibility for lm in f.pose],
                world_x=[lm.x for lm in f.pose_world] if has_world else None,
                world_y=[lm.y for lm in f.pose_world] if has_world else None,
                world_z=[lm.z for lm in f.pose_world] if has_world else None,
            ))
        return keyframes

    # ------------------------------------------------------------------
    # Kinematic helpers
    # ------------------------------------------------------------------

    def _wrist_positions(
        self, frames: list[GestureFrame], landmark_idx: int
    ) -> list[tuple[float, float]]:
        positions = []
        for f in frames:
            if len(f.pose) > landmark_idx:
                lm = f.pose[landmark_idx]
                if lm.visibility > 0.3:
                    positions.append((lm.x, lm.y))
                else:
                    positions.append(None)
            else:
                positions.append(None)
        return positions

    def _mean_velocity(
        self,
        positions: list[Optional[tuple[float, float]]],
        frames: list[GestureFrame],
    ) -> float:
        # positions may be left+right concatenated (2x len(frames)).
        # Use a fixed dt derived from position index within each half
        # rather than indexing into frames directly.
        n = len(frames)
        if n < 2:
            return 0.0
        vels = []
        for half in [positions[:n], positions[n:]]:
            for i in range(1, len(half)):
                p0, p1 = half[i - 1], half[i]
                if p0 is None or p1 is None:
                    continue
                dt = frames[i].timestamp_s - frames[i - 1].timestamp_s
                if dt <= 0:
                    continue
                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
                vels.append(math.sqrt(dx**2 + dy**2) / dt)
        return float(np.mean(vels)) if vels else 0.0

    def _max_displacement(self, positions: list[Optional[tuple[float, float]]]) -> float:
        valid = [p for p in positions if p is not None]
        if len(valid) < 2:
            return 0.0
        xs = [p[0] for p in valid]
        ys = [p[1] for p in valid]
        return math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)


