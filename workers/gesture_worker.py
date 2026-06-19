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
from core.models import GestureFeatures, GestureFrame, Landmark, TimeWindow
from core.preprocessing import VideoMeta, frames_for_window

# MediaPipe landmark indices
_LEFT_WRIST = 15
_RIGHT_WRIST = 16
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_ELBOW = 13
_RIGHT_ELBOW = 14


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
                self.store.log_event(job_id, "gesture", f"window {idx} done")
            except Exception as exc:
                logger.error(f"[gesture] Window {idx} failed: {exc}")
                self.store.log_event(job_id, "gesture", f"window {idx} ERROR: {exc}")
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

        return GestureFrame(
            frame_idx=frame_idx,
            timestamp_s=ts,
            pose=to_landmarks(result.pose_landmarks),
            left_hand=to_landmarks(result.left_hand_landmarks),
            right_hand=to_landmarks(result.right_hand_landmarks),
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
                gesture_amplitude=0.0,
                bilateral_symmetry_score=0.0,
                hands_above_shoulder_ratio=0.0,
                gesture_rate=0.0,
                pose_present_ratio=0.0,
            )

        # Wrist positions over time
        left_wrists = self._wrist_positions(pose_present, _LEFT_WRIST)
        right_wrists = self._wrist_positions(pose_present, _RIGHT_WRIST)

        mean_vel = self._mean_velocity(left_wrists + right_wrists, pose_present)
        max_disp = self._max_displacement(left_wrists + right_wrists)
        amplitude = self._gesture_amplitude(left_wrists + right_wrists)
        symmetry = self._bilateral_symmetry(left_wrists, right_wrists, pose_present)
        above_shoulder = self._hands_above_shoulder_ratio(pose_present)
        gesture_rate = self._estimate_gesture_rate(left_wrists + right_wrists, end_s - start_s)

        return GestureFeatures(
            window=window,
            mean_wrist_velocity=mean_vel,
            max_wrist_displacement=max_disp,
            gesture_amplitude=amplitude,
            bilateral_symmetry_score=symmetry,
            hands_above_shoulder_ratio=above_shoulder,
            gesture_rate=gesture_rate,
            pose_present_ratio=pose_present_ratio,
        )

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

    def _gesture_amplitude(self, positions: list[Optional[tuple[float, float]]]) -> float:
        valid = [p for p in positions if p is not None]
        if len(valid) < 2:
            return 0.0
        xs = [p[0] for p in valid]
        ys = [p[1] for p in valid]
        return math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)

    def _bilateral_symmetry(
        self,
        left: list[Optional[tuple]],
        right: list[Optional[tuple]],
        frames: list[GestureFrame],
    ) -> float:
        """
        Symmetry score: 1 - normalised mean absolute difference
        in vertical wrist position between L and R hands.
        """
        diffs = []
        for l, r in zip(left, right):
            if l is None or r is None:
                continue
            # Compare y-coordinate (vertical position)
            diffs.append(abs(l[1] - r[1]))
        if not diffs:
            return 0.0
        max_possible = max(diffs) if diffs else 1.0
        if max_possible == 0:
            return 1.0
        return float(1.0 - (np.mean(diffs) / max_possible))

    def _hands_above_shoulder_ratio(self, frames: list[GestureFrame]) -> float:
        above = 0
        total = 0
        for f in frames:
            if len(f.pose) < max(_LEFT_WRIST, _RIGHT_WRIST, _LEFT_SHOULDER, _RIGHT_SHOULDER) + 1:
                continue
            l_shoulder_y = f.pose[_LEFT_SHOULDER].y
            r_shoulder_y = f.pose[_RIGHT_SHOULDER].y
            shoulder_y = (l_shoulder_y + r_shoulder_y) / 2.0

            l_wrist = f.pose[_LEFT_WRIST]
            r_wrist = f.pose[_RIGHT_WRIST]
            total += 1
            # In image coords, smaller y = higher on screen
            if (l_wrist.visibility > 0.3 and l_wrist.y < shoulder_y) or \
               (r_wrist.visibility > 0.3 and r_wrist.y < shoulder_y):
                above += 1
        return above / max(total, 1)

    def _estimate_gesture_rate(
        self,
        positions: list[Optional[tuple[float, float]]],
        duration_s: float,
    ) -> float:
        """
        Counts velocity peaks as a proxy for discrete gesture events.
        """
        if duration_s <= 0:
            return 0.0
        valid = [p for p in positions if p is not None]
        if len(valid) < 3:
            return 0.0

        # Compute displacement series
        displacements = []
        for i in range(1, len(valid)):
            dx = valid[i][0] - valid[i - 1][0]
            dy = valid[i][1] - valid[i - 1][1]
            displacements.append(math.sqrt(dx**2 + dy**2))

        if not displacements:
            return 0.0

        arr = np.array(displacements)
        threshold = np.mean(arr) + np.std(arr)
        # Count zero-crossings above threshold as gesture events
        peaks = 0
        in_peak = False
        for d in arr:
            if d > threshold and not in_peak:
                peaks += 1
                in_peak = True
            elif d <= threshold:
                in_peak = False

        return peaks / duration_s