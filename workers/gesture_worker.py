"""
workers/gesture_worker.py
--------------------------
YOLO-Pose (Ultralytics) worker — multi-person detection with speaker
selection.

For each time window it:
  1. Reads frames from the video
  2. Runs YOLO-Pose (multi-person, COCO-17 keypoints) on every frame
  3. Selects which detected person is "the subject": most-central-to-frame
     (by bounding-box center) at the first frame of the window (voted
     once), then nearest-to-last-known-position for the rest of the
     window — no re-vote until the next window starts. Deliberately no
     "is this still plausibly the same person" ambiguity guard yet (e.g. a
     max-distance cutoff) — that's held off until there's enough real
     multi-person test data to decide whether it's actually needed.
  4. Computes kinematic features (velocity, amplitude, symmetry, etc.) from
     only the selected person's landmarks — everyone else detected in a
     frame is discarded before a GestureFrame is ever built, so nothing
     downstream (FusionEngine, the dashboard) needs to know multiple people
     were ever in frame.

Switched from MediaPipe (Holistic, then Tasks API PoseLandmarker) to
YOLO-Pose after a direct accuracy comparison showed MediaPipe's multi-person
predictions were worse than expected. Two real tradeoffs from this switch,
both worth knowing about:

  - License: Ultralytics is AGPL-3.0 (copyleft, with a network-use clause),
    a materially different license from the rest of this project's
    permissive dependencies. Adopted deliberately, tradeoff understood.
  - Topology: YOLO-Pose uses COCO's 17-keypoint topology, not MediaPipe's
    33-point BlazePose topology — fewer face points, no separate hand/finger
    or foot/toe landmarks, and critically, no 3D world-space coordinates at
    all (2D image-plane keypoints only). Every kinematic feature actually
    computed here only ever needed wrist position (a keypoint COCO-17 still
    has), so that's unaffected — but FusionEngine's camera yaw/pitch angle
    computation depended entirely on MediaPipe's 3D world landmarks, and has
    nothing to compute from now; see core/fusion_engine.py for how that
    degrades (cleanly, to None, not an error) and its shot-classification
    landmark remap (COCO-17 also has no foot/toe landmark distinct from
    ankle, so the LONG/VERY_LONG-vs-MEDIUM_LONG distinction lost a tier).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from ultralytics import YOLO

from core.feature_store import FeatureStore
from core.models import GestureFeatures, GestureFrame, Landmark, PoseKeyframe, TimeWindow
from core.preprocessing import VideoMeta, frames_for_window

# COCO-17 keypoint indices (Ultralytics YOLO-Pose topology) — different from
# MediaPipe's 33-point BlazePose topology this file used before switching.
# Standard COCO order: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows,
# 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles.
_LEFT_WRIST = 9
_RIGHT_WRIST = 10
_LEFT_HIP = 11
_RIGHT_HIP = 12
_NUM_LANDMARKS = 17

# Absolute path (not a bare filename) so the model is found regardless of the
# process's working directory — Ultralytics otherwise downloads/looks for it
# relative to CWD, which isn't reliable across how this worker gets started
# (CLI vs. dashboard vs. tests). Pre-downloaded here in the Dockerfile too.
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolo11m-pose.pt"
_FRAME_CENTER = (0.5, 0.5)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class GestureWorker:
    def __init__(self, store: FeatureStore, conf_threshold: float = 0.5):
        self.store = store
        self._conf_threshold = conf_threshold
        # Unlike MediaPipe's Tasks-API VIDEO mode (which required a fresh
        # landmarker per video — see git history), plain YOLO .predict()
        # calls are stateless: no timestamp bookkeeping, no memory carried
        # between frames. Safe to load once here and reuse across every
        # video this worker instance ever processes.
        self._model = YOLO(str(_MODEL_PATH))

    def close(self) -> None:
        pass  # no persistent per-video resource to release

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
        # None until the first frame with any detection in this window —
        # that frame runs the centrality vote; every frame after just tracks
        # whoever was picked. Reset every window (never carried across
        # windows), same design as the MediaPipe version before it.
        ref_pos: Optional[tuple[float, float]] = None

        for frame_idx, (ts, bgr) in enumerate(raw_frames):
            # YOLO's numpy-array input path assumes BGR (OpenCV's native
            # format) already — confirmed directly (near-identical keypoint
            # confidence whether fed BGR or a manually-converted RGB copy of
            # the same frame), so no cv2.cvtColor needed here, unlike the
            # MediaPipe version.
            result = self._model(bgr, verbose=False, conf=self._conf_threshold)[0]
            people = result.keypoints
            boxes = result.boxes

            if people is None or len(people) == 0:
                gesture_frames.append(self._empty_frame(frame_idx, ts))
                continue

            # Bounding-box center (normalised) as the "where is this person"
            # point — simpler and more robust than averaging hip landmarks,
            # since it's available even when specific keypoints are occluded.
            centers = [(float(b[0]), float(b[1])) for b in boxes.xywhn]
            if ref_pos is None:
                chosen = min(range(len(centers)), key=lambda i: _dist(centers[i], _FRAME_CENTER))
            else:
                chosen = min(range(len(centers)), key=lambda i: _dist(centers[i], ref_pos))
            ref_pos = centers[chosen]

            xy = people.xy[chosen]
            conf = people.conf[chosen] if people.conf is not None else None
            gf = self._build_frame(frame_idx, ts, xy, conf)
            gesture_frames.append(gf)

        return self._aggregate(start_s, end_s, gesture_frames, meta.width, meta.height)

    @staticmethod
    def _build_frame(frame_idx: int, ts: float, xy, conf) -> GestureFrame:
        pose = [
            Landmark(
                x=float(xy[i][0]),
                y=float(xy[i][1]),
                z=0.0,  # no depth/world coordinate from a 2D pose model
                visibility=float(conf[i]) if conf is not None else 1.0,
            )
            for i in range(len(xy))
        ]
        return GestureFrame(
            frame_idx=frame_idx,
            timestamp_s=ts,
            pose=pose,
            left_hand=[],
            right_hand=[],
            pose_world=[],  # YOLO-Pose has no 3D world-space output at all
        )

    @staticmethod
    def _empty_frame(frame_idx: int, ts: float) -> GestureFrame:
        return GestureFrame(
            frame_idx=frame_idx, timestamp_s=ts,
            pose=[], left_hand=[], right_hand=[], pose_world=[],
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

        # Require the full COCO-17 keypoint set so all landmark index
        # accesses below are safe.
        pose_present = [f for f in frames if len(f.pose) >= _NUM_LANDMARKS]
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
            if len(f.pose) < _NUM_LANDMARKS:
                continue
            has_world = len(f.pose_world) == _NUM_LANDMARKS
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
