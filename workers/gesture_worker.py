"""
workers/gesture_worker.py
--------------------------
MeTRAbs worker — metric-scale absolute 3D multi-person pose estimation with
speaker selection.

For each time window it:
  1. Reads frames from the video for that window
  2. Runs MeTRAbs (multi-person, 19-keypoint "coco_19" skeleton, absolute
     3D world-space coordinates in mm) on every frame
  3. Selects which detected person is "the subject": most-central-to-frame
     (by detection-box center) at the first frame of the window (voted
     once), then nearest-to-last-known-position for the rest of the window
     — no re-vote until the next window starts. Deliberately no "is this
     still plausibly the same person" ambiguity guard yet (e.g. a
     max-distance cutoff) — that's held off until there's enough real
     multi-person test data to decide whether it's actually needed. Same
     design as the RTMPose version before it, just tracking the detector's
     own box center instead of a keypoint-derived centroid (MeTRAbs already
     hands back a box per person, so there's no need to compute one).
  4. Computes kinematic features (velocity, amplitude, symmetry, etc.) from
     only the selected person's landmarks — everyone else detected in a
     frame is discarded before a GestureFrame is ever built, so nothing
     downstream (FusionEngine, the dashboard) needs to know multiple people
     were ever in frame.

Switched from RTMPose (via rtmlib) after RTMPose turned out to be strictly
2D — FusionEngine's camera yaw/pitch computation had nothing to work with
and permanently returned None. Chosen after researching alternatives:
  - MediaPipe's own multi-person Tasks API does give 3D world landmarks, but
    re-uses the same underlying model whose accuracy was the original
    reason for moving off MediaPipe in the first place.
  - NLF (NeurIPS'24, a newer relative of MeTRAbs, reportedly even more
    accurate) has MIT-licensed code but noncommercial-research-only
    pretrained weights — ruled out the same way YOLO-Pose's AGPL-3.0 was.
  - MeTRAbs: MIT license (code and weights both), natively multi-person
    with absolute metric-scale 3D output, and — per an independent 2025
    study benchmarking 16 pose frameworks head-to-head (MediaPipe, rtmlib,
    YOLOv8, MMPose, ViTPose, etc.) — the single highest-accuracy performer
    overall, with MediaPipe notably absent from the top tier.

**Runs isolated in its own subprocess, not in the dashboard's main
process** — see `workers/gesture_subprocess.py` and
`core/orchestrator.py`. TensorFlow's memory allocator doesn't reliably
release memory back to the OS even after the Python model object is
deleted and `gc.collect()` is run (a well-documented TF behavior, not
specific to this project) — so simply avoiding eager construction in
`Orchestrator.__init__` wouldn't be enough on its own to keep MeTRAbs from
staying resident indefinitely across jobs. Running it as a subprocess that
fully exits after each job is what actually guarantees the OS reclaims its
memory, at the cost of paying the ~15-22s model-load time on every job
instead of once. `Orchestrator` also deliberately runs gesture's subprocess
*after*, not concurrently with, the other three workers — isolation alone
only guarantees memory gets released *afterward*; it says nothing about
whether it was the only heavy thing running *at the time*, which is what
actually caused the crashes (MeTRAbs's own footprint compounding with
Whisper/prosody/camera all active via the same `ThreadPoolExecutor` batch).

Loaded as a standalone TensorFlow SavedModel (not the training codebase,
not tensorflow-hub) — see `_MODEL_DIR` below and the Dockerfile for how the
model file gets there. Plain `tensorflow` on PyPI is CPU-only by default
and, unlike plain `torch`, doesn't pull in a CUDA toolkit.

Two real tradeoffs from the RTMPose->MeTRAbs switch, both worth knowing
about:

  - No per-joint confidence: unlike RTMPose/MediaPipe, MeTRAbs's
    `detect_poses` returns only a per-*person* detection-box confidence —
    no per-keypoint score, occlusion flag, or uncertainty at all (confirmed
    against the model directly, not just docs). It always estimates a
    plausible position for all 19 joints per detected person, including
    ones that are occluded or entirely outside the frame (e.g. cropped-off
    legs), extrapolated from its learned body-shape prior. In place of a
    real confidence value, every landmark here gets a pseudo-visibility of
    1.0 if its 2D projection lands inside the actual frame bounds, else
    0.0 — good enough for the existing visibility-threshold logic
    downstream (shot classification, wrist-motion filtering), but it's an
    in-frame check, not a real confidence estimate.
  - Speed/footprint: uses `metrabs_mob3s_y4t` — MobileNetV3-Small backbone
    *and* YOLOv4-tiny as the person detector, not full YOLOv4. The detector
    dominates total model size far more than the pose backbone does
    (confirmed: `mob3s_y4` vs `mob3l_y4`, a backbone jump, changes the
    download by <5%; swapping to the `t`-suffixed detector shrinks it 8x,
    248MB -> 31MB). Real tradeoff, per MeTRAbs's own published numbers:
    worse multi-person detection accuracy (MuPoTS PCK 81.0 vs 86.6) than
    full YOLOv4 — acceptable here since this pipeline only needs the
    detector to reliably find *the* subject, not exhaustively catalog
    everyone in frame.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from loguru import logger

from core.feature_store import FeatureStore
from core.models import GestureFeatures, GestureFrame, Landmark, PoseKeyframe, TimeWindow
from core.preprocessing import VideoMeta, frames_for_window

# MeTRAbs "coco_19" skeleton joint order — verified directly against the
# model's own `per_skeleton_joint_names['coco_19']` output (there's no
# authoritative public spec for this ordering other than the model itself):
#   0 neck, 1 nose, 2 pelvis, 3 l_shoulder, 4 l_elbow, 5 l_wrist, 6 l_hip,
#   7 l_knee, 8 l_ankle, 9 r_shoulder, 10 r_elbow, 11 r_wrist, 12 r_hip,
#   13 r_knee, 14 r_ankle, 15 l_eye, 16 l_ear, 17 r_eye, 18 r_ear
# This has real structure MediaPipe/COCO-17 didn't give us in one topology:
# both a neck AND a pelvis point, useful as stable torso anchors.
_SKELETON = "coco_19"
_NUM_LANDMARKS = 19
_LEFT_WRIST = 5
_RIGHT_WRIST = 11
_LEFT_HIP = 6
_RIGHT_HIP = 12

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "metrabs_mob3s_y4t"
_MODEL_URL = "https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_mob3s_y4t.zip"

_FRAME_CENTER = (0.5, 0.5)
_DEFAULT_FOV_DEGREES = 55.0  # MeTRAbs's own default; used when no camera
# intrinsics are known, which is always true here (arbitrary source videos).


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class GestureWorker:
    def __init__(self, store: FeatureStore):
        self.store = store
        if not _MODEL_DIR.exists():
            raise FileNotFoundError(
                f"MeTRAbs model not found at {_MODEL_DIR}. Download it with:\n"
                f"  curl -L -o /tmp/metrabs.zip {_MODEL_URL}\n"
                f"  unzip /tmp/metrabs.zip -d {_MODEL_DIR.parent}\n"
                "(the Dockerfile does this automatically at build time)"
            )
        self._model = tf.saved_model.load(str(_MODEL_DIR))

    def close(self) -> None:
        pass  # no persistent per-video resource to release; the subprocess
        # this worker runs in (see module docstring) is what actually
        # reclaims TensorFlow's memory, by exiting entirely after the job.

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
        # windows), same design as the RTMPose/MediaPipe-multi-person
        # versions before it.
        ref_pos: Optional[tuple[float, float]] = None

        for frame_idx, (ts, bgr) in enumerate(raw_frames):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = tf.constant(rgb, dtype=tf.uint8)
            pred = self._model.detect_poses(
                image, skeleton=_SKELETON, default_fov_degrees=_DEFAULT_FOV_DEGREES
            )
            boxes = pred["boxes"].numpy()

            if len(boxes) == 0:
                gesture_frames.append(self._empty_frame(frame_idx, ts))
                continue

            poses2d = pred["poses2d"].numpy()
            poses3d = pred["poses3d"].numpy()

            centers = [self._box_center(b, meta.width, meta.height) for b in boxes]
            if ref_pos is None:
                chosen = min(range(len(centers)), key=lambda i: _dist(centers[i], _FRAME_CENTER))
            else:
                chosen = min(range(len(centers)), key=lambda i: _dist(centers[i], ref_pos))
            ref_pos = centers[chosen]

            gf = self._build_frame(
                frame_idx, ts, poses2d[chosen], poses3d[chosen], meta.width, meta.height
            )
            gesture_frames.append(gf)

        return self._aggregate(start_s, end_s, gesture_frames, meta.width, meta.height)

    @staticmethod
    def _box_center(box: np.ndarray, width: int, height: int) -> tuple[float, float]:
        """Detection box is [x, y, w, h, confidence] in pixel coords."""
        x, y, w, h = box[0], box[1], box[2], box[3]
        return ((x + w / 2.0) / width, (y + h / 2.0) / height)

    @staticmethod
    def _build_frame(
        frame_idx: int,
        ts: float,
        xy2d: np.ndarray,
        xyz3d: np.ndarray,
        width: int,
        height: int,
    ) -> GestureFrame:
        # No native per-joint confidence (see module docstring) — a joint
        # counts as "visible" here if its 2D projection actually lands
        # inside the frame, not extrapolated off-screen from occlusion.
        pose = []
        pose_world = []
        for i in range(_NUM_LANDMARKS):
            x2, y2 = float(xy2d[i][0]), float(xy2d[i][1])
            vis = 1.0 if (0.0 <= x2 < width and 0.0 <= y2 < height) else 0.0
            pose.append(Landmark(x=x2, y=y2, z=0.0, visibility=vis))
            x3, y3, z3 = float(xyz3d[i][0]), float(xyz3d[i][1]), float(xyz3d[i][2])
            pose_world.append(Landmark(x=x3, y=y3, z=z3, visibility=vis))
        return GestureFrame(
            frame_idx=frame_idx,
            timestamp_s=ts,
            pose=pose,
            left_hand=[],
            right_hand=[],
            pose_world=pose_world,
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

        # Require the full 19-keypoint set so all landmark index accesses
        # below are safe. MeTRAbs always returns all 19 for a detected
        # person (see module docstring), so in practice this is equivalent
        # to "was anyone detected at all this frame".
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
