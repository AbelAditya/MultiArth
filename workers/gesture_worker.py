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
     multi-person test data to decide whether it's actually needed.
  4. Computes kinematic features (velocity, amplitude, symmetry, etc.) from
     only the selected person's landmarks — everyone else detected in a
     frame is discarded before a GestureFrame is ever built, so nothing
     downstream (FusionEngine, the dashboard) needs to know multiple people
     were ever in frame.

Runs remotely first, local as a fallback — see "Remote MeTRAbs" below for
the full design. Whichever path actually runs, this class's aggregation
logic (_aggregate, _extract_keyframes, kinematic helpers) is identical —
only "how do we get this window's per-frame selected-person landmarks"
differs between _process_window_remote and _process_window_local.

Chosen (over RTMPose, MediaPipe, NLF, OpenPose, AlphaPose) after research
covered in wikis/Gesture-Worker.md — in short: MIT licensed (unlike
YOLO-Pose/NLF), natively multi-person with absolute 3D output (unlike
RTMPose, which is 2D-only), and the single highest-accuracy performer in
an independent 2025 16-framework benchmark study.

## Remote MeTRAbs, local fallback

`GESTURE_REMOTE_URL` (+ `GESTURE_API_KEY`), if set, points at a
colab/gesture_server.ipynb instance — MeTRAbs running on a free Colab GPU
(a T4, typically — meaningfully faster than this project's CPU-only local
setup, and sidesteps the CUDA-version mismatch and 4GB VRAM that ruled out
using a local GPU at all). Each *window* (not each frame — see below for
why) is sent as one batched HTTP request; on any failure (network, timeout,
bad response), that and every subsequent window in the same job falls back
to local MeTRAbs instead (`_remote_failed_this_job` — a per-job circuit
breaker, so a dead tunnel doesn't retry-and-fail on every single window of
a video, just once).

Per-window, not per-frame or per-video: a per-frame calling pattern was
considered and rejected outright — a window can have up to 150 frames
(`core/preprocessing.py`'s `frames_for_window`), and network latency alone
across that many individual round-trips would likely dwarf actual
inference time. Whole-video-at-once was also considered — fewer
round-trips still, and video codecs compress better than a pile of
independent JPEG frames — but committing to per-window keeps a natural
path to real per-window progress reporting later (each window's result
already arrives as its own discrete response) without needing an
HTTP-streaming redesign, at only a modest cost: dozens of round-trips per
video instead of one. MeTRAbs's own batched-inference call
(`detect_poses_batched`, confirmed directly — takes a stacked
`[N, H, W, 3]` array, returns `boxes`/`poses2d`/`poses3d` as
`RaggedTensor`s, one raggeed row per frame, each indexable exactly like the
single-frame `detect_poses` call's per-frame output) means the remote side
batches its own model call per window too, not a frame-by-frame loop.

The vote-once/track-thereafter subject-selection logic runs **on the
remote side** for the remote path (duplicated from this file into the
notebook, same trade already accepted for
`colab/sensevoice_server.ipynb` duplicating `_transcribe_alt` — keep them
in sync if this logic ever changes) — sending back only the *selected*
person's landmarks per frame, not everyone detected, keeps response
payloads down. This is safe statelessly per window: the vote always resets
fresh at the start of a window (never carried across windows, by design —
see point 3 above), so no tracking state needs to cross the network
boundary; each window's request is fully self-contained.

TensorFlow's `import` itself is deferred (inside `_get_model`, not at
module top-level) specifically so that a successful remote call never
costs anything TF-related in this process — mirrors
`VerbalWorker._get_sensevoice`'s lazy `from funasr import AutoModel`
exactly. Local MeTRAbs (whether because remote isn't configured, or a
video's remote calls failed) still runs inside an isolated subprocess —
see `workers/gesture_subprocess.py` / `workers/gesture_server.py` /
`core/orchestrator.py` — that part is unchanged; this file has no idea
which one is invoking it.

Two real tradeoffs, both worth knowing about, that apply whichever engine
actually runs the inference (remote or local — same model, same output
shape):

  - No per-joint confidence: unlike RTMPose/MediaPipe, MeTRAbs's
    `detect_poses`(`_batched`) returns only a per-*person* detection-box
    confidence — no per-keypoint score, occlusion flag, or uncertainty at
    all (confirmed against the model directly, not just docs). It always
    estimates a plausible position for all 19 joints per detected person,
    including ones that are occluded or entirely outside the frame (e.g.
    cropped-off legs), extrapolated from its learned body-shape prior. In
    place of a real confidence value, every landmark here gets a
    pseudo-visibility of 1.0 if its 2D projection lands inside the actual
    frame bounds, else 0.0 — good enough for the existing
    visibility-threshold logic downstream (shot classification,
    wrist-motion filtering), but it's an in-frame check, not a real
    confidence estimate.
  - Speed/footprint (local path only): uses `metrabs_mob3s_y4t` —
    MobileNetV3-Small backbone *and* YOLOv4-tiny as the person detector,
    not full YOLOv4. The detector dominates total model size far more than
    the pose backbone does (confirmed: `mob3s_y4` vs `mob3l_y4`, a
    backbone jump, changes the download by <5%; swapping to the
    `t`-suffixed detector shrinks it 8x, 248MB -> 31MB). Real tradeoff,
    per MeTRAbs's own published numbers: worse multi-person detection
    accuracy (MuPoTS PCK 81.0 vs 86.6) than full YOLOv4 — acceptable here
    since this pipeline only needs the detector to reliably find *the*
    subject, not exhaustively catalog everyone in frame. The remote
    Colab notebook uses this exact same checkpoint, for consistent output
    between the two paths — a bigger checkpoint would be a legitimate
    separate upgrade to consider later, given the remote path has real
    GPU headroom the local path never did.
"""

from __future__ import annotations

import base64
import math
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
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

# See module docstring's "Remote MeTRAbs, local fallback" section.
_REMOTE_URL_ENV = "GESTURE_REMOTE_URL"
_REMOTE_API_KEY_ENV = "GESTURE_API_KEY"
_REMOTE_TIMEOUT_S = 120  # one window's worth of frames (up to 150), batched
_JPEG_QUALITY = 85


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class GestureWorker:
    def __init__(self, store: FeatureStore):
        self.store = store
        self._model = None  # lazy — see _get_model; never touched at all if remote succeeds
        self._remote_url = os.environ.get(_REMOTE_URL_ENV) or None
        self._remote_api_key = os.environ.get(_REMOTE_API_KEY_ENV) or None
        self._remote_failed_this_job = False
        if self._remote_url:
            logger.info(f"[gesture] Will try remote MeTRAbs first, at {self._remote_url}")

    def close(self) -> None:
        pass  # no persistent per-video resource to release; the subprocess
        # this worker runs in (see module docstring) is what actually
        # reclaims TensorFlow's memory, by exiting entirely after the job —
        # only relevant when the local fallback actually ran at all.

    def _get_model(self):
        """Lazily loads the local MeTRAbs SavedModel — see module docstring
        for why both the model load AND the `tensorflow` import itself are
        deferred to here rather than module/instance-construction time."""
        if self._model is not None:
            return self._model
        import tensorflow as tf  # deferred — see module docstring

        if not _MODEL_DIR.exists():
            raise FileNotFoundError(
                f"MeTRAbs model not found at {_MODEL_DIR}. Download it with:\n"
                f"  curl -L -o /tmp/metrabs.zip {_MODEL_URL}\n"
                f"  unzip /tmp/metrabs.zip -d {_MODEL_DIR.parent}\n"
                "(the Dockerfile does this automatically at build time)"
            )
        logger.info("[gesture] Loading local MeTRAbs model...")
        self._model = tf.saved_model.load(str(_MODEL_DIR))
        return self._model

    def process_job(
        self,
        job_id: str,
        meta: VideoMeta,
        windows: list[tuple[float, float]],
    ) -> None:
        # Fresh chance to try remote for every video, even if a previous
        # job's remote calls failed (the failure might have been transient,
        # or the notebook might have been restarted since).
        self._remote_failed_this_job = False

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

        gesture_frames: Optional[list[GestureFrame]] = None
        if self._remote_url and not self._remote_failed_this_job:
            try:
                gesture_frames = self._process_window_remote(raw_frames, meta)
                logger.info("RUNNING POSE ESTIMATION REMOTELY")
            except Exception as exc:
                logger.warning(
                    f"[gesture] Remote MeTRAbs call failed ({exc}) — falling back to "
                    "local MeTRAbs for the rest of this job"
                )
                self._remote_failed_this_job = True

        if gesture_frames is None:
            gesture_frames = self._process_window_local(raw_frames, meta)

        return self._aggregate(start_s, end_s, gesture_frames, meta.width, meta.height)

    def _process_window_local(
        self, raw_frames: list[tuple[float, np.ndarray]], meta: VideoMeta
    ) -> list[GestureFrame]:
        import tensorflow as tf  # deferred — see module docstring; cheap
        # after the first call (module import is cached), whether that was
        # via _get_model or a previous call to this method.

        model = self._get_model()
        gesture_frames: list[GestureFrame] = []
        # None until the first frame with any detection in this window —
        # that frame runs the centrality vote; every frame after just tracks
        # whoever was picked. Reset every window (never carried across
        # windows), same design as the remote path's selection logic.
        ref_pos: Optional[tuple[float, float]] = None

        for frame_idx, (ts, bgr) in enumerate(raw_frames):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = tf.constant(rgb, dtype=tf.uint8)
            pred = model.detect_poses(
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

        return gesture_frames

    def _process_window_remote(
        self, raw_frames: list[tuple[float, np.ndarray]], meta: VideoMeta
    ) -> list[GestureFrame]:
        """
        Sends this window's frames to colab/gesture_server.ipynb as one
        batched request and returns the same list[GestureFrame] shape
        `_process_window_local` produces — see module docstring's "Remote
        MeTRAbs, local fallback" for the full design and why per-window
        (not per-frame or per-video).
        """
        frames_payload = []
        for frame_idx, (ts, bgr) in enumerate(raw_frames):
            ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
            if not ok:
                continue
            frames_payload.append({
                "frame_idx": frame_idx,
                "ts": ts,
                "jpeg_b64": base64.b64encode(jpeg.tobytes()).decode("ascii"),
            })

        response = requests.post(
            self._remote_url,
            json={"width": meta.width, "height": meta.height, "frames": frames_payload},
            headers={"X-API-Key": self._remote_api_key or ""},
            timeout=_REMOTE_TIMEOUT_S,
        )
        response.raise_for_status()
        payload = response.json()

        gesture_frames: list[GestureFrame] = []
        for entry in payload["frames"]:
            if entry["pose_2d"] is None:
                gesture_frames.append(self._empty_frame(entry["frame_idx"], entry["ts"]))
                continue
            pose = [
                Landmark(x=xy[0], y=xy[1], z=0.0, visibility=vis)
                for xy, vis in zip(entry["pose_2d"], entry["visibility"])
            ]
            pose_world = [
                Landmark(x=xyz[0], y=xyz[1], z=xyz[2], visibility=vis)
                for xyz, vis in zip(entry["pose_3d"], entry["visibility"])
            ]
            gesture_frames.append(GestureFrame(
                frame_idx=entry["frame_idx"],
                timestamp_s=entry["ts"],
                pose=pose,
                left_hand=[],
                right_hand=[],
                pose_world=pose_world,
            ))
        return gesture_frames

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
        y is pre-flipped (stored as 1 − raw_y) so the JS viewer doesn't need
        to re-flip it.
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
