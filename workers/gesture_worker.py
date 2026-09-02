"""
workers/gesture_worker.py
--------------------------
MediaPipe worker — multi-person pose estimation with speaker selection.
This branch (`light-gesture`) deliberately replaces the MeTRAbs setup used
on the main branch entirely: no TensorFlow, no local GPU concerns, no
remote-Colab-offload complexity — MediaPipe's models are small (a few MB
each) and light enough to run locally without the memory/crash history
that motivated MeTRAbs's remote-offload design in the first place. See
wikis/Gesture-Worker.md for the fuller "why this branch exists" story.

For each time window it:
  1. Reads frames from the video for that window
  2. Runs MediaPipe Tasks' PoseLandmarker (multi-person, 33-keypoint
     BlazePose topology) on every frame, in VIDEO running mode (see "VIDEO
     mode" below)
  3. Selects which detected person is "the subject": most-central-to-frame
     at the first frame of a window/since the last scene cut (voted once),
     then nearest-to-last-known-position for the rest — same selection
     design MeTRAbs used, ported over almost unchanged (see "Speaker
     selection" below for what did/didn't carry over).
  4. Computes kinematic features (velocity, amplitude, symmetry, etc.) from
     only the selected person's landmarks — everyone else detected in a
     frame is discarded before a GestureFrame is ever built, so nothing
     downstream (FusionEngine, the dashboard) needs to know multiple people
     were ever in frame.

`GestureFrame.left_hand`/`right_hand` are always empty here — MediaPipe's
HandLandmarker was tried, wired up, and then deliberately removed again:
nothing downstream (FusionEngine, the dashboard) ever consumed the real
per-finger data it produced, so it was pure added inference cost (roughly
doubling per-frame time) for no payoff. The fields are kept on
`GestureFrame` for shape compatibility with the rest of the pipeline
(same reason they were always empty on the MeTRAbs branch too), not
because something still populates them.

## VIDEO mode

MediaPipe Tasks' `PoseLandmarker` supports a `VIDEO` running mode
(`detect_for_video(image, timestamp_ms)`, strictly increasing timestamps
across calls to the same landmarker instance) instead of `IMAGE` mode's
independent-per-call detection — VIDEO mode lets MediaPipe use its own
internal tracking between consecutive frames rather than re-running full
detection from scratch every frame, which is both faster and reduces
frame-to-frame jitter (a real, measured problem for the per-frame-
independent alternative — see wikis/Gesture-Worker.md's MeTRAbs-era
history of `_extract_keyframes`' step=1 vs step=2 for the same underlying
issue on that branch). The landmarker is created once per job
(`process_job`, not per-window) specifically so its internal timestamp
counter and tracking state span the whole video coherently, and is closed
in a `finally` block at the end of that same method.

## Speaker selection

Ported from the MeTRAbs branch with one real difference: MeTRAbs's
detector gave an explicit per-person bounding box; PoseLandmarker doesn't
expose one at all (confirmed directly against the installed library — a
`PoseLandmarkerResult` has `pose_landmarks`, `pose_world_landmarks`, and
`segmentation_masks`, nothing box-shaped). A raw min/max bounding box over
all 33 landmarks was considered and rejected — BlazePose, like MeTRAbs,
always estimates a plausible position for every landmark even when
occluded or off-screen (e.g. ankles in a close-up shot), and those wildly
extrapolated points would skew a bbox center away from where the visible
person actually is. Centering instead on the mean of just the shoulder and
hip landmarks (indices 11, 12, 23, 24 — BlazePose's own stable torso
anchors) is a closer analogue to what MeTRAbs's detector box represented.

Everything downstream of "which person is the subject" — vote-once at a
window/scene-cut boundary, nearest-to-`ref_pos` tracking otherwise, the
max-jump ambiguity guard, scene-cut-aware resets via this worker's own
independent PySceneDetect pass — is unchanged from the MeTRAbs branch, but
only applies to a job with **no** speaker gallery. See the next section.

## Speaker re-identification (gallery-based)

For a job with a gallery (built interactively during dashboard bulk upload
— see wikis/Gesture-Worker.md; single-file upload never has one), the
heuristics above are not used *at all*, not even as a fallback. Instead
this worker runs one of two states per frame:

  - **Locked** — `ref_pos` is set, from an earlier confirmed gallery match.
    Continuation is cheap: nearest-to-`ref_pos` among this frame's
    detected candidates, identical to the heuristic path's tracking.
  - **Searching** — no current lock (a window just started, a scene cut
    just happened, or `_MAX_TRACK_JUMP` was exceeded — all three trigger
    the same recovery here, never a centrality vote). Every detected
    candidate is cropped via its own segmentation mask, embedded through
    OSNet (`_reid_model`, see below), and scored against the gallery by
    top-`_GALLERY_MATCH_TOP_K` mean cosine similarity. Whichever candidate
    clears `_GALLERY_MATCH_THRESHOLD` with the highest score gets locked
    onto; if nobody does, this frame is emitted as empty — deliberately
    "no speaker here" rather than a heuristic guess — and the next frame
    tries again from Searching.

This means gallery matching isn't bounded to fixed checkpoints: a stretch
of frames where the real speaker is off-screen or unmatchable pays the
embedding cost on every one of them until a real match resumes. Once
locked, cost drops back to the cheap steady state. Deliberate tradeoff —
an empty frame is a better answer than a confidently wrong guess.

`_GALLERY_MATCH_THRESHOLD` currently reuses the same 0.85 value chosen for
gallery-*building*'s own redundancy check (`gamma`, see the dashboard's
gallery-confirmation flow) — measured against real same-speaker/
different-speaker footage (see wikis/Gesture-Worker.md's re-ID section for
the actual numbers), but that experiment used max-similarity, single-person
footage with no real simultaneous-multiple-candidates data. Reusing gamma
here is a starting point, not a validated value for this specific
top-K-mean-pooled decision — flagged for revisiting once real multi-person
footage exists to calibrate it independently, same as `_GALLERY_MATCH_TOP_K`.

OSNet (`workers/_osnet.py`, vendored, MIT licensed — see that file's own
docstring) is loaded lazily, once per `GestureWorker` instance (not
per-job like the pose landmarker) — unlike `PoseLandmarker`'s VIDEO-mode
timestamp coupling, it's stateless and has no reason to be reloaded, so it
stays warm across an entire bulk batch once any job in it needs a gallery.
It's never loaded at all for a job with no gallery — the heuristic-only
path pays zero cost for this feature, same reasoning as HandLandmarker's
removal above: no cost for capability that job isn't using.

## Frame resolution

Frames are downscaled (`_resize_scale`, aspect-preserving, longer edge
capped at `_MAX_DIM` = 960px) before being held or sent anywhere — carried
over from the MeTRAbs branch, and re-verified rather than assumed to still
apply: BlazePose's landmark model is *also* a detector+crop architecture,
running on a fixed 256x256 crop per detected person regardless of the
source frame's resolution (confirmed via MediaPipe's own published
architecture description), so the same "full source resolution buys
nothing the landmark model can use" reasoning holds. This also still
matters for the same memory reason it did before: `core/preprocessing.py`'s
`frames_for_window` holds up to 150 full-resolution frames per window in
one list regardless of which model consumes them — at 1080p that's
~930MB, at 4K ~3.7GB, held raw before any inference starts. That
memory-safety motivation is independent of MeTRAbs vs. MediaPipe.

Coordinates come back from the landmarker already normalised to [0, 1]
(not raw pixels, unlike MeTRAbs's output) — downscaling doesn't need any
rescale-back step the way MeTRAbs's pixel-space output did; a normalised
coordinate means the same thing regardless of what resolution produced it.
Pixel-space `Landmark.x/y` (matching the rest of this file's existing
convention — `_aggregate`'s velocity/displacement math expects pixels, not
normalised fractions) are reconstructed by multiplying against the
*original* meta.width/meta.height once, immediately after detection.

## Per-landmark confidence — a real improvement over MeTRAbs

MeTRAbs had no per-joint confidence at all (only a per-person detection-box
score), so this file used to fake a pseudo-visibility (1.0 if a landmark's
2D projection landed inside the frame, else 0.0). MediaPipe genuinely
reports both `visibility` and `presence` per landmark; `Landmark.visibility`
here is populated directly from MediaPipe's own `visibility` field — an
actual confidence estimate, not an in-frame-bounds proxy.

## World coordinates — a real downgrade versus MeTRAbs, worth being honest about

MeTRAbs's `pose_world` was genuinely absolute, camera-relative metric 3D,
in millimetres (derived from an assumed FOV and the detected person's
real-world scale). MediaPipe's `pose_world_landmarks` are metric-*ish*
(meters) but hip-midpoint-relative, not absolute camera-space depth —
confirmed directly (sample values sit in roughly [-1, 1], consistent with
hip-relative meters, not absolute distance-from-camera). Populated here as
`pose_world` **in meters, unconverted** — deliberately not rescaled to mm
to match MeTRAbs's old convention, since these aren't the same kind of
quantity to begin with (hip-relative vs. absolute) and forcing them onto
the same unit invited exactly the false-equivalence this section is
warning about. Anything downstream that assumed true absolute depth
(there wasn't any on the MeTRAbs branch — FusionEngine's camera-angle math
uses 2D `pose`/`pose_keyframes`, not `pose_world`, and only ever takes
differences between landmarks, so it's unit-agnostic) should not assume
that's still true here, and should not assume mm either.
"""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from scenedetect import ContentDetector, SceneManager, open_video

from core.feature_store import FeatureStore
from core.models import GalleryEntry, GestureFeatures, GestureFrame, Landmark, PoseKeyframe, TimeWindow
from core.preprocessing import VideoMeta, frames_for_window
from workers import _reid

# MediaPipe's BlazePose 33-point topology — standard, documented ordering
# (https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker):
#   0 nose, 1-3 left eye (inner/center/outer), 4-6 right eye (inner/center/
#   outer), 7 left ear, 8 right ear, 9 mouth (left), 10 mouth (right),
#   11 left shoulder, 12 right shoulder, 13 left elbow, 14 right elbow,
#   15 left wrist, 16 right wrist, 17 left pinky, 18 right pinky,
#   19 left index, 20 right index, 21 left thumb, 22 right thumb,
#   23 left hip, 24 right hip, 25 left knee, 26 right knee, 27 left ankle,
#   28 right ankle, 29 left heel, 30 right heel, 31 left foot index,
#   32 right foot index.
# "Left"/"right" are the subject's own, not camera-relative (mirrored from
# the viewer's perspective when facing the camera) — same convention
# MeTRAbs used.
_NUM_LANDMARKS = 33
_LEFT_WRIST = 15
_RIGHT_WRIST = 16
_LEFT_HIP = 23
_RIGHT_HIP = 24
# Stable torso anchors used for the speaker-selection centrality/tracking
# math in place of a detector box PoseLandmarker doesn't provide — see
# module docstring's "Speaker selection".
_TORSO_LANDMARKS = (11, 12, 23, 24)

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_POSE_MODEL_PATH = _MODELS_DIR / "pose_landmarker_lite.task"
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# How many people to look for at once — a practical cap (typical talk
# footage: the speaker plus a handful of visible audience members), not an
# attempt at exhaustive detection. Not empirically tuned.
_NUM_POSES = 5

_FRAME_CENTER = (0.5, 0.5)

# Same ContentDetector default CameraWorker uses (core/camera_worker.py) —
# not shared/imported from there deliberately, see _detect_scene_cuts.
_SCENE_CUT_THRESHOLD = 27.0

# If the nearest-to-ref_pos candidate is farther than this (normalised
# [0,1] frame-fraction distance) from the last known position, it's treated
# as implausible — track loss, not a real continuation. A job with no
# gallery re-votes by centrality; a job with a gallery drops the lock and
# re-enters Searching instead (see module docstring's "Speaker
# re-identification"). Starting value, not empirically tuned.
_MAX_TRACK_JUMP = 0.3

# Frames are downscaled (aspect-preserving) to at most this on the longer
# edge before they're ever held or sent — see module docstring's "Frame
# resolution" section.
_MAX_DIM = 960

# --- Speaker re-identification (gallery-based) — see module docstring ---
# Model loading, crop extraction, and embedding math itself all live in
# workers/_reid.py, shared with core/gallery_builder.py's dashboard-side
# gallery confirmation flow — see that file's own docstring for why this
# is deliberately factored out rather than duplicated here.

# How many of the gallery's per-exemplar similarity scores to average when
# deciding whether a live candidate matches — measures "does this match
# *any* of our confirmed looks" rather than diluting across every look the
# gallery happens to contain. Chosen provisionally (see module docstring's
# "Speaker re-identification" for why); revisit once real same-frame
# multi-person footage exists to validate it against.
_GALLERY_MATCH_TOP_K = 3

# Minimum top-K mean similarity for a Searching-state candidate to be
# accepted as the speaker. Currently reuses gallery-building's own
# redundancy threshold (gamma) as a starting point — see module
# docstring for why that's not the same decision and this value should be
# independently revisited.
_GALLERY_MATCH_THRESHOLD = 0.85


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _resize_scale(width: int, height: int, max_dim: int = _MAX_DIM) -> float:
    """<=1.0 factor to shrink (width, height) so its longer edge is at most
    max_dim; 1.0 (no-op) if it's already smaller."""
    longest = max(width, height)
    return min(1.0, max_dim / longest)


def _ensure_model(path: Path, url: str, name: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[gesture] Downloading {name} model to {path}...")
    urllib.request.urlretrieve(url, str(path))


class GestureWorker:
    def __init__(self, store: FeatureStore):
        self.store = store
        self._pose_landmarker = None  # per-job, see process_job
        self._reid_model = None  # per-worker-instance, lazy — see _ensure_reid_model

    def close(self) -> None:
        # Primary cleanup is process_job's own try/finally — this only
        # matters if a job crashed badly enough to skip that, leaving
        # these set on an otherwise-idle worker instance (persistent
        # bulk-mode server, reused across many jobs). _reid_model isn't
        # touched here deliberately — it's a per-instance lazy singleton,
        # not per-job state, see _ensure_reid_model.
        self._close_landmarker()

    def _close_landmarker(self) -> None:
        if self._pose_landmarker is not None:
            self._pose_landmarker.close()
            self._pose_landmarker = None

    def _open_landmarker(self, use_segmentation: bool = False) -> None:
        """Creates a fresh PoseLandmarker instance for this job — see
        module docstring's "VIDEO mode" for why per-job, not
        per-worker-instance or per-window. use_segmentation enables
        per-person segmentation masks, needed only for gallery-matching's
        crops (see "Speaker re-identification") — off by default so a
        job with no gallery pays nothing extra for a capability it isn't
        using."""
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        _ensure_model(_POSE_MODEL_PATH, _POSE_MODEL_URL, "pose_landmarker_lite")

        self._pose_landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(_POSE_MODEL_PATH)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=_NUM_POSES,
                output_segmentation_masks=use_segmentation,
            )
        )
        self._mp = mp  # stashed for mp.Image/mp.ImageFormat use in the per-frame loop

    def _ensure_reid_model(self) -> None:
        """Lazily loads OSNet once per *worker instance*, not per-job —
        unlike PoseLandmarker it's stateless (no VIDEO-mode timestamp
        coupling), so it stays warm across an entire bulk batch the same
        way BulkOrchestrator already keeps one GestureWorker warm across
        every video. Never called at all for a job with no gallery. See
        workers/_reid.py for the actual loading logic, shared with the
        dashboard's gallery-building flow."""
        if self._reid_model is not None:
            return
        self._reid_model = _reid.load_reid_model()

    def process_job(
        self,
        job_id: str,
        meta: VideoMeta,
        windows: list[tuple[float, float]],
    ) -> None:
        logger.info(f"[gesture] Starting job {job_id} — {len(windows)} windows")
        cuts = self._detect_scene_cuts(meta.path)
        logger.info(f"[gesture] Found {len(cuts)} scene cuts (own independent pass)")

        # Empty list (no gallery for this job — single-file upload always,
        # or a bulk video whose gallery wasn't built) means the heuristic
        # -only path below, unchanged from before this feature existed —
        # see module docstring's "Speaker re-identification".
        gallery_entries = self.store.get_gallery(job_id)
        gallery: Optional[np.ndarray] = None
        if gallery_entries:
            gallery = np.array([e.embedding for e in gallery_entries], dtype=np.float32)
            logger.info(
                f"[gesture] Job {job_id} has a {len(gallery_entries)}-entry speaker "
                "gallery — using gallery-based re-identification, heuristics disabled"
            )

        self._open_landmarker(use_segmentation=gallery is not None)
        if gallery is not None:
            self._ensure_reid_model()
        try:
            for idx, (start, end) in enumerate(windows):
                try:
                    window_cuts = [c for c in cuts if start <= c < end]
                    features = self._process_window(meta, start, end, window_cuts, gallery)
                    self.store.put_gesture(job_id, idx, features)
                    logger.debug(f"[gesture] window {idx} done")
                except Exception as exc:
                    logger.error(f"[gesture] Window {idx} failed: {exc}")
        finally:
            self._close_landmarker()
        logger.info(f"[gesture] Job {job_id} complete")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_scene_cuts(video_path: str) -> list[float]:
        """
        This worker's own PySceneDetect pass — deliberately independent
        from CameraWorker's own identical pass (core/camera_worker.py),
        not wired to it. Orchestrator._run_parallel dispatches gesture and
        camera concurrently with no ordering guarantee between them, so
        sharing one worker's cut list with the other would mean
        serializing dispatch order. Running the same detection pass twice
        is a real, accepted cost for keeping the two workers' concurrency
        untouched.

        Returns just the cut timestamps (seconds) — gesture only needs
        "did a cut happen here" for ref_pos resets, not a full SceneCut
        record (frame index, cut score) the way Camera's dashboard chart
        does.
        """
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=_SCENE_CUT_THRESHOLD))
        scene_manager.detect_scenes(video, show_progress=False)
        return [start_tc.get_seconds() for start_tc, _ in scene_manager.get_scene_list()]

    def _process_window(
        self, meta: VideoMeta, start_s: float, end_s: float, window_cuts: list[float],
        gallery: Optional[np.ndarray],
    ) -> GestureFeatures:
        """
        Windows must be processed in non-decreasing start_s order across
        the whole job — confirmed directly (not assumed): VIDEO mode's
        detect_for_video raises ValueError("Input timestamp must be
        monotonically increasing") if fed an earlier timestamp than a
        previous call on the same landmarker instance. process_job's own
        loop over `windows` already guarantees this (core/preprocessing.py's
        compute_windows builds them in chronological order), so this isn't
        a constraint callers need to actively manage today — worth knowing
        if that ever changes (e.g. parallelizing windows within a job, or
        reprocessing/retrying an earlier window after a later one already
        ran).

        gallery is None for a job with no speaker gallery (the heuristic
        -only path); otherwise an (N, 512) array of L2-normalised
        exemplar embeddings — see module docstring's "Speaker
        re-identification".
        """
        raw_frames = frames_for_window(meta.path, start_s, end_s, meta.fps)
        gesture_frames = self._process_frames(raw_frames, meta, window_cuts, gallery)
        return self._aggregate(start_s, end_s, gesture_frames, meta.width, meta.height)

    def _process_frames(
        self,
        raw_frames: list[tuple[float, np.ndarray]],
        meta: VideoMeta,
        window_cuts: list[float],
        gallery: Optional[np.ndarray],
    ) -> list[GestureFrame]:
        mp = self._mp
        gesture_frames: list[GestureFrame] = []
        # None means "no current lock" — for a heuristic-only job (gallery
        # is None) that means the next frame runs a fresh centrality vote;
        # for a gallery job it means Searching (see module docstring's
        # "Speaker re-identification"). Reset every window (never carried
        # across windows) *and* at every scene cut within a window — see
        # next_cut_idx below — and, for a gallery job only, whenever a
        # tracked position jumps further than plausible (see the
        # _MAX_TRACK_JUMP branch below).
        ref_pos: Optional[tuple[float, float]] = None
        next_cut_idx = 0

        # See module docstring's "Frame resolution" section.
        scale = _resize_scale(meta.width, meta.height)
        resized_wh = (round(meta.width * scale), round(meta.height * scale))

        for frame_idx in range(len(raw_frames)):
            ts, bgr = raw_frames[frame_idx]
            raw_frames[frame_idx] = None  # release this frame's raw buffer as
            # we go, rather than keeping the whole window's raw frames alive
            # for the entire loop — see module docstring's "Frame
            # resolution" section for why this and downscaling matter
            # together (carried over from the MeTRAbs branch).

            # A cut landing anywhere at-or-before this frame's timestamp
            # invalidates whatever we were tracking — the next frame is a
            # different shot, so "nearest to ref_pos" would be measuring
            # distance in a scene ref_pos was never computed from.
            while next_cut_idx < len(window_cuts) and window_cuts[next_cut_idx] <= ts:
                ref_pos = None
                next_cut_idx += 1

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if scale < 1.0:
                rgb = cv2.resize(rgb, resized_wh, interpolation=cv2.INTER_AREA)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(round(ts * 1000))

            pose_result = self._pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            people = pose_result.pose_landmarks
            people_world = pose_result.pose_world_landmarks
            masks = pose_result.segmentation_masks if gallery is not None else None

            if not people:
                gesture_frames.append(self._empty_frame(frame_idx, ts))
                continue

            centers = [self._torso_center(p) for p in people]

            if ref_pos is None:
                # Searching (gallery job) / fresh vote (heuristic job) —
                # see module docstring's "Speaker re-identification".
                if gallery is not None:
                    chosen = self._gallery_match(rgb, masks, gallery)
                    if chosen is None:
                        gesture_frames.append(self._empty_frame(frame_idx, ts))
                        continue
                else:
                    chosen = min(range(len(centers)), key=lambda i: _dist(centers[i], _FRAME_CENTER))
            else:
                chosen = min(range(len(centers)), key=lambda i: _dist(centers[i], ref_pos))
                if _dist(centers[chosen], ref_pos) > _MAX_TRACK_JUMP:
                    # Implausible jump — more likely a track switch onto
                    # someone/something else than real motion. A
                    # heuristic-only job re-votes by centrality; a gallery
                    # job drops the lock and re-attempts gallery matching
                    # on this same frame instead — never a centrality
                    # fallback for a gallery job (see module docstring).
                    if gallery is not None:
                        chosen = self._gallery_match(rgb, masks, gallery)
                        if chosen is None:
                            ref_pos = None  # explicitly Searching for the
                            # next frame too, not still "locked" onto the
                            # stale pre-jump position.
                            gesture_frames.append(self._empty_frame(frame_idx, ts))
                            continue
                    else:
                        chosen = min(range(len(centers)), key=lambda i: _dist(centers[i], _FRAME_CENTER))
            ref_pos = centers[chosen]

            gf = self._build_frame(
                frame_idx, ts, people[chosen], people_world[chosen] if people_world else [],
                meta.width, meta.height,
            )
            gesture_frames.append(gf)

        return gesture_frames

    def _gallery_match(
        self, rgb: np.ndarray, masks, gallery: np.ndarray,
    ) -> Optional[int]:
        """
        Embeds every detected candidate via its own segmentation-mask
        crop, scores each against the gallery by top-_GALLERY_MATCH_TOP_K
        mean cosine similarity, and returns whichever index clears
        _GALLERY_MATCH_THRESHOLD with the highest score — or None if
        nobody does (this frame gets treated as "no speaker here", not a
        guess). See module docstring's "Speaker re-identification"; the
        actual crop/embed/score math lives in workers/_reid.py, shared
        with the dashboard's gallery-building flow.
        """
        best_idx: Optional[int] = None
        best_score = _GALLERY_MATCH_THRESHOLD
        for i, mask in enumerate(masks or []):
            crop = _reid.crop_via_mask(rgb, mask.numpy_view())
            if crop is None:
                continue
            emb = _reid.embed_crop(self._reid_model, crop)
            score = _reid.top_k_similarity(emb, gallery, _GALLERY_MATCH_TOP_K)
            if score > best_score:
                best_idx, best_score = i, score
        return best_idx

    @staticmethod
    def _torso_center(landmarks) -> tuple[float, float]:
        """Mean of the shoulder/hip landmarks (already-normalised [0,1]
        coords) as a stable person-center proxy — see module docstring's
        "Speaker selection" for why not a full-landmark bounding box."""
        xs = [landmarks[i].x for i in _TORSO_LANDMARKS]
        ys = [landmarks[i].y for i in _TORSO_LANDMARKS]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    @staticmethod
    def _build_frame(
        frame_idx: int,
        ts: float,
        pose_landmarks,
        pose_world_landmarks,
        width: int,
        height: int,
    ) -> GestureFrame:
        pose = [
            Landmark(x=lm.x * width, y=lm.y * height, z=lm.z, visibility=lm.visibility)
            for lm in pose_landmarks
        ]
        # Left in meters, unconverted — see module docstring's "World
        # coordinates" for why this is a hip-relative proxy, not true
        # absolute depth the way MeTRAbs's (millimetre) pose_world was, and
        # why it's deliberately not rescaled to force a shared unit.
        pose_world = [
            Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
            for lm in pose_world_landmarks
        ] if pose_world_landmarks else []

        # left_hand/right_hand are always empty — no hand model runs here
        # (removed; see module docstring). Kept on GestureFrame for shape
        # compatibility with the rest of the pipeline.
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

        # Require the full 33-keypoint set so all landmark index accesses
        # below are safe. BlazePose always returns all 33 for a detected
        # person (extrapolated for occluded/off-screen ones, same as
        # MeTRAbs did — see module docstring's "Speaker selection"), so in
        # practice this is equivalent to "was anyone detected at all this
        # frame".
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
        step: int = 2,
    ) -> list[PoseKeyframe]:
        """
        Return every `step`-th frame as a PoseKeyframe with normalised coords.
        frames here is already pose_present (real detections only). step=2
        (not 1) — carried over from the MeTRAbs branch's own finding that
        full per-frame density visibly picked up per-frame jitter with no
        temporal smoothing between displayed samples; VIDEO mode's own
        internal tracking (see module docstring) may make step=1 viable
        here even though it wasn't there, but that hasn't been tested, so
        step=2 is kept as the known-good starting point.
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
