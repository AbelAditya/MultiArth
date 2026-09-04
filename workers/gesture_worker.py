"""
workers/gesture_worker.py
--------------------------
Gesture worker — YOLO11n-seg person detection feeding single-person
MediaPipe pose estimation, with speaker selection in between.
This branch (`light-gesture`) deliberately replaces the MeTRAbs setup used
on the main branch entirely: no TensorFlow, no local GPU concerns, no
remote-Colab-offload complexity — MediaPipe's models are small (a few MB
each) and light enough to run locally without the memory/crash history
that motivated MeTRAbs's remote-offload design in the first place. See
wikis/Gesture-Worker.md for the fuller "why this branch exists" story.

For each time window it:
  1. Reads frames from the video for that window
  2. Runs the YOLO11n-seg person detector (workers/_detector.py) on every
     frame, getting a box + segmentation mask per person
  3. Selects which detection is "the subject": by gallery re-identification
     if the job has a speaker gallery, otherwise by the heuristics
     (most-central-to-frame, voted once per window/scene-cut, then
     nearest-to-last-known-position) ported from the MeTRAbs branch
  4. Runs MediaPipe Tasks' PoseLandmarker (single-person, 33-keypoint
     BlazePose topology, IMAGE running mode) on **only that one
     detection's crop**, and maps the landmarks back into frame
     coordinates
  5. Computes kinematic features (velocity, amplitude, symmetry, etc.) from
     only the selected person's landmarks — everyone else detected in a
     frame is discarded before a GestureFrame is ever built, so nothing
     downstream (FusionEngine, the dashboard) needs to know multiple people
     were ever in frame.

## Detector-first pipeline

Steps 2-4 above used to be a single call: multi-person PoseLandmarker on
the whole frame, then pick one of the poses it returned. Detection is now
a separate, earlier stage, and pose runs afterwards on one crop.

The reason is that MediaPipe's *detector* (BlazePose) was the weak link,
not its landmark model. It is trained overwhelmingly on close-range,
single-dominant-person imagery, and this project's footage is the
opposite: a small speaker on a wide stage with a seated audience. Two
concrete failures followed. First, `num_poses` was a hard cap on
candidates, so when the audience filled those slots the speaker was absent
from the results entirely and *no* downstream selection — heuristic or
gallery — could recover them. Second, an over-large ROI produced skeletons
fitted across the speaker plus background structure. See
workers/_detector.py's module docstring for the fuller diagnosis.

Consequences worth knowing:

  - Selection now runs on detector output (boxes and masks), before any
    pose inference. `_box_center` replaces the old `_torso_center`, since
    there are no landmarks yet at selection time.
  - Pose runs **once per frame**, not once per candidate. Note this was
    not a speed win: measured, `num_poses=5` and `num_poses=1` cost the
    same on single-speaker footage, because MediaPipe only runs the
    landmark model for people it actually finds. The saving comes from the
    crop being smaller than the frame (~29ms vs ~70ms), and it does not
    offset the detector's own ~120ms — the pipeline is roughly 2x slower
    than before, accepted deliberately in exchange for the accuracy.
  - Segmentation masks come from the detector, so the pose model no longer
    computes them. This also removed a real inconsistency: gallery
    exemplars were always built from the dashboard's own detections, so
    embedding runtime candidates from MediaPipe-derived masks meant the
    two sides of every cosine comparison had been cropped by different
    models.
  - A detection whose crop MediaPipe declines to fit a pose to yields an
    empty frame, not a guess. This is not rare for physically small
    subjects: measured across 152 detections on real footage, MediaPipe
    found a pose for only ~15/25 of the smallest ones (<15k mask pixels).
    Neither expanding the crop box nor upscaling it fixed this (both
    tested across margins 1.0-1.4 and minimum sides 192-384; differences
    were within noise, and expansion was slightly *worse* while adding
    back the background the detector exists to exclude). So the crop is
    tight and unscaled, and the remaining gap is a genuine limitation of
    BlazePose on small subjects rather than something tuning fixes.

`GestureFrame.left_hand`/`right_hand` are always empty here — MediaPipe's
HandLandmarker was tried, wired up, and then deliberately removed again:
nothing downstream (FusionEngine, the dashboard) ever consumed the real
per-finger data it produced, so it was pure added inference cost (roughly
doubling per-frame time) for no payoff. The fields are kept on
`GestureFrame` for shape compatibility with the rest of the pipeline
(same reason they were always empty on the MeTRAbs branch too), not
because something still populates them.

## IMAGE mode

This worker runs `PoseLandmarker` in `IMAGE` mode (`detect(image)`, each
call fully independent), having previously used `VIDEO` mode
(`detect_for_video(image, timestamp_ms)`, strictly increasing timestamps
across calls to the same landmarker instance).

VIDEO mode's appeal was that MediaPipe tracks between consecutive frames
rather than re-detecting from scratch, which is faster and damps
frame-to-frame jitter (a real, measured problem for per-frame-independent
detection — see wikis/Gesture-Worker.md's MeTRAbs-era history of
`_extract_keyframes`' step=1 vs step=2 for the same underlying issue on
that branch).

It was dropped because that tracking is exactly what makes a bad fit
*stick*. In VIDEO mode MediaPipe derives each frame's ROI from the
*previous frame's landmarks*, re-running the detector only once tracking
confidence collapses. On this project's footage (a small speaker on a wide
stage, cluttered background) a single frame whose ROI over-covers the
speaker plus background produces a skeleton with, typically, legs on the
real speaker and arms thrown onto background structure — and because the
next ROI is computed from *that* corrupted skeleton, the ROI genuinely
does now cover the background, so the error feeds itself and latches for a
run of frames instead of self-correcting. Note the background needn't look
remotely human for this: it only has to fall inside the ROI, since
BlazePose's landmark model is a single-person regressor that always emits
all 33 landmarks over whatever region it's given (there is no part-
association step that could decline to attach an arm).

IMAGE mode derives every ROI from the image itself, so a bad frame stays
one bad frame. The accepted costs are real and go the other way: it is
slower (full detection every frame, no tracking shortcut) and gives up
VIDEO mode's inter-frame damping, so per-frame jitter is expected to rise
— and MediaPipe Tasks exposes no `smooth_landmarks`-equivalent option to
compensate (confirmed directly against `PoseLandmarkerOptions`' own
fields; the legacy Solutions API's explicit `smooth_landmarks=True` has no
counterpart here). If jitter proves worse than the latching it fixes, an
explicit landmark filter (One-Euro or similar) is the route back, not a
return to VIDEO mode.

`min_tracking_confidence` is inert in IMAGE mode (there is no tracking
state for it to gate) — left at its default rather than removed, so
flipping `running_mode` back for a comparison needs no other edit.

The landmarker is still created once per job (`process_job`, not
per-window) and closed in a `finally` block there. That per-job lifetime
was originally required — VIDEO mode's internal timestamp counter had to
span the video coherently — and is now merely an optimisation, since an
IMAGE-mode landmarker is stateless across calls. Kept as-is because
re-creating it per window would pay model-load cost for nothing.

## Speaker selection

Selection is by detection-box centre (`_box_center`), which closes a gap
that existed for the whole MediaPipe era of this branch. MeTRAbs's
detector gave an explicit per-person bounding box; PoseLandmarker exposes
none at all (confirmed directly — a `PoseLandmarkerResult` has
`pose_landmarks`, `pose_world_landmarks` and `segmentation_masks`, nothing
box-shaped), so this worker approximated one from landmarks. A raw min/max
box over all 33 landmarks was rejected, because BlazePose always estimates
a plausible position for every landmark even when occluded or off-screen
(e.g. ankles in a close-up), and those extrapolated points skew a box
centre away from the visible person; the mean of the shoulder/hip
landmarks was used instead as a stabler proxy.

The detector now supplies a real box directly, so neither workaround is
needed — and a detector box has no extrapolation failure mode at all. One
practical caveat: a box centre sits at the body's midpoint, whereas the
torso-mean sat higher, at shoulder/hip level. `_MAX_TRACK_JUMP` is
measured against that quantity and was tuned for the old one, so it is on
the retune list.

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
    The nearest candidate to `ref_pos` is proposed, then **verified against
    the gallery before being accepted** (`_score_candidate`, one embedding
    rather than the whole frame's worth). Proximity is a cheap prior here,
    never the authority; failing verification drops the lock and falls
    through to a full Searching match on the same frame. See "Lock
    verification" below for why.
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

## Lock verification

The Locked state used to accept the nearest candidate to `ref_pos` on
proximity alone, consulting the gallery only when the jump exceeded
`_MAX_TRACK_JUMP`. That was a real hole, and it produced a confirmed bug
on TED-style footage with a person visible on a projection screen behind
the speaker:

  1. The lock is legitimately earned — the real speaker matches the
     gallery, `ref_pos` is set to her.
  2. She is briefly not detected (measured at 23% of frames in one scene,
     where she is small and scores 0.25-0.43 with the detector).
  3. The on-screen person is the only remaining candidate. It sits just
     inside `_MAX_TRACK_JUMP` of her last position, so it is adopted
     **without any identity check**, inheriting a lock it never earned.
  4. It is a *static projection*, so it never moves again. The jump guard
     never re-fires, the gallery is never re-consulted, and it holds the
     track for the rest of the scene. Measured: 204 of 222 frames.

Note the gallery was never fooled — that impostor scores 0.45-0.53 against
a speaker gallery and is rejected every time it is actually asked. It
simply stopped being asked. The failure was structural, not a threshold
being too loose, which is why tightening `_MAX_TRACK_JUMP` is not a fix:
at 0.25 this specific impostor happens to fall 0.012 outside the guard and
is caught, but nothing about that generalises to the next video.

The flaw predates the YOLO detector and was latent for the whole MediaPipe
era of this branch: the old multi-person PoseLandmarker found *zero*
candidates in that scene (confirmed — 0 across 8 sampled frames, versus 14
for the detector), so there was never an impostor to hijack the track. A
better detector did not introduce the bug, it supplied the conditions that
expose it.

So Locked now verifies the candidate it proposes. Cost is one OSNet
embedding per Locked frame (~5ms against a ~150ms frame budget), and
crucially the check no longer depends on motion — a stationary impostor is
rejected just as readily as a moving one.

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
per-job like the pose landmarker) — it's stateless and has no reason to be
reloaded, so it stays warm across an entire bulk batch once any job in it
needs a gallery.
It's never loaded at all for a job with no gallery — the heuristic-only
path pays zero cost for this feature, same reasoning as HandLandmarker's
removal above: no cost for capability that job isn't using.

## Frame resolution — downscaling removed, a deliberate, acknowledged risk

Frames are no longer downscaled before detection — `_resize_scale`/`_MAX_DIM`
(aspect-preserving, longer edge capped at 960px, carried over unmodified
from the MeTRAbs branch) were removed by explicit choice, after this
branch was found to produce visibly less stable/accurate pose output than
the project's own original, pre-MeTRAbs MediaPipe implementation (which
ran at full native resolution, no downscaling at all — see
wikis/Gesture-Worker.md). The reasoning that motivated downscaling in the
first place — BlazePose's *landmark* model only ever sees a fixed 256x256
crop per detected person regardless of source resolution — is still true,
but doesn't account for the separate *person-detection* step that decides
where that crop goes in the first place, which does see the frame at
whatever resolution it's given; a lower-resolution input plausibly costs
real precision there, which downscaling had been trading away for a
memory-safety guarantee without ever being benchmarked against the
alternative.

Worth being direct about what removing it actually reintroduces:
`core/preprocessing.py`'s `frames_for_window` holds up to 150
full-resolution frames per window in one list, regardless of which model
consumes them — at 1080p that's ~930MB, at 4K ~3.7GB, held raw before any
inference starts. This is the exact memory profile that was directly
confirmed (via `journalctl`/OOM-killer forensics) to have caused a real
crash on the MeTRAbs branch, and downscaling was the fix. That risk is
real again now, unmitigated — a live, accepted tradeoff made in exchange
for accuracy, not a closed question, and worth revisiting if this branch
sees a crash resembling that one.

Coordinates come back from the landmarker already normalised to [0, 1]
(not raw pixels, unlike MeTRAbs's output) — this was already resolution
-independent before, so removing the downscale step changes nothing about
how coordinates are handled. Pixel-space `Landmark.x/y` (matching the rest
of this file's existing convention — `_aggregate`'s velocity/displacement
math expects pixels, not normalised fractions) are still reconstructed by
multiplying against meta.width/meta.height, now always the frame's true
original dimensions rather than a downscaled stand-in.

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
from typing import NamedTuple, Optional

import cv2
import numpy as np
from loguru import logger
from scenedetect import ContentDetector, SceneManager, open_video

from core.feature_store import FeatureStore
from core.models import GalleryEntry, GestureFeatures, GestureFrame, Landmark, PoseKeyframe, TimeWindow
from core.preprocessing import VideoMeta, frames_for_window
from workers import _detector, _reid

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
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
# "full", not "lite" — matches the legacy pre-MeTRAbs MediaPipe branch's own
# Holistic config (`model_complexity=1`, Solutions API's 0/1/2 = lite/full/
# heavy tiering), which this branch had drifted away from onto Tasks API's
# smallest tier with no accuracy comparison ever run against it. Switched
# back deliberately after the "lite" choice was identified as a likely
# cause of this branch producing visibly less stable/accurate pose output
# than that legacy version — see wikis/Gesture-Worker.md.
_POSE_MODEL_PATH = _MODELS_DIR / "pose_landmarker_full.task"
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

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


class _MappedLandmark(NamedTuple):
    """A landmark whose x/y have been mapped out of crop space into
    frame-normalised coordinates by _crop_norm_to_frame_norm. Structurally
    compatible with MediaPipe's own landmark type for _build_frame's
    purposes (.x/.y/.z/.visibility), but a distinct type so it's obvious at
    a glance which coordinate frame a given object is in."""
    x: float
    y: float
    z: float
    visibility: float


def _box_center(
    box: tuple[int, int, int, int], frame_w: int, frame_h: int,
) -> tuple[float, float]:
    """Frame-normalised centre of a detection box — the subject-tracking
    position that `_torso_center` used to supply from pose landmarks.

    Switching to a box centre is what lets pose inference be deferred until
    after selection (there are no landmarks yet at that point), and it is
    arguably the better signal anyway: the old torso-mean was a deliberate
    workaround for BlazePose extrapolating occluded landmarks to implausible
    positions, and a detector box has no such failure mode. It does shift
    the quantity `_MAX_TRACK_JUMP` is measured against — a box centre sits
    at the body's midpoint where the torso-mean sat higher, at
    shoulder/hip level — so that threshold is on the retune list."""
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2 / frame_w, (y0 + y1) / 2 / frame_h)


def _letterbox_crop(
    rgb: np.ndarray, box: tuple[int, int, int, int],
) -> tuple[np.ndarray, int, int, int]:
    """
    Cuts `box` out of the frame and pads it into a square canvas, without
    rescaling. Returns (square, side, off_x, off_y) — the three values
    _crop_norm_to_frame_norm needs to invert this.

    Square, padded, and *not* stretched, for three separate reasons:
      - MediaPipe's landmark-projection step assumes a square ROI when it
        isn't handed IMAGE_DIMENSIONS (the "Using NORM_RECT without
        IMAGE_DIMENSIONS" warning it logs); a square input makes that
        assumption true rather than approximately true.
      - Stretching a person changes their apparent proportions, which is
        off-distribution for a model trained on real photographs.
      - `pose_world_landmarks` — the basis for every angle in
        core/fusion_engine.py — is estimated from apparent geometry, so
        anisotropic scaling would bias yaw/pitch systematically rather
        than just adding noise.

    No resize happens here: the square is max(box_w, box_h) at native
    resolution, so a small distant speaker stays exactly as many pixels as
    the frame gave us. MediaPipe rescales to its own input internally.
    """
    x0, y0, x1, y1 = box
    crop = rgb[y0:y1, x0:x1]
    bh, bw = crop.shape[:2]
    side = max(bw, bh)
    square = np.zeros((side, side, 3), dtype=rgb.dtype)
    off_x, off_y = (side - bw) // 2, (side - bh) // 2
    square[off_y:off_y + bh, off_x:off_x + bw] = crop
    return np.ascontiguousarray(square), side, off_x, off_y


def _crop_norm_to_frame_norm(
    nx: float, ny: float,
    box: tuple[int, int, int, int], side: int, off_x: int, off_y: int,
    frame_w: int, frame_h: int,
) -> tuple[float, float]:
    """
    Inverts _letterbox_crop for one landmark: crop-normalised (nx, ny) ->
    frame-normalised. Kept as a pure function, separate from the frame
    loop, precisely because this is the class of arithmetic that fails
    *silently* — an off-by-one or a forgotten padding offset yields
    landmarks that are wrong but entirely plausible-looking, with nothing
    raised anywhere. See tests/test_gesture_crop_mapping.py.

    Coordinates are deliberately not clamped to [0, 1]: MediaPipe
    legitimately extrapolates landmarks outside its input (a speaker whose
    legs are below the crop), and clamping would silently fold those onto
    the border as if they had been observed there. `visibility` is what
    downstream code uses to judge them.
    """
    px = box[0] + nx * side - off_x
    py = box[1] + ny * side - off_y
    return px / frame_w, py / frame_h


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


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
        self._detector = None    # per-worker-instance, lazy — see _ensure_detector

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

    def _open_landmarker(self) -> None:
        """Creates a fresh single-person PoseLandmarker for this job — see
        module docstring's "IMAGE mode" for why per-job, not
        per-worker-instance or per-window.

        `num_poses=1` is fixed, not a tunable: this landmarker is only ever
        run on an already-isolated single-person crop produced by the
        detector (see "Detector-first pipeline"), so there is never more
        than one person in its input to find. It is also MediaPipe's own
        default for `PoseLandmarkerOptions`.

        Segmentation masks are off unconditionally. They used to be
        enabled for gallery-matching crops, but masks now come from the
        detector, which produces them for every candidate in one pass
        rather than requiring a pose inference per person just to reach
        the mask riding alongside it."""
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        _ensure_model(_POSE_MODEL_PATH, _POSE_MODEL_URL, "pose_landmarker_full")

        self._pose_landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(_POSE_MODEL_PATH)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                output_segmentation_masks=False,
            )
        )
        self._mp = mp  # stashed for mp.Image/mp.ImageFormat use in the per-frame loop

    def _ensure_reid_model(self) -> None:
        """Lazily loads OSNet once per *worker instance*, not per-job — it
        stays warm across an entire bulk batch the same
        way BulkOrchestrator already keeps one GestureWorker warm across
        every video. Never called at all for a job with no gallery. See
        workers/_reid.py for the actual loading logic, shared with the
        dashboard's gallery-building flow."""
        if self._reid_model is not None:
            return
        self._reid_model = _reid.load_reid_model()

    def _ensure_detector(self) -> None:
        """Lazily loads the YOLO11n-seg person detector, per *worker
        instance* — same lifetime as _ensure_reid_model and for the same
        reason: it's stateless across calls, so there is nothing per-job
        to reset, and reloading it per video in a bulk run would pay the
        session-init cost repeatedly for no benefit. Unlike OSNet this is
        loaded for every job, gallery or not, since the detector is what
        finds people at all."""
        if self._detector is not None:
            return
        self._detector = _detector.load_detector()

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

        self._open_landmarker()
        self._ensure_detector()   # unconditional: the detector is now what
        # finds people at all, for gallery and heuristic jobs alike.
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
        Window ordering is unconstrained as of the switch to IMAGE mode.
        Under VIDEO mode windows had to be processed in non-decreasing
        start_s order across the whole job — detect_for_video raises
        ValueError("Input timestamp must be monotonically increasing") if
        fed an earlier timestamp than a previous call on the same
        landmarker instance (confirmed directly, not assumed). IMAGE mode's
        detect() takes no timestamp and holds no cross-call state, so that
        constraint is simply gone: windows could now be reprocessed out of
        order, retried individually, or parallelised within a job without
        touching the landmarker. process_job still feeds them
        chronologically (core/preprocessing.py's compute_windows builds
        them that way); nothing depends on it here any more.

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

        for frame_idx in range(len(raw_frames)):
            ts, bgr = raw_frames[frame_idx]
            raw_frames[frame_idx] = None  # release this frame's raw buffer as
            # we go, rather than keeping the whole window's raw frames alive
            # for the entire loop — see module docstring's "Frame
            # resolution" section: this alone doesn't bound peak memory the
            # way downscaling used to, it's just not holding onto frames any
            # longer than each one is actually needed for.

            # A cut landing anywhere at-or-before this frame's timestamp
            # invalidates whatever we were tracking — the next frame is a
            # different shot, so "nearest to ref_pos" would be measuring
            # distance in a scene ref_pos was never computed from.
            while next_cut_idx < len(window_cuts) and window_cuts[next_cut_idx] <= ts:
                ref_pos = None
                next_cut_idx += 1

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # Detection first, pose second — see module docstring's
            # "Detector-first pipeline". Nothing here runs a pose model
            # yet: selecting the subject needs positions and (for a
            # gallery job) mask crops, both of which the detector supplies
            # directly, so pose inference is deferred until exactly one
            # candidate has been chosen.
            detections = _detector.detect_people(self._detector, rgb)

            if not detections:
                gesture_frames.append(self._empty_frame(frame_idx, ts))
                continue

            centers = [_box_center(d.box, meta.width, meta.height) for d in detections]

            if ref_pos is None:
                # Searching (gallery job) / fresh vote (heuristic job) —
                # see module docstring's "Speaker re-identification".
                if gallery is not None:
                    chosen = self._gallery_match(rgb, detections, gallery)
                    if chosen is None:
                        gesture_frames.append(self._empty_frame(frame_idx, ts))
                        continue
                else:
                    chosen = min(range(len(centers)), key=lambda i: _dist(centers[i], _FRAME_CENTER))
            else:
                chosen = min(range(len(centers)), key=lambda i: _dist(centers[i], ref_pos))
                # Two independent ways to lose the lock. The jump check is
                # geometric: an implausible move is more likely a track
                # switch than real motion. The identity check below is what
                # makes proximity a *prior* rather than an authority — see
                # "Lock verification" in the module docstring for the
                # confirmed bug that motivated it.
                lost_lock = _dist(centers[chosen], ref_pos) > _MAX_TRACK_JUMP
                if gallery is not None and not lost_lock:
                    score = self._score_candidate(rgb, detections[chosen], gallery)
                    lost_lock = score is None or score < _GALLERY_MATCH_THRESHOLD

                if lost_lock:
                    # A heuristic-only job re-votes by centrality; a gallery
                    # job drops the lock and re-attempts gallery matching
                    # on this same frame instead — never a centrality
                    # fallback for a gallery job (see module docstring).
                    if gallery is not None:
                        chosen = self._gallery_match(rgb, detections, gallery)
                        if chosen is None:
                            ref_pos = None  # explicitly Searching for the
                            # next frame too, not still "locked" onto the
                            # stale pre-jump position.
                            gesture_frames.append(self._empty_frame(frame_idx, ts))
                            continue
                    else:
                        chosen = min(range(len(centers)), key=lambda i: _dist(centers[i], _FRAME_CENTER))
            ref_pos = centers[chosen]

            pose = self._pose_on_crop(rgb, detections[chosen].box, meta.width, meta.height)
            if pose is None:
                # The detector found a person here but MediaPipe declined to
                # fit a skeleton to the crop. Real and expected (heavy
                # occlusion, motion blur, a torso-only sliver at a frame
                # edge), and not a reason to guess: emit an empty frame,
                # exactly as a no-detection frame does. ref_pos is
                # deliberately left set — the detector's own track is still
                # good, so the next frame should continue from Locked
                # rather than pay a full gallery re-search over a
                # momentary pose failure.
                gesture_frames.append(self._empty_frame(frame_idx, ts))
                continue

            landmarks, world_landmarks = pose
            gf = self._build_frame(
                frame_idx, ts, landmarks, world_landmarks, meta.width, meta.height,
            )
            gesture_frames.append(gf)

        return gesture_frames

    def _pose_on_crop(
        self, rgb: np.ndarray, box: tuple[int, int, int, int],
        frame_w: int, frame_h: int,
    ):
        """
        Runs the single-person landmarker on one detection's letterboxed
        crop and maps the result back into frame-normalised coordinates.
        Returns (landmarks, world_landmarks), or None if no pose was found.

        The returned `landmarks` are plain _MappedLandmark objects rather
        than MediaPipe's own type: their coordinates have been transformed
        out of crop space, so handing back MediaPipe's objects unchanged
        would be actively misleading about what frame of reference they're
        in. `world_landmarks` pass through untouched — they are
        hip-origin and person-relative (metres), so cropping does not
        affect them, which is also why every angle in
        core/fusion_engine.py is unaffected by this change.
        """
        mp = self._mp
        square, side, off_x, off_y = _letterbox_crop(rgb, box)
        if square.size == 0:
            return None

        result = self._pose_landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=square)
        )
        if not result.pose_landmarks:
            return None

        mapped = []
        for lm in result.pose_landmarks[0]:
            fx, fy = _crop_norm_to_frame_norm(
                lm.x, lm.y, box, side, off_x, off_y, frame_w, frame_h,
            )
            # z is left in MediaPipe's own units, i.e. now scaled relative
            # to the *crop* rather than the frame. Nothing downstream reads
            # it (confirmed — _build_frame stores it and no consumer uses
            # it; all depth/angle work goes through pose_world instead), so
            # rescaling it would invent a precision this value never had.
            mapped.append(_MappedLandmark(fx, fy, lm.z, lm.visibility))

        world = result.pose_world_landmarks[0] if result.pose_world_landmarks else []
        return mapped, world

    def _gallery_match(
        self, rgb: np.ndarray, detections, gallery: np.ndarray,
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

        Masks come from the detector rather than MediaPipe now. That also
        closes a subtle mismatch: gallery *exemplars* were always built
        from the dashboard's own detections, so embedding runtime
        candidates from a differently-derived mask meant the two sides of
        every cosine comparison had been cropped by different models. Both
        sides now go through the same detector and the same
        crop_via_mask.
        """
        best_idx: Optional[int] = None
        best_score = _GALLERY_MATCH_THRESHOLD
        for i, det in enumerate(detections):
            score = self._score_candidate(rgb, det, gallery)
            if score is not None and score > best_score:
                best_idx, best_score = i, score
        return best_idx

    def _score_candidate(self, rgb: np.ndarray, det, gallery: np.ndarray) -> Optional[float]:
        """Top-`_GALLERY_MATCH_TOP_K` mean cosine similarity between one
        detection and the gallery, or None if the detection's mask is too
        small/degenerate to crop (see workers/_reid.py's MIN_MASK_PIXELS).

        Split out of _gallery_match so the Locked state can verify its
        single tracked candidate without embedding every other candidate
        in the frame — one OSNet forward pass per frame instead of one per
        person."""
        crop = _reid.crop_via_mask(rgb, det.mask)
        if crop is None:
            return None
        emb = _reid.embed_crop(self._reid_model, crop)
        return _reid.top_k_similarity(emb, gallery, _GALLERY_MATCH_TOP_K)

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
        step: int = 3,
    ) -> list[PoseKeyframe]:
        """
        Return every `step`-th frame as a PoseKeyframe with normalised coords.
        frames here is already pose_present (real detections only). step=3
        — carried over from the MeTRAbs branch's own finding that full
        per-frame density (step=1) visibly picked up per-frame jitter with
        no temporal smoothing between displayed samples. An earlier note
        here speculated that VIDEO mode's internal tracking might make a
        smaller step viable; that no longer applies at all now the worker
        runs in IMAGE mode (see module docstring), which has no inter-frame
        damping whatsoever — if anything a smaller step is now *less*
        viable, not more. Bumped from step=2 to step=3 by explicit choice,
        not a new finding — not re-benchmarked against 2, just carried
        forward as the current known-good value.
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
