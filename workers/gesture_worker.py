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
  3. Selects which detection is "the subject" by gallery re-identification,
     then holds the track by frame-to-frame appearance continuity — see
     "Speaker selection"
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
  - What MediaPipe is *fed* turned out to matter more than anything about
    the detection itself. The crop is a square region of the frame around
    the box (see _square_crop), not the box padded to square, because
    padding starved the model: on a 76x302 box — an ordinary distant
    standing speaker — a padded square is 75% black, and MediaPipe
    returned no pose at all on 25 consecutive frames where detection,
    mask, gallery match and continuity had every passed cleanly. Taking
    the same-sized square from the image recovers all 25.

    An earlier note here claimed expansion and upscaling had been tested
    and made no difference, "within noise, expansion slightly worse". That
    experiment was wrong: it expanded the box and then black-padded the
    result, so it never varied the thing that actually mattered. The real
    variable is whether the square is filled with image or with void — a
    1.0x region of the image is exactly as large as the padded one and
    still recovers 0/25.

  - A detection whose crop MediaPipe still declines to fit a pose to
    yields an empty frame, not a guess. That remains real for physically
    small subjects: measured across 152 detections, MediaPipe found a pose
    for only ~15/25 of the smallest ones (<15k mask pixels). Those were
    measured under the old padded crop, so the figure is worth re-taking.

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

**A speaker gallery is required.** `process_job` raises without one. Only
the gallery can *acquire* a lock; everything else here maintains or
invalidates an existing one.

The centrality vote this replaced — pick the candidate nearest the frame
centre — was the MeTRAbs-era fallback for jobs with no gallery, and it was
removed rather than kept as a degraded mode for two reasons. It is wrong
often enough to matter on this footage: in a crowded auditorium frame the
speaker stood at (0.25, 0.57) while the seated audience occupied the middle
of the frame, so centrality would have picked an audience member. And as a
*fallback* it fails silently — it always returns somebody, so a job with a
missing or expired gallery would produce a confident, wrong gesture track
rather than an obvious failure. Raising is louder than degrading, and an
all-empty gesture track (the alternative if selection simply never
succeeded) is indistinguishable from a video containing no people.

### Centrality returns — as a gallery-scoped tie-break only

Centrality was later reintroduced in one narrow role: when **two or more**
candidates have already cleared the gallery threshold, the one nearest the
frame centre is chosen rather than the highest scorer (see
`_gallery_match`). The trigger was live relay: TED-style stages project the
speaker onto a screen behind them, the projection is the same person and so
matches the gallery legitimately, and — being a sharp close-up — it usually
outscores the small, distant real speaker. Appearance cannot separate them;
position can, since screens hang above and beside the stage.

This is not the heuristic removed above, and the scoping is what answers
both of the original objections:

  - *It picked audience members.* Audience members are different people,
    so they fail the gallery threshold and are gone before centrality is
    consulted. It only ever chooses between candidates the gallery has
    already identified as the speaker — in practice, the speaker versus a
    picture of the speaker.
  - *It failed silently as a fallback.* It never runs without a gallery
    match, so it cannot manufacture a track. A frame where nobody clears
    the threshold is still empty.

Accepted limitation: it needs both candidates present. When the detector
misses the real speaker and returns only the projection, there is one
passer and it wins. That is a recall problem, left for a later change.

What remains, in full:

  - **Acquisition** — `_gallery_match` over every candidate, at a window
    start, after a scene cut, on lease expiry, or after a jump; with the
    centrality tie-break above when more than one candidate passes.
  - **Continuity** — the Locked-state check against the previous accepted
    frame (see "Lock verification").
  - **`_MAX_TRACK_JUMP`** — a heuristic that can only *reject*. It
    invalidates a lock when the nearest candidate has moved implausibly
    far, but never chooses who holds the lock. A rejected frame re-anchors
    against the gallery.

Positions come from `_box_center` on the detector's own box. This closed a
long-standing gap: PoseLandmarker exposes no bounding box at all (confirmed
— a `PoseLandmarkerResult` has `pose_landmarks`, `pose_world_landmarks` and
`segmentation_masks`, nothing box-shaped), so this worker used to
approximate one from the shoulder/hip landmark mean, itself chosen because
a min/max box over all 33 landmarks is skewed by the positions BlazePose
extrapolates for occluded or off-screen joints. A detector box has no such
failure mode. One caveat: a box centre sits at the body's midpoint where
the torso-mean sat at shoulder/hip level, and `_MAX_TRACK_JUMP` was tuned
against the old quantity, so it is on the retune list.

## Tracklet selection — one decision per scene, not per frame

The centrality tie-break above fixes the frames where *both* the speaker
and her projection are detected. It cannot fix the frames where only the
projection is, and those are common: her detector confidence swings with
her pose (measured 0.64 with arms raised, 0.162 with them down), so she
drops in and out of the candidate list. Per-frame selection then alternates
with whether she happened to be detected, and the track visibly jumps
between her and the screen. The lease bounds each wrong lock to ~30 frames
but cannot prevent the next one, because every re-anchor is decided from
that frame's evidence alone.

So selection is lifted to the scene. `_compute_scene_decision` streams a
whole scene once, links every detection into tracklets geometrically
(`_associate` — box overlap only, no embeddings), scores each *tracklet*
against the gallery from a handful of sampled crops, and picks one
(`_select_tracklet`). Two or more passing tracklets means the relay case,
and the most central one by **median** distance wins — median because it is
length-independent, so a short track of hers is judged on the same footing
as a long one of the screen.

Measured on a real relay scene (test_vid_39, 3.6-12.2s), three tracklets
passed a researcher-built gallery and it chose correctly:

    chosen    median centre distance 0.086, 148 frames
    rejected  0.467 (203 frames, score 0.86)
    rejected  0.477 (143 frames, score 0.89)

Both rejects were *longer* and one scored *higher* — which is exactly why
neither length nor gallery score is consulted in the choice.

Deciding once is the whole point: a scene-level decision cannot alternate.
Frames the chosen track does not cover come out empty, which is honest —
she genuinely was not detected there — and is what stops the track jumping
to the screen. Expect `pose_present_ratio` to fall on relay scenes as a
result; that is the correct number replacing a wrong one, not a
regression.

Decisions are cached in Redis under `(job_id, scene_idx)` because a scene
routinely spans several 5s windows and, with the process pool, those
windows run in different processes. Without sharing they would not only
recompute the decision but could reach *different* ones from their own
partial view, making the track flip at window boundaries. Concurrent
writers need no lock: the computation is deterministic over the same
frames, so a race wastes work and never produces disagreement.

Scene indices mean the same thing here and in the dashboard's gallery
builder because both derive them from `_detect_scene_cuts`. That coupling
is load-bearing — changing scene detection on one side only would silently
repoint every stored index at a different scene.

### Known costs and limitations, measured

  - **Non-ambiguous scenes currently pay for detection twice.** Ambiguity
    is the *output* of building tracklets, not a precondition, so the
    decision pass runs for every scene; where it concludes "not
    ambiguous", the per-frame path then re-detects the same frames.
    Estimated from measured per-stage costs, this takes a 40-minute video
    from ~1.41h to ~2.21h pooled. Two ways out are known and neither is
    implemented: use the single passing tracklet too (~1.46h, but changes
    behaviour on every scene), or trigger the scene pass only once
    ambiguity has been observed for free during the ordinary path
    (~1.48h, preserves the scoping). This is the main open question here.
  - **Identity is judged from the five *earliest* sampled crops**, not
    five spread along the track (`[:_TRACK_ID_SAMPLES]` over insertion
    order). A track is therefore assessed on its opening frames, so an
    identity switch partway through is invisible, and a speaker who
    starts a scene turned away can have her whole track misjudged.
  - **Association compares against a tracklet's last box regardless of
    how stale it is.** After a long gap a moving person may fall below
    `_TRACK_IOU_MIN` and start a spurious new track, while a static
    projection re-links trivially — a bias toward the screen. Not yet
    biting: her worst measured gap is 14 frames against a tolerance of 30.
  - **A scene where the speaker is never detected still fails.** One
    passing tracklet is not ambiguous, so the projection wins by the
    ordinary path. This rests on the researcher's observation that she is
    always detected for at least a few frames per scene.

## Speaker re-identification (gallery-based)

For a job with a gallery (built interactively during dashboard bulk upload
— see wikis/Gesture-Worker.md; single-file upload never has one), the
heuristics above are not used *at all*, not even as a fallback. Instead
this worker runs one of two states per frame:

  - **Locked** — `ref_pos` is set, from an earlier confirmed gallery match.
    The nearest candidate to `ref_pos` is proposed, then **verified against
    the previous accepted frame** (`_CONTINUITY_THRESHOLD`, one embedding
    rather than the whole frame's worth). Proximity is a cheap prior here,
    never the authority. Failing verification does not lose the frame: it
    falls through to a full gallery match on that same frame, so the four
    anchor triggers (Searching, jump, lease, continuity failure) all
    resolve identically. See "Lock verification" below for why the
    reference is the previous frame rather than the gallery.
  - **Searching** — no current lock (a window just started, a scene cut
    just happened, the lease expired, or `_MAX_TRACK_JUMP` was exceeded —
    all four trigger the same recovery here). Every detected
    candidate is cropped via its own segmentation mask, embedded through
    OSNet (`_reid_model`, see below), and scored against the gallery by
    nearest-exemplar (max) cosine similarity. If exactly one candidate
    clears `_GALLERY_MATCH_THRESHOLD` it gets locked onto; if several do,
    the one nearest the frame centre wins (see "Centrality returns — as a
    gallery-scoped tie-break only"); if nobody does, this frame is emitted
    as empty — deliberately "no speaker here" rather than a heuristic
    guess — and the next frame tries again from Searching.

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
embedding per Locked frame, and crucially the check no longer depends on
motion — a stationary impostor is rejected just as readily as a moving one.

An earlier note here put that embedding at "~5ms against a ~150ms frame
budget". Measured, it is 12.8ms in isolation and was 55ms as this pipeline
was actually configured — the gap being thread-pool contention between the
detector's, OSNet's and MediaPipe's separate pools, since each sizes itself
for a machine it assumes it owns. That is now addressed at the two loaders
(see _detector.load_detector's spin-wait note and _reid.limit_torch_threads)
rather than by changing anything here, and the check costs ~26ms. Still the
cheapest stage in the frame, but a fifth of the budget rather than a
thirtieth.

That verification is against the **previous accepted frame**
(`_CONTINUITY_THRESHOLD`), not the gallery, with the gallery reserved for
acquisition and for the periodic re-anchor `_LOCK_LEASE_FRAMES` forces.
Verifying against the gallery every frame was tried first and dropped: a
gallery is a set of discrete snapshots while appearance varies
continuously, so frames falling between two held looks were rejected even
though nothing was wrong with them — 50 of 300 frames (17%) on real
footage, appearing to the researcher as scattered dropouts. Continuity
recovered all 50. Measured on three videos, continuity-only and
continuity-with-gallery-fallback perform identically at a 0.85 threshold
(300/300, 200/200), while gallery-per-frame kept only 250/300 on the
affected clip; gallery calls drop from one per frame to ~11 per 300.

An impostor is still rejected on both paths — it scores ~0.5 against a
previous speaker frame just as it does against the gallery — so removing
the per-frame gallery check does not reopen the hijack above. Verified
directly against the TED TEST_2 scene that produced it: 0 impostor frames
posed.

`_GALLERY_MATCH_THRESHOLD` currently reuses the same 0.85 value chosen for
gallery-*building*'s own redundancy check (`gamma`, see the dashboard's
gallery-confirmation flow) — measured against real same-speaker/
different-speaker footage (see wikis/Gesture-Worker.md's re-ID section for
the actual numbers), but that experiment used max-similarity, single-person
footage with no real simultaneous-multiple-candidates data. Reusing gamma
here is now a like-for-like reuse — runtime matching is max-pooled too (see
_gallery_match), so gamma and this threshold answer the same shape of
question. Still worth calibrating independently once real multi-person
footage exists.

OSNet (`workers/_osnet.py`, vendored, MIT licensed — see that file's own
docstring) is loaded lazily, once per `GestureWorker` instance (not
per-job like the pose landmarker) — it's stateless and has no reason to be
reloaded, so it stays warm across an entire bulk batch once any job in it
needs a gallery.
It is loaded for every job, since selection always needs it: acquisition
matches candidates against the gallery, and every Locked frame embeds one
candidate for the continuity check.

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

Removing it did reintroduce a real memory risk, since
`core/preprocessing.py`'s `frames_for_window` used to hold up to 150
full-resolution frames per window in one list — at 1080p ~930MB (measured:
6.22MB per frame), at 4K ~3.7GB, held raw before any inference started.
That is the exact profile confirmed (via `journalctl`/OOM-killer
forensics) to have crashed the MeTRAbs branch, and downscaling had been
the fix.

That risk is now closed by a different route: `frames_for_window` streams,
so one decoded frame is alive at a time rather than a window's worth. The
peak fell from ~933MB to ~6MB per window without touching resolution, so
the accuracy that motivated removing the downscale is kept. Streaming is
also what makes the window pool viable — see _process_windows_pooled;
four unstreamed windows in flight would be ~3.7GB of frames on top of
~489MB of models per process.

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
import multiprocessing
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
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

# Same ContentDetector default CameraWorker uses (core/camera_worker.py) —
# not shared/imported from there deliberately, see _detect_scene_cuts.
_SCENE_CUT_THRESHOLD = 27.0

# If the nearest-to-ref_pos candidate is farther than this (normalised
# [0,1] frame-fraction distance) from the last known position, it's treated
# as implausible — track loss, not a real continuation. The lock is dropped
# and the frame re-anchored against the gallery. It is deliberately a
# heuristic that can only *reject*: it invalidates a lock but never chooses
# who holds it. Starting value, not empirically tuned.
_MAX_TRACK_JUMP = 0.3

# Frame-normalised centre, for the gallery tie-break in _gallery_match —
# consulted only when two or more candidates have *already* cleared the
# gallery threshold. See module docstring's "Speaker selection" for why this
# is not the centrality vote that was removed.
_FRAME_CENTER = (0.5, 0.5)

# --- Tracklets (see module docstring's "Tracklet selection") -------------
# Minimum box-overlap for a detection to continue an existing tracklet.
# Deliberately loose: a speaker walking at 30fps barely moves between
# frames, so anything this low is a continuation, and the alternative
# (starting a new tracklet) is the failure that matters — it fragments her
# into stubs that lose to the projection's single long track.
_TRACK_IOU_MIN = 0.3

# How many consecutive frames a tracklet survives with no detection before
# it is closed. This is load-bearing rather than a tidiness parameter: the
# speaker's detectability swings with her pose (measured 0.64 with arms
# raised, 0.162 with them down, against a 0.1 detector floor), so she
# drops out repeatedly. A short gap tolerance shatters her into stubs
# while the static projection stays one clean track, and any criterion
# that prefers longer tracks then picks the screen. One second at 30fps.
_TRACK_MAX_GAP_FRAMES = 30

# Detections a tracklet needs before it may be *selected*. A sanity floor
# against single-frame noise, not a preference for long tracks — the
# speaker is sometimes detected only briefly in a projection scene, so
# raising this hands those scenes to the screen. Starting value, expected
# to need tuning against measured tracklet lengths.
_TRACK_MIN_SUPPORT = 10

# Identity crops are taken every Nth frame while scanning a scene. Each
# costs a mask build (~4.8ms), which the rest of that pass deliberately
# avoids by working on boxes alone, so they are sampled rather than taken
# for every detection.
_TRACK_ID_STRIDE = 10

# How many of a tracklet's detections are embedded to decide its identity.
# Identity is a property of the track, not the frame: sampling a handful
# costs a fraction of embedding every candidate in every frame, which is
# what makes this affordable at _CONF_THRESHOLD = 0.1 (10-20 detections
# per frame).
_TRACK_ID_SAMPLES = 5


def _scene_index(ts: float, cuts: list[float]) -> int:
    """Which scene a timestamp falls in. `cuts` are scene *start* times as
    _detect_scene_cuts returns them (cuts[0] is 0.0), so scene i spans
    [cuts[i], cuts[i+1]).

    Both the gallery builder and this worker derive scenes from the same
    _detect_scene_cuts call, which is what lets a scene index mean the same
    thing on both sides. That coupling is load-bearing: changing this
    worker's scene detection without changing the dashboard's would
    silently repoint every stored scene index at a different scene.
    """
    lo, hi = 0, len(cuts)
    while lo < hi:
        mid = (lo + hi) // 2
        if cuts[mid] <= ts:
            lo = mid + 1
        else:
            hi = mid
    return max(0, lo - 1)


def _scene_bounds(
    scene_idx: int, cuts: list[float], duration_s: float,
) -> tuple[float, float]:
    """[start, end) of one scene. The last scene runs to the video's end."""
    start = cuts[scene_idx] if scene_idx < len(cuts) else 0.0
    end = cuts[scene_idx + 1] if scene_idx + 1 < len(cuts) else duration_s
    return start, end


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


class _Tracklet:
    """One candidate's path through a scene: frame index -> detection box.

    Deliberately holds boxes only, never masks or frames. A scene's worth
    of boxes is a few kilobytes and is safe to buffer, cache in Redis and
    hand between processes; a scene's worth of masks would be gigabytes.
    """

    __slots__ = ("boxes", "last_frame", "score")

    def __init__(self, frame_idx: int, box: tuple[int, int, int, int]):
        self.boxes: dict[int, tuple[int, int, int, int]] = {frame_idx: box}
        self.last_frame = frame_idx
        self.score: Optional[float] = None      # gallery similarity, set later

    def add(self, frame_idx: int, box: tuple[int, int, int, int]) -> None:
        self.boxes[frame_idx] = box
        self.last_frame = frame_idx

    @property
    def support(self) -> int:
        return len(self.boxes)

    def median_centre_distance(self, frame_w: int, frame_h: int) -> float:
        """Median distance of this track's boxes from the frame centre.

        Median rather than mean so one frame catching the speaker at the
        edge of frame mid-stride doesn't drag the whole track's score, and
        because it is length-independent — a 4-frame track of hers is
        judged on the same footing as a 200-frame track of the screen.
        """
        return float(np.median([
            _dist(_box_center(b, frame_w, frame_h), _FRAME_CENTER)
            for b in self.boxes.values()
        ]))


def _associate(per_frame_boxes: list[tuple[int, list]]) -> list[_Tracklet]:
    """Links per-frame detection boxes into tracklets by overlap.

    Greedy highest-overlap-first matching, which is enough here: the
    subjects in question are a person walking and a projection that barely
    moves, not a crowd crossing paths. `per_frame_boxes` is
    [(frame_idx, [box, ...]), ...] in increasing frame order.

    Purely geometric — no embeddings, no model. That is what makes it
    affordable to run over every frame of a scene; identity is decided
    per tracklet afterwards.
    """
    tracks: list[_Tracklet] = []
    for frame_idx, boxes in per_frame_boxes:
        live = [t for t in tracks if frame_idx - t.last_frame <= _TRACK_MAX_GAP_FRAMES]
        pairs = sorted(
            ((_iou(t.boxes[t.last_frame], b), ti, bi)
             for ti, t in enumerate(live) for bi, b in enumerate(boxes)),
            key=lambda p: -p[0],
        )
        used_t: set[int] = set()
        used_b: set[int] = set()
        for score, ti, bi in pairs:
            if score < _TRACK_IOU_MIN or ti in used_t or bi in used_b:
                continue
            live[ti].add(frame_idx, boxes[bi])
            used_t.add(ti)
            used_b.add(bi)
        for bi, b in enumerate(boxes):
            if bi not in used_b:
                tracks.append(_Tracklet(frame_idx, b))
    return tracks


def _select_tracklet(
    tracks: list[_Tracklet], frame_w: int, frame_h: int,
) -> Optional[_Tracklet]:
    """Picks the speaker's tracklet, or None if the scene is unambiguous
    and the caller should use the ordinary per-frame path.

    Returns a track only when **two or more** tracklets pass the gallery,
    which is the live-relay case: the speaker and her projection are the
    same person, so both match legitimately and appearance cannot separate
    them. Position can — screens hang above and beside the stage — so the
    most central track wins, exactly as _gallery_match's per-frame
    tie-break does, but decided once from the whole scene instead of
    re-decided every frame.

    Deciding once is the point. Per-frame selection alternates with
    whether she happened to be detected in that frame, which is what makes
    the track jump between her and the screen; a scene-level decision
    cannot alternate, and frames where the chosen track has no detection
    simply come out empty.
    """
    passing = [
        t for t in tracks
        if t.score is not None
        and t.score > _GALLERY_MATCH_THRESHOLD
        and t.support >= _TRACK_MIN_SUPPORT
    ]
    if len(passing) < 2:
        return None
    return min(passing, key=lambda t: t.median_centre_distance(frame_w, frame_h))

# --- Speaker re-identification (gallery-based) — see module docstring ---
# Model loading, crop extraction, and embedding math itself all live in
# workers/_reid.py, shared with core/gallery_builder.py's dashboard-side
# gallery confirmation flow — see that file's own docstring for why this
# is deliberately factored out rather than duplicated here.

# --- Lock continuity ----------------------------------------------------
# While Locked, a candidate is verified against the *previous accepted
# frame* rather than against the gallery. Consecutive frames of the same
# person are near-identical (measured: median cosine 0.987 between adjacent
# frames, 0.943 at the 5th percentile), so this is a much sharper signal
# than gallery similarity — and, crucially, it does not depend on the
# gallery happening to hold the speaker's current look.
#
# That last point is what the gallery structurally cannot do. A gallery is
# a set of discrete snapshots while appearance varies continuously, so a
# frame falling between two held looks scores below both. Measured on real
# footage, 50 of 300 frames (17%) were dropped that way with per-frame
# gallery verification; continuity recovers all of them, because each is
# ~0.99 similar to its own neighbour even when it is only ~0.80 similar to
# anything in the gallery.
#
# 0.85 rather than 0.90: the genuine mid-lease failures measured on real
# footage sat at 0.867/0.887/0.893 — ordinary frames differing from their
# neighbour by a combination of motion blur, mask wobble and pose, none
# individually large. A known impostor scores ~0.5 against a previous
# speaker frame, so the margin against a real intruder remains wide.
_CONTINUITY_THRESHOLD = 0.80

# How many frames a lock may run on continuity alone before it must be
# re-anchored against the gallery. Continuity walks its reference forward
# every frame, so errors would otherwise compound with nothing to pull them
# back — the same self-reinforcing shape as VIDEO mode's ROI latching. This
# is not hypothetical: a frame scoring only 0.819 against the gallery (below
# the match threshold) was observed being admitted by continuity at 0.975
# and then serving as the reference for the next frame. Similarity to a
# frame one second earlier is 0.897 median and two seconds earlier 0.796, so
# ~1s is about where the tracked appearance has meaningfully moved.
_LOCK_LEASE_FRAMES = 30

# Minimum nearest-exemplar similarity for a candidate to be accepted as
# the speaker — used by both Searching and Locked verification (see
# _gallery_match, which is the single place pooling is decided).
# Reuses gallery-building's own redundancy threshold (gamma), which is now
# a defensible reuse rather than a placeholder: both sides ask the same
# max-pooled question, so the two halves of the system agree on what
# "similar enough" means. Still worth calibrating against real
# multi-person footage.
_GALLERY_MATCH_THRESHOLD = 0.80


# How many worker *processes* share out a job's windows. 1 disables the
# pool and runs everything inline, which is what you want for debugging (a
# traceback from a pool child is a pickled shadow of the real one).
#
# 4 rather than 6 on a 6-physical-core machine: measured throughput was
# 4.63 frames/s at one process, 8.13 at two, 11.17 at four and 13.01 at
# six, so the last two processes buy 1.16x for 50% more memory and leave
# nothing for the three other workers Orchestrator._run_parallel is running
# alongside this one.
#
# Processes rather than threads twice over. The GIL would serialise the
# non-inference half of the loop; and the models themselves scale badly
# with threads — the detector graph only reaches ~2x across six threads
# (160ms -> 79ms) while OSNet is fastest single-threaded — so N
# single-threaded processes do far more total work than one N-threaded one.
_POOL_PROCESSES = 4

# Per-process thread caps for pool children. Every library here sizes its
# pool for a machine it assumes it owns, and four such processes on six
# cores would oversubscribe several times over.
_POOL_THREADS_PER_PROCESS = 1

# Set in each pool child by _pool_init and read by _pool_process_window.
# Module-level because ProcessPoolExecutor's initializer has nowhere else
# to leave state, and because "spawn" gives every child a fresh import of
# this module, so there is no cross-process sharing to worry about.
_POOL_STATE: dict = {}


def _pool_init(gallery: np.ndarray, job_id: str, all_cuts: list[float]) -> None:
    """Runs once per pool child: caps threads, then loads the three models
    that child will reuse for every window it is handed.

    The gallery, job id and cut list arrive here rather than as per-window
    arguments because they are the same for the whole job — passing them
    per task would re-pickle ~140KB several hundred times for nothing.
    """
    import cv2 as _cv2

    _cv2.setNumThreads(_POOL_THREADS_PER_PROCESS)

    # A child now gets its own store, solely to share per-scene tracklet
    # decisions with its siblings (see _scene_decision). Results still come
    # back to the parent to be written, so this does not make children
    # general writers: scene decisions are write-once and identical
    # whoever computes them, which is the one safe shape for concurrent
    # writes from parallel children. A store that cannot be reached is
    # not fatal — decisions are simply recomputed per window.
    try:
        store = FeatureStore()
    except Exception as exc:
        logger.warning(f"[gesture] pool child has no store ({exc}); "
                       "scene decisions will not be shared")
        store = None

    worker = GestureWorker(store=store)
    worker._open_landmarker()
    worker._detector = _detector.load_detector(
        intra_op_threads=_POOL_THREADS_PER_PROCESS
    )
    worker._ensure_reid_model()          # also caps torch to one thread

    # Deliberately no teardown hook for the landmarker. ProcessPoolExecutor
    # offers none, and atexit does not work here: MediaPipe's close()
    # dispatches through a ThreadPoolExecutor, and concurrent.futures
    # registers its own shutdown via threading._register_atexit, which
    # CPython runs *before* ordinary atexit callbacks. Registering
    # worker.close therefore cannot succeed — it raises "cannot schedule
    # new futures after shutdown" on every child, every time (observed, not
    # theorised). Letting the process exit reclaims the native resources
    # anyway, and a child builds exactly one landmarker for its whole life,
    # so there is nothing here that leaks while the process is running.
    _POOL_STATE["worker"] = worker
    _POOL_STATE["gallery"] = gallery
    _POOL_STATE["job_id"] = job_id
    _POOL_STATE["all_cuts"] = all_cuts


def _pool_process_window(
    meta: VideoMeta, start_s: float, end_s: float, window_cuts: list[float],
) -> GestureFeatures:
    """One window, in a pool child. Module-level and taking only picklable
    arguments, because that is what ProcessPoolExecutor can dispatch."""
    return _POOL_STATE["worker"]._process_window(
        meta, start_s, end_s, window_cuts, _POOL_STATE["gallery"],
        job_id=_POOL_STATE["job_id"], all_cuts=_POOL_STATE["all_cuts"],
    )


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


# How far the square pose crop extends beyond the detector box. Measured,
# and the exact value matters less than being clear of 1.0: on 25 frames
# where MediaPipe returned no pose at all, a square region of the *image*
# recovered 0/25 at 1.0x but 25/25 at 1.15x and again at 1.5x-2.5x, with a
# non-monotonic dip to 16/25 at 1.3x. 1.5x sits in the flat, reliable band
# rather than on the edge of the 1.15x cliff.
_POSE_CROP_SCALE = 1.5


def _square_crop(
    rgb: np.ndarray, box: tuple[int, int, int, int],
) -> tuple[np.ndarray, int, int]:
    """
    Cuts a square region *of the frame* centred on `box` and expanded by
    _POSE_CROP_SCALE. Returns (crop, origin_x, origin_y) — the two offsets
    _crop_norm_to_frame_norm needs to invert this.

    ## Why a region of the image, not a padded box

    This replaced a letterbox crop that cut out the box exactly and padded
    it to square with black. That failed badly on tall, thin subjects: a
    distant standing speaker gives a box like 76x302, so padding to a
    302x302 square makes **75% of MediaPipe's input black**, and BlazePose
    — trained on photographs — has nothing to work with. Measured on real
    footage, that crop returned no pose at all on 25 consecutive frames
    where detection, mask, gallery match and continuity had every passed
    cleanly (detector confidence 0.82-0.86, gallery similarity
    0.854-0.942). Taking the same-sized square from the image instead
    recovers all 25.

    Note the fix is *what fills the square*, not merely its size: a
    1.0x-scaled region of the image is exactly as large as the padded one
    and still recovers 0/25, so the model needs genuine surrounding
    context, not just fewer black pixels.

    Squareness is kept for the same three reasons the letterbox version
    had it — MediaPipe's landmark projection assumes a square ROI when not
    handed IMAGE_DIMENSIONS; stretching a person is off-distribution; and
    `pose_world_landmarks`, the basis for every angle in
    core/fusion_engine.py, is estimated from apparent geometry, so
    anisotropic scaling would bias yaw/pitch rather than just add noise.

    Near a frame edge the square is **slid inward** rather than clipped,
    so it stays square and stays full of real pixels. Only a square larger
    than the frame itself is truncated, which is why the caller must use
    the returned crop's actual shape rather than assuming it.

    No resize happens here: the region is taken at native resolution, so a
    small distant speaker keeps exactly the pixels the frame gave us.
    MediaPipe rescales to its own input internally.
    """
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = min(max(x1 - x0, y1 - y0) * _POSE_CROP_SCALE, float(w), float(h))
    half = side / 2.0
    # Slide the centre so the square lies fully inside the frame.
    cx = min(max(cx, half), w - half)
    cy = min(max(cy, half), h - half)
    ax, ay = int(round(cx - half)), int(round(cy - half))
    bx, by = ax + int(round(side)), ay + int(round(side))
    ax, ay = max(0, ax), max(0, ay)
    bx, by = min(w, bx), min(h, by)
    return np.ascontiguousarray(rgb[ay:by, ax:bx]), ax, ay


def _crop_norm_to_frame_norm(
    nx: float, ny: float,
    origin_x: int, origin_y: int, crop_w: int, crop_h: int,
    frame_w: int, frame_h: int,
) -> tuple[float, float]:
    """
    Inverts _square_crop for one landmark: crop-normalised (nx, ny) ->
    frame-normalised. Kept as a pure function, separate from the frame
    loop, precisely because this is the class of arithmetic that fails
    *silently* — an off-by-one or a wrong origin yields landmarks that are
    wrong but entirely plausible-looking, with nothing raised anywhere.
    See tests/test_gesture_crop_mapping.py.

    `crop_w`/`crop_h` come from the returned crop's real shape rather than
    the requested side, since a square larger than the frame is truncated.

    Coordinates are deliberately not clamped to [0, 1]: MediaPipe
    legitimately extrapolates landmarks outside its input (a speaker whose
    legs are below the crop), and clamping would silently fold those onto
    the border as if they had been observed there. `visibility` is what
    downstream code uses to judge them.
    """
    px = origin_x + nx * crop_w
    py = origin_y + ny * crop_h
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
    def __init__(self, store: Optional[FeatureStore]):
        # None only for a pool child (see _pool_init): it computes window
        # features and hands them back to the parent, which owns every
        # write. Anything calling process_job needs a real store.
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
                min_pose_detection_confidence = 0.60,
            )
        )
        self._mp = mp  # stashed for mp.Image/mp.ImageFormat use in the per-frame loop

    def _ensure_reid_model(self) -> None:
        """Lazily loads OSNet once per *worker instance*, not per-job — it
        stays warm across an entire bulk batch the same
        way BulkOrchestrator already keeps one GestureWorker warm across
        every video. Never called at all for a job with no gallery. See
        workers/_reid.py for the actual loading logic, shared with the
        dashboard's gallery-building flow.

        The thread cap is taken here rather than inside load_reid_model
        because it is process-global: this worker wants it (OSNet is one of
        three models taking turns on the same cores every frame), the
        dashboard's gallery flow has no such contention, and VerbalWorker's
        locally-loaded SenseVoice would rather not have it. See
        _reid.limit_torch_threads for the measurements and the tradeoff."""
        if self._reid_model is not None:
            return
        _reid.limit_torch_threads(1)
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

        # A gallery is *required*: it is the only thing that can acquire a
        # lock (see module docstring's "Speaker selection"). Raising rather
        # than degrading, because the alternative is worse — with no way to
        # select a subject, every frame would come out empty and the job
        # would "succeed" with output indistinguishable from a video
        # containing no people at all. Failing is loud; silently shipping an
        # empty gesture track is not.
        #
        # Checked before scene detection deliberately: that is a full pass
        # over the video (~3 min on a 40-minute one), and there is no point
        # paying it for a job that cannot produce anything.
        gallery_entries = self.store.get_gallery(job_id)
        if not gallery_entries:
            raise RuntimeError(
                f"Job {job_id} has no speaker gallery. Gesture analysis needs one "
                "to identify the subject — build it in the dashboard's Bulk Upload "
                "or Live Analysis flow before processing."
            )
        gallery = np.array([e.embedding for e in gallery_entries], dtype=np.float32)
        logger.info(
            f"[gesture] Job {job_id} has a {len(gallery_entries)}-entry speaker gallery"
        )

        cuts = self._detect_scene_cuts(meta.path)
        logger.info(f"[gesture] Found {len(cuts)} scene cuts (own independent pass)")
        # Sliced once here, in the parent: one pass over the cut list
        # instead of one per window, and in pooled mode a child receives
        # only the cuts for the window it was given.
        per_window_cuts = [
            [c for c in cuts if start <= c < end] for start, end in windows
        ]

        if _POOL_PROCESSES > 1:
            self._process_windows_pooled(job_id, meta, windows, per_window_cuts, gallery, cuts)
        else:
            self._process_windows_inline(job_id, meta, windows, per_window_cuts, gallery, cuts)
        logger.info(f"[gesture] Job {job_id} complete")

    def _process_windows_inline(
        self, job_id, meta, windows, per_window_cuts, gallery, cuts,
    ) -> None:
        """Every window in this process, in order — the original behaviour,
        kept for _POOL_PROCESSES == 1. Worth keeping rather than deleting:
        an exception raised in a pool child reaches the parent as a pickled
        copy with its traceback flattened to a string, so debugging the
        frame loop is much easier here."""
        self._open_landmarker()
        self._ensure_detector()   # the detector is what finds people at all
        self._ensure_reid_model() # always needed now: both acquisition and
        # the per-frame continuity check embed candidates.
        try:
            for idx, (start, end) in enumerate(windows):
                try:
                    features = self._process_window(
                        meta, start, end, per_window_cuts[idx], gallery,
                        job_id=job_id, all_cuts=cuts,
                    )
                    self.store.put_gesture(job_id, idx, features)
                    logger.debug(f"[gesture] window {idx} done")
                except Exception as exc:
                    logger.error(f"[gesture] Window {idx} failed: {exc}")
        finally:
            self._close_landmarker()

    def _process_windows_pooled(
        self, job_id, meta, windows, per_window_cuts, gallery, cuts,
    ) -> None:
        """Windows fanned out across _POOL_PROCESSES child processes.

        Safe to do at all only because windows are independent: each one
        re-acquires its own lock from the gallery at its first frame and
        carries no state in or out (see _process_window's own note on
        ordering, which the switch to IMAGE mode freed up).

        "spawn", not "fork", and not negotiable: this parent has already
        loaded onnxruntime, torch and MediaPipe, each with live thread
        pools, and forking a process with threads mid-flight is a classic
        way to inherit a lock held by a thread that does not exist in the
        child. A spawned child imports this module fresh and builds its own
        models in _pool_init.

        Results come back to the parent to be written, so a child needs no
        Redis connection and the store keeps a single writer.

        Note this pool runs *inside* one of the four worker threads
        Orchestrator._run_parallel starts, so briefly there are four
        gesture processes plus three sibling workers competing. The
        siblings finish in minutes against this worker's hours, so the
        overlap is short and not worth scheduling around.
        """
        logger.info(
            f"[gesture] Processing {len(windows)} windows across "
            f"{_POOL_PROCESSES} processes"
        )
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=_POOL_PROCESSES, mp_context=ctx,
            initializer=_pool_init, initargs=(gallery, job_id, cuts),
        ) as pool:
            futures = {
                pool.submit(
                    _pool_process_window, meta, start, end, per_window_cuts[idx],
                ): idx
                for idx, (start, end) in enumerate(windows)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    self.store.put_gesture(job_id, idx, future.result())
                    logger.debug(f"[gesture] window {idx} done")
                except Exception as exc:
                    # Same contract as the inline path: one bad window is
                    # logged and skipped, it does not abort the job.
                    logger.error(f"[gesture] Window {idx} failed: {exc}")

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

    def _scene_decision(
        self, job_id: str, meta: VideoMeta, scene_idx: int,
        scene_start: float, scene_end: float, gallery: np.ndarray,
    ) -> dict:
        """Cached per-scene speaker-track decision — see _compute_scene_decision.

        Keyed by (job_id, scene_idx) in Redis so that the several windows
        overlapping one scene agree and compute it once. A store is
        optional: without one this still works, just without the sharing.
        """
        if self.store is not None:
            try:
                cached = self.store.get_scene_track(job_id, scene_idx)
                if cached is not None:
                    return cached
            except Exception as exc:          # a cache miss must never be fatal
                logger.warning(f"[gesture] scene-track cache read failed: {exc}")

        decision = self._compute_scene_decision(meta, scene_start, scene_end, gallery)
        if self.store is not None:
            try:
                self.store.put_scene_track(job_id, scene_idx, decision)
            except Exception as exc:
                logger.warning(f"[gesture] scene-track cache write failed: {exc}")
        return decision

    def _compute_scene_decision(
        self, meta: VideoMeta, scene_start: float, scene_end: float,
        gallery: np.ndarray,
    ) -> dict:
        """
        Decides, once for a whole scene, which tracklet is the speaker.

        Returns either `{"ambiguous": False}` — meaning the caller should
        use the ordinary per-frame path — or `{"ambiguous": True, "boxes":
        {frame_index: box}}` giving the chosen tracklet's box for every
        frame it was detected in.

        ## Why a scene-level decision at all

        Per-frame selection alternates with whether the speaker happened to
        be detected in that frame. She flickers — measured, her detector
        confidence swings from 0.64 with arms raised to 0.162 with them
        down — so on a relay scene the track jumps between her and her
        projection, and the lease re-anchors onto whichever is visible at
        that instant. A decision made once from the whole scene cannot
        alternate.

        ## Why this does not double the detector cost

        The obvious objection is that this is a second pass. It is not:
        the returned boxes are what the caller then uses, so in an
        ambiguous scene the per-frame path skips detection entirely and the
        detector still runs exactly once per frame. Only the frames are
        read twice, and decoding is 2.05ms against detection's 58ms.

        Frames are never retained. Boxes are kept for every frame (a few
        kilobytes per scene) and a handful of *crops* per tracklet for
        identity scoring; the frames themselves are released as they
        stream past, so this does not reopen the memory profile that
        streaming just closed.
        """
        per_frame: list[tuple[int, list]] = []
        # Crops kept for identity scoring, keyed by the box they came from:
        # boxes are what _associate works on, and a box identifies its
        # detection uniquely within a frame.
        crops: dict[tuple[int, tuple], np.ndarray] = {}

        for ts, bgr in frames_for_window(
            meta.path, scene_start, scene_end, meta.fps, max_frames=10 ** 9,
        ):
            fi = int(round(ts * meta.fps))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dets = _detector.detect_people(self._detector, rgb)
            per_frame.append((fi, [d.box for d in dets]))
            # Sample sparsely: one crop per detection every _TRACK_ID_STRIDE
            # frames is plenty to identify a track, and each costs a mask
            # build (~4.8ms) that the rest of this pass avoids.
            if fi % _TRACK_ID_STRIDE == 0:
                for d in dets:
                    crop = _reid.crop_via_mask(rgb, d.mask, d.box)
                    if crop is not None:
                        crops[(fi, d.box)] = crop

        tracks = _associate(per_frame)
        for t in tracks:
            samples = [
                crops[(fi, box)] for fi, box in t.boxes.items()
                if (fi, box) in crops
            ][:_TRACK_ID_SAMPLES]
            if not samples:
                continue
            # Max over samples, matching runtime's max-pooled matching:
            # "does this track ever look like the speaker".
            t.score = max(
                _reid.max_similarity(_reid.embed_crop(self._reid_model, c), gallery)
                for c in samples
            )

        chosen = _select_tracklet(tracks, meta.width, meta.height)
        if chosen is None:
            logger.debug(
                f"[gesture] scene {scene_start:.1f}-{scene_end:.1f}s: "
                f"{len(tracks)} tracklets, fewer than 2 passed — per-frame path"
            )
            return {"ambiguous": False}

        passers = [t for t in tracks if t.score is not None
                   and t.score > _GALLERY_MATCH_THRESHOLD
                   and t.support >= _TRACK_MIN_SUPPORT]
        logger.info(
            f"[gesture] scene {scene_start:.1f}-{scene_end:.1f}s: {len(passers)} "
            f"tracklets passed the gallery; chose the one at median centre "
            f"distance {chosen.median_centre_distance(meta.width, meta.height):.3f} "
            f"({chosen.support} frames) over "
            + ", ".join(
                f"{t.median_centre_distance(meta.width, meta.height):.3f}"
                f"({t.support}f, score {t.score:.2f})"
                for t in passers if t is not chosen
            )
        )
        return {
            "ambiguous": True,
            "boxes": {str(fi): list(box) for fi, box in chosen.boxes.items()},
        }

    def _process_window(
        self, meta: VideoMeta, start_s: float, end_s: float, window_cuts: list[float],
        gallery: Optional[np.ndarray],
        job_id: Optional[str] = None, all_cuts: Optional[list[float]] = None,
    ) -> GestureFeatures:
        """
        `job_id` and `all_cuts` enable per-scene tracklet selection: the
        scene a frame belongs to is defined by the *whole* video's cuts,
        not the few that fall inside this window, and decisions are cached
        per (job_id, scene_idx) so the several windows overlapping one
        scene agree. Both optional — without them this falls back to the
        per-frame path, which is what a caller with no store does.

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

        gallery is an (N, 512) array of L2-normalised exemplar embeddings
        and is always present — process_job raises without one (see module
        docstring's "Speaker selection").
        """
        raw_frames = frames_for_window(meta.path, start_s, end_s, meta.fps)
        gesture_frames = self._process_frames(
            raw_frames, meta, window_cuts, gallery, job_id, all_cuts,
        )
        return self._aggregate(start_s, end_s, gesture_frames, meta.width, meta.height)

    def _process_frames(
        self,
        raw_frames: Iterator[tuple[float, np.ndarray]],
        meta: VideoMeta,
        window_cuts: list[float],
        gallery: Optional[np.ndarray],
        job_id: Optional[str] = None,
        all_cuts: Optional[list[float]] = None,
    ) -> list[GestureFrame]:
        gesture_frames: list[GestureFrame] = []
        # Per-scene tracklet decisions, resolved lazily as the frames cross
        # into each scene. Function-local on purpose: the pool reuses one
        # GestureWorker for every window a child handles, so anything kept
        # on `self` would leak across windows — and because windows are
        # distributed nondeterministically, that leak would make results
        # vary run to run. Redis is where cross-window sharing belongs
        # (see _scene_decision), not worker state.
        scene_decisions: dict[int, dict] = {}
        use_scenes = job_id is not None and all_cuts is not None and gallery is not None
        # None means "no current lock" — the next frame runs Searching,
        # i.e. a full gallery match (see module docstring's "Speaker
        # re-identification"). Reset every window (never carried
        # across windows) *and* at every scene cut within a window — see
        # next_cut_idx below — and, for a gallery job only, whenever a
        # tracked position jumps further than plausible (see the
        # _MAX_TRACK_JUMP branch below).
        ref_pos: Optional[tuple[float, float]] = None
        # Appearance of the last accepted frame, and how many frames have
        # passed since the lock was last anchored against the gallery.
        ref_emb: Optional[np.ndarray] = None
        lock_age = 0
        next_cut_idx = 0

        # raw_frames streams (core/preprocessing.py's frames_for_window is a
        # generator), so exactly one decoded frame is alive at a time and
        # this loop never sees the window as a whole. That replaced an
        # explicit `raw_frames[i] = None` after each read, which dropped
        # references as it went but could not help with the peak — the list
        # was fully built before the first frame was ever processed.
        for frame_idx, (ts, bgr) in enumerate(raw_frames):
            # A cut landing anywhere at-or-before this frame's timestamp
            # invalidates whatever we were tracking — the next frame is a
            # different shot, so "nearest to ref_pos" would be measuring
            # distance in a scene ref_pos was never computed from.
            while next_cut_idx < len(window_cuts) and window_cuts[next_cut_idx] <= ts:
                ref_pos, ref_emb, lock_age = None, None, 0
                next_cut_idx += 1

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # --- Scene-level tracklet selection ------------------------
            # If this frame's scene was decided ambiguous (two or more
            # tracklets matched the gallery — the speaker and her
            # projection), that decision already names which box is hers
            # in every frame. Use it and skip everything below: no
            # detection, no embedding, no lock. Frames the chosen track
            # does not cover are empty, which is the point — that is what
            # stops the track jumping to the screen whenever she is missed.
            if use_scenes:
                s_idx = _scene_index(ts, all_cuts)
                if s_idx not in scene_decisions:
                    s_start, s_end = _scene_bounds(s_idx, all_cuts, meta.duration_s)
                    scene_decisions[s_idx] = self._scene_decision(
                        job_id, meta, s_idx, s_start, s_end, gallery,
                    )
                decision = scene_decisions[s_idx]
                if decision.get("ambiguous"):
                    box = decision.get("boxes", {}).get(str(int(round(ts * meta.fps))))
                    if box is None:
                        gesture_frames.append(self._empty_frame(frame_idx, ts))
                        continue
                    pose = self._pose_on_crop(
                        rgb, tuple(box), meta.width, meta.height,
                    )
                    if pose is None:
                        gesture_frames.append(self._empty_frame(frame_idx, ts))
                        continue
                    landmarks, world_landmarks = pose
                    gesture_frames.append(self._build_frame(
                        frame_idx, ts, landmarks, world_landmarks,
                        meta.width, meta.height,
                    ))
                    continue

            # Detection first, pose second — see module docstring's
            # "Detector-first pipeline". Nothing here runs a pose model
            # yet: selecting the subject needs positions and (for a
            # gallery job) mask crops, both of which the detector supplies
            # directly, so pose inference is deferred until exactly one
            # candidate has been chosen.
            detections = _detector.detect_people(self._detector, rgb)

            if not detections:
                ref_pos, ref_emb, lock_age = None, None, 0
                gesture_frames.append(self._empty_frame(frame_idx, ts))
                continue

            centers = [_box_center(d.box, meta.width, meta.height) for d in detections]

            # Four things can require a gallery anchor on this frame: no
            # current lock (Searching), the jump guard firing, the lease
            # expiring, or continuity failing. They all resolve the same
            # way — match every candidate against the gallery, here, now —
            # so the anchor is written once below rather than at each
            # trigger.
            chosen = None
            needs_anchor = ref_pos is None

            if not needs_anchor:
                chosen = min(range(len(centers)), key=lambda i: _dist(centers[i], ref_pos))
                # The geometric guard is independent of identity: an
                # implausible move is more likely a track switch than real
                # motion, whatever the candidate looks like. It is the one
                # heuristic left in the selection path, and it only ever
                # *rejects* — it can invalidate a lock but never choose who
                # holds it, so it cannot put the wrong person on the track.
                needs_anchor = (
                    _dist(centers[chosen], ref_pos) > _MAX_TRACK_JUMP
                    # The lease bypasses continuity deliberately: it only
                    # bounds drift if failing it actually breaks the lock.
                    or lock_age + 1 > _LOCK_LEASE_FRAMES
                )

            if not needs_anchor:
                lock_age += 1
                emb = self._embed_candidate(rgb, detections[chosen])
                if emb is None or ref_emb is None or (
                    float(emb @ ref_emb) < _CONTINUITY_THRESHOLD
                ):
                    # Continuity failed — fall through to the gallery on
                    # *this* frame rather than dropping it. A frame the
                    # gallery still recognises is worth keeping: continuity
                    # dips on ordinary motion blur and mask wobble (measured
                    # failures sat at 0.867-0.893), and the gallery catches
                    # exactly those.
                    needs_anchor = True
                else:
                    ref_emb = emb   # the reference walks with the subject

            if needs_anchor:
                match = self._gallery_match(rgb, detections, gallery)
                if match is None:
                    # Explicitly Searching for the next frame too, not still
                    # "locked" onto the stale position.
                    ref_pos, ref_emb, lock_age = None, None, 0
                    gesture_frames.append(self._empty_frame(frame_idx, ts))
                    continue
                chosen, ref_emb = match
                lock_age = 0
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
        Runs the single-person landmarker on a square region of the frame
        around one detection (see _square_crop) and maps the result back
        into frame-normalised coordinates. Returns (landmarks,
        world_landmarks), or None if no pose was found.

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
        crop, origin_x, origin_y = _square_crop(rgb, box)
        if crop.size == 0:
            return None
        crop_h, crop_w = crop.shape[:2]

        result = self._pose_landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
        )
        if not result.pose_landmarks:
            return None

        mapped = []
        for lm in result.pose_landmarks[0]:
            fx, fy = _crop_norm_to_frame_norm(
                lm.x, lm.y, origin_x, origin_y, crop_w, crop_h, frame_w, frame_h,
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
    ) -> Optional[tuple[int, np.ndarray]]:
        """
        Embeds every detected candidate via its own segmentation-mask
        crop, scores each against the gallery by nearest-exemplar (max)
        cosine similarity, and returns `(index, embedding)` for the chosen
        candidate — or None if nobody clears _GALLERY_MATCH_THRESHOLD (this
        frame gets treated as "no speaker here", not a guess). See module
        docstring's "Speaker re-identification"; the actual
        crop/embed/score math lives in workers/_reid.py, shared with the
        dashboard's gallery-building flow.

        Which candidate is chosen depends on how many clear the threshold:

          - **none**  -> None.
          - **one**   -> that one, regardless of where it is in the frame.
          - **two+**  -> the one nearest the frame centre, *not* the highest
                         scorer. See "Why centrality breaks ties" below.

        The winner's embedding is returned rather than just its index
        because the caller needs it as the next continuity reference, and
        it has already been computed here — re-deriving it would pay a
        second ~14ms OSNet pass on every anchor.

        ## Why centrality breaks ties

        On TED-style stages the speaker is often relayed live onto a
        projection screen behind them. That projection is the *same person*,
        so it matches the gallery legitimately — re-ID cannot reject it, and
        a better re-ID model would score it higher, not lower. Worse, it is
        usually a sharp, well-lit close-up while the real speaker is small
        and distant, so "highest score wins" systematically prefers the
        screen. Once chosen, continuity then holds it: a static projection
        is perfectly self-consistent frame to frame.

        Position separates them where appearance cannot. Screens are
        mounted above and beside the stage; the speaker stands on it. On two
        frames of such a scene the distances from frame centre were
        0.057 vs 0.361 and 0.052 vs 0.363 — speaker vs projection, a ~7x
        margin both times, and mostly *vertical*, which is why it should
        generalise beyond these shots rather than being a quirk of framing.

        Scored ranking is deliberately discarded among the passers rather
        than blended with position: the screen's score advantage is exactly
        the bias being corrected, so weighing it back in would reintroduce
        it.

        Because every anchor trigger (Searching, jump, lease expiry,
        continuity failure) resolves here, this also lets the lease repair
        a wrong lock: if the track has drifted onto the screen, the next
        re-anchor pulls it back to the stage. A position prior ("nearest
        the previous ref_pos") would instead preserve the error.

        ## Known limitation, accepted

        The tie-break needs *both* candidates. When the detector finds only
        the projection there is a single passer and it wins, so the track
        goes to the screen.

        That is what drove `_CONF_THRESHOLD` down to 0.1. Measured while it
        was still 0.25: the speaker scored 0.162 — found by YOLO, then
        discarded by the floor — against the projection's 0.89, and the
        same speaker scored 0.64 a few seconds later with her arms raised.
        Her detectability swings ~4x with her pose, so at 0.25 she flickered
        out of the candidate list entirely; at 0.1 she is admitted.

        Note the detector's confidence is *anti-correlated* with
        correctness here — a sharp, well-lit, front-facing close-up on a
        screen outscores a small, dim, side-on figure on a stage — so
        raising the floor makes this worse, and any rule preferring
        higher-scoring detections prefers the screen.

        A residue remains: she can still fall below the *gallery* threshold
        while the sharper projection clears it, and a frame where she is
        genuinely undetected offers nothing to choose. See the module
        docstring's "Tracklet selection", which lifts the decision to the
        scene for exactly that case.

        ## Why max, not a top-K mean

        This used to average the top 3 similarities. The stated reason was
        noise robustness: requiring several exemplars to agree stops one
        fluke high score admitting the wrong person. The problem is that the
        same requirement breaks on a look the gallery only holds once or
        twice — the mean pulls in the gallery's other, legitimately
        different looks and drags a genuine match down. Measured on real
        footage, a correctly-tracked speaker in a scene with one matching
        exemplar scored 0.66-0.68 under top-K (rejected) versus 0.90-0.97
        under max (accepted), while a known impostor stayed at 0.47-0.49
        under both.

        That is not a tuning problem. Gallery-building stops once new looks
        stop appearing, so rare looks are left with only one or two
        exemplars at the moment sampling ends — precisely the condition a
        top-K mean cannot score fairly. Whatever the stopping rule, the
        gallery will always be thin somewhere.

        The two halves of the system pool *differently* on purpose:
        building's redundancy check is top-K (a duplicate must resemble
        several held exemplars — see core/gallery_builder.py's
        GALLERY_REDUNDANCY_TOP_K), matching here is max (one strong
        resemblance is enough to be recognised). Different questions, so the
        asymmetry is intended rather than the silent disagreement it used to
        be.

        Masks come from the detector rather than MediaPipe now. That also
        closes a subtle mismatch: gallery *exemplars* were always built
        from the dashboard's own detections, so embedding runtime
        candidates from a differently-derived mask meant the two sides of
        every cosine comparison had been cropped by different models. Both
        sides now go through the same detector and the same
        crop_via_mask.
        """
        h, w = rgb.shape[:2]
        passing: list[tuple[int, np.ndarray, float]] = []
        for i, det in enumerate(detections):
            emb = self._embed_candidate(rgb, det)
            if emb is None:
                continue
            score = _reid.max_similarity(emb, gallery)
            if score > _GALLERY_MATCH_THRESHOLD:
                passing.append((i, emb, score))

        if not passing:
            return None
        if len(passing) == 1:
            i, emb, _ = passing[0]
            return i, emb

        # Two or more gallery-confirmed candidates: the nearest to frame
        # centre wins. Score is used only to break an exact distance tie,
        # so the choice is deterministic.
        i, emb, _ = min(
            passing,
            key=lambda p: (_dist(_box_center(detections[p[0]].box, w, h), _FRAME_CENTER),
                           -p[2]),
        )
        logger.debug(
            f"[gesture] {len(passing)} candidates cleared the gallery; "
            f"centrality chose #{i} over "
            + ", ".join(f"#{p[0]}({p[2]:.2f})" for p in passing if p[0] != i)
        )
        return i, emb

    def _embed_candidate(self, rgb: np.ndarray, det) -> Optional[np.ndarray]:
        """L2-normalised OSNet embedding of one detection's mask crop, or
        None if the mask is too small/degenerate to crop (workers/_reid.py's
        MIN_MASK_PIXELS). Shared by gallery scoring and the continuity
        check so both compare embeddings built exactly the same way.

        The box is passed purely so crop_via_mask can find the mask's extent
        without scanning the whole frame — it does not change the crop."""
        crop = _reid.crop_via_mask(rgb, det.mask, det.box)
        if crop is None:
            return None
        return _reid.embed_crop(self._reid_model, crop)

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
