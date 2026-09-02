# Gesture Worker

Source: [`workers/gesture_worker.py`](../workers/gesture_worker.py),
dispatched in-process by [`core/orchestrator.py`](../core/orchestrator.py)
— same as prosody/verbal/camera, no subprocess involved.
Dashboard section: **Pose Estimation**
Feature model: [`GestureFeatures`](../core/models.py) (in `core/models.py`)

> **Branch note (`light-gesture`):** this branch replaces the main
> branch's MeTRAbs setup entirely with local MediaPipe PoseLandmarker — no
> TensorFlow, no remote-Colab option, and (after an explicit later decision
> — see "Isolation was removed" below) no subprocess isolation either.
> HandLandmarker was also tried on this branch and then deliberately
> removed again (see "Hand landmarks — tried, removed" below); nothing in
> this doc describes hand/finger data any more. This page describes the
> MediaPipe-based implementation as it exists *on this branch*; the main
> branch's `wikis/Gesture-Worker.md` (MeTRAbs, remote offload, subprocess
> isolation, the memory-growth history that motivated all of it) doesn't
> apply here. See "Why this branch exists" below for the reasoning, and
> "vs. the main branch" throughout for what did/didn't carry over.

## What it does

For every time window, `GestureWorker`:

1. Reads frames from the video for that window
   (`frames_for_window`, [`core/preprocessing.py`](../core/preprocessing.py)).
2. Runs **MediaPipe Tasks' PoseLandmarker** (multi-person, 33-point
   BlazePose topology) on each frame, in `VIDEO` running mode — not
   `IMAGE` mode — so MediaPipe uses its own internal frame-to-frame
   tracking rather than re-detecting from scratch every frame (faster, and
   reduces jitter — see "VIDEO mode" below).
3. Selects which detected person is "the subject" — same design as the
   main branch's MeTRAbs implementation, ported over: most-central at a
   window's first frame or right after a scene cut (voted once), then
   nearest-to-last-known-position otherwise, with a max-jump ambiguity
   guard and its own independent scene-cut detection pass. See "Speaker
   selection" below for the one real difference (no bounding box).
4. Computes kinematic features (velocity, amplitude, handedness) from only
   the selected person's landmarks — everyone else discarded before a
   `GestureFrame` is ever built, same as before.

No hand/finger model runs — `GestureFrame.left_hand`/`right_hand` are
always empty, same as on the main branch, though for a different reason
(see "Hand landmarks — tried, removed" below).

## Why this branch exists

The main branch moved from MediaPipe to YOLO-Pose to MeTRAbs specifically
to get genuine multi-person detection with absolute 3D output (see main's
own wiki history). That's real capability, but it comes with real cost:
TensorFlow as a dependency, a model that needed isolating into its own
subprocess to avoid crashing the host machine, and — for meaningfully
faster local inference — a whole remote-Colab-offload system (tunnel
management, a duplicated selection-logic notebook, the associated
operational friction documented at length in main's wiki history).

This branch trades some of that capability back for a simpler dependency
footprint: MediaPipe's models are a few MB each, install via a single pip
package, and need no GPU/CUDA reasoning at all. It's not a strictly
lighter setup in every dimension, though — see "Honest tradeoffs vs. the
main branch" below before assuming "lighter" means "better across the
board."

## VIDEO mode

`PoseLandmarker` supports a `VIDEO` running mode
(`detect_for_video(image, timestamp_ms)`, timestamps strictly increasing
across calls to the same landmarker instance) instead of `IMAGE` mode's
independent-per-call detection. VIDEO mode lets MediaPipe track between
consecutive frames rather than re-running full detection every time —
faster, and reduces frame-to-frame jitter (the same underlying problem
`_extract_keyframes`' `step=3` — see below — works around from the other
direction).

The landmarker is created **once per job** (`process_job`, not once per
worker instance and not per-window) and closed in a `finally` block at the
end of that same method — confirmed directly, not assumed, that this
matters: MediaPipe's monotonic-timestamp requirement applies *within one
landmarker instance's lifetime*, and each video has its own independent,
0-based timeline, so one instance genuinely can't validly span two
different videos, even though the *worker object* itself is reused across
many videos in a bulk run (see "Bulk runs" below).

## Speaker selection

Ported from the main branch's MeTRAbs implementation almost unchanged —
vote-once/track-thereafter, scene-cut-aware resets (this worker's own
independent PySceneDetect pass, same as main, for the same reason: keeping
gesture and camera's dispatch concurrent without cross-worker wiring), a
max-jump ambiguity guard. The one real difference: MeTRAbs's detector gave
an explicit per-person bounding box to compute a "center" from;
`PoseLandmarkerResult` doesn't expose one at all (confirmed directly
against the installed library — it has `pose_landmarks`,
`pose_world_landmarks`, `segmentation_masks`, nothing box-shaped).

A raw min/max bounding box over all 33 landmarks was considered and
rejected — BlazePose, like MeTRAbs, always estimates a plausible position
for every landmark even when occluded or off-screen (e.g. ankles in a
close-up shot), and those wildly extrapolated points would skew a bbox
center away from where the visible person actually is. Centering instead
on the mean of just the shoulder and hip landmarks (BlazePose indices 11,
12, 23, 24 — its own stable torso anchors) is a closer analogue to what
MeTRAbs's detector box represented.

Background-subtraction-based foreground filtering was tried and reverted
on the main branch before this one existed (measured directly: it
rejected a real, continuously-present, actively-gesturing speaker's own
bounding box far more often than it caught anything static/false, since a
speaker mostly stands still and MOG2 can't tell that apart from genuine
background). That finding wasn't model-specific, so it wasn't retried here.

## Hand landmarks — tried, removed

This branch tried adding **HandLandmarker** (21-point per hand) alongside
PoseLandmarker — a separate model with no built-in link to a detected
body, so it required its own proximity-based matching (nearest detected
hand's wrist to the selected person's pose-landmark wrist,
`_HAND_MATCH_MAX_DIST` normalised distance) to attach a hand to the
selected subject. It worked: verified directly against real footage (not
just unit-tested in isolation), a 5-second clean window tracked the same
person's hands consistently across 123/125 and 122/125 frames
respectively, no visible flicker. One real gap found only by testing: hand
landmarks carry a `visibility` attribute but it's always `None` in
practice — unlike pose landmarks, HandLandmarker doesn't populate a real
per-point confidence, so matched hand points were given a flat `1.0`
instead of a genuine score.

It was removed again anyway: nothing downstream (`FusionEngine`, the
dashboard's pose overlay) ever consumed the real per-finger data it
produced — the dashboard's "hands" highlight segment only ever drew
BlazePose's own crude wrist/fingertip-proxy pose landmarks (17-22), both
before this was added and after it was removed, since that overlay was
never wired up to the real 21-point data. Running HandLandmarker roughly
doubled per-frame inference cost for that zero downstream payoff, so it
was pulled: `workers/gesture_worker.py` no longer imports/creates a
`HandLandmarker` at all, `models/hand_landmarker.task` is no longer
downloaded (Dockerfile, `_ensure_model`), and `GestureFrame.left_hand`/
`right_hand` are back to always-empty — same as they were on the MeTRAbs
branch, for the same-shaped reason (no hand model runs), if not quite the
same underlying cause (MeTRAbs never had a hand model to remove).
Revisit if a real downstream consumer (a finger-specific kinematic
feature, or a dashboard overlay actually wired to per-finger data) is
ever built — the matching logic above worked and can be resurrected from
git history rather than re-derived from scratch.

## Frame resolution — downscaling removed, a deliberate, acknowledged risk

Frames are **no longer downscaled** before detection. `_resize_scale`/
`_MAX_DIM` (aspect-preserving, longer edge capped at 960px — carried over
unmodified from the MeTRAbs branch when this branch was first built) were
removed by explicit choice, after this branch was found to produce visibly
less stable/accurate pose output than this project's own original,
pre-MeTRAbs MediaPipe implementation — which ran at full native
resolution, no downscaling at all. The reasoning that originally motivated
downscaling — BlazePose's *landmark* model only ever sees a fixed 256x256
crop per detected person regardless of source resolution — is still true,
but it only ever covered half the pipeline: the separate *person
-detection* step that decides where that crop goes does see the frame at
whatever resolution it's given, and a lower-resolution input plausibly
costs real precision there. Downscaling had been trading that away for a
memory-safety guarantee, never benchmarked against the alternative until
this comparison against the original MediaPipe branch surfaced it as a
likely cause.

Worth being direct about what this reintroduces: `core/preprocessing.py`'s
`frames_for_window` holds up to 150 full-resolution frames per window in
one list regardless of which model consumes them — at 1080p that's
~930MB, at 4K ~3.7GB, held raw before any inference starts. This is the
exact memory profile directly confirmed (via `journalctl`/OOM-killer
forensics) to have caused a real crash on the MeTRAbs branch, and
downscaling was the fix for it. That risk is real again now, unmitigated
— a live, accepted tradeoff made in exchange for accuracy, not a closed
question, and worth revisiting if this branch sees a crash resembling
that one.

One thing that doesn't change: MediaPipe's coordinates still come back
already normalised to [0, 1] (unlike MeTRAbs's raw pixel output), so
there's still no rescale-back-to-original-resolution step needed —
removing downscaling changes nothing about how coordinates are handled,
it just means `meta.width`/`meta.height` (used to reconstruct pixel-space
`Landmark.x/y` for `_aggregate`'s velocity/displacement math) now always
reflect the frame's true original dimensions rather than a downscaled
stand-in.

## Model tier — switched from "lite" to "full"

`pose_landmarker_lite.task` was the original choice on this branch (a
handful of MB, chosen for speed with no accuracy comparison ever run
against it — see the retired wiki text this replaces). Switched to
`pose_landmarker_full.task` (~9.4MB, still small) after being identified
as a likely cause of the same stability/accuracy regression above:
Tasks API's lite/full/heavy tiering is the direct descendant of the
project's original, pre-MeTRAbs MediaPipe branch's own explicit
`model_complexity=1` (Holistic Solutions API's 0/1/2 = lite/full/heavy) —
confirmed directly from that commit's own code, not assumed. This branch
had drifted onto the smallest tier without that being a deliberate
accuracy decision; switching to `full` restores the same tier the
original branch actually used.

## Bulk runs

`workers/gesture_subprocess.py`/`workers/gesture_server.py` (main
branch's per-job spawn and persistent-server subprocess machinery,
respectively) don't exist on this branch at all — see "Isolation was
removed" below. Bulk runs (CLI `analyze bulk`, the dashboard's Bulk Upload
tab) instead get warm-across-the-batch gesture handling for free, the same
way prosody/verbal/camera already did: `BulkOrchestrator` constructs one
`Orchestrator` for the whole manifest, and `Orchestrator._gesture_worker`
is a lazy property (same pattern as the other three) — so one
`GestureWorker` instance, in-process, gets reused across every video in
the batch.

Because one landmarker instance can't span multiple videos (see "VIDEO
mode" above), `GestureWorker.process_job` still opens fresh landmarkers at
the start of every job and closes them at the end, even though the worker
object itself persists across the whole batch — unlike MeTRAbs's model,
which really did just stay loaded across every job it ever handled.
Measured directly, that reopen-per-job cost turns out to be cheap in
practice: ~4.5s the first time in a process (paying Python's one-time
`import mediapipe` cost), then ~0.09s on every subsequent job in the same
process — so a bulk batch still only pays a real load cost once, for its
first video, via near-instant recreation rather than never closing the
landmarkers at all.

## Honest tradeoffs vs. the main branch

Measured directly against real footage (`vids/test_vid1.mp4`, 1920x1080
@25fps) rather than assumed from architecture alone:

| | MediaPipe (this branch) | MeTRAbs (main branch) |
|---|---|---|
| Peak local memory (after real use) | ~1.9GB — measured while HandLandmarker was still running; not re-benchmarked after its removal (see below), presumably somewhat lower now | ~1.8-2.7GB |
| Model load time | ~4.5s first time in a process, ~0.09s after | ~15-22s, every time |
| Per-joint confidence | Real `visibility`/`presence` per landmark | Only a per-person box score; this project faked a pseudo-visibility (in-frame-bounds check) |
| World coordinates | `pose_world_landmarks` — metric (meters), but **hip-relative, not absolute camera-space depth** (confirmed: sample values sit in roughly [-1, 1] meters) | Genuinely absolute, camera-relative metric 3D, in millimetres (derived from an assumed FOV) |
| Hand/finger landmarks | None — tried (HandLandmarker), then removed again; see "Hand landmarks — tried, removed" above | None at all |
| Remote/GPU offload option | None — solely local by design | Optional, via Colab (see main's wiki) |
| Dependency footprint | `mediapipe` only, a few MB of model files | `tensorflow`, a much larger model download, GPU/CUDA reasoning for the (unused, locally) remote path |
| Process isolation | None — runs in-process, same as prosody/verbal/camera (see "Isolation was removed" below) | Isolated subprocess/persistent server, required — TensorFlow's allocator doesn't reliably release memory back to the OS otherwise |

The headline "lighter" framing for this branch is real for *load time* and
*dependencies*, not for *peak memory* — worth not overstating that in
either direction, especially given the isolation decision below.

## Isolation was removed — a deliberate, acknowledged risk

Subprocess isolation was kept initially on this branch, specifically
*because* the measured ~1.9GB figure above is comparable to what
originally caused real out-of-memory crashes when MeTRAbs ran
concurrently with the other three workers (see main branch's wiki history
for that story) — there was no measured basis to assume dropping it was
safe just because the model changed. It was subsequently removed anyway,
by explicit decision, once gesture became "just another lazy in-process
worker" felt more valuable than the safety margin isolation provided.

Worth being direct about what that means: this branch now runs gesture in
the *same* process as prosody/verbal/camera, sharing memory with all three
concurrently the same way MeTRAbs did before isolation was added — the
same category of risk that motivated isolating it in the first place,
reintroduced here on the reasoning that MediaPipe's absolute footprint
(~1.9GB) is smaller than whatever full-pipeline peak actually crashed the
original 15GB-RAM dev laptop (that number isn't in this project's own
measured record, only the MeTRAbs-alone figure is). Removing HandLandmarker
(see "Hand landmarks — tried, removed" above) trims real per-frame
inference cost, but the ~1.9GB peak-memory figure above was measured
*with* HandLandmarker running, so this branch's actual footprint now is
unmeasured-but-presumably-somewhat-lower, not re-benchmarked after its
removal. No crash has been
observed under this setup so far, but it also hasn't been stress-tested
against a long video or a large bulk batch the way the original crash was
found — this is a live, accepted tradeoff, not a closed question, and
worth revisiting if this branch ever sees a crash resembling the
MeTRAbs-era ones.

One more piece of real history worth knowing, which is what this decision
actually returns to: neither `gesture_server.py` nor
`gesture_subprocess.py` existed during this project's *original*
MediaPipe era (confirmed via git history — both were added in the same
commit that introduced MeTRAbs). Back then gesture ran in-process like
every other worker, no isolation at all — though that was a simpler,
single-person Holistic-only setup — lighter than this branch's
multi-person PoseLandmarker even after HandLandmarker's removal — so "no
isolation worked fine before" isn't quite the same claim as "no isolation
is fine for *this* setup" — the paragraph above is the honest version of
that claim.

## Implementation notes

- Windows must be processed in non-decreasing `start_s` order across a
  job — confirmed directly (not assumed): `detect_for_video` raises
  `ValueError("Input timestamp must be monotonically increasing")`
  otherwise. `process_job`'s own loop already guarantees this
  (`core/preprocessing.py`'s `compute_windows` builds windows in
  chronological order), so this isn't something callers need to actively
  manage today — worth knowing if that ever changes.
- `_extract_keyframes` keeps `step=3` (not `1`), carried over from the
  main branch's own finding that full per-frame density visibly picked up
  per-frame jitter with no temporal smoothing between displayed samples.
  VIDEO mode's own internal tracking may make a smaller step viable here
  even though it wasn't on main, but that hasn't been tested. Bumped from
  `step=2` to `step=3` by explicit choice, not a new finding — not
  re-benchmarked against `2`, just carried forward as the current
  known-good value.

## Package documentation

| Package | Role | Docs |
|---|---|---|
| mediapipe | PoseLandmarker (body only — HandLandmarker was tried and removed, see above) | https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker |
| PySceneDetect (`scenedetect`) | This worker's own independent scene-cut pass for `ref_pos` resets | https://www.scenedetect.com/docs/latest/api.html |
| OpenCV (`opencv-contrib-python`, pulled in by mediapipe itself on this branch) | Frame colour conversion, resizing | https://docs.opencv.org/4.x/ |
| NumPy | Velocity/displacement math | https://numpy.org/doc/stable/ |
| Pydantic | `GestureFeatures`/`GestureFrame`/`Landmark` models | https://docs.pydantic.dev/latest/ |
| loguru | Per-window/job logging | https://loguru.readthedocs.io/en/stable/ |

See also [Home](Home.md) for the full dependency list.
