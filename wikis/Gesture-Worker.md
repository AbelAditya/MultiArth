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
2. Runs the **YOLO11n-seg** person detector on each frame, getting a box
   and segmentation mask per person (`workers/_detector.py`). No pose model
   runs at this stage.
3. Selects which detection is "the subject" — by matching candidates
   against a human-confirmed **speaker gallery**, then holding the track by
   frame-to-frame appearance continuity, with a max-jump guard and its own
   independent scene-cut detection pass. See "Working" and "Speaker
   selection" below.
4. Runs **MediaPipe Tasks' PoseLandmarker** (single-person, 33-point
   BlazePose topology, `IMAGE` running mode — see "IMAGE mode" below) on
   **only that one detection's crop**, mapping the landmarks back into
   frame coordinates.
5. Computes kinematic features (velocity, amplitude, handedness) from only
   the selected person's landmarks — everyone else discarded before a
   `GestureFrame` is ever built, same as before.

No hand/finger model runs — `GestureFrame.left_hand`/`right_hand` are
always empty, same as on the main branch, though for a different reason
(see "Hand landmarks — tried, removed" below).

## Working

End to end, from gallery construction to a single processed frame. Constants
named here are the live values in
[`core/gallery_builder.py`](../core/gallery_builder.py) and
[`workers/gesture_worker.py`](../workers/gesture_worker.py).

### Phase 1 — Gallery construction (dashboard, interactive)

Runs once per video, before any processing, in both the Bulk Upload and
Live Analysis flows.

1. **Scene detection.** PySceneDetect `ContentDetector(threshold=27.0)`
   splits the video into scenes.
2. **Sampling schedule.** Each scene gets `round(duration / 5s)` slots,
   clamped to `[1, 6]`. Slots are laid out in *interleaved sweeps* — every
   scene gets its 1st slot before any scene gets its 2nd — shuffled within
   each sweep. Each slot owns a distinct, non-overlapping sub-interval of
   its scene's timeline, so coverage of a long scene is structural rather
   than left to chance.
3. **Draw a candidate.** `draw_next_candidate` pops one slot and samples a
   random, not-yet-shown frame index inside that slot's sub-interval,
   retrying up to 4 times *within the same slot* if nobody is detected.
4. **Show candidates.** YOLO11n-seg returns boxes + masks; `crop_via_mask`
   produces background-zeroed crops taken from *the mask's own extent*,
   not the detector box. The researcher clicks the speaker, or skips.
5. **Record.** OSNet embeds the clicked crop (512-d, L2-normalised) into
   Redis at `job:{id}:gallery:{n}`. It is marked **redundant** if
   `top_k(3) >= 0.85` against what is already held — but stored either way;
   redundancy only drives stopping.
6. **Stop** when 16 of the last 20 confirmations were redundant (sliding
   window), or at 70 entries (hard cap), or when the researcher says so.
   Floor of 3.

### Phase 2 — Runtime

**Job setup.** `process_job` **raises if there is no gallery** — it is the
only thing that can acquire a lock — and checks that *before* scene
detection, so a doomed job does not pay a full video pass first. Then its
own independent PySceneDetect pass, and loads landmarker, detector and
OSNet.

**Per window** (5s): `frames_for_window` reads frames, capped at 150. So
30fps footage is processed every frame; 60fps is decimated by 2.

**Per scene, before its frames are processed:** the scene is streamed once,
every detection is linked into tracklets geometrically, each tracklet is
scored against the gallery, and if **two or more pass** the scene is
"ambiguous" — the speaker plus her projection. The most central tracklet by
median distance is chosen and its per-frame boxes are cached in Redis under
`(job_id, scene_idx)`. Frames in such a scene then skip steps 2-4 entirely:
the box is already known, so no detector and no re-ID run, and frames the
chosen track does not cover are emitted empty. See "Tracklet selection".

**Per frame** (scenes that are *not* ambiguous):

1. **Scene cut?** Drop the lock (`ref_pos`, `ref_emb`, `lock_age` reset).
2. **Detect.** YOLO11n-seg, 640x640 letterboxed, confidence >= 0.1, at
   most 20 detections. No pose model has run yet.
3. **No detections** -> empty frame, lock dropped.
4. **Select.** Four triggers require a gallery anchor:

   ```
   needs_anchor = (no current lock)                 # Searching
                or dist(nearest, ref_pos) > 0.30    # jump guard
                or lock_age + 1 > 30                # lease expiry
                or emb . ref_emb < 0.80             # continuity failure

   if any:  _gallery_match over every candidate, max-similarity > 0.80.
            One passer -> it wins. Two+ passers -> the one nearest frame
            centre wins, not the highest scorer (see "The centrality
            tie-break"). No passer -> empty frame, lock dropped.
   else:    keep the proximity pick; ref_emb := emb (the reference walks
            forward with the subject).
   ```

5. **Pose, once.** A square region *of the frame* around the chosen box,
   expanded by `_POSE_CROP_SCALE` (1.5x) and taken at native resolution, is
   run through single-person MediaPipe in IMAGE mode. The region is slid
   inward near a frame edge rather than clipped, so it stays square and
   full of real pixels — padding the box to square with black instead
   starves the model on tall, thin subjects and was measured returning no
   pose on 25 consecutive otherwise-healthy frames. Landmarks are mapped back to frame coordinates by
   `_crop_norm_to_frame_norm`; world landmarks pass through untouched,
   being hip-origin and person-relative, which is why every angle in
   `core/fusion_engine.py` is unaffected by cropping.
6. **No pose fitted** -> empty frame, but the **lock is kept** — the
   detector's track is still good, so a momentary pose failure should not
   cost a full gallery re-search.

**Aggregate per window:** wrist velocity, max displacement, handedness and
pose-present ratio from *every* frame; `pose_keyframes` at `step=3` for the
dashboard viewer.

### The invariant

| mechanism | can it *choose* the subject? |
|---|---|
| gallery match | **yes — the only one** |
| tracklet selection | only *among gallery-passing tracklets*, when 2+ pass |
| centrality tie-break | only *among gallery passers*, when 2+ pass |
| continuity (0.80) | no — accepts or rejects the proximity pick |
| `_MAX_TRACK_JUMP` (0.30) | no — rejects only |

Nothing except a human-confirmed gallery can put a person on the track.
Centrality can choose *which* gallery-confirmed candidate, but never admits
one the gallery rejected — so it cannot introduce a person the researcher
did not confirm. See "Speaker selection" for the centrality vote that was
removed, and "The centrality tie-break" for why this scoped version is
different.

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

## IMAGE mode (was VIDEO)

`PoseLandmarker` runs in `IMAGE` mode (`detect(image)`, every call
independent). It previously ran in `VIDEO` mode
(`detect_for_video(image, timestamp_ms)`, timestamps strictly increasing
across calls to the same landmarker instance), which lets MediaPipe track
between consecutive frames rather than re-detecting every time — faster,
and it damps frame-to-frame jitter (the same underlying problem
`_extract_keyframes`' `step=3` — see below — works around from the other
direction).

That tracking turned out to be the *cause* of a worse failure. In VIDEO
mode MediaPipe derives each frame's ROI from the **previous frame's
landmarks**, re-running the detector only once tracking confidence
collapses. On this project's footage — a small speaker on a wide, cluttered
stage — one frame whose ROI over-covers the speaker plus background yields
a skeleton with legs on the speaker and arms thrown onto background
structure; the next ROI is then computed from *that* corrupted skeleton, so
the error feeds itself and latches for a run of frames rather than
self-correcting. The background needn't look remotely human: BlazePose's
landmark model is a single-person regressor that always emits all 33
landmarks over whatever region it is handed, with no part-association step
that could decline to attach a limb.

IMAGE mode derives every ROI from the image itself, so a bad frame stays
one bad frame. Measured cost was nil (51.4 ms/frame IMAGE vs 53.6 VIDEO —
the tracking shortcut was not buying much at `num_poses=5`), but it does
give up VIDEO's inter-frame damping, and MediaPipe Tasks exposes no
`smooth_landmarks` equivalent to compensate. If jitter becomes the
dominant problem, an explicit landmark filter (One-Euro or similar) is the
route forward, not a return to VIDEO mode.

Switching also removed a real constraint: windows had to be processed in
non-decreasing `start_s` order for the whole job, because
`detect_for_video` raises `ValueError("Input timestamp must be
monotonically increasing")` otherwise. `detect()` takes no timestamp and
holds no cross-call state, so windows may now be reprocessed out of order,
retried individually, or parallelised within a job.

The landmarker is still created **once per job** (`process_job`, not once
per worker instance and not per-window) and closed in a `finally` block at
the end of that same method. Under VIDEO mode that per-job lifetime was
*required* — the monotonic-timestamp requirement applies within one
landmarker instance's lifetime, and each video has its own independent
0-based timeline, so one instance could not validly span two videos. In
IMAGE mode it is merely an optimisation (avoiding repeated model loads),
kept because the worker object itself is reused across many videos in a
bulk run (see "Bulk runs" below).

## Speaker selection

The live mechanism is described in "Working" above. This section is the
history of how it got there, and what was removed.

**A speaker gallery is now required** — `process_job` raises without one.
Only the gallery can *acquire* a lock; continuity maintains it and
`_MAX_TRACK_JUMP` can only invalidate it.

**The centrality vote is gone.** The MeTRAbs-era design was
vote-once/track-thereafter: pick the candidate nearest the frame centre at
a window start or scene cut, then track nearest-to-last-position. That
survived into the gallery era as the fallback for jobs with no gallery, and
was removed for two reasons. It is wrong often enough to matter on this
footage — in a crowded auditorium frame the speaker stood at (0.25, 0.57)
while the seated audience occupied the middle of the frame, so centrality
would have picked an audience member. And as a *fallback* it fails
silently: it always returns somebody, so a job with a missing or expired
gallery would have produced a confident, wrong gesture track rather than an
obvious failure. Raising is louder than degrading.

### The centrality tie-break

Centrality came back later in one narrow role: when **two or more**
candidates have already cleared the gallery threshold, `_gallery_match`
picks the one nearest the frame centre `(0.5, 0.5)` instead of the highest
scorer.

**The problem it solves.** TED-style stages relay the speaker live onto a
projection screen behind them. The projection is the *same person*, so it
matches the gallery legitimately — re-ID cannot reject it, and a better
re-ID model would score it higher, not lower. It is also usually a sharp,
well-lit close-up while the real speaker is small and distant, so "highest
score wins" systematically chose the screen. Continuity then held it there,
since a static projection is perfectly self-consistent frame to frame.

**Why position works.** Screens hang above and beside the stage; the
speaker stands on it. Measured on two frames of such a scene (distance from
frame centre, normalised):

| frame | speaker | projection |
|---|---|---|
| arms down | 0.057 | 0.361 |
| arms raised | 0.052 | 0.363 |

A ~7x margin, and mostly *vertical* — which is why it is expected to
generalise rather than being a quirk of these shots. MediaPipe-derived
depth was tried as the discriminator first and ruled out: it estimated the
projection as **2.89x nearer** than the speaker, and assigned the flat
projection *more* 3D depth structure (`z_spread` 0.661) than the real
person (0.406). It is a single-person regressor that assumes it is looking
at a real human, so it has no way to represent a picture of one.

**Why this is not the heuristic that was removed.** The scoping answers
both original objections:

- *It picked audience members.* They are different people, so they fail
  the gallery threshold before centrality is consulted.
- *It failed silently as a fallback.* It never runs without a gallery
  match, so it cannot manufacture a track.

Score is deliberately discarded among the passers rather than blended with
position — the screen's score advantage is exactly the bias being
corrected. It is used only to break an exact distance tie.

Because every anchor trigger resolves through `_gallery_match`, the lease
also becomes self-repairing: a track that has drifted onto the screen is
pulled back to the stage at the next re-anchor. A position prior ("nearest
the previous position") was considered and rejected for this reason — it
would preserve a wrong lock rather than correct it.

**Accepted limitation.** The tie-break needs both candidates present. When
the detector misses the speaker and returns only the projection, there is
one passer and it wins.

This is what drove `_CONF_THRESHOLD` down to **0.1**. Measured on one such
frame while it was still 0.25: the speaker scored **0.162** — detected by
YOLO, then discarded by the floor — while the projection scored **0.89**,
and the same speaker scored **0.64** a few seconds later with her arms
raised. Her detectability swings roughly 4x with her pose, so at 0.25 she
flickered in and out of the candidate list entirely. At 0.1 she is
admitted, and the tie-break can do its job.

Note the detector's confidence is *anti-correlated* with correctness here:
the wrong answer (a sharp, well-lit, front-facing close-up on a screen)
scores far higher than the right one (a small, dim, side-on figure on
stage). Raising the floor makes this worse, and any rule preferring
higher-scoring detections prefers the screen.

Lowering the floor does not close the gap completely — she can still fall
below the *gallery* threshold while the sharper projection clears it, and
a frame where she is genuinely not detected has no candidate to choose.
That residue is what "Tracklet selection" below addresses.

Tests: `tests/test_gallery_centrality.py`. The one that matters most is
`test_non_passing_central_candidate_is_ignored` — it pins the scoping that
makes reintroduction safe.

### Tracklet selection

The centrality tie-break fixes frames where *both* the speaker and her
projection are detected. It cannot fix frames where only the projection is
— and those are common, because her detector confidence swings with her
pose (**0.64** with arms raised, **0.162** with them down, against a 0.1
floor). Per-frame selection then alternates with whether she happened to be
detected that frame, and the track visibly jumps between her and the
screen. The 30-frame lease bounds each wrong lock but cannot prevent the
next, since every re-anchor is decided from one frame's evidence.

So the decision moved to the scene:

1. Stream the scene once; record **every** detection's box per frame.
2. Link them into tracklets by box overlap (`_associate`) — purely
   geometric, gap-tolerant to `_TRACK_MAX_GAP_FRAMES` (30).
3. Score each *tracklet* from a few sampled crops (`_TRACK_ID_STRIDE`,
   `_TRACK_ID_SAMPLES`), not every frame.
4. If 2+ tracklets pass the gallery and the support floor
   (`_TRACK_MIN_SUPPORT`, 10), pick the one with the lowest **median**
   centre distance. Otherwise return "not ambiguous" and use the per-frame
   path.

Measured on `test_vid_39` (scene 3.6-12.2s) against a researcher-built
gallery:

| tracklet | median centre dist | frames | gallery score | outcome |
|---|---|---|---|---|
| speaker | **0.086** | 148 | — | **chosen** |
| projection (right screen) | 0.467 | 203 | 0.86 | rejected |
| projection (left screen) | 0.477 | 143 | 0.89 | rejected |

Both rejects are *longer* and one scores *higher*. Neither length nor
gallery score is consulted — the projection usually wins on both, which is
the bias being corrected.

**Why gap tolerance is load-bearing.** Her worst measured gap is 14 frames.
At a tolerance below that she fragments into stubs while the static
projection stays one clean track, and any criterion rewarding length then
picks the screen. 30 frames clears it with margin. The same inversion
applies to `_TRACK_MIN_SUPPORT`: raise it above the length of her genuine
track and the screen is all that remains.

**Caching.** A scene routinely spans several 5s windows, and with the
process pool those run in different processes. Decisions are cached in
Redis under `job:{id}:scene:{n}:track` — swept by the existing `delete_job`
wildcard, so no new teardown path. Concurrent writers need no lock: the
computation is deterministic, so a race wastes work but cannot disagree.

Scene indices are shared with the dashboard's gallery builder only because
both call `_detect_scene_cuts`. Changing scene detection on one side alone
would silently repoint every stored index.

**Expect `pose_present_ratio` to fall on relay scenes.** Frames the chosen
track does not cover are emitted empty, which is the honest answer — she
was not detected there — replacing a pose fitted to the screen.

#### Known costs and limitations

- **Non-ambiguous scenes pay for detection twice.** Ambiguity is the
  *output* of building tracklets, not a precondition, so the decision pass
  runs for every scene; where it concludes "not ambiguous" the per-frame
  path re-detects the same frames. Estimated from measured per-stage
  costs, a 40-minute video goes from ~1.41h to ~2.21h pooled. Two fixes
  are known, neither implemented — use the single passing tracklet too
  (~1.46h, but changes behaviour on every scene), or trigger the scene
  pass only after ambiguity is observed for free during the ordinary path
  (~1.48h, preserves the scoping). **This is the main open question.**
- **Identity is judged from the five *earliest* sampled crops**, not five
  spread along the track. An identity switch partway through is invisible,
  and a speaker who begins a scene turned away can have her whole track
  misjudged.
- **Association compares against a tracklet's last box however stale.**
  After a long gap a moving person may fall below `_TRACK_IOU_MIN` and
  start a spurious track while a static projection re-links trivially — a
  bias toward the screen. Not yet biting at the measured 14-frame gap.
- **A scene where the speaker is never detected still fails**: one passing
  tracklet is not ambiguous, so the projection wins by the ordinary path.
  This rests on the observation that she is always detected for at least a
  few frames per scene.

Tests: `tests/test_tracklets.py`. The two that matter are
`test_speaker_with_gaps_stays_one_tracklet` and
`test_short_speaker_track_still_beats_long_projection_track` — both guard
failures that would hand the scene to the screen while raising nothing.

**Positions come from the detector's own box** (`_box_center`). This closed
a gap that existed for the whole MediaPipe era: MeTRAbs's detector gave an
explicit per-person bounding box, but `PoseLandmarkerResult` exposes none
at all (confirmed directly against the installed library — it has
`pose_landmarks`, `pose_world_landmarks`, `segmentation_masks`, nothing
box-shaped). A raw min/max box over all 33 landmarks was considered and
rejected, because BlazePose always estimates a plausible position for every
landmark even when occluded or off-screen (e.g. ankles in a close-up), and
those extrapolated points skew a box centre away from the visible person;
the mean of the shoulder/hip landmarks (indices 11, 12, 23, 24) was used
instead as a stabler proxy. A detector box has no such failure mode, so
neither workaround is needed.

One caveat carried forward: a box centre sits at the body's midpoint where
the torso-mean sat at shoulder/hip level, and `_MAX_TRACK_JUMP` was tuned
against the old quantity — so it is on the retune list.

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
