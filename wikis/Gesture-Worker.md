# Gesture Worker

Source: [`workers/gesture_worker.py`](../workers/gesture_worker.py),
run via [`workers/gesture_subprocess.py`](../workers/gesture_subprocess.py)
/ [`workers/gesture_server.py`](../workers/gesture_server.py) and
dispatched by [`core/orchestrator.py`](../core/orchestrator.py) — see
"Running in isolation" / "Bulk runs" below. Optionally offloaded to
[`colab/gesture_server.ipynb`](../colab/gesture_server.ipynb) — see
"Remote MeTRAbs, local fallback" below.
Dashboard section: **Pose Estimation**
Feature model: [`GestureFeatures`](../core/models.py) (in `core/models.py`)

## What it does

For every time window, `GestureWorker`:

1. Reads frames from the video for that window
   (`frames_for_window`, [`core/preprocessing.py`](../core/preprocessing.py)).
2. Runs **MeTRAbs** on each frame — a metric-scale, absolute 3D, multi-person
   pose model (it detects every person in frame and estimates a 19-point
   skeleton, in millimetre-scale 3D world coordinates, for each one in a
   single pass) — and selects which detected person is "the subject":
   - **First frame of the window**: whoever's detection-box center is
     closest to the frame center is picked. This is the one and only
     "vote" for the whole window.
   - **Every subsequent frame**: no re-vote — whoever's box center is
     closest to the *previously selected* person's position is picked
     instead, and that becomes the new reference position. This
     deliberately doesn't re-open "who's the subject" mid-window, so a
     second person becoming briefly more central doesn't cause the
     selection to flip. There's currently no "is this still plausibly the
     same person" ambiguity guard (e.g. a max-distance cutoff) — held off
     pending more real multi-person test data to judge whether it's
     actually needed.
   - The vote resets fresh at the start of every window (never carried
     across windows), bounding how long a bad pick can persist to at most
     one window (5s by default) before self-correcting.
   - Everyone *not* selected in a given frame is discarded before a
     `GestureFrame` is ever built — downstream code (`FusionEngine`, the
     dashboard) has no idea multiple people were ever in frame.
3. Aggregates the selected person's per-frame landmarks into window-level
   kinematic features:
   - `mean_wrist_velocity` — mean speed (px/s) of both wrists across the window.
   - `max_wrist_displacement` — largest bounding displacement of either wrist.
   - `pose_present_ratio` — fraction of frames where a full 19-point pose was detected.
   - `handedness_ratio` — ratio of right- vs. left-hand motion (−1 fully
     left-dominant → +1 fully right-dominant).
   - `pose_keyframes` — subsampled (every 3rd frame), normalised pose
     snapshots (2D **and** 3D world coordinates) used to drive the animated
     pose overlay in the dashboard and `FusionEngine`'s camera-angle math.
4. Stores the resulting `GestureFeatures` record in the Redis
   [`FeatureStore`](../core/feature_store.py) under `job:{id}:gesture:{w}`.

## Values shown on the dashboard

The **Pose Estimation** section has one KPI card and two charts, plus the
video-overlaid skeleton:

| Where | What it shows |
|---|---|
| KPI card: **Wrist Vel.** | Average wrist speed across the whole video. |
| Pose overlay (on the video itself) | The tracked skeleton drawn over the video at the current playback moment, toggleable by body segment (Head/Face, Arms, Hands, Torso, Gaze) so you can isolate the part you care about. |
| **Wrist Velocity** chart | How fast the hands are moving, per time window — a higher value means more energetic/animated gesturing in that window. |
| **Handedness** chart | Which hand is doing more of the moving in each window, from −1 (fully left-dominant) through 0 (both hands equally) to +1 (fully right-dominant). |

`max_wrist_displacement` and `pose_present_ratio` are computed per window but
aren't currently surfaced anywhere in the dashboard UI.

## Why MeTRAbs, and what changed getting there

This worker has been through several pose-estimation engines, each swapped
in response to a concrete problem with the previous one:

1. **MediaPipe Holistic** (original) — a single-person model, which handled
   multiple people in frame poorly (no concept of tracking a specific
   person; whichever region its internal detector picked could silently
   swap between people frame to frame, corrupting velocity/displacement
   features and shot-type classification alike).
2. **MediaPipe's own multi-person API** (Tasks API `PoseLandmarker`,
   `num_poses`) — lowest migration cost at the time (same vendor, same
   BlazePose lineage), but a side-by-side accuracy comparison showed its
   predictions were still worse than expected.
3. **YOLO-Pose** (Ultralytics) — better accuracy, but AGPL-3.0-licensed,
   which wasn't accepted on this branch (built and kept on a separate one).
4. **RTMPose** (via `rtmlib`, not the full `mmpose` framework — see below
   for why) — Apache 2.0, good multi-person accuracy, no PyTorch
   dependency. Landed here for a while, but it's a **2D-only** model:
   `FusionEngine`'s camera yaw/pitch angle computation needs 3D world-space
   landmarks and had nothing to compute from, permanently returning `None`
   (rendered as empty "UNKNOWN" bars in the dashboard).
5. **MeTRAbs**, landed on here — chosen after researching alternatives that
   support 3D world-space output specifically:
   - MediaPipe's multi-person API does give 3D world landmarks, but reusing
     it would reopen the same accuracy complaint that motivated leaving it
     in step 2.
   - **NLF** (NeurIPS'24, a newer relative of MeTRAbs, reportedly even more
     accurate) has MIT-licensed code but **noncommercial-research-only**
     pretrained weights — ruled out the same way YOLO-Pose's AGPL-3.0 was.
   - **MeTRAbs**: MIT license (code *and* weights), natively multi-person,
     absolute metric-scale 3D output — and per an independent 2025 study
     benchmarking 16 pose frameworks head-to-head (MediaPipe, `rtmlib`,
     YOLOv8, MMPose, ViTPose, etc.), the single highest-accuracy performer
     overall, with MediaPipe notably absent from the top tier.

   Loaded as a **standalone TensorFlow SavedModel**, not via the training
   codebase or `tensorflow-hub` — `tf.saved_model.load()` on a model
   directory downloaded once at build time (see Implementation notes).
   Plain `tensorflow` on PyPI is CPU-only by default and, unlike plain
   `torch`, doesn't pull in a CUDA toolkit — confirmed via a clean install.

   (Aside, from the RTMPose era, still true here: `mmcv`/`mmdet`/`mmpose`
   — the full framework MeTRAbs's underlying research area is normally
   associated with — haven't been co-released since ~early 2024, and
   getting a working combination against a current toolchain means
   precisely pinning `setuptools`/`numpy`/`torch`/`mmcv`/`mmdet` all at
   once, several of which conflict with each other — confirmed hands-on to
   be a real dead end. MeTRAbs sidesteps this entirely by shipping as a
   plain TensorFlow SavedModel with no dependency on any of it.)

Two real, current tradeoffs worth knowing about:

- **No per-joint confidence.** Unlike RTMPose/MediaPipe, MeTRAbs's
  `detect_poses` returns only a per-*person* detection-box confidence — no
  per-keypoint score, occlusion flag, or uncertainty (confirmed directly
  against the model, not just its docs). It always estimates a plausible
  position for all 19 joints of a detected person, including ones that are
  occluded or entirely outside the frame (e.g. cropped-off legs),
  extrapolated from its learned body-shape prior. In place of a real
  confidence value, every landmark gets a pseudo-visibility of 1.0 if its
  2D projection actually lands inside the frame, else 0.0 — good enough for
  the existing visibility-threshold logic downstream (shot classification,
  wrist-motion filtering), but it's an in-frame check, not a confidence
  estimate.
- **Speed/footprint.** This is a TensorFlow model running on CPU — real,
  unavoidable baseline overhead regardless of checkpoint choice (~1.8GB
  resident just from loading the runtime + model, measured directly). Uses
  `metrabs_mob3s_y4t`: MobileNetV3-Small backbone *and* YOLOv4-**tiny** as
  the person detector. The detector dominates total footprint far more
  than backbone choice does — confirmed by comparing published checkpoint
  sizes, where a backbone jump (`mob3s_y4` -> `mob3l_y4`) changes the
  download by <5%, but the `t`-suffixed (tiny-detector) variant is 8x
  smaller (248MB -> 31MB). Originally shipped with the full-YOLOv4
  `metrabs_mob3s_y4` instead; its ~5.9GB resident footprint during
  inference caused a real out-of-memory crash on a development laptop
  (15Gi total RAM) — switching to the tiny-detector variant brought peak
  RSS for a full window down to ~2.7GB and roughly halved per-window
  processing time, with detection output looking equivalent on the
  (single-subject) test videos tried so far. Real tradeoff, per MeTRAbs's
  own published numbers: worse multi-person detection accuracy (MuPoTS
  PCK@150mm 76.8 vs 81.8 — see "Benchmark accuracy" below) — acceptable
  here since this pipeline only needs the detector to reliably find *the*
  subject, not exhaustively catalog everyone in frame, but worth
  revisiting if that stops being true.
  Larger backbones (EfficientNetV2-based, up to `eff2l`) exist for more
  accuracy at more cost, but backbone was never really the lever that
  mattered here — see `docs/MODELS_6_DATASETS.md` in the MeTRAbs repo.

And one real, current *gain* over every prior engine: MeTRAbs's `coco_19`
skeleton includes both a **neck** and a **pelvis** landmark, which neither
MediaPipe's 33-point topology nor RTMPose's COCO-17 had as dedicated
points — useful stable torso anchors, and camera yaw/pitch
(`core/fusion_engine.py`) work again, computed from real 3D data instead of
always returning `None`.

## Running in isolation, not in the dashboard's main process

Even the tiny-detector checkpoint's ~2.7GB footprint (see above) was still
enough to crash a 15GB-RAM development laptop **twice** in practice —
including once as a full system hang requiring a hard reboot (no swap
configured on that machine at the time). The actual cause wasn't the model
size alone: `Orchestrator` used to construct every worker — including
`GestureWorker` — eagerly in `__init__`, meaning MeTRAbs's TensorFlow
runtime was loaded and resident for the *entire dashboard server
lifetime*, and `core/orchestrator.py`'s `_run_parallel` ran all four
workers' `process_job` calls concurrently via a `ThreadPoolExecutor` —
so MeTRAbs's own inference-time memory compounded with Whisper
(`VerbalWorker`, also loaded eagerly at startup) and the other workers all
being active at once.

Before landing on MeTRAbs's current setup, two lighter-weight 3D
alternatives were checked and ruled out: **NLF** (a newer, reportedly even
more accurate relative of MeTRAbs) — its only genuinely multi-person
checkpoints are ~287MB (PyTorch) or ~613MB (TensorFlow), comparable to or
bigger than the MeTRAbs checkpoint that already crashed the laptop, and
PyTorch would reintroduce the "pip torch pulls the full CUDA toolkit"
problem already avoided once (by choosing `rtmlib` over YOLO-Pose,
earlier). **OpenPose** and **AlphaPose** were also considered as
non-TensorFlow options, but both were ruled out on licensing grounds first
(noncommercial-research-only and restrictive/unclear licenses
respectively) before their weight was even evaluated.

The actual fix has two parts, both necessary — one alone doesn't solve it:

1. **`GestureWorker` is never constructed in `Orchestrator.__init__`.**
   It doesn't exist anywhere in the main dashboard process's memory.
2. **Gesture analysis runs in its own subprocess, once per job** —
   `workers/gesture_subprocess.py`, invoked via `subprocess.run(["python",
   "-m", "workers.gesture_subprocess", job_id, video_path,
   window_size_s])` from `Orchestrator._run_gesture_isolated`. That
   subprocess imports `tensorflow`, loads MeTRAbs fresh, runs
   `GestureWorker.process_job`, writes results straight to the same
   Redis-backed `FeatureStore` the main process uses (same
   `REDIS_HOST`/`REDIS_PORT` env vars), and then **exits completely**.

Point 2 matters for a specific, verified reason: TensorFlow's memory
allocator does **not** reliably release memory back to the OS even after
the Python model object is deleted and `gc.collect()` is run — a
well-documented TensorFlow behavior, not something specific to this
project. An in-process "load lazily, then `del` it" approach would not
have been a reliable fix; only a subprocess that fully terminates
guarantees the OS reclaims everything. This was confirmed directly: a real
end-to-end run's memory was sampled every 5s, and usage dropped by
~550MB the moment the gesture subprocess exited.

`Orchestrator.analyze()` originally ran the gesture subprocess **strictly
after**, never alongside, the other three workers — isolation by itself
only guarantees memory gets released *afterward*, it says nothing about
whether MeTRAbs was the only heavy thing running *at the time*, which is
what actually caused the crashes, so both parts (isolation + not
overlapping) were needed together. That "run strictly after" half has
since been deliberately reverted: gesture's subprocess is now dispatched
*alongside* prosody/verbal/camera (a 4th task in `_run_parallel`'s
`ThreadPoolExecutor`), a known, explicitly chosen tradeoff — accepting the
real risk of the crash pattern recurring in exchange for shorter total
wall-clock time per job, on the reasoning that isolation still guarantees
the memory gets reclaimed once the job ends, even though it no longer
guarantees gesture was the only heavy thing running while it was active.
Worth reverting back to strictly-after if that tradeoff stops being
acceptable, e.g. on a machine without much RAM headroom and no swap
configured.

The real cost of the *isolation* itself (independent of the above): a
fresh spawn-per-job subprocess pays MeTRAbs's ~15-22s model-load time on
**every job**, not once — a deliberate trade of latency for memory safety,
for single-video jobs. See the next section for how bulk runs avoid paying
that cost per video while keeping the same reclaim-on-exit guarantee.

**Logs stream live, not in one dump at the end.** `Orchestrator._run_gesture_isolated`
uses `Popen` with a piped stdout/stderr, reading and re-logging each line
as the subprocess produces it (`for line in process.stdout: logger.info(...)`),
rather than `subprocess.run(..., capture_output=True)` — that call is fully
blocking and only hands back output once the whole subprocess has already
exited, so nothing would be visible while a job is running, only a dump
afterward. Matters in practice for actually seeing, live, whether a given
job ran locally or hit remote MeTRAbs successfully (see "Remote MeTRAbs"
below) — `subprocess.run`'s capture used to silently swallow that on the
(common) success path entirely, only printing it on failure. Getting real
streaming also needed the child's own invocation to include `-u`
(`python -u -m workers.gesture_subprocess ...`) — Python block-buffers
stdout by default when it isn't an interactive terminal (a pipe never is),
so without it the child would sit on its own output internally regardless
of how promptly the parent reads its end of the pipe. Verified directly:
timestamped log lines arrive progressively across a job's real runtime
(model load at t≈2.5s, window completions spread across the full ~100s a
job took), not all at once at the end.

## Bulk runs: a persistent gesture server instead

Single-video jobs (the dashboard's main Analyze tab, CLI's `analyze run`)
use the spawn-per-job subprocess above — reasonable for occasional,
casual use, where you don't want MeTRAbs resident in memory in between.
Bulk runs (CLI `analyze bulk`, the dashboard's Bulk Upload tab) are
different: potentially dozens of videos processed back to back, where
paying that ~15-22s load cost *per video* adds up to real, wasted time
across a whole manifest.

For these, `Orchestrator(persistent_gesture=True)` — set automatically by
both bulk entrypoints — routes gesture dispatch to
[`workers/gesture_server.py`](../workers/gesture_server.py) instead of
`workers/gesture_subprocess.py`: a long-lived, localhost-only HTTP server
(`Orchestrator._ensure_gesture_server` starts it on first use, picking a
free port) that loads MeTRAbs once, on its first `/process_job` request,
and stays warm for every subsequent video in the batch — verified directly
(two jobs sent to the same running server reused the same OS process, and
the second job showed no reload cost). `Orchestrator.close()` — already
called by both `BulkOrchestrator.run()` and the CLI's `run`/`bulk` commands
in a `finally` block — terminates that server process when the whole batch
finishes, so the OS reclaims its memory at that point (confirmed directly:
system memory dropped back to its pre-batch baseline immediately after
`close()`), same reclaim guarantee as the per-job subprocess, just on a
batch-scoped cycle instead of a job-scoped one rather than lingering
indefinitely the way an in-process `GestureWorker` reused across a batch
would have (TensorFlow's allocator not reliably releasing memory applies
here exactly as it does everywhere else in this file — a process actually
exiting is what makes reclaim reliable, so `gesture_server.py` being a
separate process, not a plain object living inside the long-running
dashboard process, is what makes this safe for the dashboard's Bulk Upload
tab specifically, not just the CLI's naturally-short-lived process).

## Remote MeTRAbs, local fallback

`GESTURE_REMOTE_URL` (+ `GESTURE_API_KEY`), if set, points at a
[`colab/gesture_server.ipynb`](../colab/gesture_server.ipynb) instance —
the same MeTRAbs checkpoint, running on Colab's GPU (typically a T4)
instead of this project's CPU-only local setup. Chosen deliberately, not
by default: a local GPU was already ruled out earlier (CUDA-version
mismatch against the system install, and only 4GB VRAM on the dev
laptop) — Colab sidesteps both.

**HTTP requests are per window, not per frame or per video — but the
*model* is called per frame within that.** These are two independent
decisions, worth separating clearly:

- *Network batching (per window)*: a per-frame HTTP calling pattern was
  considered and rejected outright — a window can have up to 150 frames
  (`frames_for_window`'s cap), and network latency across that many
  individual round-trips would likely dwarf actual inference time; this is
  exactly why gesture wasn't the first worker remoted (SenseVoice was,
  specifically because it needs only one call per *video*). Whole-video-
  at-once was also considered — fewer round-trips still, and video codecs
  compress better than a pile of independent JPEG frames — but per-window
  was chosen instead because it keeps a natural, low-effort path to real
  per-window progress reporting later (each window's result already
  arrives as its own discrete response), for a modest cost: dozens of
  round-trips per video instead of one.
- *Model batching (per frame, not per window)*: the notebook originally
  used MeTRAbs's own batched-inference call (`detect_poses_batched` —
  confirmed directly against the model: takes a stacked `[N, H, W, 3]`
  array, returns `boxes`/`poses2d`/`poses3d` as `RaggedTensor`s, one
  ragged row per frame) to process a whole window's frames in one model
  call. This caused a real, measured problem — see "A memory-growth dead
  end, and what actually fixed it" below — and was replaced with a loop
  calling the single-image `detect_poses` once per frame, still within
  the one per-window HTTP request.

## A memory-growth dead end, and what actually fixed it

Colab's system RAM (not GPU RAM — checked explicitly) climbed until the
notebook crashed partway through real use. Root cause, confirmed by direct
measurement, not guessed: `detect_poses_batched` is a `tf.function`
internally, and TensorFlow permanently caches a freshly-compiled graph for
every *new* input shape it sees — it never gets freed. A window's frame
count varies almost every request (the last window of a video is shorter,
`frames_for_window`'s fps-dependent subsampling varies, etc.), so nearly
every request hit a batch size TensorFlow hadn't seen before:

```
RSS after model load: 1870 MB
call  1 (batch=129): RSS = 6587 MB   <- first-ever trace, huge one-time jump
call  2 (batch=133): RSS = 7161 MB
...
call 12 (batch=144): RSS = 7860 MB   <- still climbing, no sign of levelling off
```

First fix tried: pad every window's frames to a **fixed** batch size
(`_MAX_BATCH = 150`) with dummy frames, so there's only ever one shape to
trace. This measurably worked *for the retracing problem specifically* —
growth dropped from a continuous climb to a plateau by around the 10th
call — but real Colab usage still ran out of RAM. The batched call itself,
even at a constant shape, still requires TensorFlow to allocate buffers
sized for a 150-image batch on every single call — a large, unavoidable
per-call cost that padding didn't address, since it only fixed *which*
shapes got traced, not how much memory one call needs regardless of shape.

The fix that actually worked: stop batching the model call at all. Loop
over the window's frames and call plain `detect_poses` once per frame —
the same design `_process_window_local` already used locally, which never
showed this problem, for a simple reason: a single video's frames are
always the same resolution, so a per-frame call's input shape is constant
within a video, meaning at most one trace per resolution, not one per
window. Measured directly, same test scenario as above (varying "real"
frame counts, now processed one at a time instead of batched-with-
padding):

```
RSS after model load: 1871 MB
window 1 (30 frames): RSS = 2456 MB
window 2 (30 frames): RSS = 2474 MB
window 3 (30 frames): RSS = 2479 MB
window 4 (30 frames): RSS = 2487 MB   <- plateaued, and at a much lower level
```

Real tradeoff: `detect_poses_batched` was almost certainly faster per
window than 150 sequential single-frame calls (better GPU utilisation from
batching) — worth it given the alternative was fast until it crashed. The
**local** paths (`gesture_subprocess.py`/`gesture_server.py`) were never
affected by any of this — they always called `detect_poses` per frame, not
batched, so this whole investigation only changed the notebook.

**Selection logic runs on the remote side.** The vote-once/track-
thereafter subject-selection logic (see "What it does" above) is
duplicated into the notebook rather than only living in this file — same
trade already accepted for `colab/sensevoice_server.ipynb` duplicating
`_transcribe_alt`'s call; **keep the two in sync** if this logic ever
changes. This keeps response payloads down (only the *selected* person's
landmarks per frame, not everyone detected) and is safe to do statelessly
per window, since the vote always resets fresh at a window's start —
never carried across windows, by design — so no tracking state needs to
cross the network boundary at all; each window's request is fully
self-contained.

**Local fallback, not a hard requirement once remote is configured.** Any
failure calling remote (network, timeout, bad response) falls back to
running MeTRAbs locally for that window — and every subsequent window in
the same job, via a per-job flag (`_remote_failed_this_job`), so a dead
tunnel doesn't retry-and-fail on every single window of a video, just
once. The flag resets at the start of the next job, so a later video gets
a fresh chance in case the failure was transient. Local MeTRAbs, when it
does run, still runs inside the same isolated-subprocess/persistent-server
setup described above ("Running in isolation" / "Bulk runs") — unchanged;
neither `Orchestrator` nor `gesture_subprocess.py`/`gesture_server.py`
have any idea whether a given call ended up hitting Colab or running
locally, that decision is entirely internal to `GestureWorker`.

**TensorFlow's own `import` is deferred**, not just the model load — moved
from module top-level into `_get_model()`, mirroring
`VerbalWorker._get_sensevoice`'s lazy `from funasr import AutoModel`
exactly. This means a successful remote call costs nothing TF-related in
whichever process runs it, even though that process (the per-job
subprocess, or the persistent bulk server) still technically exists and
still gets spawned — the process-spawn overhead itself (~1-2s of Python
startup) was judged an acceptable, much smaller cost than the alternative
of restructuring `Orchestrator`/the subprocess entrypoints to skip
spawning entirely on a remote success, which would have meant a bigger
refactor for a comparatively small win.

## Benchmark accuracy (published)

MeTRAbs's own published numbers (`docs/MODELS_6_DATASETS.md` in the
[MeTRAbs repo](https://github.com/isarandi/metrabs)), verified directly
against the table, for the exact checkpoint family this project uses
(MobileNetV3-Small backbone) — 3DPW is single-person body-shape accuracy,
MuPoTS is multi-person detection accuracy:

| Checkpoint | Detector | 3DPW PCK@50mm | 3DPW MPJPE | MuPoTS PCK@150mm |
|---|---|---|---|---|
| `metrabs_mob3s_y4t` (**used by this project**) | YOLOv4-**tiny** | 36.3% | 87.3 mm | 76.8 |
| `metrabs_mob3s_y4` | YOLOv4 (full) | 36.5% | 86.4 mm | 81.8 |

Reading this: switching to the tiny detector (done specifically to fix the
out-of-memory crashes described above) cost almost nothing on single-
person body-shape accuracy (36.3% vs 36.5% PCK, 87.3mm vs 86.4mm MPJPE —
both come from the *same* pose backbone, only the detector changed) but a
real, measurable ~5-point drop in multi-person detection accuracy (76.8
vs 81.8 MuPoTS PCK). Consistent with why this tradeoff was accepted: this
pipeline only needs the detector to reliably find *the* subject in frame,
not to exhaustively catalogue every person present, so the accuracy this
project actually depends on barely moved.

For context, larger backbones on the *same* full detector (`y4`) trade
more compute for more single-person accuracy — backbone was never the
lever that mattered for this project's memory problems, only detector
choice was:

| Backbone | 3DPW PCK@50mm | 3DPW MPJPE |
|---|---|---|
| MobileNetV3-Small (`mob3s_y4`) | 36.5% | 86.4 mm |
| MobileNetV3-Large (`mob3l_y4`) | 44.6% | 73.1 mm |
| EfficientNetV2-Large (`eff2l_y4`) | 53.3% | 61.9 mm |

An independent 2025 study benchmarking 16 pose frameworks head-to-head
(MediaPipe, `rtmlib`, YOLOv8, MMPose, ViTPose, MeTRAbs, and others) — the
study already cited in "Why MeTRAbs" above, motivating the switch away
from MediaPipe in the first place — found MeTRAbs the single
highest-accuracy performer overall across that comparison, with MediaPipe
notably absent from the top tier.

## Implementation notes

- A pose is only counted if all 19 `coco_19` keypoints are present, so
  downstream landmark-index lookups (e.g. left/right wrist, index 5/11)
  are always safe. In practice this is equivalent to "was anyone detected
  at all this frame", since MeTRAbs always returns the full 19-point set
  for a detected person.
- Wrist positions are only used when the pseudo-visibility flag is 1.0
  (i.e. the wrist's 2D projection is actually inside the frame).
- `pose_keyframes.pose_y` is pre-flipped (`1 − raw_y`) so the browser-side
  canvas overlay doesn't need to re-flip the Y axis.
- World coordinates are in millimetres, camera-relative: `x` increases
  rightward, `y` increases downward (same convention MediaPipe used), `z`
  increases *away* from the camera (the opposite of MediaPipe's "toward
  camera positive" — verified from a real detection where a facing-camera
  subject's nose had a smaller `z` than their neck, i.e. nearer to camera).
  `FusionEngine`'s shoulder-yaw formula was hand-checked against real
  output and needed its subtraction order flipped (left-minus-right, not
  right-minus-left) to correctly land frontal poses near 0°.
- The local MeTRAbs `tf.saved_model` is loaded lazily (`_get_model`, not
  `__init__`) on first actual local-inference need, then reused across
  every window *within that instance's job* — inference itself is
  stateless per frame (unlike MediaPipe's Tasks API `PoseLandmarker` in
  `VIDEO` mode), so there's no cross-window timestamp bookkeeping to worry
  about, and `close()` is a no-op. If a job's every window is served by
  remote MeTRAbs successfully, the local model is never touched at all. A
  `GestureWorker` instance's whole lifetime is scoped to a single
  subprocess handling a single job (see "Running in isolation" above) —
  unlike `rtmlib` before it, there's no reuse *across* jobs; the local
  model, if it loads at all, gets reloaded from scratch every time.
- The model file itself (`metrabs_mob3s_y4t`, ~50MB unzipped) is **not**
  distributed via pip — it's downloaded once from
  `https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_mob3s_y4t.zip`
  and unzipped into `models/metrabs_mob3s_y4t/` (gitignored). The Dockerfile
  does this automatically at build time; for local development without
  `GESTURE_REMOTE_URL` set, download and unzip it manually first
  (`GestureWorker._get_model` raises a clear `FileNotFoundError` with the
  exact commands the first time local inference is actually attempted).

## Package documentation

| Package | Role | Docs |
|---|---|---|
| tensorflow | Runs the MeTRAbs SavedModel (multi-person 3D pose), local path only | https://www.tensorflow.org/api_docs |
| fastapi / uvicorn | `gesture_server.py`'s persistent-bulk HTTP server, and `colab/gesture_server.ipynb`'s remote server | https://fastapi.tiangolo.com/ · https://www.uvicorn.org/ |
| requests | `_process_window_remote`'s HTTP client (calls `colab/gesture_server.ipynb`) | https://requests.readthedocs.io/en/latest/ |
| OpenCV (`opencv-contrib-python`) | Frame I/O, BGR→RGB conversion (pulled in by scenedetect/camera_worker) | https://docs.opencv.org/4.x/ |
| NumPy | Velocity/displacement math (`np.mean`, `np.sqrt`) | https://numpy.org/doc/stable/ |
| Pydantic | `GestureFeatures` / `GestureFrame` / `PoseKeyframe` models | https://docs.pydantic.dev/latest/ |
| loguru | Per-window logging | https://loguru.readthedocs.io/en/stable/ |

MeTRAbs itself: https://github.com/isarandi/metrabs (model/API docs under `docs/`).

See also [Home](Home.md) for the full dependency list.
