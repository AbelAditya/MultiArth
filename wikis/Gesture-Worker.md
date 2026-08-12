# Gesture Worker

Source: [`workers/gesture_worker.py`](../workers/gesture_worker.py),
run via [`workers/gesture_subprocess.py`](../workers/gesture_subprocess.py)
and dispatched by [`core/orchestrator.py`](../core/orchestrator.py) — see
"Running in isolation" below.
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
  own published numbers: worse multi-person detection accuracy (MuPoTS PCK
  81.0 vs 86.6) — acceptable here since this pipeline only needs the
  detector to reliably find *the* subject, not exhaustively catalog
  everyone in frame, but worth revisiting if that stops being true.
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

`Orchestrator.analyze()` also runs the gesture subprocess **strictly
after**, never alongside, the other three workers (prosody/verbal/camera,
still dispatched together via the existing `ThreadPoolExecutor` — none of
them load anything close to MeTRAbs's weight, so that concurrency was
never the problem). Isolation by itself only guarantees memory gets
released *afterward* — it says nothing about whether MeTRAbs was the only
heavy thing running *at the time*, which is what actually caused the
crashes. Both parts together are what make this safe.

The real cost of this design: MeTRAbs's ~15-22s model-load time is paid on
**every job**, not once at server startup — a deliberate trade of latency
for memory safety.

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
- The MeTRAbs `tf.saved_model` is loaded once per `GestureWorker` instance
  and reused across every window *within that instance's job* — inference
  itself is stateless per frame (unlike MediaPipe's Tasks API
  `PoseLandmarker` in `VIDEO` mode), so there's no cross-window timestamp
  bookkeeping to worry about, and `close()` is a no-op. But a
  `GestureWorker` instance's whole lifetime is now scoped to a single
  subprocess handling a single job (see "Running in isolation" above) —
  unlike `rtmlib` before it, there's no reuse *across* jobs; the model gets
  reloaded from scratch every time, by design.
- The model file itself (`metrabs_mob3s_y4t`, ~50MB unzipped) is **not**
  distributed via pip — it's downloaded once from
  `https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_mob3s_y4t.zip`
  and unzipped into `models/metrabs_mob3s_y4t/` (gitignored). The Dockerfile
  does this automatically at build time; for local development, download
  and unzip it manually first (`GestureWorker.__init__` raises a clear
  `FileNotFoundError` with the exact commands if it's missing).

## Package documentation

| Package | Role | Docs |
|---|---|---|
| tensorflow | Runs the MeTRAbs SavedModel (multi-person 3D pose) | https://www.tensorflow.org/api_docs |
| OpenCV (`opencv-contrib-python`) | Frame I/O, BGR→RGB conversion (pulled in by scenedetect/camera_worker) | https://docs.opencv.org/4.x/ |
| NumPy | Velocity/displacement math (`np.mean`, `np.sqrt`) | https://numpy.org/doc/stable/ |
| Pydantic | `GestureFeatures` / `GestureFrame` / `PoseKeyframe` models | https://docs.pydantic.dev/latest/ |
| loguru | Per-window logging | https://loguru.readthedocs.io/en/stable/ |

MeTRAbs itself: https://github.com/isarandi/metrabs (model/API docs under `docs/`).

See also [Home](Home.md) for the full dependency list.
