# Camera Worker

Source: [`workers/camera_worker.py`](../workers/camera_worker.py)
Dashboard section: **Camera**
Feature model: [`CameraFeatures`](../core/models.py) (in `core/models.py`)

> **Branch note (`light-gesture`):** shot classification and camera-angle
> computation (point 2 below) are powered by MediaPipe's pose landmarks on
> this branch, not MeTRAbs's — see [Gesture-Worker.md](Gesture-Worker.md).

## What it does

`CameraWorker` analyses editorial/camera behaviour: scene cuts and
framing/zoom over time.

1. **Scene cut detection** (`_detect_cuts`) — runs once per job over the
   whole video using **PySceneDetect**'s `ContentDetector`
   (`detection_threshold=27.0` by default) via `SceneManager`, producing a
   list of `SceneCut`s (timestamp, frame index, and an index-based
   `cut_score` proxy, since PySceneDetect doesn't expose per-cut scores
   directly).
2. **Per-window features** (`_process_window`), for each time window:
   - `cut_count` / `cut_rate` — number of detected cuts falling in the
     window, normalised to cuts-per-minute.
   - Samples up to 30 frames from the window
     (`frames_for_window`, [`core/preprocessing.py`](../core/preprocessing.py))
     and runs an **OpenCV Haar cascade** face detector on each
     (`haarcascade_frontalface_default.xml`) to get:
     - `mean_face_bbox_area` — mean largest-face bounding-box area as a
       fraction of the frame, a lightweight proxy for zoom level.
     - `face_bbox_trend` — linear-regression slope of face area over the
       sampled frames (positive = zooming in, negative = zooming out).
   - `dominant_shot_type` is left as `ShotType.UNKNOWN` here — actual shot
     classification (extreme close-up → very long) as well as
     `horizontal_angle`/`vertical_angle` (shoulder yaw / face pitch) are
     computed later in the `FusionEngine` using MediaPipe pose keypoints
     from the Gesture worker, which give finer-grained framing information
     than face detection alone.

## Values shown on the dashboard

The **Camera** section has one KPI card and six charts:

| Where | What it shows |
|---|---|
| KPI card: **Cuts** | Total number of scene cuts detected across the whole video. |
| **Shot Type** chart | The dominant framing of each window — extreme close-up through very long shot — based on how much of the frame the subject fills. |
| **Horizontal Angle** chart | How far the subject is turned away from the camera, from shoulder orientation — frontal, transitional, or oblique (see below). |
| **Vertical Angle** chart | Whether the camera is positioned above, level with, or below the subject's face (high / eye-level / low angle). |
| **Scene Cuts** chart | How many cuts fall in each time window. |
| **Face Area** chart | A rough zoom-level proxy: how much of the frame the largest detected face occupies. |
| **Zoom Trend** chart | Whether the framing is zooming in or zooming out within each window. |

## Horizontal angle bands

`horizontal_angle` is classified from mean shoulder yaw by
`_classify_horizontal_angle` in `core/fusion_engine.py`. Only the
*magnitude* of the yaw matters — a turn to either side scores identically:

| Band | Mean shoulder yaw | Reading |
|---|---|---|
| `frontal` | \|yaw\| < 10° | Squarely toward the camera |
| `transitional` | 10° ≤ \|yaw\| < 20° | Turning — off-axis but still broadly addressing the audience |
| `oblique` | \|yaw\| ≥ 20° | Committed side-on stance, body turned away |

`transitional` was split out of `oblique`, which previously started at 10°
and so scored a subject merely *mid-turn* the same as one standing fully
side-on. That distinction matters for discourse analysis: a speaker turning
is usually still engaging the audience, whereas a sustained oblique stance
often marks disengagement or attention directed elsewhere (a slide, another
participant).

The thresholds are analyst-chosen rather than empirically derived, and are
the two constants `_YAW_FRONTAL_MAX_DEG` and `_YAW_TRANSITIONAL_MAX_DEG` if
they need revisiting. The dashboard's Horizontal Angle chart colours the
three bands as an ordinal ramp (green → olive-gold → amber) rather than
three unrelated hues, since they express a progression.

## Implementation notes

- The Haar cascade is a fast, GPU-free, but low-accuracy face detector; the
  module docstring notes it as a stand-in and recommends a deep-learning
  detector (e.g. RetinaFace via `insightface`) for production accuracy —
  see "Benchmark accuracy" below for how much of a gap that actually is.
- Only the **largest** detected face per frame is used for the area/trend
  calculations, to avoid background faces skewing the zoom proxy.

## Benchmark accuracy (published)

`haarcascade_frontalface_default.xml` is the classic Viola-Jones detector
(2001) — hand-crafted features, not a trained deep-learning model, and its
accuracy on standard face-detection benchmarks reflects that age gap
clearly:

| Detector | Benchmark | Reported accuracy |
|---|---|---|
| Haar cascade (Viola-Jones), OpenCV implementation | FDDB | ~67% precision / 0.67 positive-detection rate in one direct evaluation; other studies report anywhere from ~34% to ~94% depending on how "accuracy" is defined and how challenging the test images are (lighting, angle, occlusion) |
| RetinaFace (the module docstring's suggested replacement) | FDDB / WIDER FACE | Consistently reported as substantially ahead of Haar cascade and other classical/lightweight detectors across every comparative study checked — one head-to-head put Haar cascade at 34% vs. RetinaFace at 65% on the same evaluation set |

The wide spread in Haar cascade's own numbers across studies isn't
inconsistency in the citation — it's a real property of the algorithm:
unlike a benchmark-trained deep model, its accuracy swings hard with
lighting/pose/occlusion conditions in the specific test images used, which
is exactly why `_process_window` here only ever uses it as a coarse zoom
proxy (largest detected face's bounding-box area) rather than for anything
requiring reliable detection of *every* face in frame.

Sources: ["Evaluation of Human and Machine Face Detection using a Novel
Distinctive Human Appearance
Dataset"](https://arxiv.org/pdf/2111.00660) (FDDB precision figure);
["Comparative Analysis of Multi-Face Detection Methods in Classroom
Environments"](https://ieeexplore.ieee.org/document/10823781/) (Haar vs.
RetinaFace head-to-head); [RetinaFace paper](https://arxiv.org/abs/1905.00641).

**PySceneDetect's `ContentDetector`** (the other detector this worker uses,
for scene cuts) doesn't have a comparable published benchmark table — it's
a threshold-based heuristic over HSV colour-space frame-to-frame
difference, not a trained model evaluated against a labelled dataset, so
"accuracy on public benchmarks" isn't a meaningful framing for it the way
it is for the face detector.

## Package documentation

| Package | Role | Docs |
|---|---|---|
| PySceneDetect (`scenedetect`) | Global scene-cut detection (`ContentDetector`, `SceneManager`) | https://www.scenedetect.com/docs/latest/api.html |
| OpenCV (`opencv-python`) | Haar cascade face detection, frame colour conversion | https://docs.opencv.org/4.x/ |
| OpenCV Haar cascades (tutorial) | Background on `CascadeClassifier` | https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html |
| NumPy | Face-area averaging and trend regression (`np.polyfit`) | https://numpy.org/doc/stable/ |
| Pydantic | `CameraFeatures` / `SceneCut` models | https://docs.pydantic.dev/latest/ |
| loguru | Per-window/job logging | https://loguru.readthedocs.io/en/stable/ |

See also [Home](Home.md) for the full dependency list.
</content>
