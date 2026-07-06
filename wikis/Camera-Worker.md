# Camera Worker

Source: [`workers/camera_worker.py`](../workers/camera_worker.py)
Dashboard section: **Camera**
Feature model: [`CameraFeatures`](../core/models.py) (in `core/models.py`)

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

## Implementation notes

- The Haar cascade is a fast, GPU-free, but low-accuracy face detector; the
  module docstring notes it as a stand-in and recommends a deep-learning
  detector (e.g. RetinaFace via `insightface`) for production accuracy.
- Only the **largest** detected face per frame is used for the area/trend
  calculations, to avoid background faces skewing the zoom proxy.

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
