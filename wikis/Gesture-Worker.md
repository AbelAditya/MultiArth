# Gesture Worker

Source: [`workers/gesture_worker.py`](../workers/gesture_worker.py)
Dashboard section: **Pose Estimation**
Feature model: [`GestureFeatures`](../core/models.py) (in `core/models.py`)

## What it does

For every time window, `GestureWorker`:

1. Reads frames from the video for that window
   (`frames_for_window`, [`core/preprocessing.py`](../core/preprocessing.py)).
2. Converts each frame BGR → RGB and runs **MediaPipe Holistic**
   (`mp.solutions.holistic.Holistic`) to get pose, left-hand, and right-hand
   landmarks, plus metric "world" pose landmarks.
3. Aggregates the per-frame landmarks into window-level kinematic features:
   - `mean_wrist_velocity` — mean speed (px/s) of both wrists across the window.
   - `max_wrist_displacement` — largest bounding displacement of either wrist.
   - `pose_present_ratio` — fraction of frames where a full 33-point pose was detected.
   - `handedness_ratio` — ratio of right- vs. left-hand motion (−1 fully
     left-dominant → +1 fully right-dominant).
   - `pose_keyframes` — subsampled (every 3rd frame), normalised pose
     snapshots used to drive the animated pose overlay in the dashboard.
4. Stores the resulting `GestureFeatures` record in the Redis
   [`FeatureStore`](../core/feature_store.py) under `job:{id}:gesture:{w}`.

## Implementation notes

- A pose is only counted if all 33 MediaPipe pose landmarks are present, so
  downstream landmark-index lookups (e.g. left/right wrist) are always safe.
- Wrist positions are only used when landmark `visibility > 0.3`, filtering
  out low-confidence detections.
- `pose_keyframes.pose_y` is pre-flipped (`1 − raw_y`) so the browser-side
  canvas overlay doesn't need to re-flip the Y axis.
- The Holistic model is created once per worker instance
  (`model_complexity=1`, `smooth_landmarks=True`) and must be closed via
  `.close()` to release its resources.

## Package documentation

| Package | Role | Docs |
|---|---|---|
| MediaPipe | Holistic pose/hand landmark detection | https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker |
| OpenCV (`opencv-python`) | Frame colour conversion (`cv2.cvtColor`) | https://docs.opencv.org/4.x/ |
| NumPy | Velocity/displacement math (`np.mean`, `np.sqrt`) | https://numpy.org/doc/stable/ |
| Pydantic | `GestureFeatures` / `GestureFrame` / `PoseKeyframe` models | https://docs.pydantic.dev/latest/ |
| loguru | Per-window logging | https://loguru.readthedocs.io/en/stable/ |

See also [Home](Home.md) for the full dependency list.
</content>
