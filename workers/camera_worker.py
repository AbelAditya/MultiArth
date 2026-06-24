"""
workers/camera_worker.py
------------------------
PySceneDetect + OpenCV camera/editorial analysis worker.

Detects scene cuts globally then attributes them to windows.
Uses Haar cascade face detection to track face area over time
(zoom-level proxy for the dashboard charts).
Shot type classification is handled in FusionEngine using MediaPipe
pose keypoints, which give finer-grained framing information.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from loguru import logger
from scenedetect import open_video, SceneManager, ContentDetector

from core.feature_store import FeatureStore
from core.models import CameraFeatures, SceneCut, ShotType, TimeWindow
from core.preprocessing import VideoMeta, frames_for_window


class CameraWorker:
    def __init__(self, store: FeatureStore, detection_threshold: float = 27.0):
        self.store = store
        self.threshold = detection_threshold
        # OpenCV's Haar cascade for face detection (fast, no GPU needed)
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def process_job(
        self,
        job_id: str,
        meta: VideoMeta,
        windows: list[tuple[float, float]],
    ) -> None:
        logger.info(f"[camera] Detecting scene cuts in {meta.path}")
        cuts = self._detect_cuts(meta.path)
        logger.info(f"[camera] Found {len(cuts)} cuts")
        self.store.log_event(job_id, "camera", f"detected {len(cuts)} cuts")

        for idx, (start, end) in enumerate(windows):
            try:
                features = self._process_window(meta, start, end, cuts)
                self.store.put_camera(job_id, idx, features)
                self.store.log_event(job_id, "camera", f"window {idx} done")
            except Exception as exc:
                logger.error(f"[camera] Window {idx} failed: {exc}")
                self.store.log_event(job_id, "camera", f"window {idx} ERROR: {exc}")

        logger.info(f"[camera] Job {job_id} complete")

    # ------------------------------------------------------------------

    def _detect_cuts(self, video_path: str) -> list[SceneCut]:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        cuts: list[SceneCut] = []
        for i, (start_tc, _) in enumerate(scene_list):
            ts = start_tc.get_seconds()
            frame = start_tc.get_frames()
            # PySceneDetect doesn't expose per-cut scores easily; use index as proxy
            cuts.append(SceneCut(timestamp_s=ts, frame_idx=frame, cut_score=float(i)))
        return cuts

    def _process_window(
        self,
        meta: VideoMeta,
        start_s: float,
        end_s: float,
        all_cuts: list[SceneCut],
    ) -> CameraFeatures:
        window = TimeWindow(start_s=start_s, end_s=end_s)
        duration_s = end_s - start_s
        duration_min = duration_s / 60.0

        # Cuts within this window
        window_cuts = [c for c in all_cuts if start_s <= c.timestamp_s < end_s]
        cut_count = len(window_cuts)
        cut_rate = cut_count / max(duration_min, 1e-6)

        # Sample frames for shot-type classification
        sampled = frames_for_window(meta.path, start_s, end_s, meta.fps, max_frames=30)
        face_areas: list[float] = []
        for _, frame in sampled:
            area = self._detect_face_area(frame, meta.width, meta.height)
            if area is not None:
                face_areas.append(area)

        mean_face_area = float(np.mean(face_areas)) if face_areas else None
        face_trend = self._face_area_trend(face_areas) if len(face_areas) > 2 else None

        return CameraFeatures(
            window=window,
            scene_cuts=window_cuts,
            cut_count=cut_count,
            cut_rate=cut_rate,
            dominant_shot_type=ShotType.UNKNOWN,
            mean_face_bbox_area=mean_face_area,
            face_bbox_trend=face_trend,
        )

    def _detect_face_area(
        self, frame: np.ndarray, width: int, height: int
    ) -> Optional[float]:
        """Returns face bounding box area as a fraction of total frame area."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            return None
        # Take largest face
        areas = [w * h for (_, _, w, h) in faces]
        max_area = max(areas)
        frame_area = width * height
        return max_area / frame_area

    def _face_area_trend(self, face_areas: list[float]) -> float:
        """
        Linear regression slope of face area over time.
        Positive = zooming in, negative = zooming out.
        """
        x = np.arange(len(face_areas), dtype=float)
        y = np.array(face_areas)
        if np.std(x) == 0:
            return 0.0
        slope = float(np.polyfit(x, y, 1)[0])
        return slope
