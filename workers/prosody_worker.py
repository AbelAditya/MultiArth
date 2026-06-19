"""
workers/prosody_worker.py
--------------------------
Parselmouth (Praat) prosody worker.

For each time window it extracts:
  - F0 contour (pitch)
  - Intensity
  - Jitter, shimmer, HNR (voice quality)
  - Voiced fraction
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import parselmouth
from parselmouth.praat import call
from loguru import logger

from core.feature_store import FeatureStore
from core.models import PitchFrame, ProsodyFeatures, TimeWindow
from core.preprocessing import VideoMeta


class ProsodyWorker:
    def __init__(self, store: FeatureStore):
        self.store = store

    def process_job(
        self,
        job_id: str,
        meta: VideoMeta,
        windows: list[tuple[float, float]],
    ) -> None:
        logger.info(f"[prosody] Starting job {job_id} — loading audio")
        sound = parselmouth.Sound(meta.audio_path)

        for idx, (start, end) in enumerate(windows):
            try:
                features = self._process_window(sound, start, end)
                self.store.put_prosody(job_id, idx, features)
                self.store.log_event(job_id, "prosody", f"window {idx} done")
            except Exception as exc:
                logger.error(f"[prosody] Window {idx} failed: {exc}")
                self.store.log_event(job_id, "prosody", f"window {idx} ERROR: {exc}")

        logger.info(f"[prosody] Job {job_id} complete")

    # ------------------------------------------------------------------

    def _process_window(
        self, sound: parselmouth.Sound, start_s: float, end_s: float
    ) -> ProsodyFeatures:
        window = TimeWindow(start_s=start_s, end_s=end_s)
        segment: parselmouth.Sound = sound.extract_part(
            from_time=start_s,
            to_time=end_s,
            window_shape=parselmouth.WindowShape.RECTANGULAR,
            relative_width=1.0,
            preserve_times=True,
        )

        # ---- F0 --------------------------------------------------------
        pitch_obj = segment.to_pitch(
            time_step=0.01,
            pitch_floor=75.0,    # Hz — below typical human voice
            pitch_ceiling=500.0, # Hz — above typical human voice
        )
        f0_values = pitch_obj.selected_array["frequency"]  # 0 = unvoiced
        voiced = f0_values[f0_values > 0]

        mean_f0 = float(np.mean(voiced)) if len(voiced) > 0 else None
        f0_range = float(np.ptp(voiced)) if len(voiced) > 1 else None
        f0_std = float(np.std(voiced)) if len(voiced) > 1 else None
        voiced_fraction = len(voiced) / max(len(f0_values), 1)

        # ---- Intensity -------------------------------------------------
        intensity_obj = segment.to_intensity(time_step=0.01)
        intensity_values = intensity_obj.values.T.flatten()
        mean_intensity = float(np.mean(intensity_values)) if len(intensity_values) > 0 else 0.0
        intensity_range = float(np.ptp(intensity_values)) if len(intensity_values) > 0 else 0.0

        # ---- Voice quality (jitter, shimmer, HNR) ----------------------
        jitter = self._get_jitter(segment)
        shimmer = self._get_shimmer(segment)
        hnr = self._get_hnr(segment)

        return ProsodyFeatures(
            window=window,
            mean_f0=mean_f0,
            f0_range=f0_range,
            f0_std=f0_std,
            voiced_fraction=voiced_fraction,
            mean_intensity_db=mean_intensity,
            intensity_range_db=intensity_range,
            speech_rate_syl_per_s=None,  # filled by fusion engine after ASR
            jitter_local=jitter,
            shimmer_local=shimmer,
            hnr_db=hnr,
        )

    def _get_jitter(self, segment: parselmouth.Sound) -> Optional[float]:
        """Local jitter: mean absolute F0 period difference."""
        try:
            point_process = call(segment, "To PointProcess (periodic, cc)", 75, 500)
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            return float(jitter) if jitter is not None else None
        except Exception:
            return None

    def _get_shimmer(self, segment: parselmouth.Sound) -> Optional[float]:
        """Local shimmer: mean absolute amplitude difference."""
        try:
            point_process = call(segment, "To PointProcess (periodic, cc)", 75, 500)
            shimmer = call(
                [segment, point_process],
                "Get shimmer (local)",
                0, 0, 0.0001, 0.02, 1.3, 1.6,
            )
            return float(shimmer) if shimmer is not None else None
        except Exception:
            return None

    def _get_hnr(self, segment: parselmouth.Sound) -> Optional[float]:
        """Harmonics-to-noise ratio in dB."""
        try:
            harmonicity = call(segment, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            return float(hnr) if hnr is not None and not np.isnan(hnr) else None
        except Exception:
            return None
