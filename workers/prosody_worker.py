"""
workers/prosody_worker.py
--------------------------
Parselmouth (Praat) prosody worker.

For each time window it extracts:
  - F0 contour (pitch)
  - Intensity
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

        try:
            sg = self._compute_spectrogram(sound)
            self.store.put_spectrogram(job_id, sg)
            self.store.log_event(job_id, "prosody", "spectrogram done")
        except Exception as exc:
            logger.error(f"[prosody] Spectrogram failed: {exc}")

        logger.info(f"[prosody] Job {job_id} complete")

    # ------------------------------------------------------------------

    def _compute_spectrogram(self, sound: parselmouth.Sound) -> dict:
        """
        Compute a downsampled wide-band spectrogram for the full audio.
        Returns times (s), freqs (kHz), and dB values as a JSON-safe dict.
        Downsampled to ≤1000 time steps × ≤200 frequency bins to keep
        payload manageable for the dashboard.
        """
        sg = sound.to_spectrogram(
            window_length=0.050,     # 25 ms → narrow-band (resolves harmonics)
            maximum_frequency=2000.0,
            time_step=0.01,          # 10 ms time step
        )
        values = sg.values           # shape: (n_freq, n_time), power Pa²/Hz
        n_freq, n_time = values.shape

        t_step = max(1, n_time // 1000)
        f_step = 1
        ds = values[::f_step, ::t_step]
        db = 10.0 * np.log10(ds + 1e-10)

        # Clip to a sensible dB range so colour scale isn't dominated by silence
        db_max = float(np.percentile(db, 99))
        db_min = db_max - 60.0
        db = np.clip(db, db_min, db_max)

        return {
            "times": sg.xs()[::t_step].tolist(),
            "freqs": (sg.ys()[::f_step] / 1000.0).tolist(),  # Hz → kHz
            "data":  db.tolist(),
        }

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
        # ---- Intensity -------------------------------------------------
        intensity_obj = segment.to_intensity(time_step=0.01)
        intensity_values = intensity_obj.values.T.flatten()
        mean_intensity = float(np.mean(intensity_values)) if len(intensity_values) > 0 else 0.0
        intensity_range = float(np.ptp(intensity_values)) if len(intensity_values) > 0 else 0.0

        return ProsodyFeatures(
            window=window,
            mean_f0=mean_f0,
            f0_range=f0_range,
            f0_std=f0_std,
            mean_intensity_db=mean_intensity,
            intensity_range_db=intensity_range,
            speech_rate_syl_per_s=None,  # filled by fusion engine after ASR
        )
