"""
core/fusion_engine.py
---------------------
Merges per-window outputs from all four workers into FusedWindow
records, then enriches them with cross-modal derived features
(e.g. speech rate backfilled into prosody, emphasis detection).
"""

from __future__ import annotations

from loguru import logger

from core.feature_store import FeatureStore
from core.models import FusedWindow, TimeWindow


class FusionEngine:
    def __init__(self, store: FeatureStore):
        self.store = store

    def fuse_job(self, job_id: str, num_windows: int) -> list[FusedWindow]:
        logger.info(f"[fusion] Fusing {num_windows} windows for job {job_id}")
        results: list[FusedWindow] = []

        for idx in range(num_windows):
            gesture = self.store.get_gesture(job_id, idx)
            prosody = self.store.get_prosody(job_id, idx)
            verbal = self.store.get_verbal(job_id, idx)
            camera = self.store.get_camera(job_id, idx)

            # Derive a window from whichever modality is available
            window = self._resolve_window(gesture, prosody, verbal, camera, idx)

            fused = FusedWindow(
                window=window,
                gesture=gesture,
                prosody=prosody,
                verbal=verbal,
                camera=camera,
            )

            # Cross-modal enrichment
            fused = self._enrich(fused)

            self.store.put_fused(job_id, idx, fused)
            results.append(fused)

        logger.info(f"[fusion] Done — {len(results)} fused windows")
        return results

    # ------------------------------------------------------------------
    # Cross-modal enrichment
    # ------------------------------------------------------------------

    def _enrich(self, fused: FusedWindow) -> FusedWindow:
        """
        Back-fill and cross-modal derivations:
          1. Speech rate (syllables/s) from verbal token timing → prosody
          2. Annotation flags (emphasis, hesitation, engaged) added as
             simple heuristics for dashboard colour coding.
        """
        # 1. Speech rate: word count / window duration as a proxy
        if fused.verbal and fused.prosody:
            duration = fused.window.duration
            if duration > 0:
                words_per_s = fused.verbal.word_count / duration
                # Rough syllables/s: English avg ~1.5 syllables/word
                fused.prosody.speech_rate_syl_per_s = words_per_s * 1.5

        return fused

    # ------------------------------------------------------------------

    def _resolve_window(self, gesture, prosody, verbal, camera, idx: int) -> TimeWindow:
        for obj in (verbal, prosody, gesture, camera):
            if obj is not None:
                return obj.window
        # Fallback: construct a dummy window
        return TimeWindow(start_s=float(idx * 5), end_s=float((idx + 1) * 5))
