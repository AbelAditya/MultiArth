"""
core/feature_store.py
---------------------
Thin wrapper around Redis that workers use to publish features and
the fusion engine uses to read them.

Key schema
----------
  job:{job_id}:status          STRING  JobStatus value
  job:{job_id}:gesture:{w}     STRING  GestureFeatures JSON   (w = window index)
  job:{job_id}:prosody:{w}     STRING  ProsodyFeatures JSON
  job:{job_id}:verbal:{w}      STRING  VerbalFeatures JSON
  job:{job_id}:camera:{w}      STRING  CameraFeatures JSON
  job:{job_id}:fused:{w}       STRING  FusedWindow JSON
  job:{job_id}:meta            STRING  AnalysisJob JSON
  job:{job_id}:stream          STREAM  progress/log events

All keys are set with a 24-hour TTL by default.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import redis
from loguru import logger

from .models import (
    AnalysisJob,
    CameraFeatures,
    FusedWindow,
    GestureFeatures,
    JobStatus,
    ProsodyFeatures,
    VerbalFeatures,
)

_TTL = 86_400  # 24 hours


class FeatureStore:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def create_job(self, job: AnalysisJob) -> None:
        key = f"job:{job.job_id}:meta"
        self.r.set(key, job.model_dump_json(), ex=_TTL)
        self.r.set(f"job:{job.job_id}:status", job.status.value, ex=_TTL)

    def get_job(self, job_id: str) -> Optional[AnalysisJob]:
        raw = self.r.get(f"job:{job_id}:meta")
        return AnalysisJob.model_validate_json(raw) if raw else None

    def set_status(self, job_id: str, status: JobStatus, error: str = None) -> None:
        self.r.set(f"job:{job_id}:status", status.value, ex=_TTL)
        if error:
            meta_raw = self.r.get(f"job:{job_id}:meta")
            if meta_raw:
                meta = AnalysisJob.model_validate_json(meta_raw)
                meta.status = status
                meta.error = error
                meta.completed_at = time.time()
                self.r.set(f"job:{job_id}:meta", meta.model_dump_json(), ex=_TTL)

    def get_status(self, job_id: str) -> Optional[JobStatus]:
        val = self.r.get(f"job:{job_id}:status")
        return JobStatus(val) if val else None

    # ------------------------------------------------------------------
    # Per-modality write
    # ------------------------------------------------------------------

    def put_gesture(self, job_id: str, window_idx: int, features: GestureFeatures) -> None:
        self.r.set(f"job:{job_id}:gesture:{window_idx}", features.model_dump_json(), ex=_TTL)

    def put_prosody(self, job_id: str, window_idx: int, features: ProsodyFeatures) -> None:
        self.r.set(f"job:{job_id}:prosody:{window_idx}", features.model_dump_json(), ex=_TTL)

    def put_verbal(self, job_id: str, window_idx: int, features: VerbalFeatures) -> None:
        self.r.set(f"job:{job_id}:verbal:{window_idx}", features.model_dump_json(), ex=_TTL)

    def put_camera(self, job_id: str, window_idx: int, features: CameraFeatures) -> None:
        self.r.set(f"job:{job_id}:camera:{window_idx}", features.model_dump_json(), ex=_TTL)

    def put_fused(self, job_id: str, window_idx: int, fused: FusedWindow) -> None:
        self.r.set(f"job:{job_id}:fused:{window_idx}", fused.model_dump_json(), ex=_TTL)

    def put_spectrogram(self, job_id: str, data: dict) -> None:
        self.r.set(f"job:{job_id}:spectrogram", json.dumps(data), ex=_TTL)

    def put_wordlist(self, job_id: str, data: dict) -> None:
        self.r.set(f"job:{job_id}:wordlist", json.dumps(data), ex=_TTL)

    def put_ngrams(self, job_id: str, data: dict) -> None:
        self.r.set(f"job:{job_id}:ngrams", json.dumps(data), ex=_TTL)

    def put_collocations(self, job_id: str, data: dict) -> None:
        self.r.set(f"job:{job_id}:collocations", json.dumps(data), ex=_TTL)

    # ------------------------------------------------------------------
    # Per-modality read
    # ------------------------------------------------------------------

    def get_gesture(self, job_id: str, window_idx: int) -> Optional[GestureFeatures]:
        raw = self.r.get(f"job:{job_id}:gesture:{window_idx}")
        return GestureFeatures.model_validate_json(raw) if raw else None

    def get_prosody(self, job_id: str, window_idx: int) -> Optional[ProsodyFeatures]:
        raw = self.r.get(f"job:{job_id}:prosody:{window_idx}")
        return ProsodyFeatures.model_validate_json(raw) if raw else None

    def get_verbal(self, job_id: str, window_idx: int) -> Optional[VerbalFeatures]:
        raw = self.r.get(f"job:{job_id}:verbal:{window_idx}")
        return VerbalFeatures.model_validate_json(raw) if raw else None

    def get_camera(self, job_id: str, window_idx: int) -> Optional[CameraFeatures]:
        raw = self.r.get(f"job:{job_id}:camera:{window_idx}")
        return CameraFeatures.model_validate_json(raw) if raw else None

    def get_fused(self, job_id: str, window_idx: int) -> Optional[FusedWindow]:
        raw = self.r.get(f"job:{job_id}:fused:{window_idx}")
        return FusedWindow.model_validate_json(raw) if raw else None

    def get_spectrogram(self, job_id: str) -> Optional[dict]:
        raw = self.r.get(f"job:{job_id}:spectrogram")
        return json.loads(raw) if raw else None

    def get_wordlist(self, job_id: str) -> Optional[dict]:
        raw = self.r.get(f"job:{job_id}:wordlist")
        return json.loads(raw) if raw else None

    def get_ngrams(self, job_id: str) -> Optional[dict]:
        raw = self.r.get(f"job:{job_id}:ngrams")
        return json.loads(raw) if raw else None

    def get_collocations(self, job_id: str) -> Optional[dict]:
        raw = self.r.get(f"job:{job_id}:collocations")
        return json.loads(raw) if raw else None

    # ------------------------------------------------------------------
    # Bulk read (for dashboard / fusion)
    # ------------------------------------------------------------------

    def get_all_fused(self, job_id: str) -> list[FusedWindow]:
        pattern = f"job:{job_id}:fused:*"
        keys = self.r.keys(pattern)
        results = []
        for key in sorted(keys, key=lambda k: int(k.split(":")[-1])):
            raw = self.r.get(key)
            if raw:
                try:
                    results.append(FusedWindow.model_validate_json(raw))
                except Exception as exc:
                    logger.warning(f"[store] Skipping stale fused window {key}: {exc}")
        return results

    def count_windows(self, job_id: str, modality: str) -> int:
        return len(self.r.keys(f"job:{job_id}:{modality}:*"))

    # ------------------------------------------------------------------
    # Progress stream
    # ------------------------------------------------------------------

    def log_event(self, job_id: str, worker: str, message: str) -> None:
        self.r.xadd(
            f"job:{job_id}:stream",
            {"worker": worker, "msg": message, "ts": str(time.time())},
        )

    def read_events(self, job_id: str, last_id: str = "0") -> list[dict]:
        entries = self.r.xread({f"job:{job_id}:stream": last_id}, count=100)
        if not entries:
            return []
        events = []
        for _, messages in entries:
            for msg_id, fields in messages:
                events.append({"id": msg_id, **fields})
        return events
