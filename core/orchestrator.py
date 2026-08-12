"""
core/orchestrator.py
--------------------
Coordinates the full analysis pipeline for a single video.

Architecture (Option B — event-driven modular):
  1. Ingest: extract audio, probe video, compute windows
  2. Dispatch: run prosody/verbal/camera (optionally in parallel via
     threading), THEN gesture — in its own isolated subprocess, never
     concurrently with the other three
  3. Fuse: merge all modality outputs per window
  4. Report: store final results in FeatureStore

Prosody/verbal/camera run as threads so they can be parallelised without
the GIL bottleneck (each worker is I/O or C-extension bound, not pure
Python). Gesture (MeTRAbs) is deliberately excluded from that pool and run
afterward instead, in a separate OS process — see `_run_gesture_isolated`
below and `workers/gesture_subprocess.py` for why: MeTRAbs's TensorFlow
runtime is heavy enough (~1.8-2.7GB) that running it concurrently with
Whisper (loaded once at startup in VerbalWorker) and the other workers is
what caused real out-of-memory crashes in practice, and TensorFlow doesn't
reliably release memory back to the OS even when the model object is
explicitly deleted — only a subprocess that fully exits guarantees that.
Running gesture strictly after, not alongside, the others ensures its
memory footprint never overlaps with anything else's peak, which isolation
alone does not guarantee (isolation only fixes what happens *afterward*).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from core.feature_store import FeatureStore
from core.fusion_engine import FusionEngine
from core.models import AnalysisJob, FusedWindow, JobStatus
from core.preprocessing import compute_windows, extract_audio, probe_video
from workers.camera_worker import CameraWorker
from workers.prosody_worker import ProsodyWorker
from workers.verbal_worker import VerbalWorker


class Orchestrator:
    def __init__(
        self,
        store: FeatureStore,
        work_dir: str | None = None,
        window_size_s: float = 5.0,
        whisper_model: str = "small",
        whisper_device: str = "cpu",
        parallel: bool = True,
    ):
        self.store = store
        self.work_dir = Path(work_dir or os.environ.get("WORK_DIR", "/tmp/mannerism"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.window_size_s = window_size_s
        self.parallel = parallel

        # Gesture (MeTRAbs) is deliberately NOT constructed here — it runs
        # in its own subprocess per job, see _run_gesture_isolated. This
        # process never imports tensorflow at all.
        self._prosody_worker = ProsodyWorker(store)
        self._verbal_worker = VerbalWorker(store, model_size=whisper_model, device=whisper_device)
        self._camera_worker = CameraWorker(store)
        self._fusion = FusionEngine(store)

    def analyze(self, video_path: str, job_id: str | None = None) -> str:
        """
        Full analysis pipeline. Returns the job_id.
        Pass job_id to pre-assign one (e.g. so callers can poll before analyze() returns).
        """
        if job_id is None:
            job_id = str(uuid.uuid4())[:8]
        job = AnalysisJob(
            job_id=job_id,
            video_path=video_path,
            window_size_s=self.window_size_s,
            status=JobStatus.PENDING,
            created_at=time.time(),
        )
        self.store.create_job(job)
        logger.info(f"[orchestrator] Created job {job_id} for {video_path}")

        try:
            self.store.set_status(job_id, JobStatus.RUNNING)

            # 1. Ingest
            audio_path = extract_audio(video_path, str(self.work_dir))
            meta = probe_video(video_path, audio_path)
            windows = compute_windows(meta.duration_s, self.window_size_s)
            self.store.set_total_windows(job_id, len(windows))

            logger.info(
                f"[orchestrator] Video: {meta.duration_s:.1f}s, "
                f"{meta.fps:.1f}fps, {len(windows)} windows"
            )

            # 2a. Dispatch the three lightweight workers
            if self.parallel:
                self._run_parallel(job_id, meta, windows)
            else:
                self._run_sequential(job_id, meta, windows)

            # 2b. THEN gesture, alone, in its own subprocess — never
            # overlapping with the above (see module docstring).
            self._run_gesture_isolated(job_id, video_path)

            # 3. Fuse
            fused = self._fusion.fuse_job(job_id, len(windows))
            logger.info(f"[orchestrator] Fusion complete — {len(fused)} records")

            self.store.set_status(job_id, JobStatus.DONE)
            logger.info(f"[orchestrator] Job {job_id} DONE")

        except Exception as exc:
            logger.exception(f"[orchestrator] Job {job_id} FAILED: {exc}")
            self.store.set_status(job_id, JobStatus.FAILED, error=str(exc))

        return job_id

    # ------------------------------------------------------------------

    def _run_parallel(self, job_id, meta, windows) -> None:
        tasks = {
            "prosody": lambda: self._prosody_worker.process_job(job_id, meta, windows),
            "verbal":  lambda: self._verbal_worker.process_job(job_id, meta, windows),
            "camera":  lambda: self._camera_worker.process_job(job_id, meta, windows),
        }
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                exc = future.exception()
                if exc:
                    logger.error(f"[orchestrator] Worker '{name}' raised: {exc}")
                else:
                    logger.info(f"[orchestrator] Worker '{name}' finished")

    def _run_sequential(self, job_id, meta, windows) -> None:
        self._prosody_worker.process_job(job_id, meta, windows)
        self._verbal_worker.process_job(job_id, meta, windows)
        self._camera_worker.process_job(job_id, meta, windows)

    def _run_gesture_isolated(self, job_id: str, video_path: str) -> None:
        """
        Runs gesture analysis (MeTRAbs) in a fresh `python -m
        workers.gesture_subprocess` process and waits for it to finish —
        see workers/gesture_subprocess.py and this module's docstring for
        why. The child inherits this process's environment (REDIS_HOST/
        REDIS_PORT etc.), so it connects to the same FeatureStore and
        writes gesture results there directly; nothing is passed back
        through this call except the exit code.
        """
        logger.info(f"[orchestrator] Starting isolated gesture subprocess for job {job_id}")
        result = subprocess.run(
            [
                sys.executable, "-m", "workers.gesture_subprocess",
                job_id, video_path, str(self.window_size_s),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(
                f"[orchestrator] Gesture subprocess for job {job_id} exited "
                f"{result.returncode}:\n{result.stderr}"
            )
        else:
            logger.info(f"[orchestrator] Gesture subprocess for job {job_id} finished")

    def close(self) -> None:
        pass  # nothing held open — gesture's subprocess exits on its own
        # after every job; prosody/verbal/camera hold no closeable resources.
