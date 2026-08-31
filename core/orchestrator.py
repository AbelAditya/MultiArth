"""
core/orchestrator.py
--------------------
Coordinates the full analysis pipeline for a single video.

Architecture (Option B — event-driven modular):
  1. Ingest: extract audio, probe video, compute windows
  2. Dispatch: run all four workers together as threads — optionally in
     parallel
  3. Fuse: merge all modality outputs per window
  4. Report: store final results in FeatureStore

All four workers run as threads (not subprocesses) so they can be
parallelised without the GIL bottleneck — each worker is I/O or
C-extension bound, not pure Python.

**Branch note (`light-gesture`):** gesture used to run in an isolated OS
subprocess (`workers/gesture_subprocess.py`/`workers/gesture_server.py`,
both removed on this branch) — main branch's MeTRAbs needed that because
TensorFlow's allocator doesn't reliably release memory back to the OS even
after the model object is deleted, only a process actually exiting
guarantees that. This branch's GestureWorker (MediaPipe) doesn't carry
that same hard requirement, so it's just another lazy in-process worker
now, identical in shape to prosody/verbal/camera — see
`wikis/Gesture-Worker.md`'s "Honest tradeoffs vs. the main branch" for the
measured memory numbers this decision rests on (comparable order of
magnitude to MeTRAbs's own footprint, not a guaranteed-safe reduction —
worth knowing if this ever needs revisiting).

**Lazy workers.** `_prosody_worker`/`_verbal_worker`/`_camera_worker`/
`_gesture_worker` are properties, not attributes set in `__init__` — each
one's actual worker (and whatever model it loads — Whisper, in
VerbalWorker's case) is only constructed the first time it's actually
used, not eagerly when `Orchestrator` itself is constructed. This matters
most for the dashboard's module-level `_orch = Orchestrator(store=store)`:
without this, Whisper would load the moment the dashboard process starts
and stay resident for its entire lifetime, whether or not a job is ever
run. It's also what makes bulk runs keep every worker warm across a whole
manifest for free — `BulkOrchestrator` reuses one `Orchestrator` instance
for the entire batch, so each lazy worker (GestureWorker included, now)
gets constructed once and reused for every video in it, no separate
persistent-server machinery needed the way MeTRAbs required.
"""

from __future__ import annotations

import os
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
from workers.gesture_worker import GestureWorker
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
        self._whisper_model = whisper_model
        self._whisper_device = whisper_device

        # All four workers are lazy (see the properties below) — none of
        # them, including this process itself, load anything until the
        # first job actually needs it.
        self._prosody_worker_inst: ProsodyWorker | None = None
        self._verbal_worker_inst: VerbalWorker | None = None
        self._camera_worker_inst: CameraWorker | None = None
        self._gesture_worker_inst: GestureWorker | None = None
        self._fusion = FusionEngine(store)

    # ------------------------------------------------------------------
    # Lazy worker construction — see module docstring's "Lazy workers"
    # ------------------------------------------------------------------

    @property
    def _prosody_worker(self) -> ProsodyWorker:
        if self._prosody_worker_inst is None:
            self._prosody_worker_inst = ProsodyWorker(self.store)
        return self._prosody_worker_inst

    @property
    def _verbal_worker(self) -> VerbalWorker:
        if self._verbal_worker_inst is None:
            self._verbal_worker_inst = VerbalWorker(
                self.store, model_size=self._whisper_model, device=self._whisper_device
            )
        return self._verbal_worker_inst

    @property
    def _camera_worker(self) -> CameraWorker:
        if self._camera_worker_inst is None:
            self._camera_worker_inst = CameraWorker(self.store)
        return self._camera_worker_inst

    @property
    def _gesture_worker(self) -> GestureWorker:
        if self._gesture_worker_inst is None:
            self._gesture_worker_inst = GestureWorker(self.store)
        return self._gesture_worker_inst

    # ------------------------------------------------------------------

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

            # 2. Dispatch all four workers together, in parallel by default.
            if self.parallel:
                self._run_parallel(job_id, meta, windows)
            else:
                self._run_sequential(job_id, meta, windows)

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
            "gesture": lambda: self._gesture_worker.process_job(job_id, meta, windows),
        }
        # Collect every worker's outcome before raising anything — waiting
        # for all of them (not bailing on the first failure) means one
        # worker's exception doesn't leave the others' threads to keep
        # running detached from anything watching them.
        failures: dict[str, BaseException] = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                exc = future.exception()
                if exc:
                    logger.error(f"[orchestrator] Worker '{name}' raised: {exc}")
                    failures[name] = exc
                else:
                    logger.info(f"[orchestrator] Worker '{name}' finished")

        if failures:
            # Re-raise (rather than just log, as this used to) — otherwise
            # analyze() has no way to know a worker failed, and proceeds to
            # fuse/mark the job DONE/let it ship with that worker's data
            # missing or incomplete. See analyze()'s own try/except, which
            # already handles marking the job FAILED for exactly this.
            names = ", ".join(failures)
            raise RuntimeError(f"Worker(s) failed: {names}") from next(iter(failures.values()))

    def _run_sequential(self, job_id, meta, windows) -> None:
        self._prosody_worker.process_job(job_id, meta, windows)
        self._verbal_worker.process_job(job_id, meta, windows)
        self._camera_worker.process_job(job_id, meta, windows)
        self._gesture_worker.process_job(job_id, meta, windows)

    def close(self) -> None:
        # Defensive only — GestureWorker.process_job already opens/closes
        # its own landmarkers per job (see workers/gesture_worker.py), so
        # there's nothing routinely left open here. This only matters if a
        # job crashed badly enough to skip that cleanup, leaving a
        # constructed-but-not-yet-reused GestureWorker instance holding
        # landmarkers open on an Orchestrator that's about to be discarded
        # (see cli.py / core/bulk_orchestrator.py's own finally: close()).
        # None of the other three workers hold anything needing explicit
        # cleanup at this level.
        if self._gesture_worker_inst is not None:
            self._gesture_worker_inst.close()
