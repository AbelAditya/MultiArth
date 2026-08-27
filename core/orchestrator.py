"""
core/orchestrator.py
--------------------
Coordinates the full analysis pipeline for a single video.

Architecture (Option B — event-driven modular):
  1. Ingest: extract audio, probe video, compute windows
  2. Dispatch: run all four workers together — prosody/verbal/camera as
     threads, gesture as an isolated subprocess — optionally in parallel
  3. Fuse: merge all modality outputs per window
  4. Report: store final results in FeatureStore

Prosody/verbal/camera run as threads so they can be parallelised without
the GIL bottleneck (each worker is I/O or C-extension bound, not pure
Python). Gesture (MeTRAbs) still runs in a separate OS process rather than
in-process — see `_run_gesture` below and `workers/gesture_subprocess.py` —
because TensorFlow's allocator doesn't reliably release memory back to the
OS even after the model object is explicitly deleted; only a process
actually exiting guarantees that.

Gesture's subprocess is dispatched *alongside* the other three
(`parallel=True`, the default) rather than strictly after them. This is a
deliberate, known tradeoff, not an oversight: running MeTRAbs concurrently
with Whisper and the other workers is exactly what caused real
out-of-memory crashes earlier in this project's history — subprocess
isolation guarantees MeTRAbs's memory gets released *after* its job
finishes, but says nothing about peak memory *while* everything is running
together, which is what actually matters for avoiding a crash.

Two further things worth knowing about, both explicit tradeoffs:

- **Lazy workers.** `_prosody_worker`/`_verbal_worker`/`_camera_worker`
  are properties, not attributes set in `__init__` — each one's actual
  worker (and whatever model it loads — Whisper, in VerbalWorker's case)
  is only constructed the first time it's actually used, not eagerly when
  `Orchestrator` itself is constructed. This matters most for the
  dashboard's module-level `_orch = Orchestrator(store=store)`: without
  this, Whisper would load the moment the dashboard process starts and
  stay resident for its entire lifetime, whether or not a job is ever run.
- **`persistent_gesture`.** Single-video jobs (the default, `False`) use
  `workers/gesture_subprocess.py` — spawn fresh, load MeTRAbs, run one job,
  exit — so MeTRAbs's memory doesn't linger afterward for what's likely
  casual, interspersed-with-other-laptop-use usage. Bulk runs (CLI
  `analyze bulk`, the dashboard's Bulk Upload tab) pass `True`: a single
  `workers/gesture_server.py` subprocess is started once, kept warm across
  every video in the batch (so MeTRAbs's ~15-22s load time is paid once,
  not once per video), and explicitly terminated in `close()` when the
  whole batch finishes — see `_ensure_gesture_server`.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from loguru import logger

from core.feature_store import FeatureStore
from core.fusion_engine import FusionEngine
from core.models import AnalysisJob, FusedWindow, JobStatus
from core.preprocessing import compute_windows, extract_audio, probe_video
from workers.camera_worker import CameraWorker
from workers.prosody_worker import ProsodyWorker
from workers.verbal_worker import VerbalWorker

_GESTURE_SERVER_TIMEOUT_S = 4800  # covers a whole video's worth of per-window
# inference, not just the first call's ~15-22s model load — this scales with
# video length/window count, so 300s (fine for short clips) silently timed
# out on long ones. See _run_gesture_persistent: a timeout here now fails
# the job properly instead of letting analyze()/fuse/ship proceed as if
# gesture had actually finished.
_GESTURE_SERVER_HEALTH_TIMEOUT_S = 30  # server process boot, not model load — should be quick


class Orchestrator:
    def __init__(
        self,
        store: FeatureStore,
        work_dir: str | None = None,
        window_size_s: float = 5.0,
        whisper_model: str = "small",
        whisper_device: str = "cpu",
        parallel: bool = True,
        persistent_gesture: bool = False,
    ):
        self.store = store
        self.work_dir = Path(work_dir or os.environ.get("WORK_DIR", "/tmp/mannerism"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.window_size_s = window_size_s
        self.parallel = parallel
        self.persistent_gesture = persistent_gesture
        self._whisper_model = whisper_model
        self._whisper_device = whisper_device

        # Gesture (MeTRAbs) is never constructed in-process here — it
        # always runs in a subprocess, see _run_gesture. The other three
        # are lazy (see _prosody_worker/_verbal_worker/_camera_worker
        # properties below) — none of them, including this process itself,
        # load anything until the first job actually needs it.
        self._prosody_worker_inst: ProsodyWorker | None = None
        self._verbal_worker_inst: VerbalWorker | None = None
        self._camera_worker_inst: CameraWorker | None = None
        self._fusion = FusionEngine(store)

        # Only used when persistent_gesture=True — see _ensure_gesture_server.
        self._gesture_proc: subprocess.Popen | None = None
        self._gesture_port: int | None = None

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

            # 2. Dispatch all four workers together — see module docstring
            # for why gesture (a subprocess, not an in-process call like the
            # other three) being included here instead of run afterward is
            # a deliberate memory/throughput tradeoff.
            if self.parallel:
                self._run_parallel(job_id, meta, windows, video_path)
            else:
                self._run_sequential(job_id, meta, windows, video_path)

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

    def _run_parallel(self, job_id, meta, windows, video_path) -> None:
        tasks = {
            "prosody": lambda: self._prosody_worker.process_job(job_id, meta, windows),
            "verbal":  lambda: self._verbal_worker.process_job(job_id, meta, windows),
            "camera":  lambda: self._camera_worker.process_job(job_id, meta, windows),
            "gesture": lambda: self._run_gesture(job_id, video_path),
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

    def _run_sequential(self, job_id, meta, windows, video_path) -> None:
        self._prosody_worker.process_job(job_id, meta, windows)
        self._verbal_worker.process_job(job_id, meta, windows)
        self._camera_worker.process_job(job_id, meta, windows)
        self._run_gesture(job_id, video_path)

    # ------------------------------------------------------------------
    # Gesture dispatch — spawn-per-job (default) or persistent-server
    # (persistent_gesture=True) — see module docstring.
    # ------------------------------------------------------------------

    def _run_gesture(self, job_id: str, video_path: str) -> None:
        if self.persistent_gesture:
            self._run_gesture_persistent(job_id, video_path)
        else:
            self._run_gesture_isolated(job_id, video_path)

    def _run_gesture_isolated(self, job_id: str, video_path: str) -> None:
        """
        Runs gesture analysis (MeTRAbs) in a fresh `python -m
        workers.gesture_subprocess` process and waits for it to finish —
        see workers/gesture_subprocess.py and this module's docstring for
        why it's a subprocess at all. The child inherits this process's
        environment (REDIS_HOST/REDIS_PORT etc.), so it connects to the
        same FeatureStore and writes gesture results there directly;
        nothing is passed back through this call except the exit code.

        Streams the subprocess's output line-by-line as it's produced
        (`Popen` + iterating its pipe), rather than `subprocess.run(...,
        capture_output=True)`, which is fully blocking — it hands back
        everything only once the whole subprocess has already exited, so
        nothing is visible while a job is still running, only in one dump
        at the end. `-u` on the child's own invocation matters here too:
        Python block-buffers stdout by default when it isn't an
        interactive terminal (which a pipe never is), so without it the
        *child* would sit on its own output internally regardless of how
        promptly we read our end of the pipe.
        """
        logger.info(f"[orchestrator] Starting isolated gesture subprocess for job {job_id}")
        process = subprocess.Popen(
            [
                sys.executable, "-u", "-m", "workers.gesture_subprocess",
                job_id, video_path, str(self.window_size_s),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merged: loguru defaults to stderr,
            # and one merged stream avoids needing two reader threads (or
            # select()-based multiplexing) just to preserve line ordering.
            text=True,
            bufsize=1,  # line-buffered on our read side
        )
        for line in process.stdout:
            logger.info(f"[gesture-subprocess] {line.rstrip()}")
        process.wait()

        if process.returncode != 0:
            logger.error(f"[orchestrator] Gesture subprocess for job {job_id} exited {process.returncode}")
            # Raise (rather than just log, as this used to) — same reasoning
            # as _run_gesture_persistent's except block: a silent return
            # here looks like success to whatever called _run_gesture, and
            # the job proceeds to fuse/DONE/ship with gesture data missing.
            raise RuntimeError(
                f"Gesture subprocess for job {job_id} exited {process.returncode}"
            )
        logger.info(f"[orchestrator] Gesture subprocess for job {job_id} finished")

    def _run_gesture_persistent(self, job_id: str, video_path: str) -> None:
        """
        Sends this job to the long-lived workers/gesture_server.py instance
        (starting it first if this is the first job of the batch) instead
        of spawning a fresh subprocess — see module docstring's
        `persistent_gesture` explanation.
        """
        self._ensure_gesture_server()
        logger.info(f"[orchestrator] Sending job {job_id} to persistent gesture server")
        try:
            response = requests.post(
                f"http://127.0.0.1:{self._gesture_port}/process_job",
                json={"job_id": job_id, "video_path": video_path, "window_size_s": self.window_size_s},
                timeout=_GESTURE_SERVER_TIMEOUT_S,
            )
            response.raise_for_status()
            logger.info(f"[orchestrator] Persistent gesture server finished job {job_id}")
        except Exception:
            logger.exception(f"[orchestrator] Persistent gesture server failed on job {job_id}")
            # Re-raise (rather than swallow, as this used to) — otherwise
            # this looks like a successful "finished" to _run_parallel's
            # future / _run_sequential's caller, and analyze() proceeds to
            # fuse and mark the job DONE while gesture may still be running
            # server-side (on a timeout, the server never got the memo that
            # we gave up) or may have genuinely failed — either way the job
            # should fail loudly, not ship with missing/incomplete gesture
            # data. See analyze()'s own try/except, which already handles
            # marking the job FAILED for exactly this kind of propagation.
            raise

    def _ensure_gesture_server(self) -> None:
        """Starts workers/gesture_server.py if it isn't already running for this Orchestrator."""
        if self._gesture_proc is not None and self._gesture_proc.poll() is None:
            return  # already running

        self._gesture_port = self._pick_free_port()
        logger.info(f"[orchestrator] Starting persistent gesture server on port {self._gesture_port}")
        self._gesture_proc = subprocess.Popen(
            [sys.executable, "-m", "workers.gesture_server", str(self._gesture_port)],
        )

        health_url = f"http://127.0.0.1:{self._gesture_port}/health"
        deadline = time.time() + _GESTURE_SERVER_HEALTH_TIMEOUT_S
        while time.time() < deadline:
            if self._gesture_proc.poll() is not None:
                raise RuntimeError(
                    f"gesture server exited early (code {self._gesture_proc.returncode}) before becoming healthy"
                )
            try:
                requests.get(health_url, timeout=1).raise_for_status()
                logger.info("[orchestrator] Persistent gesture server is up")
                return
            except requests.exceptions.RequestException:
                time.sleep(0.5)
        raise TimeoutError(f"gesture server did not become healthy within {_GESTURE_SERVER_HEALTH_TIMEOUT_S}s")

    @staticmethod
    def _pick_free_port() -> int:
        # Standard "ask the OS for an unused port" idiom: bind to port 0,
        # read back whatever it assigned, then release it. There's a small,
        # unavoidable race between closing this socket and the gesture
        # server binding the same port — accepted as low-risk here since
        # only one gesture server is ever started per Orchestrator instance.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def close(self) -> None:
        # Only the persistent gesture server holds a resource that needs
        # explicit cleanup — terminating it is what reclaims MeTRAbs's
        # memory for a persistent_gesture batch (see module docstring).
        # Everything else (prosody/verbal/camera, and the plain per-job
        # gesture subprocess) has nothing left open once process_job returns.
        if self._gesture_proc is not None and self._gesture_proc.poll() is None:
            logger.info("[orchestrator] Terminating persistent gesture server")
            self._gesture_proc.terminate()
            try:
                self._gesture_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("[orchestrator] Persistent gesture server didn't exit in time, killing it")
                self._gesture_proc.kill()
                self._gesture_proc.wait()
