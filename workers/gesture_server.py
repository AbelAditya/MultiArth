"""
workers/gesture_server.py
--------------------------
Long-lived HTTP server wrapping GestureWorker (MeTRAbs), for bulk
processing runs (CLI `analyze bulk`, the dashboard's Bulk Upload tab) where
reloading the model per video would waste real time — a whole manifest can
be dozens of videos, and MeTRAbs's own model-load alone takes ~15-22s.
Loads MeTRAbs once, on the first request, and stays warm for as many
/process_job calls as the parent process sends it, until the parent
explicitly terminates this process (see core/orchestrator.py's
`persistent_gesture` mode, specifically `Orchestrator.close()`).

Single-video jobs deliberately do NOT use this — they still use
workers/gesture_subprocess.py's spawn-fresh-then-exit-per-job pattern, so
MeTRAbs's memory doesn't linger afterward for what's likely much more
casual, interspersed-with-other-laptop-use usage. This module exists
specifically for the case where the reload-per-video cost is large enough
(many videos, back to back) that trading "reclaim memory the instant this
one job finishes" for "reclaim it once, at the end of the whole batch"
is worth it — the parent is still expected to terminate this process when
the batch ends, so memory is still reliably reclaimed, just on a longer
(batch-scoped, not job-scoped) cycle.

Bound to 127.0.0.1 only — unlike colab/sensevoice_server.ipynb (deliberately
reachable over the internet), this never leaves the machine, so no API key
is needed.

Usage: python -m workers.gesture_server <port>
"""

from __future__ import annotations

import sys

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from core.feature_store import FeatureStore
from core.preprocessing import compute_windows, probe_video
from workers.gesture_worker import GestureWorker

app = FastAPI()
_worker: GestureWorker | None = None


class ProcessJobRequest(BaseModel):
    job_id: str
    video_path: str
    window_size_s: float


@app.get("/health")
def health():
    # Deliberately doesn't report whether the model is loaded yet — the
    # parent only needs to know the server process itself is up and can
    # accept a request; the first /process_job call absorbs MeTRAbs's load
    # time as part of that request's latency, same total cost as today's
    # per-job subprocess, just paid once instead of once per video.
    return {"status": "ok"}


@app.post("/process_job")
def process_job(req: ProcessJobRequest):
    global _worker
    if _worker is None:
        store = FeatureStore()
        _worker = GestureWorker(store)
        logger.info("[gesture-server] MeTRAbs loaded, ready for subsequent jobs")

    logger.info(f"[gesture-server] job {req.job_id} starting")
    try:
        # Gesture analysis never touches audio — see gesture_subprocess.py.
        meta = probe_video(req.video_path, audio_path="")
        windows = compute_windows(meta.duration_s, req.window_size_s)
        _worker.process_job(req.job_id, meta, windows)
    except Exception as exc:
        logger.exception(f"[gesture-server] job {req.job_id} failed")
        raise HTTPException(500, str(exc)) from exc

    logger.info(f"[gesture-server] job {req.job_id} complete")
    return {"status": "ok"}


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m workers.gesture_server <port>", file=sys.stderr)
        sys.exit(2)
    port = int(sys.argv[1])

    # This process is launched via a bare subprocess.Popen with no
    # stdout/stderr redirection (core/orchestrator.py's
    # _ensure_gesture_server) — unlike the isolated gesture_subprocess.py
    # path, nothing re-logs its output through a parent process's own
    # logger, so without this it's invisible the moment its console goes
    # away (see wikis/Gesture-Worker.md's memory-growth history — this is
    # exactly the gap that made diagnosing a real crash here a dead end).
    from core.logging_setup import setup_file_logging
    setup_file_logging("gesture_server")

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
