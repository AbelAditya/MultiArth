"""
core/logging_setup.py
----------------------
Adds a rotating file sink to loguru's logger, on top of its default console
sink (never replaces it). Every entrypoint that can run independently of
some other process's own logger — cli.py, dashboard/app.py, and
workers/gesture_server.py (the persistent bulk-mode gesture server, which
runs detached from the CLI/dashboard process and has never had its own log
capture) — calls this once at startup, so a crash leaves an actual trail on
disk instead of nothing. This gap is what made the first couple of memory-
crash diagnoses in this project dead ends: everything only ever went to
whichever console happened to be open, gone the instant that process died.

workers/gesture_subprocess.py (isolated single-video gesture) doesn't need
its own call — its stdout is already piped and re-logged through whichever
process invoked it (see core/orchestrator.py's _run_gesture_isolated), so
it's covered transparently once that parent process has a file sink.
"""
from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

_LOG_DIR = Path(os.environ.get("LOG_DIR", "/tmp/mannerism/logs"))


def setup_file_logging(name: str) -> None:
    """
    Adds a file sink named after *name* (e.g. "cli", "dashboard",
    "gesture_server") — separate files per entrypoint so a bulk run's
    gesture_server output isn't interleaved line-by-line with the
    orchestrator's own in one file, but everything is still on disk and
    timestamped. Safe to call more than once (loguru just adds another sink,
    it doesn't error) — each real entrypoint below only calls it once.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        _LOG_DIR / f"{name}_{{time:YYYY-MM-DD}}.log",
        rotation="50 MB",
        retention="7 days",
        level="DEBUG",
        enqueue=True,  # thread-safe — _run_parallel's ThreadPoolExecutor
        # workers all log concurrently from separate threads.
    )
    logger.info(f"[logging] File logging enabled -> {_LOG_DIR}/{name}_*.log")
