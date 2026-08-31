"""
core/logging_setup.py
----------------------
Adds a rotating file sink to loguru's logger, on top of its default console
sink (never replaces it). cli.py and dashboard/app.py each call this once
at startup, so a crash leaves an actual trail on disk instead of nothing.
This gap is what made the first couple of memory-crash diagnoses in this
project dead ends: everything only ever went to whichever console happened
to be open, gone the instant that process died.

On the `light-gesture` branch, gesture runs in-process (see
core/orchestrator.py) rather than in its own subprocess/persistent server
the way MeTRAbs did on the main branch, so its logs are already covered by
whichever of the two calls above is active — there's no separate
gesture-specific entrypoint needing its own call here anymore.
"""
from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

_LOG_DIR = Path(os.environ.get("LOG_DIR", "/tmp/mannerism/logs"))


def setup_file_logging(name: str) -> None:
    """
    Adds a file sink named after *name* (e.g. "cli", "dashboard") —
    separate files per entrypoint, but everything is still on disk and
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
