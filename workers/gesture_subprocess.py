"""
workers/gesture_subprocess.py
------------------------------
Standalone entrypoint that runs GestureWorker (MeTRAbs) for a single job in
a fresh OS process, then exits.

Why: TensorFlow's memory allocator doesn't reliably release memory back to
the OS even after the Python model object is deleted and `gc.collect()` is
run — a well-documented TensorFlow behavior, not specific to this project.
Just avoiding eager construction of GestureWorker in Orchestrator.__init__
wouldn't be enough on its own to stop MeTRAbs's ~1.8-2.7GB footprint from
accumulating/staying resident across jobs. Running it in a subprocess that
fully exits when the job is done is what actually guarantees the OS
reclaims 100% of its memory before the next job starts — at the cost of
paying MeTRAbs's ~15-22s model-load time on every job instead of once.

`core/orchestrator.py` also deliberately runs this subprocess *after*, not
concurrently with, the other three workers (verbal/prosody/camera) — this
script being an isolated process only guarantees memory gets released
*afterward*; it says nothing about whether MeTRAbs was the only heavy thing
running *at the time*, which is what actually caused the original crashes
(MeTRAbs's footprint compounding with Whisper/prosody/camera all active at
once via the orchestrator's ThreadPoolExecutor).

This connects to the same Redis-backed FeatureStore as the main dashboard
process (same REDIS_HOST/REDIS_PORT env vars) and writes gesture results
there directly — the main process picks them up from Redis exactly like
any other worker's output, it just never has GestureWorker (or
TensorFlow) loaded in its own memory at all.

Usage:
    python -m workers.gesture_subprocess <job_id> <video_path> <window_size_s>

Exit code 0 on success (including "processed with some per-window errors
logged", matching GestureWorker.process_job's own error-tolerant design —
see its per-window try/except), non-zero only if something prevented
running at all (bad args, model missing, can't open the video, can't
reach Redis).
"""

from __future__ import annotations

import sys

from loguru import logger

from core.feature_store import FeatureStore
from core.preprocessing import compute_windows, probe_video
from workers.gesture_worker import GestureWorker


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            "usage: python -m workers.gesture_subprocess <job_id> <video_path> <window_size_s>",
            file=sys.stderr,
        )
        return 2

    job_id, video_path, window_size_s_raw = argv[1], argv[2], argv[3]
    try:
        window_size_s = float(window_size_s_raw)
    except ValueError:
        print(f"invalid window_size_s: {window_size_s_raw!r}", file=sys.stderr)
        return 2

    logger.info(f"[gesture-subprocess] job {job_id} starting (pid={__import__('os').getpid()})")

    try:
        store = FeatureStore()  # reads REDIS_HOST/REDIS_PORT from env, same as the main process
        # Gesture analysis never touches audio, so there's no need to
        # extract it here (unlike the main orchestrator, which does for
        # prosody/verbal) — an empty audio_path is fine, VideoMeta just
        # carries it along unused.
        meta = probe_video(video_path, audio_path="")
        windows = compute_windows(meta.duration_s, window_size_s)

        worker = GestureWorker(store)
        try:
            worker.process_job(job_id, meta, windows)
        finally:
            worker.close()
    except Exception:
        logger.exception(f"[gesture-subprocess] job {job_id} failed to run")
        return 1

    logger.info(f"[gesture-subprocess] job {job_id} complete, exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
