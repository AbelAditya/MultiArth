"""
core/preprocessing.py
---------------------
Shared helpers for extracting audio from video and splitting
frames into time windows before dispatching to workers.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


@dataclass
class VideoMeta:
    path: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration_s: float
    audio_path: str  # extracted WAV


def extract_audio(video_path: str, out_dir: str) -> str:
    """
    Extract audio track from video to a 16 kHz mono WAV using ffmpeg.
    Returns path to the WAV file.
    """
    out_path = Path(out_dir) / (Path(video_path).stem + "_audio.wav")
    if out_path.exists():
        logger.info(f"Audio already extracted: {out_path}")
        return str(out_path)

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1",          # mono
        "-ar", "16000",      # 16 kHz — optimal for Whisper + parselmouth
        "-vn",               # no video
        str(out_path),
    ]
    logger.info(f"Extracting audio: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    return str(out_path)


def probe_video(video_path: str, audio_path: str) -> VideoMeta:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration_s = total_frames / fps if fps > 0 else 0.0
    return VideoMeta(
        path=video_path,
        fps=fps,
        total_frames=total_frames,
        width=width,
        height=height,
        duration_s=duration_s,
        audio_path=audio_path,
    )


def compute_windows(duration_s: float, window_size_s: float) -> list[tuple[float, float]]:
    """
    Returns list of (start_s, end_s) half-open intervals covering the full duration.
    The last window may be shorter than window_size_s.
    """
    windows = []
    t = 0.0
    while t < duration_s:
        end = min(t + window_size_s, duration_s)
        windows.append((t, end))
        t += window_size_s
    return windows


def frames_for_window(
    video_path: str,
    start_s: float,
    end_s: float,
    fps: float,
    max_frames: int = 150,
) -> Iterator[tuple[float, np.ndarray]]:
    """
    Yield frames in [start_s, end_s) from a video, as (timestamp_s, BGR
    frame). Downsamples if the window would exceed max_frames.

    ## Streaming, not a list

    This used to build and return the whole window as a list, and was
    changed because that list *was* the memory profile that OOM-killed the
    MeTRAbs branch (confirmed at the time via journalctl/OOM-killer
    forensics, and documented in workers/gesture_worker.py's "Frame
    resolution" section). Measured on 1080p footage, one decoded frame is
    6.22MB and a full 150-frame window is 933MB — all of it live before any
    inference has started, since the list was built to completion first.
    Yielding holds one frame instead.

    Note the old consumer already dropped its references as it went; that
    shortened the tail but not the peak, which is at construction. The fix
    has to be not building the list at all.

    Nothing else changes. Every frame in the window is still decoded and
    `step` still decides only which ones are handed on, so this is not
    faster — decode is 2.05ms/frame against a ~100ms/frame inference
    budget. The seek, the ordering and the timestamps are identical.

    What it *enables* is running windows in separate processes
    (GestureWorker's pool): six unstreamed windows would hold 5.6GB of
    frames at once, which does not fit alongside ~489MB of models per
    process on a 16GB machine. Streamed, the same six hold ~37MB.

    Two consequences of being a generator worth knowing. The VideoCapture
    stays open across the *consumer's* loop rather than just the read loop,
    so it is released when the generator is exhausted or closed (the
    `finally` covers early `break` and exceptions alike). And the call is
    lazy: nothing is read until the first `next()`. A bad path yields
    nothing where it previously returned [] — the same observable outcome,
    just deferred.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        start_frame = int(start_s * fps)
        end_frame = int(end_s * fps)
        total = end_frame - start_frame

        step = max(1, total // max_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        idx = start_frame
        while idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (idx - start_frame) % step == 0:
                yield idx / fps, frame
            idx += 1
    finally:
        cap.release()
