# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
# --frozen: install exactly what's pinned in uv.lock rather than re-resolving
# from pyproject.toml alone. Without a lockfile in the build context, `uv
# sync` resolves fresh inside the container — which isn't guaranteed to match
# the (tested, working) local resolution, and has in practice landed on a
# broken one: llvmlite==0.36.0 (a 2021 release with a hard guard against
# Python >=3.10, pulled in transitively via funasr -> umap-learn ->
# pynndescent), instead of the modern 0.48.0 the local lockfile pins.
RUN uv sync --frozen --no-dev --no-install-project

# scenedetect pulls in opencv-python; on this branch mediapipe pulls in
# opencv-contrib-python itself too (confirmed directly against its own
# PyPI metadata) — but camera_worker.py needs Haar cascade data specifically,
# which isn't guaranteed to win whichever variant's file set install order
# happens to leave in site-packages/cv2/. Installing opencv-contrib-python
# explicitly, last, forces its complete file set to win that merge
# deterministically rather than leaving the outcome to install order.
RUN uv pip install --reinstall-package opencv-contrib-python "opencv-contrib-python==4.13.0.92"

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# System deps: ffmpeg for audio extraction, OpenCV runtime libs, curl to
# fetch the models below (no unzip needed — .task/.pth files, not zips).
# libegl1/libgles2 specifically for MediaPipe: its native
# tasks/c/libmediapipe.so dynamically links against libEGL.so.1 and
# libGLESv2.so.2 unconditionally, even for CPU-only PoseLandmarker use —
# confirmed directly via `ldd` against the real installed .so inside a
# python:3.11-slim container matching this image, which failed to load at
# all (ctypes.CDLL OSError) with just libgl1. This whole branch's
# MediaPipe pipeline was only ever verified on the host machine's own
# desktop Linux environment (which already has full GL/EGL libraries
# installed), never inside an actual built-and-run container until this
# gap surfaced — so this wasn't something the gallery-building feature
# introduced, just the first thing that actually exercised
# PoseLandmarker.create_from_options inside a real container.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Download spaCy models
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download zh_core_web_sm

# Pre-download Whisper small model so first run doesn't need internet
RUN python - <<'EOF'
from faster_whisper import WhisperModel
WhisperModel("small", device="cpu", compute_type="int8")
EOF

# Pre-download SenseVoice (Chinese ASR — see VerbalWorker._transcribe_alt)
# and its VAD model, same reasoning as Whisper above.
RUN python - <<'EOF'
from funasr import AutoModel
AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cpu",
    disable_update=True,
)
EOF
# Pre-download the MediaPipe pose model (see workers/gesture_worker.py) —
# a small .task file, not distributed via pip. workers/gesture_worker.py's
# _ensure_model would also download this lazily at first use if this step
# were skipped, but baking it in at build time avoids paying that ~ a few
# hundred ms->seconds download cost (network-dependent) on a container's
# first real job. No hand model — HandLandmarker was tried and removed
# again (see workers/gesture_worker.py's module docstring).
#
# Also pre-download OSNet (workers/_reid.py) — the speaker-re-identification
# embedding model used by the dashboard's Bulk Upload gallery-building flow
# and by GestureWorker's gallery-based matching (see
# wikis/Gesture-Worker.md's "Speaker re-identification"). Same reasoning as
# the pose model: workers/_reid.py's ensure_reid_weights() would download
# this lazily on first use otherwise — real, since bulk gallery-building is
# the *first* thing a dashboard session does with a fresh manifest, not
# something deferred until later the way this project's other lazy-loaded
# models are.
RUN mkdir -p models && \
    curl -sL -o models/pose_landmarker_lite.task \
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task && \
    curl -sL -o models/osnet_x0_25_msmt17.pth \
      https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth

# Copy application code
COPY . .

# Uploads and work dirs (overridable via env; matched to compose volumes)
ENV UPLOAD_DIR=/data/uploads \
    WORK_DIR=/data/work \
    REDIS_HOST=redis

EXPOSE 8050

CMD ["python", "-m", "gunicorn", \
     "--bind", "0.0.0.0:8050", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "0", \
     "dashboard.app:server"]
