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

# scenedetect pulls in opencv-python; opencv-contrib-python isn't pulled in
# by anything else in the dependency tree (MeTRAbs uses TensorFlow directly,
# not OpenCV), but camera_worker.py needs its Haar cascade data, which
# plain opencv-python doesn't ship. Installing it explicitly, after
# scenedetect, forces its complete file set to win the site-packages/cv2/
# merge deterministically instead of leaving the outcome to install order.
RUN uv pip install --reinstall-package opencv-contrib-python "opencv-contrib-python==4.13.0.92"

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# System deps: ffmpeg for audio extraction, OpenCV runtime libs, curl+unzip
# to fetch the MeTRAbs model below
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    unzip \
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
# Pre-download the MeTRAbs pose model (see workers/gesture_worker.py) —
# a standalone TensorFlow SavedModel, not distributed via pip. Uses the
# YOLOv4-tiny-detector variant (mob3s_y4t, ~31MB) rather than the
# full-YOLOv4 one (mob3s_y4, ~248MB) — the latter's runtime footprint
# was part of why this needed to move to an isolated subprocess (see
# workers/gesture_subprocess.py, core/orchestrator.py).
RUN mkdir -p models && \
    curl -sL -o /tmp/metrabs.zip \
      https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_mob3s_y4t.zip && \
    unzip -q /tmp/metrabs.zip -d models && \
    rm /tmp/metrabs.zip

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
