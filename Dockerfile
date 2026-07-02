# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# System deps: ffmpeg for audio extraction, OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
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
