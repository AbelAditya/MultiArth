# Multimodal Linguistic Mannerism Analyzer

Automated analysis of speech mannerisms across four modalities:

| Modality | Tool |
|---|---|
| Gestures | MediaPipe Holistic |
| Auditory tone / prosody | parselmouth (Praat) |
| Verbal language | faster-whisper + spaCy |
| Camera displacement | PySceneDetect + OpenCV |

System design: **Option B — event-driven modular workers**, coordinated by
an orchestrator, communicating via a Redis feature store, with a Plotly Dash
dashboard for visualisation.

---

## Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- Redis server running locally (`redis-server`)
- `ffmpeg` on PATH (for audio extraction)

---

## Setup

```bash
# Clone / enter project
cd mannerism_analyzer

# Create virtual environment and install all dependencies
uv sync

# Download spaCy English model
uv run python -m spacy download en_core_web_sm
```

---

## Usage

### Run analysis

```bash
uv run analyze run path/to/tedtalk.mp4
# → prints a job_id on completion, e.g. "a3f2b1c9"
```

Options:
```
--window      Window size in seconds (default: 5.0)
--whisper-model  Whisper model size: tiny / base / small / medium (default: base)
--device      cpu or cuda (default: cpu)
--sequential  Disable parallel workers (useful for debugging)
--work-dir    Directory for extracted audio cache (default: /tmp/mannerism)
```

### Check status

```bash
uv run analyze status a3f2b1c9
```

### Launch dashboard

```bash
uv run analyze dashboard
# Open http://localhost:8050, enter job ID
```

### Export results to JSON

```bash
uv run analyze export a3f2b1c9 --out results.json
```

---

## Architecture

```
Video File
    │
    ▼
[ Orchestrator ]
    │  extract audio (ffmpeg)
    │  probe video metadata
    │  compute time windows
    │
    ├──────────────────────────────────────────────┐
    │  ThreadPoolExecutor (4 workers in parallel)  │
    │                                              │
    │  ┌─────────────────┐  ┌──────────────────┐  │
    │  │  GestureWorker  │  │  ProsodyWorker   │  │
    │  │  (MediaPipe)    │  │  (parselmouth)   │  │
    │  └────────┬────────┘  └────────┬─────────┘  │
    │           │                    │             │
    │  ┌────────┴────────┐  ┌────────┴─────────┐  │
    │  │  VerbalWorker   │  │  CameraWorker    │  │
    │  │  (Whisper+spaCy)│  │  (PySceneDetect) │  │
    │  └────────┬────────┘  └────────┬─────────┘  │
    │           └──────────┬─────────┘             │
    └──────────────────────┼───────────────────────┘
                           ▼
                  [ Redis FeatureStore ]
                    job:{id}:gesture:{w}
                    job:{id}:prosody:{w}
                    job:{id}:verbal:{w}
                    job:{id}:camera:{w}
                           │
                           ▼
                  [ FusionEngine ]
                    cross-modal enrichment
                    speech rate backfill
                           │
                           ▼
                  job:{id}:fused:{w}
                           │
                           ▼
                  [ Dash Dashboard ]
```

---

## Project Structure

```
mannerism_analyzer/
├── pyproject.toml          # uv / hatch build config + all dependencies
├── cli.py                  # click CLI entry point
├── core/
│   ├── models.py           # Pydantic data models (shared across workers)
│   ├── feature_store.py    # Redis interface
│   ├── preprocessing.py    # ffmpeg, cv2 video/audio utilities
│   ├── orchestrator.py     # Pipeline coordinator
│   └── fusion_engine.py    # Cross-modal merging + enrichment
├── workers/
│   ├── gesture_worker.py   # MediaPipe Holistic
│   ├── prosody_worker.py   # parselmouth (Praat)
│   ├── verbal_worker.py    # faster-whisper + spaCy
│   └── camera_worker.py    # PySceneDetect + Haar cascade
├── dashboard/
│   └── app.py              # Plotly Dash dashboard
└── tests/
    └── test_models.py      # Pydantic model unit tests
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

---

## Notes

- Audio is extracted to `--work-dir` and cached; re-running on the same
  video skips re-extraction.
- All Redis keys expire after 24 hours. Use `export` to persist results.
- For GPU inference, install `faster-whisper` with CUDA support and pass
  `--device cuda`.
- The Haar cascade face detector in `camera_worker.py` is a lightweight
  proxy for shot-type classification. For production accuracy, replace
  with a deep-learning face detector (e.g. RetinaFace via `insightface`).
