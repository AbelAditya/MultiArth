# MultiArth — Multimodal Linguistic Mannerism Analyzer

Automated analysis of speech mannerisms across four modalities:

| Modality | Tool | Dashboard section |
|---|---|---|
| Gestures | MediaPipe Holistic | Pose Estimation |
| Acoustic (formerly "Prosody") | parselmouth (Praat) | Acoustic Properties |
| Verbal language | faster-whisper + spaCy (English & Chinese) | Verbal Language |
| Camera displacement | PySceneDetect + OpenCV (Haar cascade) | Camera |

System design: **Option B — event-driven modular workers**, coordinated by
an orchestrator, communicating via a Redis feature store, with a Plotly Dash
dashboard (branded **MultiArth**) for visualisation.

> Naming note: the project was rebranded **MultiArth** — this is the name
> shown in the dashboard title/logo and used for the packaged distribution
> (`pyproject.toml` project name, Docker image `abx13/multiarth`). The
> repository directory, CLI docstrings, and internal modules/APIs still use
> the original `mannerism_analyzer` / `prosody` naming (e.g.
> `ProsodyWorker`, `ProsodyFeatures`, `job:{id}:prosody:{w}` Redis keys) —
> only the dashboard label was changed to "Acoustic" for end users.

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

### Run with Docker instead

Pre-built image: **[abx13/multiarth](https://hub.docker.com/r/abx13/multiarth)**
on Docker Hub (bundles ffmpeg, the spaCy English/Chinese models, and a
pre-downloaded Whisper `small` model so the first run doesn't need internet
access).

```bash
# Copy and adjust environment variables (Redis host, storage paths, etc.)
cp .env.example .env

# Start Redis + the dashboard app (pulls abx13/multiarth automatically)
docker compose up

# Dashboard available at http://localhost:8050
```

`docker-compose.yml` runs two services:
- `redis` — the feature store backing all workers.
- `app` — the `abx13/multiarth` image, serving the Dash dashboard via
  gunicorn on port `8050`, with `app-data` and `redis-data` named volumes
  for persistence across restarts.

To run the image directly instead of via compose (e.g. against an existing
Redis instance):

```bash
docker run -p 8050:8050 --env-file .env \
  -e REDIS_HOST=<your-redis-host> \
  -v multiarth-data:/data \
  abx13/multiarth
```

For GPU-accelerated Whisper inference, uncomment the NVIDIA `deploy` block
in `docker-compose.yml` and pass `--device cuda` where applicable.

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

### Bulk-process a corpus of videos

Sequentially processes a manifest of already-staged local videos and ships
each finished result to **MongoDB Atlas** — the durable store for a corpus
of many videos, as opposed to Redis's 24h-TTL scratch store for whichever
single job is currently in flight.

```bash
cp .env.example .env   # then set MONGO_URI to your Atlas connection string

uv run analyze bulk vids/manifest.yml
```

The manifest is a YAML (`.yml`/`.yaml`) or JSON list of videos to process,
each already downloaded locally (e.g. from a shared Google Drive folder),
tagged with the named corpus its results should ship into. A starter file
with dummy entries to overwrite lives at `vids/manifest.yml`.

```yaml
- path: /data/staging/talk1.mp4
  collection: TedX
  drive_url: https://drive.google.com/file/d/xxxx/view
  label: "Speaker A — Talk 1"
- path: /data/staging/talk2.mp4
  collection: Yixi
  drive_url: https://drive.google.com/file/d/yyyy/view
  label: "Speaker B — Talk 1"
```

(equivalently, as JSON:)

```json
[
  {"path": "/data/staging/talk1.mp4", "collection": "TedX", "drive_url": "https://drive.google.com/file/d/xxxx/view", "label": "Speaker A — Talk 1"},
  {"path": "/data/staging/talk2.mp4", "collection": "Yixi", "drive_url": "https://drive.google.com/file/d/yyyy/view", "label": "Speaker B — Talk 1"}
]
```

- `path` (required) — local file to process.
- `collection` (required) — names the corpus this video belongs to (e.g.
  `"TedX"`, `"Yixi"`). Each distinct collection gets its own set of three
  MongoDB collections (`{collection}_videos`, `{collection}_fused_windows`,
  `{collection}_artifacts`) in the `MONGO_DB` database, so corpora stay
  fully separate — a video shipped under `"TedX"` is invisible when browsing
  `"Yixi"`, including for dedup checks. Letters, digits, `_` and `-` only.
- `drive_url` (optional) — a shareable Drive link, kept only so the
  dashboard's Browse Corpus tab can embed playback later; it is not used to
  fetch the video.
- `label` (optional) — free-text tag shown in the Browse Corpus list.

Videos are processed **one at a time**. Once a video's fused results are
shipped to Mongo successfully, its local video file and extracted-audio
cache are deleted automatically (the durable copies are now Drive + Mongo).
If shipping fails after a few inline retries, the local files are left in
place and the failure is reported in the run summary — just re-run the same
manifest later; already-shipped videos are skipped automatically (dedup'd
by `drive_url`, falling back to the local path, scoped per `collection`).

Options:
```
--force           Reprocess and re-ship even if already present in Mongo
--mongo-uri       MongoDB Atlas connection string (default: $MONGO_URI)
--mongo-db        MongoDB database name (default: $MONGO_DB or "multiarth")
--window          Window size in seconds (default: 5.0)
--whisper-model   Whisper model size (default: small)
--device          cpu or cuda (default: cpu)
--work-dir        Directory for extracted audio cache (default: /tmp/mannerism)
```

### Browse a processed corpus in the dashboard

With `MONGO_URI` configured, the dashboard's **Browse Corpus** tab (next to
**Live Analysis**) shows a dropdown of every collection that has at least
one shipped video (e.g. `TedX`, `Yixi`); picking one lists its videos —
label, filename, duration, processed date. Clicking a row loads that
video's charts, transcript, and word-sketch tools exactly like a live
analysis job, with video playback embedded from its Google Drive link
instead of a local file.

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
              [ MultiArth Dash Dashboard ]
                Video Upload → Verbal Language →
                Pose Estimation → Acoustic Properties →
                Camera
```

---

## Project Structure

```
mannerism_analyzer/
├── pyproject.toml          # uv / hatch build config; package name "multiarth"
├── cli.py                  # click CLI entry point
├── core/
│   ├── models.py               # Pydantic data models (shared across workers)
│   ├── feature_store.py        # Redis interface (24h-TTL working store)
│   ├── results_repository.py   # MongoDB Atlas interface (durable corpus store)
│   ├── preprocessing.py        # ffmpeg, cv2 video/audio utilities
│   ├── orchestrator.py         # Single-video pipeline coordinator
│   ├── bulk_orchestrator.py    # Sequential multi-video runner + Mongo shipping
│   └── fusion_engine.py        # Cross-modal merging + enrichment
├── workers/
│   ├── gesture_worker.py   # MediaPipe Holistic
│   ├── prosody_worker.py   # parselmouth (Praat) — shown as "Acoustic" in the UI
│   ├── verbal_worker.py    # faster-whisper + spaCy (English & Chinese)
│   └── camera_worker.py    # PySceneDetect + Haar cascade
├── dashboard/
│   ├── app.py               # Plotly Dash dashboard (branded "MultiArth")
│   └── assets/              # Logo, favicon
├── Dockerfile / docker-compose.yml   # Container build, image "abx13/multiarth"
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
- All Redis keys expire after 24 hours. Use `export` to persist a single
  job's results, or `analyze bulk` to process and durably persist a whole
  corpus of videos to MongoDB (see "Bulk-process a corpus of videos" above).
- For GPU inference, install `faster-whisper` with CUDA support and pass
  `--device cuda`.
- The Haar cascade face detector in `camera_worker.py` is a lightweight
  proxy for shot-type classification. For production accuracy, replace
  with a deep-learning face detector (e.g. RetinaFace via `insightface`).
- Verbal language analysis supports both English and Chinese, including a
  dedicated Chinese word-sketch/stop-word list.
</content>
