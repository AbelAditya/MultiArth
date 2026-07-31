# Bulk Upload (dashboard tab)

Source: [`dashboard/app.py`](../dashboard/app.py) (UI + callbacks),
[`core/bulk_orchestrator.py`](../core/bulk_orchestrator.py) (processing/shipping),
[`core/drive_download.py`](../core/drive_download.py) (Drive fetch)

The **Bulk Upload** tab runs the same sequential multi-video pipeline as
`analyze bulk` from the CLI, but from the dashboard: upload a manifest file,
click Start, and watch each video get downloaded (if needed), analysed, and
shipped to MongoDB — no terminal required.

## Prerequisites

- `MONGO_URI` (and optionally `MONGO_DB`) configured — see the main
  [README](../README.md) — since this is what videos get shipped to.
- If any manifest entry uses `drive_url` instead of a pre-staged local
  `path`, **`GOOGLE_DRIVE_API_KEY`** must be set (`.env.example`). The video
  must be shared **"Anyone with the link"** on Drive — this only needs the
  Drive API enabled on a Google Cloud project, no billing account.

## Step by step

1. Open the **Bulk Upload** tab (next to Live Analysis and Browse Corpus).
2. Drop your manifest file onto the upload zone, or click it to browse —
   accepts `.yml`/`.yaml`/`.json`. See [Manifest format](#manifest-format)
   below if you don't have one yet.
3. Once parsed, a summary line shows how many entries were loaded and which
   collections they'll ship into, e.g.
   `Loaded 12 entries from "manifest.yml" — collections: TedX, Yixi`.
   If the file couldn't be parsed (bad YAML/JSON, missing required fields),
   an error message replaces this line instead — fix the file and re-upload.
4. Optionally tick **"Reprocess already-shipped videos"** if you want to
   force-reprocess entries that have already been shipped (matched by a hash
   of `drive_url`, or the resolved local `path` if there's no `drive_url`).
   Leave it unticked to skip anything already shipped — this is what makes
   re-running the same manifest later, after adding new entries, safe and
   fast.
5. Click **Start**. The button and manifest upload stay locked while a run
   is in progress.
6. While running, the status area shows the current video —
   `[3/12] Talk Title — downloading` / `— analysing` / `— skipped` — plus the
   same per-worker progress bars (Pose Est. / Acoustic / Verbal / Camera)
   Live Analysis shows for a single video, once that video's window-level
   processing is underway.
7. When the whole manifest finishes, a summary replaces the status area:
   succeeded / skipped / failed counts, plus the specific error for each
   failed entry (e.g. `download failed: ...`, `shipping failed: ...`).
8. Shipped videos show up under **Browse Corpus** immediately, grouped by
   whichever `collection` each entry named.

Note: the per-video analysis section (KPI strip, Verbal Language, Acoustic,
Gestural Kinematics, Camera charts) is hidden while on this tab — those show
one video's own windowed analysis, which isn't meaningful while batch
processing many videos at once.

## Manifest format

A YAML or JSON list of entries. Every entry needs `collection`, and **either**
`path` **or** `drive_url` (not necessarily both):

```yaml
- collection: TedX
  drive_url: https://drive.google.com/file/d/XXXXXXXX/view
  label: "Speaker A — Talk 1"

- collection: TedX
  path: /data/staging/already_downloaded.mp4
  drive_url: https://drive.google.com/file/d/YYYYYYYY/view
  label: "Speaker B — Talk 2"
```

| Field | Required | Meaning |
|---|---|---|
| `collection` | yes | Named corpus results ship into (e.g. `"TedX"`, `"Yixi"`) — letters, digits, `_`/`-` only. |
| `path` | no* | A local file to use if it already exists. If omitted (or the file isn't there yet), the video is downloaded from `drive_url` first. |
| `drive_url` | no* | Shareable Google Drive link. Required if `path` is omitted, or if `path` is given but the file doesn't exist locally yet. |
| `label` | no | Free-text tag shown in the Browse Corpus video list. |

\* at least one of `path`/`drive_url` must be present.

## What happens behind the scenes

- Videos are processed **one at a time**, reusing one set of loaded models
  (MediaPipe, Whisper, spaCy) across the whole manifest rather than reloading
  them per video.
- A video downloaded automatically from `drive_url` is **always** deleted
  locally afterward, whether that video succeeded or failed — the durable
  copy is Drive, so there's no reason to keep a local copy around either way.
- A video that was already staged locally (`path` existed before the run
  started) keeps the original `analyze bulk` behaviour: deleted on success,
  **kept in place on failure** so you can fix the problem and re-run the
  manifest without re-downloading or re-staging it.
- See [`core/bulk_orchestrator.py`](../core/bulk_orchestrator.py)'s module
  docstring for the full reasoning behind this split.
