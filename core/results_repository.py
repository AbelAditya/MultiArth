"""
core/results_repository.py
---------------------------
MongoDB Atlas persistence layer for fully-processed videos.

Where FeatureStore (Redis) is the 24h-TTL scratch store for whichever job is
currently in flight, ResultsRepository is the durable system of record that
`core/bulk_orchestrator.py` ships a video's results to once it's DONE. The
dashboard's Browse Corpus tab reads from here for already-shipped videos.

Videos are grouped into named corpora — e.g. "TedX", "Yixi" — set per video
in the bulk manifest. Each corpus gets its own set of three physical Mongo
collections (database name configurable, default "multiarth"), so corpora
are visibly separate in Atlas's collection browser and stay independently
queryable, rather than mixed into one collection with a discriminator field:

  {collection}_videos          one doc per shipped job: _id=job_id,
                                video_filename, drive_url, label, dedupe_key,
                                window_size_s, total_windows, duration_s,
                                created_at, completed_at
  {collection}_fused_windows   one doc per (job_id, window_idx), a FusedWindow
  {collection}_artifacts       one doc per job_id: wordlist, ngrams,
                                collocations, spectrogram, waveform

Fused windows stay a separate per-window collection (not embedded in the
video doc) regardless of corpus, so long videos with many windows never
approach the 16MB BSON document cap.

Only successfully-shipped videos get a `videos` doc — failed ships are not
persisted here at all; the caller logs them and the source Redis job simply
expires on its existing TTL.
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

from pymongo import MongoClient

from .models import AnalysisJob, FusedWindow

_COLLECTION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_VIDEOS_SUFFIX = "_videos"
_FUSED_WINDOWS_SUFFIX = "_fused_windows"
_ARTIFACTS_SUFFIX = "_artifacts"


def _validate_collection(collection: str) -> None:
    if not collection or not _COLLECTION_NAME_RE.match(collection):
        raise ValueError(
            f"Invalid collection name {collection!r} — use letters, digits, "
            "underscores or hyphens only (e.g. 'TedX', 'Yixi')"
        )


class ResultsRepository:
    def __init__(
        self,
        uri: str | None = None,
        db_name: str | None = None,
        server_selection_timeout_ms: int = 5000,
    ):
        uri = uri or os.environ.get("MONGO_URI")
        db_name = db_name or os.environ.get("MONGO_DB", "multiarth")
        if not uri:
            raise ValueError("MongoDB URI not provided (pass uri= or set MONGO_URI)")

        # A short server-selection timeout means an unreachable/misconfigured
        # Atlas cluster fails fast here instead of blocking callers — e.g.
        # the dashboard's startup — for pymongo's default ~30s timeout.
        self.client = MongoClient(uri, serverSelectionTimeoutMS=server_selection_timeout_ms)
        self.db = self.client[db_name]
        self._indexed_collections: set[str] = set()

    # ------------------------------------------------------------------
    # Per-corpus collection handles, created + indexed lazily on first use
    # ------------------------------------------------------------------

    def _collections(self, collection: str):
        _validate_collection(collection)
        videos = self.db[collection + _VIDEOS_SUFFIX]
        fused_windows = self.db[collection + _FUSED_WINDOWS_SUFFIX]
        artifacts = self.db[collection + _ARTIFACTS_SUFFIX]

        if collection not in self._indexed_collections:
            fused_windows.create_index("job_id")
            fused_windows.create_index([("job_id", 1), ("window_idx", 1)], unique=True)
            videos.create_index("dedupe_key")
            self._indexed_collections.add(collection)

        return videos, fused_windows, artifacts

    # ------------------------------------------------------------------
    # Write (bulk ship)
    # ------------------------------------------------------------------

    def save_job(
        self,
        collection: str,
        job: AnalysisJob,
        *,
        drive_url: str | None,
        label: str | None,
        dedupe_key: str,
        duration_s: float | None,
    ) -> None:
        videos, _, _ = self._collections(collection)
        doc = job.model_dump(mode="json")
        doc["_id"] = job.job_id
        # Prefer the manifest's human label — for a Drive-sourced entry
        # (no local `path` given), job.video_path is core/bulk_orchestrator
        # .py's _resolve_path destination, named after the Drive file ID
        # (deliberately, for a stable re-download path — see that
        # function's own docstring), not anything human-readable. Falling
        # back to the real basename keeps the CLI's manually-staged-path
        # workflow unchanged, where the basename already *is* meaningful.
        doc["video_filename"] = f"{label}.mp4" if label else os.path.basename(job.video_path)
        doc["drive_url"] = drive_url
        doc["label"] = label
        doc["dedupe_key"] = dedupe_key
        doc["duration_s"] = duration_s
        doc["shipped_at"] = time.time()
        videos.replace_one({"_id": job.job_id}, doc, upsert=True)

    def save_fused_windows(self, collection: str, job_id: str, windows: list[FusedWindow]) -> None:
        _, fused_windows, _ = self._collections(collection)
        fused_windows.delete_many({"job_id": job_id})
        if not windows:
            return
        docs = [
            {"job_id": job_id, "window_idx": idx, **w.model_dump(mode="json")}
            for idx, w in enumerate(windows)
        ]
        fused_windows.insert_many(docs)

    def save_artifacts(
        self,
        collection: str,
        job_id: str,
        *,
        wordlist: Optional[dict],
        ngrams: Optional[dict],
        collocations: Optional[dict],
        spectrogram: Optional[dict],
        waveform: Optional[dict],
        segmented_tokens: Optional[list] = None,
    ) -> None:
        _, _, artifacts = self._collections(collection)
        artifacts.replace_one(
            {"_id": job_id},
            {
                "_id": job_id,
                "wordlist": wordlist,
                "ngrams": ngrams,
                "collocations": collocations,
                "spectrogram": spectrogram,
                "waveform": waveform,
                "segmented_tokens": segmented_tokens,
            },
            upsert=True,
        )

    # ------------------------------------------------------------------
    # Dedup lookup (bulk skip-if-already-shipped)
    # ------------------------------------------------------------------

    def find_by_dedupe_key(self, collection: str, dedupe_key: str) -> Optional[str]:
        videos, _, _ = self._collections(collection)
        doc = videos.find_one({"dedupe_key": dedupe_key}, {"_id": 1})
        return doc["_id"] if doc else None

    def delete_job_data(self, collection: str, job_id: str) -> None:
        """Removes every stored document for one job — video doc, fused
        windows, artifacts. Used when a `--force`/bulk-force reprocess of
        an already-shipped video should *overwrite* its previous run
        rather than accumulate alongside it under a different job_id
        (every run mints a fresh one) — see core/bulk_orchestrator.py's
        _ship, which looks up the previous run via find_by_dedupe_key and
        calls this before saving the new one."""
        videos, fused_windows, artifacts = self._collections(collection)
        videos.delete_one({"_id": job_id})
        fused_windows.delete_many({"job_id": job_id})
        artifacts.delete_one({"_id": job_id})

    # ------------------------------------------------------------------
    # Read (dashboard Browse Corpus)
    # ------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        """Corpus names with at least one shipped video, discovered from
        existing `{collection}_videos` collections in the database."""
        names = []
        for name in self.db.list_collection_names():
            if name.endswith(_VIDEOS_SUFFIX):
                names.append(name[: -len(_VIDEOS_SUFFIX)])
        return sorted(names)

    def list_videos(self, collection: str) -> list[dict]:
        videos, _, _ = self._collections(collection)
        return list(videos.find({}))

    def get_job(self, collection: str, job_id: str) -> Optional[AnalysisJob]:
        videos, _, _ = self._collections(collection)
        doc = videos.find_one({"_id": job_id})
        if not doc:
            return None
        return AnalysisJob(**{k: v for k, v in doc.items() if k in AnalysisJob.model_fields})

    def get_video_doc(self, collection: str, job_id: str) -> Optional[dict]:
        """Raw video doc, unfiltered — unlike get_job, keeps fields outside
        AnalysisJob's own schema (drive_url, label, video_filename, ...)."""
        videos, _, _ = self._collections(collection)
        return videos.find_one({"_id": job_id})

    def get_all_fused(self, collection: str, job_id: str) -> list[FusedWindow]:
        _, fused_windows, _ = self._collections(collection)
        docs = fused_windows.find({"job_id": job_id}).sort("window_idx", 1)
        return [
            FusedWindow(**{k: v for k, v in d.items() if k not in ("_id", "job_id", "window_idx")})
            for d in docs
        ]

    def get_artifacts(self, collection: str, job_id: str) -> Optional[dict]:
        _, _, artifacts = self._collections(collection)
        return artifacts.find_one({"_id": job_id})

    def close(self) -> None:
        self.client.close()
