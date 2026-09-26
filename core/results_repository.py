"""
core/results_repository.py
---------------------------
MongoDB Atlas persistence layer for fully-processed videos.

Where FeatureStore (Redis) is the 24h-TTL scratch store for whichever job is
currently in flight, ResultsRepository is the durable system of record that
`core/bulk_orchestrator.py` ships a video's results to once it's DONE. The
dashboard's Browse Corpus tab reads from here for already-shipped videos.

Videos are grouped into named corpora — e.g. "Ted", "YiXi" — set per video
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
  {collection}_prosody_dense   one doc per job_id: f0 + intensity at a fixed
                                hop (~10ms), as float32 binary — see
                                put_dense_prosody

Flagging runs are deliberately NOT stored here: they are cheap to recompute,
they change whenever the manifest does, and the clusters are nearly full.
They live as files instead — see core/flag_runner.py's save_run.

Fused windows stay a separate per-window collection (not embedded in the
video doc) regardless of corpus, so long videos with many windows never
approach the 16MB BSON document cap.

Only successfully-shipped videos get a `videos` doc — failed ships are not
persisted here at all; the caller logs them and the source Redis job simply
expires on its existing TTL.

Shards — one corpus across several clusters
-------------------------------------------
A free Atlas cluster caps out at 512MB, which a few dozen talks fill. So the
repository can span several clusters ("shards"), configured in fill order:

  MONGO_URI      shard 1 (the original, and still the only one required)
  MONGO_URI_2    shard 2
  MONGO_URI_3    ...  any number, ordered by their suffix

Every shard uses the same database name (MONGO_DB) and the same collection
layout above; a shard is just more room, not a different schema.

  Writes   One job lands *entirely* on one shard — its video doc, fused
           windows and artifacts together, so a job is never split and can
           be deleted or read from a single place. The shard is the first,
           in fill order, whose used space plus this job's estimated size
           stays under MONGO_SHARD_LIMIT_MB × _FILL_RATIO. That estimate is
           only a first guess at how Atlas counts: if Atlas itself refuses a
           write for quota, the partial job is deleted from that shard, the
           shard is marked full for the rest of the process, and the whole
           job goes to the next shard. When no shard has room, shipping
           fails loudly (add a MONGO_URI_n).
  Reads    Merged across every shard: Browse Corpus lists every corpus and
           video wherever it lives, and a job is read from whichever shard
           holds it. Callers never name a shard.
  Dedupe   find_by_dedupe_key and delete_job_data search *all* shards, so a
           video shipped to shard 1 is still skipped — or, on a forced
           reprocess, removed — when new writes go to shard 2.

Failure handling is deliberately asymmetric. Listing for the dashboard
tolerates an unreachable shard (logged, and its videos simply don't appear)
so one cluster being down doesn't take the whole Browse tab with it. Dedupe,
delete and write do *not*: skipping an unreachable shard there would mean
reprocessing a video that is already shipped, or leaving a stale copy
behind, with nothing raised.

Used space is measured as dataSize + indexSize summed over every database on
the cluster (Atlas's free-tier quota is per cluster, not per database — a
cluster holding Atlas's `sample_mflix` spends quota on it too).
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

import bson
import numpy as np
from loguru import logger
from pymongo import MongoClient
from pymongo.errors import OperationFailure

from .models import AnalysisJob, FusedWindow

_COLLECTION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_VIDEOS_SUFFIX = "_videos"
_DENSE_PROSODY_SUFFIX = "_prosody_dense"
_FUSED_WINDOWS_SUFFIX = "_fused_windows"
_ARTIFACTS_SUFFIX = "_artifacts"

_EXTRA_URI_RE = re.compile(r"^MONGO_URI_(\d+)$")
_DEFAULT_SHARD_LIMIT_MB = 512  # Atlas M0 free tier
# Only fill a shard to this fraction of its limit. The job-size estimate
# (BSON bytes of the documents) is not exactly what Atlas bills, and a
# cluster that hits its quota mid-ship refuses writes — so leave margin.
_FILL_RATIO = 0.9
_SYSTEM_DBS = {"admin", "local", "config"}


def _is_quota_error(exc: Exception) -> bool:
    """Atlas refusing a write because the cluster is over its storage quota
    ("you are over your space quota, using 513 MB of 512 MB")."""
    return isinstance(exc, OperationFailure) and "space quota" in str(exc).lower()


def _validate_collection(collection: str) -> None:
    if not collection or not _COLLECTION_NAME_RE.match(collection):
        raise ValueError(
            f"Invalid collection name {collection!r} — use letters, digits, "
            "underscores or hyphens only (e.g. 'Ted', 'YiXi')"
        )


def _uris_from_env() -> list[str]:
    """MONGO_URI first, then MONGO_URI_<n> in numeric order."""
    uris = [os.environ["MONGO_URI"]] if os.environ.get("MONGO_URI") else []
    extras = sorted(
        (int(m.group(1)), v)
        for k, v in os.environ.items()
        if (m := _EXTRA_URI_RE.match(k)) and v
    )
    return uris + [v for _, v in extras]


def _host(uri: str) -> str:
    """Host part of a connection string, for logs — never the credentials."""
    rest = uri.split("://", 1)[-1]
    return rest.rsplit("@", 1)[-1].split("/", 1)[0].split("?", 1)[0]


class _Shard:
    def __init__(self, uri: str, db_name: str, timeout_ms: int):
        self.name = _host(uri)
        self.client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        self.db = self.client[db_name]
        self.indexed_collections: set[str] = set()
        # Set when Atlas refuses a write for quota — authoritative, unlike
        # used_bytes, which is only our estimate of how Atlas counts.
        self.full = False

    def used_bytes(self) -> int:
        total = 0
        for name in self.client.list_database_names():
            if name in _SYSTEM_DBS:
                continue
            stats = self.client[name].command("dbStats")
            total += int(stats.get("dataSize", 0)) + int(stats.get("indexSize", 0))
        return total


class ResultsRepository:
    def __init__(
        self,
        uri: str | None = None,
        db_name: str | None = None,
        server_selection_timeout_ms: int = 5000,
        *,
        uris: list[str] | None = None,
        shard_limit_mb: float | None = None,
    ):
        """`uris` (fill order) wins over `uri`; with neither, shards come
        from MONGO_URI / MONGO_URI_<n>. Passing a single `uri` gives an
        ordinary one-cluster repository."""
        if uris:
            uri_list = list(uris)
        elif uri:
            uri_list = [uri]
        else:
            uri_list = _uris_from_env()
        db_name = db_name or os.environ.get("MONGO_DB", "multiarth")
        if not uri_list:
            raise ValueError("MongoDB URI not provided (pass uri= or set MONGO_URI)")

        # A short server-selection timeout means an unreachable/misconfigured
        # Atlas cluster fails fast here instead of blocking callers — e.g.
        # the dashboard's startup — for pymongo's default ~30s timeout.
        self.shards = [_Shard(u, db_name, server_selection_timeout_ms) for u in uri_list]
        limit_mb = shard_limit_mb or float(
            os.environ.get("MONGO_SHARD_LIMIT_MB", _DEFAULT_SHARD_LIMIT_MB)
        )
        self._shard_limit_bytes = int(limit_mb * 1024 * 1024)
        # job_id -> shard, filled by lookups and writes. A job never moves
        # between shards, so this is safe to keep for the process lifetime;
        # delete_job_data evicts.
        self._job_shard: dict[str, _Shard] = {}

        if len(self.shards) > 1:
            logger.info(
                f"[repo] {len(self.shards)} MongoDB shards, fill order: "
                + ", ".join(s.name for s in self.shards)
            )

    # ------------------------------------------------------------------
    # Per-corpus collection handles, created + indexed lazily on first use
    # ------------------------------------------------------------------

    def _collections(self, collection: str, shard: _Shard, *, for_write: bool = False):
        """Indexes are created only on the write path: create_index makes the
        collection if it doesn't exist, so doing it on reads would plant empty
        `{corpus}_*` collections on every shard just by browsing."""
        _validate_collection(collection)
        videos = shard.db[collection + _VIDEOS_SUFFIX]
        fused_windows = shard.db[collection + _FUSED_WINDOWS_SUFFIX]
        artifacts = shard.db[collection + _ARTIFACTS_SUFFIX]

        if for_write and collection not in shard.indexed_collections:
            fused_windows.create_index("job_id")
            fused_windows.create_index([("job_id", 1), ("window_idx", 1)], unique=True)
            videos.create_index("dedupe_key")
            shard.indexed_collections.add(collection)

        return videos, fused_windows, artifacts

    # ------------------------------------------------------------------
    # Shard routing
    # ------------------------------------------------------------------

    def _locate(self, collection: str, job_id: str) -> Optional[_Shard]:
        """The shard holding any document of this job, or None."""
        _validate_collection(collection)
        cached = self._job_shard.get(job_id)
        if cached is not None:
            return cached
        for shard in self.shards:
            videos, fused_windows, artifacts = self._collections(collection, shard)
            if (
                videos.find_one({"_id": job_id}, {"_id": 1})
                or artifacts.find_one({"_id": job_id}, {"_id": 1})
                or fused_windows.find_one({"job_id": job_id}, {"_id": 1})
            ):
                self._job_shard[job_id] = shard
                return shard
        return None

    def reserve_shard(self, collection: str, job_id: str, estimated_bytes: int = 0) -> str:
        """Pick (and remember) the shard a job will be written to; returns
        its host name. A job that already has documents somewhere stays on
        that shard — so a retried ship never splits a job across two.

        Called by `ship_job` with the real document size; the individual
        save_* methods call it with no estimate if nothing was reserved."""
        existing = self._locate(collection, job_id)
        if existing is not None:
            return existing.name

        budget = self._shard_limit_bytes * _FILL_RATIO
        for shard in self.shards:
            if shard.full:
                continue
            used = shard.used_bytes() if len(self.shards) > 1 else 0
            if used + estimated_bytes <= budget:
                self._job_shard[job_id] = shard
                return shard.name
            logger.info(
                f"[repo] Shard {shard.name} full for job {job_id}: "
                f"{used / 1e6:.1f}MB used + {estimated_bytes / 1e6:.1f}MB "
                f"> {budget / 1e6:.0f}MB budget"
            )
        raise RuntimeError(
            f"No MongoDB shard has room for job {job_id} "
            f"({estimated_bytes / 1e6:.1f}MB) — add another cluster as "
            f"MONGO_URI_{len(self.shards) + 1}"
        )

    def _write_shard(self, collection: str, job_id: str) -> _Shard:
        self.reserve_shard(collection, job_id)
        return self._job_shard[job_id]

    # ------------------------------------------------------------------
    # Write (bulk ship)
    # ------------------------------------------------------------------

    def ship_job(
        self,
        collection: str,
        job: AnalysisJob,
        windows: list[FusedWindow],
        *,
        drive_url: str | None,
        label: str | None,
        dedupe_key: str,
        duration_s: float | None,
        artifacts: dict,
    ) -> str:
        """Write one job's video doc, fused windows and artifacts to a single
        shard chosen by their combined size. Returns the shard's host name.
        `artifacts` takes save_artifacts' keyword arguments.

        The size estimate picks the shard, but Atlas's own accounting is the
        authority: if a shard refuses a write for quota, the partial job is
        removed from it, the shard is marked full, and the whole job is
        written to the next shard instead. (Deletes are still allowed on an
        over-quota Atlas cluster.)"""
        estimate = sum(len(bson.encode(w.model_dump(mode="json"))) for w in windows)
        estimate += len(bson.encode({k: v for k, v in artifacts.items() if v is not None}))

        while True:
            shard_name = self.reserve_shard(collection, job.job_id, estimate)
            try:
                self.save_job(
                    collection, job, drive_url=drive_url, label=label,
                    dedupe_key=dedupe_key, duration_s=duration_s,
                )
                self.save_fused_windows(collection, job.job_id, windows)
                self.save_artifacts(collection, job.job_id, **artifacts)
                return shard_name
            except OperationFailure as exc:
                if not _is_quota_error(exc):
                    raise
                shard = self._job_shard.pop(job.job_id)
                logger.warning(
                    f"[repo] Shard {shard.name} refused job {job.job_id} for quota "
                    f"({exc}) — removing the partial write and trying the next shard"
                )
                videos, fused_windows, arts = self._collections(collection, shard)
                videos.delete_one({"_id": job.job_id})
                fused_windows.delete_many({"job_id": job.job_id})
                arts.delete_one({"_id": job.job_id})
                shard.full = True

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
        videos, _, _ = self._collections(collection, self._write_shard(collection, job.job_id), for_write=True)
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
        _, fused_windows, _ = self._collections(collection, self._write_shard(collection, job_id), for_write=True)
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
        _, _, artifacts = self._collections(collection, self._write_shard(collection, job_id), for_write=True)
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
    # Dense prosody (word-level pitch; see scripts/backfill_dense_prosody.py)
    # ------------------------------------------------------------------

    def put_dense_prosody(
        self, collection: str, job_id: str, *,
        hop_s: float, f0: "np.ndarray", intensity_db: "np.ndarray",
        params: dict,
    ) -> str:
        """Store one job's f0 and intensity contours at their native hop.

        Everything else here is per-5s-window; this is the raw contour, kept
        because word-level questions ("what was the pitch while she said
        *this*") cannot be answered from a window mean — a 5s window spans
        ~15 words. See MultiArth_Search_Bar_QUANTITATIVE.docx section 7.1.

        Stored as float32 **binary**, not JSON numbers: a 13-minute talk at a
        10ms hop is ~78k samples per contour, which as BSON doubles would be
        several MB per video and push a nearly-full free-tier cluster over
        its quota. Binary float32 is ~310KB per contour, and NaN marks
        unvoiced frames (Praat reports 0 there, which would otherwise be
        read as a real 0Hz measurement).

        Placement is by capacity, not by where the job's other documents
        live: the older shards are close to full, and a contour is
        self-contained — `get_dense_prosody` searches every shard, so it does
        not need to sit beside its job.
        """
        _validate_collection(collection)
        doc = {
            "_id": job_id,
            "hop_s": float(hop_s),
            "n_samples": int(len(f0)),
            "f0": bson.Binary(np.asarray(f0, dtype=np.float32).tobytes()),
            "intensity_db": bson.Binary(
                np.asarray(intensity_db, dtype=np.float32).tobytes()
            ),
            "params": params,
            "created_at": time.time(),
        }
        estimate = len(doc["f0"]) + len(doc["intensity_db"]) + 1024
        budget = self._shard_limit_bytes * _FILL_RATIO
        for shard in self.shards:
            if shard.full:
                continue
            used = shard.used_bytes() if len(self.shards) > 1 else 0
            if used + estimate > budget:
                continue
            shard.db[collection + _DENSE_PROSODY_SUFFIX].replace_one(
                {"_id": job_id}, doc, upsert=True
            )
            return shard.name
        raise RuntimeError(
            f"No shard has room for {job_id}'s dense prosody "
            f"({estimate / 1e6:.1f}MB) — add another cluster as "
            f"MONGO_URI_{len(self.shards) + 1}"
        )

    def get_dense_prosody(self, collection: str, job_id: str) -> Optional[dict]:
        """Returns {hop_s, f0, intensity_db, params} with the contours as
        float32 arrays (NaN = unvoiced), or None if this job has none."""
        _validate_collection(collection)
        for shard in self.shards:
            doc = shard.db[collection + _DENSE_PROSODY_SUFFIX].find_one({"_id": job_id})
            if doc:
                return {
                    "hop_s": doc["hop_s"],
                    "f0": np.frombuffer(doc["f0"], dtype=np.float32),
                    "intensity_db": np.frombuffer(doc["intensity_db"], dtype=np.float32),
                    "params": doc.get("params", {}),
                }
        return None

    def has_dense_prosody(self, collection: str, job_id: str) -> bool:
        _validate_collection(collection)
        return any(
            shard.db[collection + _DENSE_PROSODY_SUFFIX].find_one(
                {"_id": job_id}, {"_id": 1}
            )
            for shard in self.shards
        )

    # ------------------------------------------------------------------
    # Dedup lookup (bulk skip-if-already-shipped) — every shard, fail hard
    # ------------------------------------------------------------------

    def find_by_dedupe_key(self, collection: str, dedupe_key: str) -> Optional[str]:
        for shard in self.shards:
            videos, _, _ = self._collections(collection, shard)
            doc = videos.find_one({"dedupe_key": dedupe_key}, {"_id": 1})
            if doc:
                self._job_shard[doc["_id"]] = shard
                return doc["_id"]
        return None

    def delete_job_data(self, collection: str, job_id: str) -> None:
        """Removes every stored document for one job — video doc, fused
        windows, artifacts — from every shard. Used when a `--force`/bulk-force
        reprocess of an already-shipped video should *overwrite* its previous
        run rather than accumulate alongside it under a different job_id
        (every run mints a fresh one) — see core/bulk_orchestrator.py's
        _ship, which looks up the previous run via find_by_dedupe_key and
        calls this before saving the new one."""
        for shard in self.shards:
            videos, fused_windows, artifacts = self._collections(collection, shard)
            videos.delete_one({"_id": job_id})
            fused_windows.delete_many({"job_id": job_id})
            artifacts.delete_one({"_id": job_id})
            shard.db[collection + _DENSE_PROSODY_SUFFIX].delete_one({"_id": job_id})
        self._job_shard.pop(job_id, None)

    # ------------------------------------------------------------------
    # Read (dashboard Browse Corpus) — merged across shards
    # ------------------------------------------------------------------

    # Listings skip (and log) an unreachable shard; nothing else does — see
    # the module docstring's "Failure handling".

    def list_collections(self) -> list[str]:
        """Corpus names with at least one shipped video on any shard,
        discovered from existing `{collection}_videos` collections."""
        names: set[str] = set()
        for shard in self.shards:
            try:
                shard_names = shard.db.list_collection_names()
            except Exception as exc:
                logger.warning(f"[repo] Shard {shard.name} unreachable, its corpora are hidden: {exc}")
                continue
            names.update(n[: -len(_VIDEOS_SUFFIX)] for n in shard_names if n.endswith(_VIDEOS_SUFFIX))
        return sorted(names)

    def list_videos(self, collection: str) -> list[dict]:
        """Every shard's videos for this corpus, in shard fill order — so
        broadly in shipping order, since shards fill one after another."""
        _validate_collection(collection)
        out: list[dict] = []
        for shard in self.shards:
            try:
                videos, _, _ = self._collections(collection, shard)
                docs = list(videos.find({}))
            except Exception as exc:
                logger.warning(f"[repo] Shard {shard.name} unreachable, its videos are hidden: {exc}")
                continue
            for d in docs:
                self._job_shard[d["_id"]] = shard
            out.extend(docs)
        return out

    def get_job(self, collection: str, job_id: str) -> Optional[AnalysisJob]:
        doc = self.get_video_doc(collection, job_id)
        if not doc:
            return None
        return AnalysisJob(**{k: v for k, v in doc.items() if k in AnalysisJob.model_fields})

    def get_video_doc(self, collection: str, job_id: str) -> Optional[dict]:
        """Raw video doc, unfiltered — unlike get_job, keeps fields outside
        AnalysisJob's own schema (drive_url, label, video_filename, ...)."""
        shard = self._locate(collection, job_id)
        if shard is None:
            return None
        videos, _, _ = self._collections(collection, shard)
        return videos.find_one({"_id": job_id})

    def get_all_fused(self, collection: str, job_id: str) -> list[FusedWindow]:
        shard = self._locate(collection, job_id)
        if shard is None:
            return []
        _, fused_windows, _ = self._collections(collection, shard)
        docs = fused_windows.find({"job_id": job_id}).sort("window_idx", 1)
        return [
            FusedWindow(**{k: v for k, v in d.items() if k not in ("_id", "job_id", "window_idx")})
            for d in docs
        ]

    # ------------------------------------------------------------------
    # In-place patches (see scripts/backfill_verbal.py)
    # ------------------------------------------------------------------
    # Targeted $set rather than a re-ship. Both of these exist so that a
    # quantity which can be recomputed from what is already stored — no audio,
    # no video, no re-download — can be corrected without rewriting documents
    # whose other fields are large and unchanged. save_artifacts in particular
    # replaces the whole document, which would drop the ~2MB spectrogram just
    # to change a word list.

    def update_wordlist(self, collection: str, job_id: str, wordlist: dict) -> bool:
        """Replace one job's word list, leaving its other artifacts untouched."""
        shard = self._locate(collection, job_id)
        if shard is None:
            raise KeyError(f"{job_id} not found in {collection} on any shard")
        _, _, artifacts = self._collections(collection, shard)
        result = artifacts.update_one({"_id": job_id}, {"$set": {"wordlist": wordlist}})
        return result.matched_count > 0

    def update_segmented_tokens(self, collection: str, job_id: str,
                                tokens: list) -> bool:
        """Replace one job's segmented tokens, leaving its other artifacts be.

        Rewritten whenever the tokenisation changes — they now carry each
        token's part of speech and the index of the recogniser token it came
        from, which is what lets a word list count words as they were spoken.
        """
        shard = self._locate(collection, job_id)
        if shard is None:
            raise KeyError(f"{job_id} not found in {collection} on any shard")
        _, _, artifacts = self._collections(collection, shard)
        result = artifacts.update_one({"_id": job_id},
                                      {"$set": {"segmented_tokens": tokens}})
        return result.matched_count > 0

    def update_window_fields(
        self, collection: str, job_id: str, updates: dict[int, dict],
    ) -> int:
        """Patch fields on individual fused windows. Returns windows modified.

        *updates* maps window_idx to a dict of dotted field paths and values,
        e.g. {0: {"verbal.word_count": 14}}. Sent as one bulk write, because a
        13-minute talk is ~160 windows and a round trip each would dominate.
        """
        from pymongo import UpdateOne

        shard = self._locate(collection, job_id)
        if shard is None:
            raise KeyError(f"{job_id} not found in {collection} on any shard")
        if not updates:
            return 0
        _, fused_windows, _ = self._collections(collection, shard)
        ops = [
            UpdateOne({"job_id": job_id, "window_idx": idx}, {"$set": fields})
            for idx, fields in sorted(updates.items())
        ]
        return fused_windows.bulk_write(ops, ordered=False).modified_count

    def get_artifacts(self, collection: str, job_id: str) -> Optional[dict]:
        shard = self._locate(collection, job_id)
        if shard is None:
            return None
        _, _, artifacts = self._collections(collection, shard)
        return artifacts.find_one({"_id": job_id})

    def close(self) -> None:
        for shard in self.shards:
            shard.client.close()
