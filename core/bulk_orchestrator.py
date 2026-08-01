"""
core/bulk_orchestrator.py
--------------------------
Sequential multi-video runner for `analyze bulk` and the dashboard's Bulk
Upload tab.

Each manifest entry is tagged with a "collection" — the named corpus (e.g.
"TedX", "Yixi") its results should be shipped into; see
core/results_repository.py for how that maps to Mongo collections. A manifest
entry's local `path` may either already exist (manually staged in advance,
the `analyze bulk` CLI workflow) or not — if it doesn't and a `drive_url` is
given, it's downloaded there first (core/drive_download.py), the Bulk Upload
tab's workflow, where entries only carry a Drive link.

Videos are processed one at a time, reusing Orchestrator.analyze() unchanged
for the per-video pipeline. Once a video finishes, its full result is
shipped to MongoDB (via ResultsRepository) with a few inline retries. That
job's Redis scratch data (FeatureStore) is deleted on a successful ship,
rather than waiting out its 24h TTL, so a long bulk run doesn't pile up
Redis memory for jobs that are already durably stored.

Local file cleanup differs by how the video got there: a video this run
downloaded itself is always deleted afterwards, success or failure — the
durable copy is on Drive regardless, so there's no reason to keep a local
copy around after a failed attempt (and no expectation that whoever
triggered the run will manually retry the same path later). A
manually-staged video (the CLI workflow, no download involved) is only
deleted on success, exactly as before, since a maintainer re-running the
same manifest after fixing a failure expects their staged files to still
be there — its Redis job data is likewise left in place on failure so the
job can be inspected/retried before its 24h TTL expires.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import yaml
from loguru import logger

from core.drive_download import download_drive_file, extract_file_id
from core.feature_store import FeatureStore
from core.models import JobStatus
from core.orchestrator import Orchestrator
from core.results_repository import ResultsRepository

_SHIP_ATTEMPTS = 3
_SHIP_BACKOFF_S = 2.0
_PROGRESS_POLL_S = 1.0
_YAML_SUFFIXES = (".yml", ".yaml")

ProgressCallback = Callable[[dict], None]


def load_manifest(path: str) -> list[dict]:
    with open(path) as fh:
        if Path(path).suffix.lower() in _YAML_SUFFIXES:
            entries = yaml.safe_load(fh)
        else:
            entries = json.load(fh)

    if not isinstance(entries, list):
        raise ValueError(
            f"Manifest {path} must be a YAML or JSON list of "
            "{path, collection, drive_url, label} objects"
        )
    for entry in entries:
        if "collection" not in entry:
            raise ValueError(f"Manifest entry missing required 'collection': {entry}")
        # `path` is only required for a manually-staged local video (the CLI
        # workflow); an entry with no local file yet must give a drive_url so
        # BulkOrchestrator knows what to download.
        if "path" not in entry and "drive_url" not in entry:
            raise ValueError(
                f"Manifest entry needs a 'path' (pre-staged local file) or a "
                f"'drive_url' (downloaded automatically): {entry}"
            )
    return entries


def _dedupe_key(entry: dict) -> str:
    basis = entry.get("drive_url") or str(Path(entry["path"]).resolve())
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


def _resolve_path(entry: dict, work_dir: str) -> str:
    """Local path for a manifest entry — its own `path` if given, otherwise
    a stable destination derived from the Drive file ID (so re-running the
    same manifest downloads to the same place rather than a fresh temp name
    each time)."""
    if "path" in entry:
        return entry["path"]
    file_id = extract_file_id(entry["drive_url"])
    return str(Path(work_dir) / f"{file_id}.mp4")


def _audio_cache_path(video_path: str, work_dir: str) -> Path:
    return Path(work_dir) / (Path(video_path).stem + "_audio.wav")


class BulkOrchestrator:
    def __init__(
        self,
        feature_store: FeatureStore,
        repo: ResultsRepository,
        orchestrator_kwargs: dict,
    ):
        self.store = feature_store
        self.repo = repo
        self.orchestrator_kwargs = orchestrator_kwargs

    def run(
        self,
        entries: list[dict],
        force: bool = False,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> dict:
        summary = {"succeeded": [], "skipped": [], "failed": []}
        total = len(entries)

        # One Orchestrator (and its workers — MediaPipe Holistic, the Whisper
        # model, the Haar cascade, spaCy's per-language cache) for the whole
        # manifest, reused across every video, instead of reloading these
        # models from scratch for each one.
        orchestrator = Orchestrator(store=self.store, **self.orchestrator_kwargs)
        try:
            for index, entry in enumerate(entries, start=1):
                self._run_one(entry, index, total, force, progress_cb, orchestrator, summary)
        finally:
            orchestrator.close()

        return summary

    def _run_one(
        self,
        entry: dict,
        index: int,
        total: int,
        force: bool,
        progress_cb: Optional[ProgressCallback],
        orchestrator: Orchestrator,
        summary: dict,
    ) -> None:
        work_dir = self._work_dir()
        path = _resolve_path(entry, work_dir)
        collection = entry["collection"]
        drive_url = entry.get("drive_url")
        label = entry.get("label")
        dedupe_key = _dedupe_key(entry)
        downloaded_by_us = False

        def _emit(evt: dict) -> None:
            if progress_cb:
                progress_cb({
                    "index": index, "total": total, "path": path,
                    "label": label, "collection": collection, **evt,
                })

        if not force:
            existing_job_id = self.repo.find_by_dedupe_key(collection, dedupe_key)
            if existing_job_id:
                logger.info(f"[bulk] Skipping {path} — already shipped as job {existing_job_id}")
                summary["skipped"].append(path)
                _emit({"type": "skip", "job_id": existing_job_id})
                return

        if not os.path.exists(path):
            if not drive_url:
                logger.error(f"[bulk] File not found, skipping: {path}")
                summary["failed"].append({"path": path, "error": "file not found"})
                _emit({"type": "done", "result": "failed", "error": "file not found"})
                return
            downloaded_by_us = True
            _emit({"type": "download"})
            try:
                download_drive_file(drive_url, path)
            except Exception as exc:
                logger.error(f"[bulk] Download failed for {drive_url}: {exc}")
                summary["failed"].append({"path": path, "error": f"download failed: {exc}"})
                _emit({"type": "done", "result": "failed", "error": f"download failed: {exc}"})
                self._cleanup_local_files(path)  # in case a partial file was left behind
                return

        _emit({"type": "start"})

        succeeded = False
        try:
            job_id = self._analyze_with_progress(orchestrator, path, _emit)

            status = self.store.get_status(job_id)
            if status != JobStatus.DONE:
                job = self.store.get_job(job_id)
                error = job.error if job else "unknown error"
                logger.error(f"[bulk] Processing failed for {path} (job {job_id}): {error}")
                summary["failed"].append({"path": path, "error": error})
                _emit({"type": "done", "result": "failed", "job_id": job_id, "error": error})
                return

            try:
                self._ship(collection, job_id, path, drive_url, label, dedupe_key)
            except Exception as exc:
                logger.error(f"[bulk] Shipping failed for {path} (job {job_id}) after retries: {exc}")
                summary["failed"].append({"path": path, "error": f"shipping failed: {exc}"})
                _emit({"type": "done", "result": "failed", "job_id": job_id, "error": f"shipping failed: {exc}"})
                return

            self.store.delete_job(job_id)
            logger.info(f"[bulk] Shipped {path} as job {job_id}")
            summary["succeeded"].append(job_id)
            _emit({"type": "done", "result": "succeeded", "job_id": job_id})
            succeeded = True
        finally:
            # A video we downloaded ourselves is always cleaned up — the
            # durable copy is on Drive regardless of outcome. A manually
            # staged video (no download involved) is only cleaned up on
            # success, so a failed attempt stays in place for a future
            # re-run of the same manifest (see module docstring).
            if downloaded_by_us or succeeded:
                self._cleanup_local_files(path)

    # ------------------------------------------------------------------

    def _analyze_with_progress(self, orchestrator: Orchestrator, path: str, emit: Callable[[dict], None]) -> str:
        """
        Runs orchestrator.analyze() in a background thread (pre-assigning a
        job_id, as Orchestrator.analyze() supports) so this thread can poll
        Redis for per-modality window counts and emit "tick" progress events
        while the video is being processed.
        """
        job_id = str(uuid.uuid4())[:8]
        result: dict = {}

        def _target():
            result["job_id"] = orchestrator.analyze(path, job_id=job_id)

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()

        while thread.is_alive():
            job = self.store.get_job(job_id)
            total_windows = job.total_windows if job else None
            counts = None
            if total_windows:
                counts = {
                    "gesture": self.store.count_windows(job_id, "gesture"),
                    "prosody": self.store.count_windows(job_id, "prosody"),
                    "verbal":  self.store.count_windows(job_id, "verbal"),
                    "camera":  self.store.count_windows(job_id, "camera"),
                }
            emit({"type": "tick", "job_id": job_id, "total_windows": total_windows, "counts": counts})
            thread.join(timeout=_PROGRESS_POLL_S)

        return result["job_id"]

    # ------------------------------------------------------------------

    def _ship(
        self, collection: str, job_id: str, path: str,
        drive_url: str | None, label: str | None, dedupe_key: str,
    ) -> None:
        job = self.store.get_job(job_id)
        windows = self.store.get_all_fused(job_id)
        duration_s = max((w.window.end_s for w in windows), default=None)

        last_exc = None
        for attempt in range(1, _SHIP_ATTEMPTS + 1):
            try:
                self.repo.save_job(
                    collection, job, drive_url=drive_url, label=label,
                    dedupe_key=dedupe_key, duration_s=duration_s,
                )
                self.repo.save_fused_windows(collection, job_id, windows)
                self.repo.save_artifacts(
                    collection, job_id,
                    wordlist=self.store.get_wordlist(job_id),
                    ngrams=self.store.get_ngrams(job_id),
                    collocations=self.store.get_collocations(job_id),
                    spectrogram=self.store.get_spectrogram(job_id),
                    waveform=self.store.get_waveform(job_id),
                    segmented_tokens=self.store.get_segmented_tokens(job_id),
                )
                return
            except Exception as exc:
                last_exc = exc
                logger.warning(f"[bulk] Ship attempt {attempt}/{_SHIP_ATTEMPTS} failed for job {job_id}: {exc}")
                if attempt < _SHIP_ATTEMPTS:
                    time.sleep(_SHIP_BACKOFF_S * attempt)

        raise last_exc

    def _work_dir(self) -> str:
        return self.orchestrator_kwargs.get("work_dir") or os.environ.get("WORK_DIR", "/tmp/mannerism")

    def _cleanup_local_files(self, video_path: str) -> None:
        work_dir = self._work_dir()
        for p in (Path(video_path), _audio_cache_path(video_path, work_dir)):
            try:
                if p.exists():
                    p.unlink()
            except OSError as exc:
                logger.warning(f"[bulk] Could not delete {p}: {exc}")
