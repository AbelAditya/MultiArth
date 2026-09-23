"""
scripts/reprocess_with_gallery.py
----------------------------------
Re-run a full analysis on a video, reusing a gallery you already have
instead of building a new one in the dashboard.

    # reuse another job's gallery (while it is still in Redis)
    uv run python scripts/reprocess_with_gallery.py VIDEO --gallery-from 64c20643

    # or one saved by scripts/inspect_selection.py, which never expires
    uv run python scripts/reprocess_with_gallery.py VIDEO \
        --gallery-npz /tmp/sel/gallery.npz --set tie_break=vertical

## Why this exists

The gesture worker reads its gallery from Redis under the job's own id, and
every run mints a new id — so ordinarily a reprocess means rebuilding the
gallery by hand, and a rebuilt gallery is a *different operating point*:
it scores the same footage differently, so thresholds tuned against one do
not transfer to the other. Reusing the exact gallery is what makes a
before/after comparison mean anything.

It also pairs with per-video tuning: judge a parameter with
scripts/inspect_selection.py against a gallery, then reprocess with that
same gallery and `--set` the parameter you chose.

## Why a fresh job id rather than re-running the old one

Re-running under the gallery's own id would inherit that job's leftovers —
in particular the per-scene tracklet decisions cached at
`job:{id}:scene:{n}:track`, which are exactly what you are trying to
recompute. The cache would be served instead, and the run would reproduce
the old decisions while appearing to re-derive them. So this copies the
gallery to a new id and leaves the old job untouched.

## Seeing the results

Every run ends with a quality summary computed from the fused windows —
pose coverage, and any window whose pose is anatomically impossible (see
`_plausibility_scan`). That is usually enough to answer "did the reprocess
fix the scene that was wrong".

Beyond that, without `--ship` the results live in Redis under the new job
id:

    uv run analyze status <job_id>
    uv run analyze export <job_id> --out results.json

`--ship` writes them to MongoDB instead, where Browse Corpus and the
notebooks read from. Shipping a video that is already in the corpus needs
`--replace`, which deletes the previous run's documents first — a reprocess
mints a new job id, so without that the corpus would hold the same video
twice under different ids.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from core.bulk_orchestrator import _dedupe_key  # noqa: E402
from core.feature_store import FeatureStore  # noqa: E402
from core.models import GalleryEntry  # noqa: E402
from core.orchestrator import Orchestrator  # noqa: E402
from core.results_repository import ResultsRepository  # noqa: E402
from workers._gesture_params import GestureParams  # noqa: E402

# Ratio of thigh length to torso length, both measured vertically so the
# frame's aspect ratio cancels. Measured across a whole talk it sits in a
# tight band (0.77 median, 0.72-0.84 for the 10th-90th percentile); the
# malformed pose that prompted all this scored 0.19 — a skeleton stitched
# across several seated people. Outside this range is not a human body.
_PLAUSIBLE_THIGH_TORSO = (0.45, 1.2)


def _parse_overrides(pairs: list[str]) -> dict:
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        for cast in (int, float):
            try:
                out[key.strip()] = cast(raw)
                break
            except ValueError:
                continue
        else:
            out[key.strip()] = raw.strip()
    return out


def _entries_from_npz(path: str) -> list[GalleryEntry]:
    """Rebuild entries from saved embeddings. The metadata fields are
    placeholders: the worker reads only `.embedding` (see
    GestureWorker.process_job), and the rest exists for the dashboard's
    confirmation UI, which is not involved in a reprocess."""
    embeddings = np.load(path)["gallery"]
    return [
        GalleryEntry(embedding=[float(v) for v in row], timestamp_s=-1.0,
                     scene_idx=-1, thumbnail_jpeg_b64="")
        for row in embeddings
    ]


def _plausibility_scan(store: FeatureStore, job_id: str) -> None:
    """Post-run quality summary, printed instead of requiring the results to
    be shipped and opened somewhere. Flags windows whose median pose is
    anatomically impossible — the signature of a pose fitted across several
    people, which no confidence score in the pipeline reports."""
    windows = store.get_all_fused(job_id)
    if not windows:
        print("no fused windows — nothing to summarise")
        return

    posed, ratios, suspect = 0, [], []
    for w in windows:
        g = w.gesture
        if not g or not g.pose_keyframes:
            continue
        posed += 1
        per_kf = []
        for k in g.pose_keyframes:
            y = k.pose_y
            torso, thigh = abs(y[11] - y[23]), abs(y[23] - y[25])
            if torso > 1e-6:
                per_kf.append(thigh / torso)
        if not per_kf:
            continue
        median = float(np.median(per_kf))
        ratios.append(median)
        lo, hi = _PLAUSIBLE_THIGH_TORSO
        if not lo <= median <= hi:
            suspect.append((w.window.start_s, median, g.pose_present_ratio))

    coverage = float(np.mean([w.gesture.pose_present_ratio
                              for w in windows if w.gesture]))
    print(f"\nwindows: {len(windows)}  with a pose: {posed}  "
          f"mean pose coverage: {coverage:.2f}")
    if ratios:
        print(f"thigh/torso ratio: median {np.median(ratios):.2f}  "
              f"10th-90th [{np.percentile(ratios, 10):.2f}, "
              f"{np.percentile(ratios, 90):.2f}]")
    if suspect:
        print(f"IMPLAUSIBLE POSES in {len(suspect)} window(s) — a pose fitted "
              f"across more than one person looks like this:")
        for t0, ratio, ppr in suspect:
            print(f"   t={int(t0) // 60}:{int(t0) % 60:02d}  "
                  f"thigh/torso={ratio:.2f}  pose_present_ratio={ppr:.2f}")
    else:
        print("no implausible poses found")


def _ship(job_id: str, args, store: FeatureStore) -> None:
    """Write the run to MongoDB, replacing a previous copy of the same video
    when asked. Mirrors core/bulk_orchestrator.py's own _ship so a
    reprocessed video is stored exactly like a bulk-processed one."""
    repo = ResultsRepository()
    entry = ({"drive_url": args.drive_url} if args.drive_url
             else {"path": args.video})
    dedupe = _dedupe_key(entry)
    existing = repo.find_by_dedupe_key(args.collection, dedupe)
    if existing and existing != job_id:
        if not args.replace:
            raise SystemExit(
                f"{args.collection} already holds this video as job {existing}. "
                f"Every run mints a new job id, so shipping now would store it "
                f"twice. Re-run with --replace to delete the previous run's "
                f"documents first."
            )
        print(f"replacing previous run (job {existing}) in {args.collection}")
        repo.delete_job_data(args.collection, existing)

    job = store.get_job(job_id)
    windows = store.get_all_fused(job_id)
    shard = repo.ship_job(
        args.collection, job, windows,
        drive_url=args.drive_url, label=args.label, dedupe_key=dedupe,
        duration_s=max((w.window.end_s for w in windows), default=None),
        artifacts=dict(
            wordlist=store.get_wordlist(job_id), ngrams=store.get_ngrams(job_id),
            collocations=store.get_collocations(job_id),
            spectrogram=store.get_spectrogram(job_id),
            waveform=store.get_waveform(job_id),
            segmented_tokens=store.get_segmented_tokens(job_id),
        ),
    )
    print(f"shipped {len(windows)} windows to {args.collection} on shard {shard}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("##")[0].strip())
    ap.add_argument("video")
    ap.add_argument("--gallery-from", metavar="JOB_ID",
                    help="copy this job's gallery out of Redis")
    ap.add_argument("--gallery-npz",
                    help="gallery saved by scripts/inspect_selection.py")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a GestureParams field; repeatable")
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--job-id", default=None,
                    help="pre-assign the new job id (default: random)")
    ap.add_argument("--ship", action="store_true",
                    help="write the results to MongoDB when the run finishes")
    ap.add_argument("--collection", help="corpus to ship into, e.g. TedX")
    ap.add_argument("--label", help="human label stored with the video")
    ap.add_argument("--drive-url", help="Drive link; also the dedupe key's "
                                        "basis, so pass the same one the "
                                        "original run used")
    ap.add_argument("--replace", action="store_true",
                    help="with --ship: delete a previous copy of this video "
                         "from the corpus first")
    args = ap.parse_args()

    if args.ship and not args.collection:
        raise SystemExit("--ship needs --collection (e.g. --collection TedX)")
    if args.ship:
        # Fail before the hours of processing, not after.
        ResultsRepository()

    if bool(args.gallery_from) == bool(args.gallery_npz):
        raise SystemExit("pass exactly one of --gallery-from / --gallery-npz")

    params = GestureParams.from_dict(_parse_overrides(args.set))
    store = FeatureStore()

    if args.gallery_from:
        entries = store.get_gallery(args.gallery_from)
        if not entries:
            raise SystemExit(
                f"Job {args.gallery_from} has no gallery in Redis — entries "
                "expire 24h after they are written. Use --gallery-npz if you "
                "saved one with scripts/inspect_selection.py."
            )
        source = f"job {args.gallery_from}"
    else:
        entries = _entries_from_npz(args.gallery_npz)
        source = args.gallery_npz

    job_id = args.job_id or str(uuid.uuid4())[:8]
    for entry in entries:
        store.add_gallery_entry(job_id, entry)
    print(f"copied {len(entries)} gallery entries from {source} -> job {job_id}")
    print("params " + json.dumps(params.to_dict(), sort_keys=True))
    print(f"analysing {args.video} ...")

    started = time.time()
    orch = Orchestrator(
        store=store, work_dir=args.work_dir, window_size_s=args.window,
        gesture_params=params,
    )
    orch.analyze(args.video, job_id=job_id)
    print(f"\ndone in {(time.time() - started) / 60:.1f} min — job {job_id}")

    _plausibility_scan(store, job_id)
    if args.ship:
        _ship(job_id, args, store)
    else:
        print(f"\nnot shipped. Inspect with:\n"
              f"   uv run analyze export {job_id} --out results.json")
    print(f"gallery kept at job:{job_id}:gallery:* (24h TTL), so a follow-up "
          f"reprocess can use --gallery-from {job_id}")


if __name__ == "__main__":
    main()
