"""
scripts/run_flags.py
---------------------
Flag a corpus's videos against a manifest and write one Excel workbook each.

    # every video in a corpus
    uv run python scripts/run_flags.py --collection TedX

    # one video, a different rulebook, somewhere else
    uv run python scripts/run_flags.py --collection TedX --job-id 80e15587 \
        --manifest flags/my_rules.yaml --out ~/flag_reports

Nothing is written to MongoDB and no intermediate files are kept: the workbook
is the deliverable, and re-running is cheap (seconds per video once the corpus
is cached locally).

Each workbook has four sheets — Windows, Flag summary, Overlaps, Provenance —
see core/flag_runner.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from core.flag_runner import pool_runs, pooled_summary_table, pooled_to_excel, run_corpus  # noqa: E402
from core.flagging import Manifest, ManifestError  # noqa: E402

_DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "flags" / "multiarth_cda.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip())
    ap.add_argument("--collection", required=True, help="corpus, e.g. TedX or Yixi")
    ap.add_argument("--manifest", default=str(_DEFAULT_MANIFEST),
                    help=f"flagging manifest (default: {_DEFAULT_MANIFEST.name})")
    ap.add_argument("--job-id", action="append", default=[],
                    help="only these videos; repeatable (default: the whole corpus)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N videos")
    ap.add_argument("--out", default=None,
                    help="output directory (default: flag_reports/<corpus>/)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch the corpus instead of using the local cache")
    ap.add_argument("--no-pooled", action="store_true",
                    help="skip the corpus-level summary and its workbook")
    args = ap.parse_args()

    # Fail on a bad manifest before spending minutes loading a corpus.
    try:
        manifest = Manifest.from_yaml(args.manifest)
    except ManifestError as exc:
        raise SystemExit(f"manifest error: {exc}")
    print(f"manifest {manifest.id} v{manifest.version} — {len(manifest.flags)} flags, "
          f"rule: {manifest.min_elements} elements, macros "
          f"{', '.join(manifest.require_each_macro)}")

    import _corpus as C

    data = C.load_corpus(args.collection, refresh=args.refresh)
    if data.windows.empty:
        raise SystemExit(f"{args.collection} has no windows")

    ids = args.job_id or list(data.videos.job_id)
    unknown = set(ids) - set(data.videos.job_id)
    if unknown:
        raise SystemExit(f"not in {args.collection}: {', '.join(sorted(unknown))}")
    if args.limit:
        ids = ids[:args.limit]

    out_dir = Path(args.out or (Path.cwd() / "flag_reports" / args.collection))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(ids)} video(s) -> {out_dir}\n")

    runs = run_corpus(
        data.windows, manifest,
        job_ids=ids,
        out_dir=out_dir,
        labels={j: data.label(j) for j in ids},
    )

    if not runs:
        raise SystemExit("nothing was flagged")

    pooled = pool_runs(runs, manifest)
    print(f"\n{len(runs)} per-video workbook(s) written.")

    if args.no_pooled:
        return

    # ── corpus-level summary ────────────────────────────────────────────
    print(f"\n{'=' * 78}\n{args.collection}: {pooled['flagged_windows']} of "
          f"{pooled['n_windows']} windows flagged "
          f"({100 * pooled['flagged_windows'] / max(pooled['n_windows'], 1):.1f}%) "
          f"across {pooled['n_videos']} videos\n{'=' * 78}")

    # Verbal coverage first: it is the ceiling on every flag needing a subject
    # or a verb, so a flag count read without it can mislead badly.
    if pooled["verbal_coverage"]:
        print("verbal coverage — the ceiling on verbal criteria:")
        for element, stats in pooled["verbal_coverage"].items():
            print(f"   {element:<10} {stats['filled']:>6} windows  ({stats['pct']}%)")

    table = pooled_summary_table(pooled, manifest)
    print("\nflags:")
    print(table.to_string(index=False, max_colwidth=32))

    if pooled["pairs"]:
        print("\nmost frequent co-occurrences:")
        for pair, n in list(pooled["pairs"].items())[:8]:
            a, b = pair.split("|")
            print(f"   {a}–{b:<5} {n:>5} windows   ({manifest.overlap_kind(a, b)})")

    path = out_dir / f"_corpus__{manifest.id}_v{manifest.version}.xlsx"
    pooled_to_excel(pooled, manifest, path, corpus=args.collection,
                    labels={j: data.label(j) for j in runs})
    print(f"\ncorpus workbook -> {path}")


if __name__ == "__main__":
    main()
