"""
core/flag_runner.py
--------------------
Runs a flagging manifest over a video or a corpus, stores the result, and
exports the per-window table in the layout the analysts already use.

The pieces it joins:

    load windows -> categorise (core/flagging)      numbers  -> labels
                 -> annotate_verbal (core/flag_verbal)  words -> lexicon hits
                 -> evaluate + summarise             labels  -> flags
                 -> to_excel                         flags   -> the workbook

## Why categorisation happens per video

Every default in the manifest is speaker-relative: "low pitch" means low *for
this speaker*, "high velocity" high for this body. So the tertile edges are
computed over one video's windows, and running a corpus means running each
video separately rather than pooling first. Pool the flags afterwards, never
the thresholds.

## One export per video: the workbook

A run is cheap to recompute (seconds per video) and is invalid the moment the
manifest changes, so nothing is persisted as a database row or a side file.
The Excel workbook is the deliverable, and it is self-contained:

  Windows       one row per 5s window — timestamp, transcript, the four
                macro-categories, the flags, and the evidence for each
  Flag summary  per-flag counts, shares, near misses and what blocked them
  Overlaps      which flag pairs co-occurred, against which the manifest
                expected to
  Provenance    manifest id, version and content hash, the rule, verbal
                coverage, and the threshold edges actually used

The last sheet is what makes the first three checkable by someone who was not
in the room: "low pitch" is a different frequency for every speaker, and
without the edges a flag cannot be reconstructed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from loguru import logger

from .flag_verbal import annotate_verbal, verbal_coverage
from .flagging import Manifest, categorise, evaluate, summarise

# Columns the exported table carries beyond the manifest's own elements.
_CONTEXT_COLUMNS = ("start_s", "end_s", "transcript")


def _timestamp(seconds: float) -> str:
    """00:01:02 — the format the analysts' spreadsheet uses."""
    seconds = int(round(seconds or 0))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def run_video(
    windows: pd.DataFrame, manifest: Manifest, *, annotate: bool = True,
) -> dict:
    """Flag one video's windows. `windows` is that video's rows only.

    Returns the run document: manifest identity, thresholds, per-window
    records, summary and verbal coverage.
    """
    if windows.empty:
        raise ValueError("no windows to flag")

    categorised, thresholds = categorise(windows, manifest)
    annotated = (annotate_verbal(categorised, manifest) if annotate
                 else categorised)
    evaluated = evaluate(annotated, manifest)

    elements = sorted(set(manifest.variables) | set(manifest.elements))
    records = []
    for _, row in evaluated.iterrows():
        profile = {}
        for element in elements:
            value = row.get(element)
            if isinstance(value, list):
                profile[element] = value
            elif pd.isna(value):
                continue
            else:
                profile[element] = value
        records.append({
            "w": int(row.get("window_idx", -1)),
            "t": [float(row.get("start_s", 0)), float(row.get("end_s", 0))],
            "transcript": str(row.get("transcript") or ""),
            "profile": profile,
            "flags": row["flag_detail"],
            "near": row["near_miss"],
        })

    return {
        "manifest": {"id": manifest.id, "version": manifest.version,
                     "sha256": manifest.sha256, "window_s": manifest.window_s},
        "thresholds": thresholds,
        "verbal_coverage": verbal_coverage(annotated, manifest) if annotate else {},
        "windows": records,
        "summary": summarise(evaluated, manifest),
    }


def run_corpus(
    windows: pd.DataFrame,
    manifest: Manifest,
    *,
    job_ids: Optional[Iterable[str]] = None,
    out_dir: Optional[str | Path] = None,
    labels: Optional[dict[str, str]] = None,
    annotate: bool = True,
    progress: bool = True,
) -> dict[str, dict]:
    """Flag every video in `windows`, one at a time, writing one workbook each
    when `out_dir` is given.

    Per video rather than pooled, because the thresholds are per speaker —
    see this module's docstring.
    """
    runs: dict[str, dict] = {}
    ids = list(job_ids) if job_ids is not None else list(windows.job_id.unique())
    for i, job_id in enumerate(ids, 1):
        subset = windows[windows.job_id == job_id]
        if subset.empty:
            continue
        run = run_video(subset, manifest, annotate=annotate)
        runs[job_id] = run
        where = ""
        if out_dir:
            path = (Path(out_dir) /
                    f"{job_id}__{manifest.id}_v{manifest.version}.xlsx")
            to_excel(run, manifest, path, label=labels.get(job_id) if labels else None)
            where = f" -> {path.name}"
        if progress:
            s = run["summary"]
            logger.info(
                f"[flags] {i}/{len(ids)} {job_id}: {s['flagged_windows']}/"
                f"{s['n_windows']} windows flagged{where}")
    return runs


# ──────────────────────────────────────────────────────────────────────────
# Pooling across videos
# ──────────────────────────────────────────────────────────────────────────

def pool_runs(runs: dict[str, dict], manifest: Manifest) -> dict:
    """Merge per-video runs into one corpus-level summary.

    Counts, co-occurring pairs and near misses add up across videos; the
    *thresholds* do not, and are deliberately absent from the result. Each
    video's categories were cut against that speaker's own distribution, so
    there is no corpus-wide "low pitch" to report — only the flags that those
    per-speaker cuts produced.

    `by_video` keeps the per-video counts so a corpus figure can always be
    resolved back to which talks it came from, and a flag carried by one
    speaker is not mistaken for a corpus-wide pattern.
    """
    pooled = {
        "n_videos": len(runs),
        "n_windows": 0,
        "flagged_windows": 0,
        "by_flag": {f.id: {"name": f.name, "n": 0, "videos": 0} for f in manifest.flags},
        "pairs": {},
        "near_miss": {},
        "verbal_coverage": {},
        "by_video": {},
    }
    for job_id, run in runs.items():
        summary = run["summary"]
        pooled["n_windows"] += summary["n_windows"]
        pooled["flagged_windows"] += summary["flagged_windows"]
        pooled["by_video"][job_id] = {
            "n_windows": summary["n_windows"],
            "flagged": summary["flagged_windows"],
            "by_flag": {fid: rec["n"] for fid, rec in summary["by_flag"].items()},
        }
        for fid, rec in summary["by_flag"].items():
            entry = pooled["by_flag"].setdefault(fid, {"name": rec["name"], "n": 0, "videos": 0})
            entry["n"] += rec["n"]
            entry["videos"] += 1 if rec["n"] else 0
        for pair, n in summary["pairs"].items():
            pooled["pairs"][pair] = pooled["pairs"].get(pair, 0) + n
        for fid, rec in summary["near_miss"].items():
            entry = pooled["near_miss"].setdefault(fid, {"n": 0, "missing": {}})
            entry["n"] += rec["n"]
            for element, n in rec["missing"].items():
                entry["missing"][element] = entry["missing"].get(element, 0) + n
        for element, stats in (run.get("verbal_coverage") or {}).items():
            if isinstance(stats, dict) and "filled" in stats:
                entry = pooled["verbal_coverage"].setdefault(element, {"filled": 0, "total": 0})
                entry["filled"] += stats["filled"]
                entry["total"] += summary["n_windows"]

    total = max(pooled["n_windows"], 1)
    for rec in pooled["by_flag"].values():
        rec["pct"] = round(100 * rec["n"] / total, 2)
    for entry in pooled["verbal_coverage"].values():
        entry["pct"] = round(100 * entry["filled"] / max(entry["total"], 1), 1)
    pooled["pairs"] = dict(sorted(pooled["pairs"].items(), key=lambda kv: -kv[1]))
    return pooled


def pooled_summary_table(pooled: dict, manifest: Manifest) -> pd.DataFrame:
    """Corpus-level flag counts, with the spread across videos beside them.

    `Videos` matters as much as `Windows`: a flag firing 300 times in one talk
    and never elsewhere is that speaker's habit, not a property of the corpus.
    """
    rows = []
    n_videos = max(pooled["n_videos"], 1)
    for fid, rec in pooled["by_flag"].items():
        near = pooled["near_miss"].get(fid, {})
        top = sorted(near.get("missing", {}).items(), key=lambda kv: -kv[1])[:1]
        rows.append({
            "Flag": fid, "Name": rec["name"],
            "Windows": rec["n"], "% of windows": rec.get("pct", 0.0),
            "Videos": rec["videos"], "% of videos": round(100 * rec["videos"] / n_videos, 1),
            "Near misses": near.get("n", 0),
            "Top blocking element": top[0][0] if top else "",
        })
    return pd.DataFrame(rows).sort_values("Windows", ascending=False)


def pooled_video_table(pooled: dict, manifest: Manifest,
                       labels: Optional[dict[str, str]] = None) -> pd.DataFrame:
    """One row per video: how much each flag claimed there."""
    ids = [f.id for f in manifest.flags]
    rows = []
    for job_id, rec in pooled["by_video"].items():
        row = {"Video": (labels or {}).get(job_id, job_id), "job_id": job_id,
               "Windows": rec["n_windows"], "Flagged": rec["flagged"],
               "% flagged": round(100 * rec["flagged"] / max(rec["n_windows"], 1), 1)}
        row.update({fid: rec["by_flag"].get(fid, 0) for fid in ids})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Flagged", ascending=False)


def pooled_to_excel(pooled: dict, manifest: Manifest, path: str | Path, *,
                    corpus: str = "", labels: Optional[dict[str, str]] = None) -> str:
    """Corpus-level workbook: flag summary, per-video breakdown, overlaps and
    provenance. No per-window sheet — that lives in each video's own file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = pooled_summary_table(pooled, manifest)
    per_video = pooled_video_table(pooled, manifest, labels)
    overlaps = overlap_table({"summary": {"pairs": pooled["pairs"],
                                          "flagged_windows": pooled["flagged_windows"]}},
                             manifest)

    n_flags = len(manifest.flags)
    provenance = [
        {"Field": "corpus", "Value": corpus},
        {"Field": "videos", "Value": pooled["n_videos"]},
        {"Field": "windows", "Value": pooled["n_windows"]},
        {"Field": "flagged windows", "Value": pooled["flagged_windows"]},
        {"Field": "manifest id", "Value": manifest.id},
        {"Field": "manifest version", "Value": manifest.version},
        {"Field": "manifest sha256", "Value": manifest.sha256[:16]},
        {"Field": "rule", "Value": f"{manifest.min_elements} elements, macros: "
                                   f"{', '.join(manifest.require_each_macro)}"},
        {"Field": "flag pairs observed",
         "Value": f"{int((overlaps['Windows'] > 0).sum()) if not overlaps.empty else 0}"
                  f" of {n_flags * (n_flags - 1) // 2} possible"},
    ]
    for element, stats in pooled["verbal_coverage"].items():
        provenance.append({"Field": f"verbal coverage: {element}",
                           "Value": f"{stats['filled']} windows ({stats['pct']}%)"})
    provenance.append({
        "Field": "thresholds",
        "Value": "per video — see each video's own workbook; categories are cut "
                 "against that speaker's distribution, so there is no corpus-wide edge"})

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Flag summary", index=False)
        per_video.to_excel(writer, sheet_name="By video", index=False)
        overlaps.to_excel(writer, sheet_name="Overlaps", index=False)
        pd.DataFrame(provenance).to_excel(writer, sheet_name="Provenance", index=False)
        for sheet, widths in {
            "Flag summary": {"A": 8, "B": 38, "C": 12, "D": 14, "E": 10, "F": 12,
                             "G": 14, "H": 24},
            "By video": {"A": 46, "B": 12, "C": 10, "D": 10, "E": 11},
            "Overlaps": {"A": 12, "B": 34, "C": 34, "D": 14, "E": 12, "F": 20},
            "Provenance": {"A": 30, "B": 70},
        }.items():
            ws = writer.sheets[sheet]
            for column, width in widths.items():
                ws.column_dimensions[column].width = width
            ws.freeze_panes = "A2"
    return str(path)


# ──────────────────────────────────────────────────────────────────────────
# The table people read
# ──────────────────────────────────────────────────────────────────────────

def window_table(run: dict, manifest: Manifest) -> pd.DataFrame:
    """One row per window, in the layout of FLAGS SYS X ABEL.xlsx's
    "Practical Example x Video" sheet: timestamp, transcript, then one column
    per macro-category, then the flags.

    Grouping the profile by macro-category rather than listing every element
    separately is deliberate — it is how the analysts read a window, and it
    keeps the table legible when a manifest defines twenty elements.
    """
    by_macro = manifest.macro_categories
    rows = []
    for record in run["windows"]:
        profile = record["profile"]
        row = {
            "Timestamp": f"{_timestamp(record['t'][0])}–{_timestamp(record['t'][1])}",
            "Transcript": record["transcript"],
        }
        for macro, members in by_macro.items():
            parts = []
            for element in members:
                value = profile.get(element)
                if not value:
                    continue
                if isinstance(value, list):        # lexicon hits
                    names = sorted({h["lex"] if isinstance(h, dict) else str(h)
                                    for h in value})
                    parts.append(f"{element}: {', '.join(names)}")
                else:
                    parts.append(f"{element}: {value}")
            row[macro.capitalize()] = "; ".join(parts)
        row["FLAG"] = " - ".join(f["id"] for f in record["flags"])
        row["Flag names"] = "; ".join(
            next((f.name for f in manifest.flags if f.id == rec["id"]), rec["id"])
            for rec in record["flags"])
        row["Evidence"] = " | ".join(
            f"{rec['id']}: {rec['n']}/{len(rec['matched']) + len(rec['missed'])}"
            f" [{', '.join(rec['matched'])}]" for rec in record["flags"])
        rows.append(row)
    return pd.DataFrame(rows)


def summary_table(run: dict, manifest: Manifest) -> pd.DataFrame:
    """Pooled counts for one video: how many windows each flag claimed."""
    summary = run["summary"]
    rows = [
        {"Flag": fid, "Name": rec["name"], "Windows": rec["n"],
         "% of windows": rec["pct"],
         "Near misses": summary["near_miss"].get(fid, {}).get("n", 0),
         "Top blocking element": (
             max(summary["near_miss"].get(fid, {}).get("missing", {}).items(),
                 key=lambda kv: kv[1])[0]
             if summary["near_miss"].get(fid, {}).get("missing") else "")}
        for fid, rec in summary["by_flag"].items()
    ]
    return pd.DataFrame(rows).sort_values("Windows", ascending=False)


def overlap_table(run: dict, manifest: Manifest) -> pd.DataFrame:
    """Which flag pairs actually co-occurred, against which the manifest
    expected to.

    FLAGS SYS X ABEL.xlsx works this out by hand — 19 of 45 pairs can overlap,
    6 designed, 6 between related functions, 7 residual. This measures it
    instead: a pair the manifest never anticipated showing up often is a sign
    two functions are not as distinct as intended, and a designed overlap that
    never occurs says the corpus does not contain that combination.

    Pairs the manifest predicts but the video never shows are listed with a
    count of 0, because their absence is as informative as their presence.
    """
    summary = run["summary"]
    names = {f.id: f.name for f in manifest.flags}
    flagged = max(summary.get("flagged_windows", 0), 1)

    rows = []
    seen = set()
    for pair, count in summary.get("pairs", {}).items():
        a, b = pair.split("|")
        seen.add(frozenset((a, b)))
        rows.append({
            "Pair": f"{a}–{b}",
            "Flag A": names.get(a, a), "Flag B": names.get(b, b),
            "Expected": manifest.overlap_kind(a, b),
            "Windows": count,
            "% of flagged windows": round(100 * count / flagged, 2),
        })

    for kind, pairs in manifest.expected_overlaps.items():
        for pair in pairs:
            if pair in seen:
                continue
            a, b = sorted(pair)
            rows.append({
                "Pair": f"{a}–{b}",
                "Flag A": names.get(a, a), "Flag B": names.get(b, b),
                "Expected": kind, "Windows": 0, "% of flagged windows": 0.0,
            })

    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=["Pair", "Flag A", "Flag B", "Expected",
                                     "Windows", "% of flagged windows"])
    return table.sort_values(["Windows", "Pair"], ascending=[False, True])


def to_excel(run: dict, manifest: Manifest, path: str | Path, *,
             label: Optional[str] = None) -> str:
    """Write the per-window table and the pooled stats to one workbook.

    Three sheets: the window table, the per-flag summary, and the provenance
    (manifest id/version/hash and the threshold edges used) — because a table
    of flags without the rules that produced them cannot be checked by anyone
    who was not in the room.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    windows = window_table(run, manifest)
    summary = summary_table(run, manifest)
    overlaps = overlap_table(run, manifest)

    provenance = [
        {"Field": "video", "Value": label or ""},
        {"Field": "manifest id", "Value": run["manifest"]["id"]},
        {"Field": "manifest version", "Value": run["manifest"]["version"]},
        {"Field": "manifest sha256", "Value": run["manifest"]["sha256"][:16]},
        {"Field": "window size (s)", "Value": run["manifest"]["window_s"]},
        {"Field": "rule", "Value": f"{manifest.min_elements} elements, "
                                   f"macros: {', '.join(manifest.require_each_macro)}"},
        {"Field": "windows", "Value": run["summary"]["n_windows"]},
        {"Field": "flagged windows", "Value": run["summary"]["flagged_windows"]},
    ]
    for element, coverage in (run.get("verbal_coverage") or {}).items():
        if isinstance(coverage, dict) and "pct" in coverage:
            provenance.append({"Field": f"verbal coverage: {element}",
                               "Value": f"{coverage['filled']} windows ({coverage['pct']}%)"})
    for name, spec in (run.get("thresholds") or {}).items():
        edges = spec.get("edges")
        if isinstance(edges, dict):                       # per speaker
            edges = next(iter(edges.values()), None)
        if edges:
            provenance.append({
                "Field": f"threshold: {name}",
                "Value": f"{[round(float(e), 3) for e in edges]} -> "
                         f"{', '.join(spec.get('labels', []))}"})

    n_flags = len(manifest.flags)
    possible_pairs = n_flags * (n_flags - 1) // 2
    provenance.append({"Field": "flag pairs observed",
                       "Value": f"{int((overlaps['Windows'] > 0).sum()) if not overlaps.empty else 0}"
                                f" of {possible_pairs} possible"})

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        windows.to_excel(writer, sheet_name="Windows", index=False)
        summary.to_excel(writer, sheet_name="Flag summary", index=False)
        overlaps.to_excel(writer, sheet_name="Overlaps", index=False)
        pd.DataFrame(provenance).to_excel(writer, sheet_name="Provenance", index=False)

        # Column widths: the transcript column is unreadable at the default.
        widths = {"Windows": {"A": 22, "B": 60, "C": 34, "D": 34, "E": 30,
                              "F": 26, "G": 14, "H": 34, "I": 60},
                  "Flag summary": {"A": 8, "B": 38, "C": 12, "D": 14, "E": 14, "F": 24},
                  "Overlaps": {"A": 12, "B": 34, "C": 34, "D": 14, "E": 12, "F": 20},
                  "Provenance": {"A": 30, "B": 60}}
        for sheet, spec in widths.items():
            ws = writer.sheets[sheet]
            for column, width in spec.items():
                ws.column_dimensions[column].width = width
            ws.freeze_panes = "A2"
    return str(path)
