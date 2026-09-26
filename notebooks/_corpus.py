"""
notebooks/_corpus.py
---------------------
Shared loading layer for the analysis notebooks. Kept as a module rather
than copied into each notebook so the three of them cannot drift on the
things that would silently invalidate a comparison — particularly the
missing-data handling below, which is not cosmetic.

Reads the same per-corpus MongoDB collections core/results_repository.py
writes ({corpus}_videos / _fused_windows / _artifacts). Read-only: nothing
here writes, deletes, or creates a collection.

## Sharded clusters

A corpus can span several Atlas clusters — MONGO_URI, MONGO_URI_2, … — once
one fills up (see core/results_repository.py's "Shards"). Every loader here
reads *all* of them and concatenates, so a corpus that outgrew one cluster
still comes back as one frame. Shard membership is deliberately not exposed:
which cluster a video happens to live on is an artefact of when it was
processed, and nothing in an analysis should ever condition on it.

Unlike the dashboard, which hides an unreachable cluster so Browse Corpus
keeps working, every function here raises instead. Quietly analysing the
subset of a corpus that happened to be reachable is how you publish a mean
over two thirds of your videos without noticing.

## Missing data — read this before trusting a number

A pose-absent window is stored with gesture metrics at 0.0, not null. That
is a faithful record of "the pipeline produced nothing here", but treating
it as a measurement is wrong in a specific, damaging way: 0.0 mean wrist
velocity reads as "the speaker held perfectly still" when it actually means
"no speaker was found". Averaging those zeros in silently drags every
gesture statistic toward zero in proportion to how often detection failed,
which varies by video and by corpus — so it would bias exactly the
cross-corpus comparison these notebooks exist to make.

`load_windows` therefore masks gesture metrics to NaN wherever
`pose_present_ratio == 0`, and maps the sentinel string "unknown" to NaN
for the camera angle/shot categoricals. Pandas then excludes them from
means rather than counting them as observations. `pose_present_ratio`
itself is deliberately left unmasked — it is the record of coverage, and
is what `coverage_report` reports on.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Gesture metrics that are only meaningful when a pose was actually found.
_POSE_DEPENDENT = [
    "mean_wrist_velocity",
    "max_wrist_displacement",
    "handedness_ratio",
]

# Categoricals whose "unknown" is a missing-data sentinel, not a category.
_CATEGORICAL_SENTINEL = {
    "dominant_shot_type": "unknown",
    "horizontal_angle": "unknown",
    "vertical_angle": "unknown",
}


_SHARDS: Optional[list] = None


def _shards() -> list:
    """Every configured shard's database handle, in fill order.

    Connects using the project's own .env *and its own URI resolution*
    (results_repository._uris_from_env), so notebooks and the app can never
    point at different clusters — including when a cluster is added.

    Cached: without this, every loader call opened a fresh connection to
    every cluster, which on a three-shard setup is slow enough to be felt in
    a notebook. Call reset_connection() after editing .env mid-session.
    """
    global _SHARDS
    if _SHARDS is not None:
        return _SHARDS

    import sys

    from dotenv import load_dotenv
    from pymongo import MongoClient

    load_dotenv(_PROJECT_ROOT / ".env")
    sys.path.insert(0, str(_PROJECT_ROOT))
    # _host keeps the cluster's name out of the credentials, and
    # _uris_from_env is the same resolution the app uses.
    from core.results_repository import _host, _uris_from_env

    uris = _uris_from_env()
    if not uris:
        raise RuntimeError(
            "MONGO_URI is not set. The notebooks read the project's .env — "
            f"expected at {_PROJECT_ROOT / '.env'}"
        )
    db_name = os.environ.get("MONGO_DB", "multiarth")
    _SHARDS = [
        (_host(u), MongoClient(u, serverSelectionTimeoutMS=8000)[db_name])
        for u in uris
    ]
    return _SHARDS


def _dbs() -> list:
    """Just the database handles, in fill order."""
    return [db for _, db in _shards()]


def reset_connection() -> None:
    """Drop the cached connections — use after changing .env in a running
    kernel (e.g. adding MONGO_URI_3)."""
    global _SHARDS
    _SHARDS = None


def _each_shard(fn, what: str) -> list:
    """Run `fn(db)` on every shard, failing loudly on any that is down —
    see this module's "Sharded clusters". The shard is named from its URI
    rather than from the client, which reports no nodes precisely when it
    could not connect."""
    out = []
    for name, db in _shards():
        try:
            out.append(fn(db))
        except Exception as exc:
            raise RuntimeError(
                f"Shard {name} is unreachable, so {what} would be "
                f"incomplete — refusing to return a partial corpus. "
                f"Underlying error: {exc}"
            ) from exc
    return out


def available_corpora() -> list[str]:
    """Corpus names that have a videos collection on any shard."""
    per_shard = _each_shard(lambda db: db.list_collection_names(), "the corpus list")
    names = {n for shard in per_shard for n in shard if n.endswith("_videos")}
    return sorted(n[: -len("_videos")] for n in names)


def load_videos(corpus: str) -> pd.DataFrame:
    """One row per shipped video, across every shard. Empty DataFrame if the
    corpus has none."""
    docs = [
        d
        for shard in _each_shard(
            lambda db: list(db[f"{corpus}_videos"].find({})), f"{corpus}'s video list"
        )
        for d in shard
    ]
    if not docs:
        return pd.DataFrame(
            columns=["job_id", "label", "duration_s", "total_windows", "window_size_s"]
        )
    rows = [
        {
            "job_id": d["_id"],
            "label": d.get("label"),
            "duration_s": d.get("duration_s"),
            "total_windows": d.get("total_windows"),
            "window_size_s": d.get("window_size_s"),
            "corpus": corpus,
        }
        for d in docs
    ]
    # A job lives entirely on one shard, so a repeated job_id means two
    # clusters hold copies of the same video — every per-window statistic
    # would then count it twice, invisibly. Cheap to check, and the failure
    # it catches is not one you would spot in a mean.
    ids = [r["job_id"] for r in rows]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise RuntimeError(
            f"{corpus}: job(s) {sorted(duplicates)} exist on more than one "
            "shard. Delete the stale copy (ResultsRepository.delete_job_data "
            "removes a job from every shard) before analysing."
        )
    return pd.DataFrame(rows)


def load_windows(corpus: str, mask_missing: bool = True) -> pd.DataFrame:
    """
    One tidy row per analysis window, flattened across all four modalities
    and joined to its video's label/duration.

    mask_missing=True (the default, and what every notebook uses) applies
    the NaN masking described in this module's docstring. Pass False only
    to inspect the raw stored values — any statistic computed on the
    unmasked frame will be biased by pose-absent windows.
    """
    # Count the big arrays server-side instead of downloading them. A window
    # carries ~50 pose keyframes of 33 landmarks each, which is the bulk of the
    # corpus by bytes — and this function only ever uses their *length*.
    # Measured on Ted: minutes with the arrays, seconds without.
    pipeline = [
        {"$addFields": {
            "n_keyframes": {"$size": {"$ifNull": ["$gesture.pose_keyframes", []]}},
            "n_tokens": {"$size": {"$ifNull": ["$verbal.tokens", []]}},
        }},
        {"$project": {"gesture.pose_keyframes": 0, "verbal.tokens": 0}},
    ]
    docs = [
        d
        for shard in _each_shard(
            lambda db: list(db[f"{corpus}_fused_windows"].aggregate(pipeline)),
            f"{corpus}'s windows",
        )
        for d in shard
    ]
    if not docs:
        return pd.DataFrame()

    rows = []
    for d in docs:
        g = d.get("gesture") or {}
        p = d.get("prosody") or {}
        v = d.get("verbal") or {}
        c = d.get("camera") or {}
        w = d.get("window") or {}
        rows.append({
            "job_id": d.get("job_id"),
            "window_idx": d.get("window_idx"),
            "start_s": w.get("start_s"),
            "end_s": w.get("end_s"),
            # gesture
            "pose_present_ratio": g.get("pose_present_ratio"),
            "mean_wrist_velocity": g.get("mean_wrist_velocity"),
            "max_wrist_displacement": g.get("max_wrist_displacement"),
            "handedness_ratio": g.get("handedness_ratio"),
            "n_keyframes": d.get("n_keyframes", 0),
            # prosody
            "mean_f0": p.get("mean_f0"),
            "f0_range": p.get("f0_range"),
            "f0_std": p.get("f0_std"),
            "mean_intensity_db": p.get("mean_intensity_db"),
            "intensity_range_db": p.get("intensity_range_db"),
            "speech_rate_syl_per_s": p.get("speech_rate_syl_per_s"),
            # verbal
            "word_count": v.get("word_count"),
            "transcript": v.get("transcript"),
            "n_tokens": d.get("n_tokens", 0),
            # camera
            "cut_count": c.get("cut_count"),
            "cut_rate": c.get("cut_rate"),
            "dominant_shot_type": c.get("dominant_shot_type"),
            "mean_face_bbox_area": c.get("mean_face_bbox_area"),
            "face_bbox_trend": c.get("face_bbox_trend"),
            "mean_shoulder_yaw_deg": c.get("mean_shoulder_yaw_deg"),
            "horizontal_angle": c.get("horizontal_angle"),
            "mean_face_pitch_deg": c.get("mean_face_pitch_deg"),
            "vertical_angle": c.get("vertical_angle"),
        })

    # Sorted here rather than in the query: with more than one shard the
    # documents arrive one cluster at a time, so a per-query sort would only
    # order within a shard. Sorting by (job_id, window_idx) also makes the
    # frame independent of which cluster a video happens to live on — two
    # runs give the same row order even if a video is re-shipped elsewhere.
    df = pd.DataFrame(rows).sort_values(["job_id", "window_idx"], kind="stable")
    df = df.reset_index(drop=True)
    df["corpus"] = corpus

    videos = load_videos(corpus)
    if not videos.empty:
        df = df.merge(
            videos[["job_id", "label", "duration_s"]], on="job_id", how="left"
        )
        # Position within the talk, so videos of different lengths can be
        # overlaid on a common 0-1 axis.
        df["progress"] = df["start_s"] / df["duration_s"]

    if mask_missing:
        absent = df["pose_present_ratio"].fillna(0) == 0
        df.loc[absent, _POSE_DEPENDENT] = np.nan
        for col, sentinel in _CATEGORICAL_SENTINEL.items():
            df[col] = df[col].replace(sentinel, np.nan)

    return df


def load_artifacts(
    corpus: str, job_id: str, fields: Optional[list[str]] = None,
) -> dict:
    """The per-job wordlist / ngrams / collocations / spectrogram /
    waveform / segmented_tokens blob. Searches every shard — a job's
    artifacts live on whichever one holds the rest of that job.

    `fields` restricts what is fetched, and matters more than it looks:
    the whole document is ~2.5MB per video, of which the spectrogram alone
    is 2.1MB. Reading every video's wordlist without a projection moves
    ~100MB across the network to use ~1.4MB of it.
    """
    projection = {f: 1 for f in fields} if fields else None
    for doc in _each_shard(
        lambda db: db[f"{corpus}_artifacts"].find_one({"_id": job_id}, projection),
        f"{corpus}'s artifacts",
    ):
        if doc:
            return doc
    return {}


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Non-null count and share per column — the first thing to look at,
    since a confident-looking mean over 12% of windows is not a finding.
    Run this on the *masked* frame, so it reports usable observations
    rather than stored rows."""
    if df.empty:
        return pd.DataFrame(columns=["non_null", "total", "coverage"])
    counts = df.notna().sum()
    out = pd.DataFrame({"non_null": counts, "total": len(df)})
    out["coverage"] = (out["non_null"] / out["total"]).round(3)
    return out.sort_values("coverage")


def corpus_status(corpus: str) -> dict:
    """Cheap counts without loading everything — used by each notebook's
    guard cell so an empty corpus reports itself clearly instead of
    failing somewhere further down with an opaque pandas error.

    Counts are summed over every shard, so they describe the corpus, not a
    cluster."""
    per_shard = _each_shard(
        lambda db: {
            "videos": db[f"{corpus}_videos"].count_documents({}),
            "fused_windows": db[f"{corpus}_fused_windows"].count_documents({}),
            "artifacts": db[f"{corpus}_artifacts"].count_documents({}),
        },
        f"{corpus}'s counts",
    )
    return {
        "corpus": corpus,
        "videos": sum(s["videos"] for s in per_shard),
        "fused_windows": sum(s["fused_windows"] for s in per_shard),
        "artifacts": sum(s["artifacts"] for s in per_shard),
        "shards": len(per_shard),
    }


def describe_numeric(df: pd.DataFrame, cols: Optional[list[str]] = None) -> pd.DataFrame:
    """describe() restricted to the columns that carry signal, with the
    count column kept visible so a small-n statistic is never read as if
    it rested on the full corpus."""
    if df.empty:
        return pd.DataFrame()
    cols = cols or [c for c in df.select_dtypes("number").columns
                    if c not in ("window_idx", "start_s", "end_s", "duration_s", "progress")]
    return df[cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(3)


# ── Dense within-window signals (for coupling analysis) ──────────────────
#
# Everything above is per-window: one row per 5s. Coupling between
# modalities lives at 0.2-2s (a gesture stroke landing on a prosodic peak),
# so a 5s mean cannot show it — see notebooks/04_multimodal_coupling.ipynb.
# The gesture side of that resolution IS stored: pose_keyframes hold every
# 3rd pose-present frame, i.e. ~8-10Hz. These loaders expose it.
#
# The prosody side is NOT stored densely (ProsodyWorker aggregates to the
# window before writing), which is the one blocker for true sub-second
# cross-modal analysis.

# MediaPipe BlazePose indices, repeated here so notebooks do not import the
# worker (which would pull in mediapipe, onnxruntime and torch).
_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP = 11, 12, 23, 24
_L_WRIST, _R_WRIST = 15, 16


def wrist_speed_series(
    corpus: str, job_id: str, scale_window_s: float = 2.0,
) -> pd.DataFrame:
    """Dense wrist speed for one video, in **body units per second**.

    Columns: ts, speed, left_speed, right_speed, shoulder_w, torso_h, cx, cy.

    ## Why not the stored mean_wrist_velocity

    That metric is pixels/second, which confounds two things with gesture
    magnitude: shot scale (a close-up makes the same physical movement sweep
    more pixels — it correlates with mean_face_bbox_area at ~0.47 in Ted)
    and video resolution (the same framing at 720p yields two thirds the
    pixel velocity of 1080p). Neither has anything to do with the speaker.

    Keyframe coordinates are frame-normalised, which removes resolution but
    NOT shot scale. Dividing by a body length removes both, and the body
    length must be measured per axis: pose_x is normalised by frame width
    and pose_y by frame height, so on 16:9 footage every vertical distance
    is inflated ~1.78x relative to horizontal. Hence dx is divided by
    shoulder width and dy by torso height, each measured on the same axis,
    rather than dividing a Euclidean distance by one scalar.

    Gaps: keyframes are every 3rd *pose-present* frame, so a tracking
    dropout leaves a hole. Speed is only computed across consecutive
    keyframes less than `_MAX_KEYFRAME_GAP_S` apart — otherwise a dropout
    would read as one enormous gesture.

    ## The body scale must be LOCAL, not per video

    Measured on Ted, shoulder width varies ~8x *within* a single talk as the
    camera cuts between close-ups and wide shots. An earlier version divided
    by one median for the whole video, which corrected nothing within it: the
    resulting metric still tracked shot scale (rho 0.71 against face area,
    worse than the raw pixel metric's 0.57). The scale is therefore a rolling
    median over `scale_window_s` seconds, which follows the cuts.

    The rolling median is floored at a fraction of the video median
    (_MIN_SCALE_FRACTION): when the speaker turns side-on her shoulders
    foreshorten to almost nothing, and dividing by that manufactures enormous
    speeds from ordinary movement.

    y is stored pre-flipped (1 - raw_y); that affects the sign of dy, never
    its magnitude, so speed is unaffected.
    """
    # Pull only the six landmarks this uses, server-side. A stored keyframe
    # carries 33 landmarks x (x, y, visibility) plus 33 world xyz — ~198
    # numbers — of which this needs 12. Extracting them with $arrayElemAt
    # cuts the payload by more than an order of magnitude, which stops
    # mattering the moment Atlas is slow: measured at 2.5s per video on a
    # good link and 150s per video on a bad one.
    def _elem(field, i):
        return {"$arrayElemAt": [f"$$k.{field}", i]}

    pipeline = [
        {"$match": {"job_id": job_id}},
        {"$sort": {"window_idx": 1}},
        {"$project": {"_id": 0, "kf": {"$map": {
            "input": {"$ifNull": ["$gesture.pose_keyframes", []]},
            "as": "k",
            "in": {
                "ts": "$$k.ts",
                "lwx": _elem("pose_x", _L_WRIST), "lwy": _elem("pose_y", _L_WRIST),
                "rwx": _elem("pose_x", _R_WRIST), "rwy": _elem("pose_y", _R_WRIST),
                "lsx": _elem("pose_x", _L_SHOULDER), "lsy": _elem("pose_y", _L_SHOULDER),
                "rsx": _elem("pose_x", _R_SHOULDER), "rsy": _elem("pose_y", _R_SHOULDER),
                "lhy": _elem("pose_y", _L_HIP), "rhy": _elem("pose_y", _R_HIP),
            },
        }}}},
    ]
    windows = _each_shard(
        lambda db: list(db[f"{corpus}_fused_windows"].aggregate(pipeline)),
        f"{corpus}/{job_id}'s keyframes",
    )
    rows = []
    for shard in windows:
        for doc in shard:
            for k in doc.get("kf") or []:
                if k.get("lsx") is None:
                    continue
                rows.append({
                    "ts": k["ts"],
                    "lx": k["lwx"], "ly": k["lwy"],
                    "rx": k["rwx"], "ry": k["rwy"],
                    "shoulder_w": abs(k["lsx"] - k["rsx"]),
                    "torso_h": abs((k["lsy"] + k["rsy"]) / 2 - (k["lhy"] + k["rhy"]) / 2),
                    "cx": (k["lsx"] + k["rsx"]) / 2,
                    "cy": (k["lsy"] + k["rsy"]) / 2,
                })
    if not rows:
        return pd.DataFrame(columns=["ts", "speed", "left_speed", "right_speed"])

    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)

    # Rolling median body scale, so a cut to a close-up rescales with it.
    idx = pd.to_timedelta(df["ts"], unit="s")
    window = f"{int(scale_window_s * 1000)}ms"
    sw_video = float(np.nanmedian(df["shoulder_w"].replace(0, np.nan)))
    th_video = float(np.nanmedian(df["torso_h"].replace(0, np.nan)))
    sw = (df["shoulder_w"].replace(0, np.nan).set_axis(idx)
          .rolling(window, center=True, min_periods=3).median()
          .to_numpy())
    th = (df["torso_h"].replace(0, np.nan).set_axis(idx)
          .rolling(window, center=True, min_periods=3).median()
          .to_numpy())
    sw = np.where(np.isnan(sw), sw_video, sw)
    th = np.where(np.isnan(th), th_video, th)
    sw = np.maximum(sw, sw_video * _MIN_SCALE_FRACTION)
    th = np.maximum(th, th_video * _MIN_SCALE_FRACTION)
    df["scale_w"], df["scale_h"] = sw, th

    dt = df["ts"].diff()
    ok = (dt > 0) & (dt <= _MAX_KEYFRAME_GAP_S)

    for side, (xc, yc) in {"left": ("lx", "ly"), "right": ("rx", "ry")}.items():
        # The scale of the *later* frame of each pair, matching the
        # displacement being measured.
        dx = df[xc].diff() / df["scale_w"]
        dy = df[yc].diff() / df["scale_h"]
        speed = np.sqrt(dx ** 2 + dy ** 2) / dt
        df[f"{side}_speed"] = speed.where(ok)

    df["speed"] = df[["left_speed", "right_speed"]].max(axis=1)
    return df[["ts", "speed", "left_speed", "right_speed",
               "shoulder_w", "torso_h", "scale_w", "scale_h", "cx", "cy"]]


# Floor on the local body scale, as a fraction of the video's median. A
# speaker turned side-on foreshortens to a fraction of her frontal shoulder
# width; without this, ordinary movement in those frames divides by almost
# nothing.
_MIN_SCALE_FRACTION = 0.4

# Longest gap between consecutive keyframes still treated as continuous.
# Keyframes are every 3rd pose-present frame (~0.1s at 30fps); 0.5s allows a
# short dropout while refusing to measure "speed" across a real gap.
_MAX_KEYFRAME_GAP_S = 0.5


# Han characters. One character is one syllable in Mandarin, which is what
# makes a counted syllable rate possible for Chinese but not for English.
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def chinese_syllable_rate(df: pd.DataFrame) -> pd.Series:
    """Syllables per second per window, counted — **Mandarin only**.

    Named for the language because the method is language-specific, not
    general: it counts Han characters, and the one-character-one-syllable
    identity that makes that a syllable count holds for Chinese. It does not
    hold for Japanese, where a single kanji is read as one to four morae, so
    this would undercount a Japanese corpus badly while looking like it worked.
    For English it returns all-NaN by construction (no Han characters), which
    is deliberate: English syllabification needs a pronunciation dictionary or
    a vowel-group heuristic, and inventing one here would swap a visible gap
    for an invisible error.

    ## Why not use the stored column

    `speech_rate_syl_per_s` is `word_count / duration * 1.5`
    (core/fusion_engine.py), and both halves assume English:

      * `word_count` is the number of raw ASR tokens, and SenseVoice emits
        roughly one token per Han character — so for Chinese it is already a
        syllable count, not a word count.
      * Multiplying that by an assumed 1.5 syllables-per-word inflates the
        rate by about half again. YiXi's stored median is 7.2 syl/s; counted,
        4.4, which is the range the literature reports for Mandarin.

    The error is near-monotone rather than a clean constant (stored/counted
    runs 1.56-1.97 over the 5th-95th percentile, Spearman 0.95), so it
    disturbs rank analyses only mildly but makes every quoted value wrong.

    Counting happens on the window's own stored transcript, so no reprocessing
    and no re-download is needed.
    """
    if "transcript" not in df.columns:
        return pd.Series(np.nan, index=df.index, name="syl_per_s")
    syl = df.transcript.fillna("").astype(str).map(lambda s: len(_CJK.findall(s)))
    dur = (df.end_s - df.start_s).replace(0, np.nan)
    rate = (syl / dur).where(syl > 0)
    return rate.rename("syl_per_s")


def segmented_tokens(corpus: str, job_id: str) -> pd.DataFrame:
    """Words with timestamps, as **spaCy** segments them. Columns: word, start_s, end_s.

    Prefer this over `word_tokens` for anything that searches or counts words.

    The recogniser's own tokens are words in English but roughly one per
    character in Chinese, so an index built from them cannot match 女性 or
    婚姻 — it holds 女, 性, 婚 and 姻 separately, and every Chinese query
    silently returns nothing. The verbal worker already solves this: it joins
    the transcript without separators, runs spaCy (pkuseg for Chinese), then
    maps each resulting word back onto the ASR tokens it spans to recover its
    timing (`VerbalWorker._segment_words`). That is what this reads.

    Using it also keeps word-level work on the same tokenisation as the word
    list, collocations and the word sketch, so counts cannot disagree between
    two parts of the system.

    Punctuation is dropped: spaCy emits 。and , as tokens, and they are not
    words. Returns an empty frame for a job processed before segmented tokens
    were stored — callers should fall back to `word_tokens` and say so.

    Rows are **spoken words**, not spaCy tokens: pieces sharing an `asr_idx`
    are rejoined, so "don't" comes back as one row tagged AUX+PART rather than
    "do" and "n't" separately. A `pos` column carries that composite tag. See
    notebooks/DECISIONS.md §4.
    """
    art = load_artifacts(corpus, job_id, fields=["segmented_tokens"]) or {}
    rows = art.get("segmented_tokens") or []
    if not rows:
        return pd.DataFrame(columns=["word", "start_s", "end_s", "pos"])
    df = pd.DataFrame(rows)

    # Regroup onto the word that was actually spoken. spaCy splits "don't"
    # into "do" + "n't"; both carry the same asr_idx, so the group rebuilds the
    # contraction and tags it AUX+PART. In Chinese spaCy merges instead, every
    # group is a singleton, and this is a no-op. Videos processed before
    # asr_idx was stored fall through unchanged, one token per row.
    if "asr_idx" in df.columns and df.asr_idx.notna().any():
        agg = {"word": ("word", lambda x: "".join(map(str, x))),
               "start_s": ("start_s", "first"), "end_s": ("end_s", "last")}
        if "pos" in df.columns:
            agg["pos"] = ("pos", lambda x: "+".join(
                p for p in map(str, x) if p not in ("PUNCT", "nan")) or None)
        df = (df.sort_values(["asr_idx"], kind="stable")
                .groupby("asr_idx", sort=True).agg(**agg).reset_index(drop=True))

    keep = df.word.astype(str).str.contains(r"[\w\u3400-\u4dbf\u4e00-\u9fff]", regex=True)
    return df[keep].reset_index(drop=True)


def word_tokens(corpus: str, job_id: str) -> pd.DataFrame:
    """Every word with its start/end, for aligning speech against gesture.
    Columns: word, start_s, end_s, confidence."""
    windows = _each_shard(
        lambda db: list(db[f"{corpus}_fused_windows"].find(
            {"job_id": job_id}, {"verbal.tokens": 1, "window_idx": 1},
        ).sort("window_idx", 1)),
        f"{corpus}/{job_id}'s tokens",
    )
    rows = []
    for shard in windows:
        for doc in shard:
            for t in ((doc.get("verbal") or {}).get("tokens") or []):
                rows.append(t)
    if not rows:
        return pd.DataFrame(columns=["word", "start_s", "end_s", "confidence"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["word", "start_s"])
    return df.sort_values("start_s").reset_index(drop=True)


_REPO = None


def _repo():
    """A ResultsRepository sharing this module's .env resolution.

    _shards() is called first purely for its side effect: it loads the
    project's .env. Constructing the repository before that raises "MONGO_URI
    not provided" when a notebook has not touched any other loader yet.
    """
    global _REPO
    if _REPO is None:
        import sys as _sys

        _shards()
        _sys.path.insert(0, str(_PROJECT_ROOT))
        from core.results_repository import ResultsRepository

        _REPO = ResultsRepository()
    return _REPO


def dense_prosody(corpus: str, job_id: str) -> pd.DataFrame:
    """f0 and intensity at their native ~10ms hop, or an empty frame if this
    job has not been backfilled (scripts/backfill_dense_prosody.py).

    Columns: ts, f0, intensity_db. f0 is NaN where Praat found no voicing —
    that is "not measured", not 0Hz, and every statistic should skip it
    rather than average it in.

    This is the signal every *word-level* question needs. The per-window
    mean_f0 in load_windows covers 5 seconds, roughly 15 words, so it cannot
    say what the pitch was on any particular one.
    """
    data = _repo().get_dense_prosody(corpus, job_id)
    if not data:
        return pd.DataFrame(columns=["ts", "f0", "intensity_db"])
    t0 = float(data["params"].get("first_sample_s", 0.0))
    n = len(data["f0"])
    return pd.DataFrame({
        "ts": t0 + np.arange(n) * data["hop_s"],
        "f0": data["f0"],
        "intensity_db": data["intensity_db"],
    })


def has_dense_prosody(corpus: str) -> pd.DataFrame:
    """Which videos in a corpus have been backfilled — run before any
    word-level analysis, so a partial backfill is visible rather than
    silently excluding videos."""
    repo = _repo()
    rows = [
        {"job_id": v["_id"], "label": v.get("label"),
         "dense_prosody": repo.has_dense_prosody(corpus, v["_id"])}
        for v in repo.list_videos(corpus)
    ]
    return pd.DataFrame(rows)


# ── Corpus-level measures (MultiArth_Search_Bar_QUANTITATIVE.docx §2,4,5) ──

def _fetch_per_video_words(corpus: str, videos: pd.DataFrame):
    """One row per (video, word), plus each video's token total.

    Projected to `wordlist` alone: the full artifacts document is ~2.5MB per
    video, of which the spectrogram is 2.1MB, so fetching whole documents to
    read word counts moves ~100MB to use ~1.4MB of it.
    """
    rows, totals = [], []
    for job_id in videos.job_id:
        wl = (load_artifacts(corpus, job_id, fields=["wordlist"]) or {}).get("wordlist") or {}
        words = wl.get("words") or []
        if not words:
            continue
        # Deliberately NOT wl["total_tokens"]: that is the count of raw ASR
        # tokens, while every `count` below is a spaCy-segmented word. The two
        # agree in English (ratio 1.005 on Ted) but not in Chinese, where the
        # ASR emits one token per character and spaCy re-segments into words —
        # 234k ASR tokens against 130k words on YiXi. Dividing segmented counts
        # by an ASR total understated every Chinese frequency by ~1.8x, and
        # made freq_per_1000 disagree with mean_video_freq (which the worker
        # computes against its own segmented total) for reasons that had
        # nothing to do with one talk dominating a word.
        total = sum(w["count"] for w in words)
        totals.append(total)
        for w in words:
            # video_tokens travels with every row so a per-video rate can be
            # recomputed downstream. The artifact's own freq_per_1000 is kept
            # for reference but deliberately not used: it is rounded to two
            # decimals at write time, and it is one row per (word, POS), which
            # is the wrong denominator once a word carries several tags.
            rows.append({"job_id": job_id, "word": w["word"], "pos": w.get("pos"),
                         "count": w["count"], "video_tokens": total,
                         "video_freq": w.get("freq_per_1000")})
    return pd.DataFrame(rows), totals


def corpus_wordlist(
    corpus: str, pos: Optional[list[str]] = None, by_pos: bool = True,
) -> pd.DataFrame:
    """Word frequencies for a whole corpus, normalised (spec §2).

    Columns: word, pos, count, freq_per_1000, mean_video_freq, sd_video_freq,
    n_videos, range_pct.

    Two different normalisations, because they answer different questions and
    the spec's "frequency per 1,000 tokens" is ambiguous between them:

      freq_per_1000    pooled — every token in the corpus weighted equally, so
                       a 27-minute talk counts for more than a 6-minute one.
      mean_video_freq  each video's own rate, averaged — every speaker counts
                       equally, regardless of how long they spoke.

    They diverge exactly where one long talk is idiosyncratic, which is worth
    seeing rather than hiding. `n_videos` (document frequency) and `range_pct`
    are the dispersion check: a word carried by a single speaker is not a
    property of the corpus, however high its count.

    `by_pos=True` (the default) keeps one row per (word, part-of-speech), so
    "to" as an infinitive marker and "to" as a preposition stay distinct —
    which also means neither row alone is that word's corpus frequency. Pass
    `by_pos=False` for one row per surface form.

    Tags may be composite: a word is recorded as it was spoken and tagged with
    every part of speech spaCy gave its pieces, joined by "+", so "don't" is
    AUX+PART. The `pos` filter matches a tag's **first** component, the host
    word's class. See notebooks/DECISIONS.md §4.

    Chinese is segmented by pkuseg (via spaCy's zh_core_web_sm) upstream, not
    jieba as the spec suggests — same role, different tokeniser, worth stating
    in a methods section.
    """
    videos = load_videos(corpus)
    if videos.empty:
        return pd.DataFrame(columns=["word", "pos", "count", "freq_per_1000",
                                     "mean_video_freq", "sd_video_freq",
                                     "n_videos", "range_pct"])
    if corpus in _WORDLIST_CACHE:
        per_video, totals = _WORDLIST_CACHE[corpus]
        per_video = per_video.copy()
        return _aggregate_wordlist(per_video, totals, pos, by_pos)

    per_video, totals = _fetch_per_video_words(corpus, videos)
    if per_video.empty:
        return pd.DataFrame()
    # Cached per corpus: the notebook calls this at least twice (all words,
    # then content words only), and the fetch is the expensive half.
    _WORDLIST_CACHE[corpus] = (per_video, totals)
    return _aggregate_wordlist(per_video, totals, pos, by_pos)


_WORDLIST_CACHE: dict = {}


def _aggregate_wordlist(per_video, totals, pos, by_pos) -> pd.DataFrame:
    """Group an already-fetched per-video frame; see corpus_wordlist."""
    if pos:
        # Match on the FIRST component of the tag. A word is tagged with every
        # part of speech spaCy gave its pieces — "don't" is AUX+PART — and the
        # host word carries the lexical class while the clitic does not. So
        # "women's" (NOUN+PART) is a noun and belongs in a content-word cut,
        # while "there's" (PRON+VERB) is not a verb and does not. Matching any
        # component would admit the second; see notebooks/DECISIONS.md §4.
        # A plain tag has one component, so pre-composite data is unaffected.
        per_video = per_video[per_video.pos.astype(str).str.split("+").str[0].isin(pos)]
    corpus_tokens = float(sum(totals))
    n_videos_total = per_video.job_id.nunique()

    keys = ["word", "pos"] if by_pos else ["word"]

    # Collapse to one row per (key, video) before computing a rate, rather
    # than averaging the per-row rates the artifact already carries.
    #
    # The difference only appears once a word holds more than one tag inside
    # one video, which is exactly what fixing the first-occurrence POS bug
    # makes possible. With by_pos=False, "watch" used twice as a verb and once
    # as a noun in a 10k-word talk is two rows at 0.2 and 0.1 per 1,000; their
    # mean is 0.15, half the word's true 0.3 rate for that video — and that
    # video is counted twice in the average, outweighing videos where the word
    # carries a single tag. Summing first gives the rate the video actually
    # had, and weights every speaker once.
    per_key_video = (per_video.groupby(keys + ["job_id"], dropna=False)
                     .agg(count=("count", "sum"),
                          video_tokens=("video_tokens", "first"))
                     .reset_index())
    per_key_video["video_rate"] = (per_key_video["count"]
                                   / per_key_video["video_tokens"] * 1000)

    agg = (per_key_video.groupby(keys, dropna=False)
           .agg(count=("count", "sum"),
                mean_video_freq=("video_rate", "mean"),
                sd_video_freq=("video_rate", "std"),
                n_videos=("job_id", "nunique"))
           .reset_index())
    agg["freq_per_1000"] = agg["count"] / corpus_tokens * 1000
    agg["range_pct"] = agg["n_videos"] / n_videos_total * 100
    agg.attrs["corpus_tokens"] = corpus_tokens
    agg.attrs["n_videos"] = n_videos_total
    return agg.sort_values("count", ascending=False).reset_index(drop=True)


def shot_segments(corpus: str) -> pd.DataFrame:
    """One row per **shot segment**, not per window (spec §3).

    Columns: job_id, shot_idx, start_s, end_s, duration_s, shot_type.

    §3 asks for "percentages of total shots" and for shots-per-minute before
    comparing corpora. Both need shots, and the pipeline stores per-window
    *dominant* labels — so a 30-second close-up held across six windows counts
    six times in a by-window tally, and a corpus whose talks are simply longer
    accumulates more of everything.

    Segments are rebuilt from the cut timestamps the camera worker already
    stores (`camera.scene_cuts`): a segment runs from one cut to the next, with
    the video's start and end as the outer boundaries.

    The one approximation: a segment's shot type is taken from the window
    containing its **midpoint**, because shot type is classified per window and
    not per segment. The midpoint is the right choice rather than the start —
    a window straddling a cut is labelled for whichever shot dominates it, so
    the window at a segment's start may still be describing the previous shot.
    Segments shorter than a window are therefore the least reliable, and
    `duration_s` is kept so they can be filtered.
    """
    pipeline = [{"$project": {
        "job_id": 1, "window": 1,
        "camera.scene_cuts.timestamp_s": 1,
        "camera.dominant_shot_type": 1,
    }}]
    docs = [
        d
        for shard in _each_shard(
            lambda db: list(db[f"{corpus}_fused_windows"].aggregate(pipeline)),
            f"{corpus}'s scene cuts",
        )
        for d in shard
    ]
    if not docs:
        return pd.DataFrame(columns=["job_id", "shot_idx", "start_s", "end_s",
                                     "duration_s", "shot_type"])

    per_job: dict = {}
    for d in docs:
        w = d.get("window") or {}
        c = d.get("camera") or {}
        job = per_job.setdefault(d.get("job_id"), {"cuts": set(), "windows": []})
        # Same sentinel handling as load_windows: "unknown" means the
        # classifier failed, not that there is a category called unknown.
        shot = c.get("dominant_shot_type")
        if shot == _CATEGORICAL_SENTINEL["dominant_shot_type"]:
            shot = None
        job["windows"].append((w.get("start_s"), w.get("end_s"), shot))
        for cut in (c.get("scene_cuts") or []):
            ts = cut.get("timestamp_s")
            if ts is not None:
                job["cuts"].add(round(float(ts), 3))

    rows = []
    for job_id, job in per_job.items():
        windows = sorted((a, b, t) for a, b, t in job["windows"] if a is not None)
        if not windows:
            continue
        start, end = windows[0][0], windows[-1][1]
        # Cuts outside the windowed span would create empty segments.
        bounds = sorted({start, end} | {c for c in job["cuts"] if start < c < end})
        for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            mid = (a + b) / 2
            shot = next((t for ws, we, t in windows if ws <= mid < we), None)
            rows.append({"job_id": job_id, "shot_idx": i, "start_s": a, "end_s": b,
                         "duration_s": b - a, "shot_type": shot})
    return pd.DataFrame(rows)


def gesture_normalised(
    corpus: str, window_s: float = 5.0, cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """Per-window gesture magnitude in **body units**, not pixels (spec §5).

    Columns: job_id, window_idx, start_s, wrist_speed_mean, wrist_speed_p90,
    wrist_speed_max, n_samples.

    The stored `mean_wrist_velocity` is pixels/second, which confounds gesture
    magnitude with shot scale and video resolution — the spec's §5 asks for
    exactly this correction ("normalize velocity relative to body scale in
    frame ... not only frame rate"). This rebuilds the metric from the dense
    keyframes via wrist_speed_series, so no reprocessing is needed.

    Windows are reconstructed by binning keyframe timestamps, which reproduces
    the pipeline's own windowing (fixed `window_s` slices from t=0).

    Fetching keyframes for a whole corpus takes ~2s per video; pass
    `cache_path` to write/read a pickled copy so a notebook re-run is instant.
    Delete that file after reprocessing any video, or the cache will serve the
    old gesture track.
    """
    if cache_path and Path(cache_path).exists():
        return pd.read_pickle(cache_path)

    videos = load_videos(corpus)
    out = []
    for job_id in videos.job_id:
        speed = wrist_speed_series(corpus, job_id)
        speed = speed.dropna(subset=["speed"])
        if speed.empty:
            continue
        idx = (speed.ts // window_s).astype(int)
        grouped = speed.groupby(idx)["speed"]
        part = pd.DataFrame({
            "window_idx": grouped.mean().index.astype(int),
            "wrist_speed_mean": grouped.mean().to_numpy(),
            "wrist_speed_p90": grouped.quantile(0.9).to_numpy(),
            "wrist_speed_max": grouped.max().to_numpy(),
            "n_samples": grouped.size().to_numpy(),
        })
        part["job_id"] = job_id
        part["start_s"] = part["window_idx"] * window_s
        out.append(part)

    df = (pd.concat(out, ignore_index=True) if out else
          pd.DataFrame(columns=["job_id", "window_idx", "start_s",
                                "wrist_speed_mean", "wrist_speed_p90",
                                "wrist_speed_max", "n_samples"]))
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
    return df


# Minimum keyframe samples in a window before its gesture summary is trusted.
# A window holding two samples can report a large "mean speed" from a single
# noisy pair.
_MIN_GESTURE_SAMPLES = 5


def pitch_relative(corpus: str, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per-window pitch expressed **relative to each speaker** (spec §4).

    Adds to the window frame:
      pitch_st        mean_f0 in semitones from that speaker's own median
      pitch_var_st    within-window variation, in semitones
      pitch_level     low / mid / high      (tertiles within the speaker)
      pitch_variation stable / mid / variable

    ## Why semitones rather than z-scores

    Both remove the speaker's baseline, but semitones are a log scale, which
    is how pitch is perceived and how the phonetics literature reports it —
    so "+3 semitones" means the same perceptual step for a low and a high
    voice, whereas "+1 SD" depends on how variable that speaker happens to be.

    ## Why tertiles rather than fixed thresholds

    A fixed Hz cut-off would mostly sort speakers by voice type. Tertiles are
    computed **within each speaker**, so "high" means high for them.

    ## Caveat for Chinese

    Mandarin is tonal: a large share of f0 movement is lexical tone, not
    discourse style. Comparing `pitch_variation` between an English and a
    Chinese corpus therefore compares two different things, and the working
    assumption "high + variable = emotional/engaging" does not transfer across
    that boundary unexamined.

    `pitch_var_st` is derived from the stored per-window f0 SD in Hz, which is
    an approximation of a true semitone spread; it is exact only in the limit
    of small variation. Where dense contours exist (dense_prosody), computing
    it from those is strictly better.
    """
    df = load_windows(corpus) if df is None else df.copy()
    if df.empty or "mean_f0" not in df:
        return df

    med = df.groupby("job_id")["mean_f0"].transform("median")
    df["pitch_st"] = 12 * np.log2(df["mean_f0"] / med)
    # 12*log2(1 + sd/mean): the semitone distance from the window's own mean to
    # one SD above it.
    df["pitch_var_st"] = 12 * np.log2(1 + df["f0_std"] / df["mean_f0"])

    def _tertile(group, col, labels):
        try:
            return pd.qcut(group[col], 3, labels=labels)
        except (ValueError, IndexError):      # too few distinct values
            return pd.Series(np.nan, index=group.index, dtype="object")

    df["pitch_level"] = (df.groupby("job_id", group_keys=False)
                         .apply(_tertile, "pitch_st", ["low", "mid", "high"]))
    df["pitch_variation"] = (df.groupby("job_id", group_keys=False)
                             .apply(_tertile, "pitch_var_st",
                                    ["stable", "mid", "variable"]))
    return df



# ── One download point ───────────────────────────────────────────────────

_DEFAULT_CACHE_DIR = Path(os.environ.get("WORK_DIR", "/tmp/mannerism")) / "corpus_cache"

# Bumped whenever a change here alters what is cached or how it is computed, so
# that old pickles are ignored rather than silently serving superseded numbers.
# 2: word-frequency denominator switched from ASR tokens to segmented words.
# 3: per-video word rows carry video_tokens, so mean_video_freq is computed
#    per (word, video) instead of averaging the artifact's per-tag rates.
_CACHE_SCHEMA = 3


class CorpusData:
    """Everything a corpus notebook reads, fetched once.

    Built by `load_corpus`. Holds the frames rather than re-querying, so a
    notebook can be re-run cell by cell without going back to Atlas — which
    matters because the corpus lives on four free-tier clusters whose
    throughput varies by an order of magnitude hour to hour, and has twice
    been unreachable outright.

    Attributes:
      videos    one row per video
      windows   one row per 5s window: all four modalities, plus the pitch
                columns from pitch_relative and the normalised gesture
                columns merged in
      gesture   per-window body-normalised gesture on its own
      corpus    the corpus name

    Deliberately not held: spectrograms, waveforms, collocations, segmented
    tokens and raw keyframes. They are ~90% of the corpus by bytes, no
    section of the trends notebook uses them, and they stay one call away —
    load_artifacts(corpus, job_id) and wrist_speed_series(corpus, job_id).
    """

    def __init__(self, corpus, videos, windows, gesture, per_video_words):
        self.corpus, self.videos, self.windows, self.gesture = (
            corpus, videos, windows, gesture,
        )
        self._per_video_words = per_video_words

    def wordlist(self, pos: Optional[list[str]] = None, by_pos: bool = True) -> pd.DataFrame:
        """Corpus word frequencies, from the copy fetched by load_corpus.

        Filtering and grouping happen here rather than at fetch time, so
        asking for all words, then content words, then merged parts of
        speech costs one download and three groupbys.
        """
        per_video, totals = self._per_video_words
        return _aggregate_wordlist(per_video.copy(), totals, pos, by_pos)

    def label(self, job_id: str) -> str:
        row = self.videos.loc[self.videos.job_id == job_id, "label"]
        return row.iloc[0] if len(row) and isinstance(row.iloc[0], str) else job_id

    def __repr__(self) -> str:
        words = len(self._per_video_words[0]) if self._per_video_words else 0
        return (f"<CorpusData {self.corpus}: {len(self.videos)} videos, "
                f"{len(self.windows)} windows, {len(self.gesture)} with "
                f"normalised gesture, {words} per-video word rows>")


def load_corpus(
    corpus: str,
    *,
    gesture: bool = True,
    cache_dir: Optional[str] = None,
    refresh: bool = False,
    offline: bool = False,
    progress: bool = True,
) -> CorpusData:
    """Fetch a corpus once, with progress, and cache it on disk.

    Measured on Ted (39 videos): ~2.8 min cold — windows 37s, keyframes
    ~97s, wordlists ~16s — and ~3s warm, nearly all of which is the
    fingerprint check.

    ## Cache invalidation

    The cache key is the corpus' job ids and each one's window count, so
    adding, reprocessing or deleting a video invalidates it automatically.
    That check is the reason the warm path still contacts Atlas: a stale
    cache quietly analysing a deleted video is a worse failure than a slow
    notebook. `offline=True` skips it and trusts whatever is on disk — for
    working through an outage, not for producing final numbers.
    `refresh=True` forces a re-fetch.
    """
    from tqdm.auto import tqdm

    cache_root = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    steps = 3 + (1 if gesture else 0)          # windows, pitch, words [, gesture]
    bar = tqdm(total=steps, desc=f"loading {corpus}", disable=not progress,
               bar_format="{desc}: {n_fmt}/{total_fmt} |{bar}| {postfix}")

    videos = load_videos(corpus)
    if videos.empty:
        bar.close()
        return CorpusData(corpus, videos, pd.DataFrame(), pd.DataFrame(),
                          (pd.DataFrame(), []))

    cache_file = None
    if not offline:
        counts = _each_shard(
            lambda db: list(db[f"{corpus}_fused_windows"].aggregate(
                [{"$group": {"_id": "$job_id", "n": {"$sum": 1}}}])),
            f"{corpus}'s window counts",
        )
        fingerprint = str(sorted((c["_id"], c["n"]) for shard in counts for c in shard))
        import hashlib
        key = hashlib.sha1(
            f"{corpus}|{fingerprint}|{gesture}|v{_CACHE_SCHEMA}".encode()
        ).hexdigest()[:16]
        cache_file = cache_root / f"{corpus}_{key}.pkl"
    else:
        existing = sorted(cache_root.glob(f"{corpus}_*.pkl"))
        if not existing:
            raise RuntimeError(f"offline=True but no cached copy of {corpus} in {cache_root}")
        cache_file = existing[-1]

    if cache_file.exists() and not refresh:
        bar.set_postfix_str("cache hit"); bar.update(steps); bar.close()
        windows, gest, words = pd.read_pickle(cache_file)
        if progress:
            print(f"loaded from cache: {cache_file}"
                  + ("   (offline — not checked against MongoDB)" if offline else ""))
        return CorpusData(corpus, videos, windows, gest, words)

    windows = load_windows(corpus)
    bar.set_postfix_str("windows"); bar.update(1)

    windows = pitch_relative(corpus, windows)
    bar.set_postfix_str("pitch"); bar.update(1)

    words = _fetch_per_video_words(corpus, videos)
    bar.set_postfix_str("wordlists"); bar.update(1)

    gest = pd.DataFrame()
    if gesture:
        rows = []
        for job_id in tqdm(list(videos.job_id), desc="  keyframes", leave=False,
                           disable=not progress):
            sp = wrist_speed_series(corpus, job_id).dropna(subset=["speed"])
            if sp.empty:
                continue
            idx = (sp.ts // 5.0).astype(int)
            g = sp.groupby(idx)["speed"]
            part = pd.DataFrame({
                "window_idx": g.mean().index.astype(int),
                "wrist_speed_mean": g.mean().to_numpy(),
                "wrist_speed_p90": g.quantile(0.9).to_numpy(),
                "wrist_speed_max": g.max().to_numpy(),
                "n_samples": g.size().to_numpy(),
            })
            part["job_id"] = job_id
            rows.append(part)
        if rows:
            gest = pd.concat(rows, ignore_index=True)
            gest = gest[gest.n_samples >= _MIN_GESTURE_SAMPLES]
            windows = windows.merge(gest, on=["job_id", "window_idx"], how="left")
        bar.set_postfix_str("gesture"); bar.update(1)

    bar.close()
    cache_root.mkdir(parents=True, exist_ok=True)
    pd.to_pickle((windows, gest, words), cache_file)
    if progress:
        print(f"cached to {cache_file}")
    return CorpusData(corpus, videos, windows, gest, words)
