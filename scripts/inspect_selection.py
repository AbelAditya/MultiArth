"""
scripts/inspect_selection.py
-----------------------------
Selection-only inspection mode: who would this job's parameters pick, and
why — without running pose.

    uv run python scripts/inspect_selection.py VIDEO --job-id JOB \
        --start 130 --end 135 --set tie_break=vertical --out /tmp/sel

## Why this exists

Reprocessing a video to see the effect of a threshold takes hours, nearly
all of it MediaPipe. Selection needs none of it: detection, re-ID and the
tie-break decide who gets tracked, and pose only draws a skeleton on the
box that selection already chose. Cutting pose out turns "try a parameter"
from an overnight job into seconds per scene, which is what makes tuning
per video (workers/_gesture_params.py) practical rather than theoretical.

It also answers *why* a frame went wrong, which stored results cannot: the
output keeps every candidate, its detector confidence, its gallery score
and its tie-break key — not just the winner.

## The two modes

  per-frame (default)   Runs the frame-by-frame selection rule: every
                        candidate embedded and scored, the winner chosen
                        among those over the threshold. This is the
                        `_gallery_match` path, minus the lock/continuity
                        state machine, so it shows acquisition decisions.

  --scene               Runs the real per-scene mechanism instead
                        (`_compute_scene_decision`): boxes every frame,
                        links tracklets, scores each by sampled crops, and
                        applies the tie-break once for the whole range.
                        Use this for a scene where the track stuck to the
                        wrong person for its whole duration — that is the
                        signature of this path, not the per-frame one.

                        `--scene` alone treats [start, end) as ONE scene,
                        which is only faithful if you passed the real scene
                        bounds. Add `--auto-scenes` to detect cuts the way
                        the pipeline does and evaluate each scene inside
                        the range separately — a range spanning a cut
                        otherwise pools tracklets from different shots,
                        and both the scores and the support counts then
                        describe something the pipeline never computes.

## The rule for using it

Judge a parameter by whether the box lands on the right person, never by
what it does to a downstream gesture statistic — see the tuning note in
workers/_gesture_params.py.

## Gallery

`--job-id` reads the job's gallery from Redis, which is where galleries
live for 24h after a job runs. Past that they are gone, and a rebuilt
gallery will not reproduce the original decisions exactly — you will be
reproducing the failure mode, not the run. `--gallery-npz` loads one saved
by an earlier invocation of this script (it always saves the gallery it
used, so a diagnosis stays repeatable).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.feature_store import FeatureStore  # noqa: E402
from core.preprocessing import frames_for_window, probe_video  # noqa: E402
from workers import _detector, _reid  # noqa: E402
from workers._gesture_params import GestureParams  # noqa: E402
from workers.gesture_worker import (  # noqa: E402
    GestureWorker,
    _associate,
    _box_center,
    _gallery_passing_tracks,
    _scene_bounds,
    _scene_index,
    _select_tracklet,
)

# BGR, OpenCV order.
_CHOSEN = (0, 220, 0)       # green — what the pipeline would track
_PASSED = (0, 200, 255)     # amber — cleared the gallery, lost the tie-break
_REJECTED = (120, 120, 120)  # grey — detected, failed the gallery


def _parse_overrides(pairs: list[str]) -> dict:
    """`--set key=value` into typed values. Everything arrives as a string,
    so ints stay ints and floats stay floats rather than silently becoming
    strings that compare wrong."""
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        try:
            out[key.strip()] = int(raw)
        except ValueError:
            try:
                out[key.strip()] = float(raw)
            except ValueError:
                out[key.strip()] = raw.strip()
    return out


def _load_gallery(args) -> np.ndarray:
    if args.gallery_npz:
        return np.load(args.gallery_npz)["gallery"]
    if not args.job_id:
        raise SystemExit("need --job-id (gallery from Redis) or --gallery-npz")
    entries = FeatureStore().get_gallery(args.job_id)
    if not entries:
        raise SystemExit(
            f"No gallery for job {args.job_id} in Redis — galleries expire after "
            "24h. Use --gallery-npz, or rebuild one in the dashboard (note that a "
            "rebuilt gallery will not reproduce the original run exactly)."
        )
    return np.array([e.embedding for e in entries], dtype=np.float32)


def _draw(bgr, box, colour, label, thickness=2):
    x0, y0, x1, y1 = box
    cv2.rectangle(bgr, (x0, y0), (x1, y1), colour, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(bgr, (x0, max(0, y0 - th - 6)), (x0 + tw + 4, y0), colour, -1)
    cv2.putText(bgr, label, (x0 + 2, max(th, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def run_per_frame(worker, meta, gallery, args, out_dir) -> list[dict]:
    """Acquisition decision on each sampled frame, independently."""
    params = worker.params
    key = params.tie_break_key
    rows = []
    # Per-frame mode embeds every candidate on every frame, so it keeps a
    # cap by default — unlike the scene pass, which must mirror the worker.
    for ts, bgr in frames_for_window(
        meta.path, args.start, args.end, meta.fps,
        max_frames=args.max_frames if args.max_frames is not None else 300,
    ):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dets = _detector.detect_people(
            worker._detector, rgb, conf_threshold=params.conf_threshold
        )

        scored = []
        for i, det in enumerate(dets):
            emb = worker._embed_candidate(rgb, det)
            score = None if emb is None else float(_reid.max_similarity(emb, gallery))
            centre = _box_center(det.box, meta.width, meta.height)
            scored.append({
                "i": i, "box": [int(v) for v in det.box], "conf": float(det.score),
                "gallery": score, "centre": [round(c, 3) for c in centre],
                "tie_key": round(key(centre), 3),
            })

        passers = [c for c in scored
                   if c["gallery"] is not None
                   and c["gallery"] > params.gallery_match_threshold]
        chosen = min(passers, key=lambda c: (c["tie_key"], -c["gallery"])) if passers else None

        canvas = bgr.copy()
        for c in scored:
            is_chosen = chosen is not None and c["i"] == chosen["i"]
            colour = _CHOSEN if is_chosen else (_PASSED if c in passers else _REJECTED)
            g = "—" if c["gallery"] is None else f"{c['gallery']:.2f}"
            _draw(canvas, c["box"], colour,
                  f"#{c['i']} c{c['conf']:.2f} g{g} k{c['tie_key']:+.2f}",
                  thickness=3 if is_chosen else 1)
        cv2.putText(canvas, f"t={ts:.2f}s  {len(dets)} det  {len(passers)} passed"
                            f"  tie_break={params.tie_break}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"f_{ts:08.2f}.jpg"), canvas)

        rows.append({"ts": round(ts, 2), "n_det": len(dets), "n_passed": len(passers),
                     "chosen": chosen, "candidates": scored})
        print(f"  t={ts:6.2f}s  {len(dets):>3} det  {len(passers):>2} passed  "
              + (f"chose #{chosen['i']} (gallery {chosen['gallery']:.2f}, "
                 f"key {chosen['tie_key']:+.2f})" if chosen else "nobody — empty frame"))
    return rows


def run_scene(worker, meta, gallery, args, out_dir) -> list[dict]:
    """The real per-scene mechanism, over [start, end) treated as one scene."""
    params = worker.params
    decision = worker._compute_scene_decision(meta, args.start, args.end, gallery)
    print(f"\n  ambiguous={decision.get('ambiguous')}  "
          f"(ambiguous means 2+ tracklets passed the gallery, so the whole "
          f"scene follows one chosen tracklet)")

    # Re-derive the tracklets for reporting: _compute_scene_decision keeps
    # only the winner's boxes, and the rejected ones are the interesting half.
    # Every frame, matching _compute_scene_decision's own max_frames=10**9.
    # A cap here would silently halve every tracklet's support (frames_for_window
    # downsamples to fit), so `support` would no longer be the number the
    # track_min_support floor is checked against in the real run.
    scan_cap = 10 ** 9 if args.max_frames is None else args.max_frames
    per_frame = []
    crops: dict[tuple[int, tuple], np.ndarray] = {}
    for ts, bgr in frames_for_window(
        meta.path, args.start, args.end, meta.fps, max_frames=scan_cap
    ):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dets = _detector.detect_people(
            worker._detector, rgb, conf_threshold=params.conf_threshold
        )
        fi = int(round(ts * meta.fps))
        per_frame.append((fi, [d.box for d in dets]))
        # Identity crops on a stride, exactly as _compute_scene_decision
        # samples them — scoring every detection in every frame would cost
        # a mask build per candidate and change what is being measured.
        if fi % params.track_id_stride == 0:
            for d in dets:
                crop = _reid.crop_via_mask(rgb, d.mask, d.box)
                if crop is not None:
                    crops[(fi, d.box)] = crop

    tracks = _associate(per_frame, params)
    # Two scores per track, because they routinely disagree and the
    # disagreement is the thing people trip over:
    #   t.score   what the gate uses — max over the FIRST track_id_samples
    #             crops, which is all _compute_scene_decision computes
    #   available max over EVERY sampled crop of that track
    # A track whose `available` clears the threshold while `t.score` does
    # not is one the scene gate missed for want of sampling, and it will
    # very likely be picked up frame-by-frame instead.
    available: dict[int, tuple[float, int]] = {}
    for t in tracks:
        all_crops = [crops[(fi, b)] for fi, b in sorted(t.boxes.items())
                     if (fi, b) in crops]
        if not all_crops:
            continue
        scores = [
            float(_reid.max_similarity(
                _reid.embed_crop(worker._reid_model, c), gallery))
            for c in all_crops
        ]
        t.score = max(scores[:params.track_id_samples])
        available[id(t)] = (max(scores), len(scores))
    key = params.tie_break_key
    chosen = _select_tracklet(tracks, meta.width, meta.height, params)
    # The gate has two independent criteria and they fail for different
    # reasons, so colour by identity and say "short" for the support floor:
    # a track that looks like the speaker but is 6 frames long is a very
    # different situation from one that looks like somebody else.
    gallery_ok = {id(t) for t in tracks
                  if t.score is not None
                  and t.score > params.gallery_match_threshold}
    passing = set(id(t) for t in _gallery_passing_tracks(tracks, params))
    rows = []
    missed = [t for t in tracks
              if id(t) in available
              and available[id(t)][0] > params.gallery_match_threshold
              and (t.score or 0) <= params.gallery_match_threshold]
    print(f"\n  {len(tracks)} tracklets, {len(gallery_ok)} over the gallery "
          f"threshold ({params.gallery_match_threshold}), {len(passing)} of those "
          f"also over the {params.track_min_support}-frame support floor")
    if chosen is None:
        why = ("nobody cleared the gallery" if not gallery_ok else
               f"only {len(passing)} track passed both criteria; this branch "
               "needs 2+")
        print(f"  NO GREEN BOX: no scene-level selection was made — {why}, so "
              f"the pipeline falls back to the per-frame path here. Re-run "
              f"without --scene to see what that path would choose.")
    if missed:
        print(f"  NOTE: {len(missed)} track(s) would clear the threshold on some "
              f"sampled crop, but not within the first {params.track_id_samples} "
              f"the gate looks at. Those are invisible to the scene decision and "
              f"are exactly what the per-frame path picks up instead — which is "
              f"why the processed video can show a pose here while this table "
              f"shows no passer.")
    for t in sorted(tracks, key=lambda t: -t.support):
        if t.support < 3:
            continue
        mark = "CHOSEN" if t is chosen else ""
        best, n_crops = available.get(id(t), (None, 0))
        row = {"support": t.support, "score": t.score,
               "best_sampled_score": best, "n_sampled_crops": n_crops,
               "tie_key": round(t.median_key(meta.width, meta.height, key), 3),
               "median_y": round(t.median_key(meta.width, meta.height, lambda c: c[1]), 3),
               "median_centrality": round(t.median_centre_distance(meta.width, meta.height), 3),
               "chosen": t is chosen}
        rows.append(row)
        print(f"   {t.support:>4}f  gate={('%.2f' % t.score) if t.score else '  — '}"
              f"  best/{n_crops or 0}crops={('%.2f' % best) if best else '  — '}"
              f"  key={row['tie_key']:+.3f}  median_y={row['median_y']:.3f}"
              f"  centrality={row['median_centrality']:.3f}  {mark}")

    # Annotated frames for the whole range, always — including (especially)
    # when no tracklet passed. A scene that came out unambiguous is the case
    # you most need to look at: it means the gallery rejected everyone, and
    # the numbers alone do not show you who was in frame.
    ranked = sorted(tracks, key=lambda t: -t.support)
    labels = {id(t): i for i, t in enumerate(ranked)}
    every = max(1, args.draw_every)
    written = 0
    for n, (ts, bgr) in enumerate(frames_for_window(
        meta.path, args.start, args.end, meta.fps, max_frames=scan_cap
    )):
        if n % every:
            continue
        fi = int(round(ts * meta.fps))
        canvas = bgr.copy()
        for t in ranked:
            box = t.boxes.get(fi)
            if box is None:
                continue
            is_chosen = t is chosen
            colour = _CHOSEN if is_chosen else (
                _PASSED if id(t) in gallery_ok else _REJECTED)
            score = f"{t.score:.2f}" if t.score is not None else "—"
            short = "" if id(t) in passing or id(t) not in gallery_ok else " short"
            _draw(canvas, box, colour,
                  f"t{labels[id(t)]} g{score}"
                  f" k{t.median_key(meta.width, meta.height, key):+.2f}{short}",
                  thickness=3 if is_chosen else 1)
        cv2.putText(canvas,
                    f"t={ts:.2f}s  {len(tracks)} tracklets  {len(passing)} passed"
                    f"  tie_break={params.tie_break}"
                    + ("  AMBIGUOUS" if decision.get("ambiguous") else "  per-frame path"),
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"f_{ts:08.2f}.jpg"), canvas)
        written += 1
    print(f"\n  wrote {written} annotated frames "
          f"(every {every} sampled frame{'s' if every > 1 else ''})")
    return rows


def _cuts_for(worker, meta, out_dir) -> list[float]:
    """The pipeline's own cut list, cached beside the output — detection is
    a full decode pass, and you will run this script many times over the
    same video while tuning."""
    cache = out_dir / "cuts.json"
    if cache.exists():
        cuts = json.loads(cache.read_text())
        print(f"cuts: {len(cuts)} (cached in {cache})")
        return cuts
    print("detecting scene cuts (full pass over the video)...")
    cuts = worker._detect_scene_cuts(meta.path)
    cache.write_text(json.dumps(cuts))
    print(f"cuts: {len(cuts)}")
    return cuts


def run_auto_scenes(worker, meta, gallery, args, out_dir) -> list[dict]:
    """Every scene overlapping [start, end), each judged on its own real
    bounds — what the pipeline actually does."""
    cuts = _cuts_for(worker, meta, out_dir)
    first, last = _scene_index(args.start, cuts), _scene_index(
        max(args.start, args.end - 1e-3), cuts
    )
    print(f"range covers scenes {first}..{last}")

    rows = []
    for scene_idx in range(first, last + 1):
        s_start, s_end = _scene_bounds(scene_idx, cuts, meta.duration_s)
        print(f"\n=== scene {scene_idx}: {s_start:.2f}-{s_end:.2f}s "
              f"({s_end - s_start:.2f}s)")
        scene_args = argparse.Namespace(**vars(args))
        scene_args.start, scene_args.end = s_start, s_end
        # Per-scene subdirectory: frames from different scenes would
        # otherwise interleave by timestamp and be impossible to tell apart.
        scene_dir = out_dir / f"scene_{scene_idx:04d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        for row in run_scene(worker, meta, gallery, scene_args, scene_dir):
            rows.append({"scene_idx": scene_idx,
                         "scene_start": round(s_start, 2),
                         "scene_end": round(s_end, 2), **row})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("##")[0].strip())
    ap.add_argument("video")
    ap.add_argument("--job-id", help="read this job's gallery from Redis")
    ap.add_argument("--gallery-npz", help="gallery saved by an earlier run")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--scene", action="store_true",
                    help="run the per-scene tracklet mechanism instead of per-frame")
    ap.add_argument("--auto-scenes", action="store_true",
                    help="with --scene: split the range on the pipeline's own "
                         "detected cuts and evaluate each scene separately "
                         "(a full decode pass over the video, cached in --out)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a GestureParams field; repeatable")
    ap.add_argument("--draw-every", type=int, default=5,
                    help="scene mode: write every Nth frame (default 5) — a 20s "
                         "range is ~500 frames, and you rarely need them all")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap on frames pulled from the range. Default: no cap "
                         "in --scene mode (the worker scans every frame, and a "
                         "cap would understate every tracklet's support), 300 "
                         "in per-frame mode")
    ap.add_argument("--out", default="/tmp/inspect_selection")
    args = ap.parse_args()

    params = GestureParams.from_dict(_parse_overrides(args.set))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gallery = _load_gallery(args)
    # Saved next to the output so the same decision can be re-run after the
    # Redis copy expires.
    np.savez_compressed(out_dir / "gallery.npz", gallery=gallery)

    meta = probe_video(args.video, audio_path="")
    print(f"video {Path(args.video).name}  {meta.width}x{meta.height} @ {meta.fps:.2f}fps")
    print(f"range {args.start}-{args.end}s   gallery {len(gallery)} entries")
    print("params " + json.dumps(params.to_dict(), sort_keys=True))

    worker = GestureWorker(store=None, params=params)
    worker._ensure_detector()
    worker._ensure_reid_model()

    if args.scene and args.auto_scenes:
        rows = run_auto_scenes(worker, meta, gallery, args, out_dir)
    else:
        rows = (run_scene if args.scene else run_per_frame)(
            worker, meta, gallery, args, out_dir
        )
    (out_dir / "report.json").write_text(json.dumps(
        {"video": args.video, "start": args.start, "end": args.end,
         "params": params.to_dict(), "mode": "scene" if args.scene else "per-frame",
         "rows": rows}, indent=2, default=str))
    print(f"\nwrote annotated frames + report.json to {out_dir}")


if __name__ == "__main__":
    main()
