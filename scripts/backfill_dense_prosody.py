"""
scripts/backfill_dense_prosody.py
----------------------------------
Compute and store the f0 and intensity contours at their native ~10ms hop,
for videos already shipped to MongoDB.

    # one corpus, skipping videos that already have contours
    uv run python scripts/backfill_dense_prosody.py --collection Ted

    # a single video, recomputing even if present
    uv run python scripts/backfill_dense_prosody.py --collection Ted \
        --job-id 1a39b362 --force

## Why this exists

Every prosody number in the corpus is a **5-second window mean**. A 5s
window spans roughly 15 words, so no window-level statistic can answer a
word-level question — "what was her pitch while she said *rights*" — which
is precisely what the multimodal search bar specifies
(MultiArth_Search_Bar_QUANTITATIVE.docx section 7.1: the pitch window is
the word's own start/end plus a 150-200ms margin). The same missing signal
blocks the sub-second gesture-speech cross-correlation in
notebooks/04_multimodal_coupling.ipynb section 7.

ProsodyWorker computes these contours already, at time_step=0.01 with a
75-500Hz range, and then throws them away after aggregating. This script
recomputes them with **identical Praat settings**, so a window mean derived
from the stored contour reproduces the number already in the corpus — which
`--verify` checks rather than assumes.

## Where the audio comes from

Bulk processing deletes a video and its extracted WAV once the results are
shipped, so most of the corpus has no local audio any more. Resolution
order per video:

  1. `WORK_DIR/<stem>_audio.wav`          — already extracted, free
  2. the local video file, if it exists   — extract audio with ffmpeg
  3. `drive_url` from the video document  — download, extract, then delete
     the video again (keep it with `--keep-video`)

Step 3 needs GOOGLE_DRIVE_API_KEY and the file shared "anyone with the
link", same as the bulk flow.

## Cost

Praat pitch tracking runs around 20-40x realtime, so a 13-minute talk takes
roughly 30 seconds once the audio is local; downloading dominates. Storage
is ~310KB per contour (float32 binary), i.e. ~30MB for 49 videos.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import parselmouth  # noqa: E402
from loguru import logger  # noqa: E402

from core.drive_download import download_drive_file  # noqa: E402
from core.preprocessing import extract_audio  # noqa: E402
from core.results_repository import ResultsRepository  # noqa: E402

# Identical to ProsodyWorker._process_window. Changing any of these makes the
# contours disagree with the stored window means, which --verify would then
# (correctly) flag.
_TIME_STEP_S = 0.01
_PITCH_FLOOR_HZ = 75.0
_PITCH_CEILING_HZ = 500.0


def _resolve_audio(doc: dict, work_dir: Path, keep_video: bool) -> tuple[str, list[Path]]:
    """Returns (wav_path, paths_to_clean_up_afterwards)."""
    video_path = doc.get("video_path") or ""
    cached = work_dir / (Path(video_path).stem + "_audio.wav")
    if video_path and cached.exists():
        return str(cached), []

    if video_path and Path(video_path).exists():
        return extract_audio(video_path, str(work_dir)), []

    drive_url = doc.get("drive_url")
    if not drive_url:
        raise FileNotFoundError(
            f"no local audio, no local video, and no drive_url for {doc['_id']}"
        )
    dest = work_dir / f"{doc['_id']}_backfill.mp4"
    logger.info(f"[prosody-backfill] downloading {drive_url}")
    _download_with_backoff(drive_url, dest)
    wav = extract_audio(str(dest), str(work_dir))
    return wav, ([] if keep_video else [dest])


# Drive throttles, and a run of back-to-back downloads reads as automated
# traffic — observed in practice: three files refused mid-run. Retries wait
# minutes rather than seconds because a refusal is a rate limit, not a blip,
# and hammering it extends the block.
_DOWNLOAD_ATTEMPTS = 3
_BACKOFF_S = (60, 300)


def _download_with_backoff(drive_url: str, dest: Path) -> None:
    last = None
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            download_drive_file(drive_url, str(dest))
            return
        except Exception as exc:
            last = exc
            if attempt == _DOWNLOAD_ATTEMPTS:
                break
            wait = _BACKOFF_S[min(attempt - 1, len(_BACKOFF_S) - 1)]
            print(f"    download refused ({str(exc)[:70]}); "
                  f"waiting {wait}s before attempt {attempt + 1}")
            time.sleep(wait)
    raise RuntimeError(f"download failed after {_DOWNLOAD_ATTEMPTS} attempts: {last}")


def _contours(wav_path: str) -> tuple[np.ndarray, np.ndarray, float]:
    """f0 (NaN where unvoiced) and intensity in dB, on a common hop grid.

    Praat's pitch and intensity objects do not share a time base — intensity
    frames are centred differently and the object starts later — so the
    intensity contour is resampled onto the pitch grid rather than assumed
    to line up. Getting this wrong would shift intensity against f0 by tens
    of milliseconds, which at word scale is a real error.
    """
    sound = parselmouth.Sound(wav_path)
    pitch = sound.to_pitch(
        time_step=_TIME_STEP_S,
        pitch_floor=_PITCH_FLOOR_HZ,
        pitch_ceiling=_PITCH_CEILING_HZ,
    )
    grid = pitch.xs()
    f0 = pitch.selected_array["frequency"].astype(np.float32)
    # Praat reports 0 for unvoiced frames; NaN says "not measured" instead of
    # "measured 0Hz", so means and plots skip them.
    f0[f0 <= 0] = np.nan

    intensity = sound.to_intensity(time_step=_TIME_STEP_S)
    db = np.interp(
        grid, intensity.xs(), intensity.values.T.flatten(),
        left=np.nan, right=np.nan,
    ).astype(np.float32)
    return f0, db, float(grid[0] if len(grid) else 0.0)


def _verify(repo: ResultsRepository, collection: str, job_id: str,
            f0: np.ndarray, hop_s: float, t0: float) -> str:
    """Rebuild each window's mean_f0 from the contour and compare with what
    the corpus already stores. A mismatch means the contour is not the same
    signal the rest of the corpus was computed from."""
    windows = repo.get_all_fused(collection, job_id)
    stored, rebuilt = [], []
    for w in windows:
        if not w.prosody or w.prosody.mean_f0 is None:
            continue
        i0 = int(round((w.window.start_s - t0) / hop_s))
        i1 = int(round((w.window.end_s - t0) / hop_s))
        seg = f0[max(0, i0):max(0, i1)]
        seg = seg[~np.isnan(seg)]
        if len(seg) == 0:
            continue
        stored.append(w.prosody.mean_f0)
        rebuilt.append(float(np.mean(seg)))
    if len(stored) < 5:
        return "too few windows to verify"
    stored_a, rebuilt_a = np.array(stored), np.array(rebuilt)
    r = float(np.corrcoef(stored_a, rebuilt_a)[0, 1])
    mad = float(np.mean(np.abs(stored_a - rebuilt_a)))
    flag = "" if (r > 0.98 and mad < 5.0) else "   <-- CHECK: contour disagrees with stored windows"
    return (f"verify: r={r:.4f} mean|diff|={mad:.2f}Hz over {len(stored)} windows{flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("##")[0].strip())
    ap.add_argument("--collection", required=True, help="corpus, e.g. Ted")
    ap.add_argument("--job-id", action="append", default=[],
                    help="only these jobs; repeatable (default: the whole corpus)")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if contours are already stored")
    ap.add_argument("--limit", type=int, default=None, help="stop after N videos")
    ap.add_argument("--work-dir", default=os.environ.get("WORK_DIR", "/tmp/mannerism"))
    ap.add_argument("--keep-video", action="store_true",
                    help="keep any video downloaded from Drive")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the window-mean reconciliation check")
    ap.add_argument("--sleep", type=float, default=20.0, metavar="SECONDS",
                    help="pause between videos that needed a Drive download "
                         "(default 20, jittered +/-50%%). Downloading dozens of "
                         "files back to back is what Drive flags as bot "
                         "traffic; 0 disables the pause")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    repo = ResultsRepository()

    videos = repo.list_videos(args.collection)
    if args.job_id:
        videos = [v for v in videos if v["_id"] in set(args.job_id)]
    if not videos:
        raise SystemExit(f"no videos matched in {args.collection}")

    todo = [v for v in videos
            if args.force or not repo.has_dense_prosody(args.collection, v["_id"])]
    skipped = len(videos) - len(todo)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{args.collection}: {len(videos)} videos, {skipped} already done, "
          f"{len(todo)} to process\n")

    ok, failed = 0, []
    for i, doc in enumerate(todo, 1):
        job_id = doc["_id"]
        label = (doc.get("label") or job_id)[:50]
        print(f"[{i}/{len(todo)}] {job_id}  {label}")
        started = time.time()
        cleanup: list[Path] = []
        try:
            wav, cleanup = _resolve_audio(doc, work_dir, args.keep_video)
            f0, db, t0 = _contours(wav)
            shard = repo.put_dense_prosody(
                args.collection, job_id,
                hop_s=_TIME_STEP_S, f0=f0, intensity_db=db,
                params={
                    "pitch_floor_hz": _PITCH_FLOOR_HZ,
                    "pitch_ceiling_hz": _PITCH_CEILING_HZ,
                    "time_step_s": _TIME_STEP_S,
                    "first_sample_s": t0,
                    "source": "parselmouth/praat, matched to ProsodyWorker",
                },
            )
            voiced = float(np.mean(~np.isnan(f0))) * 100
            print(f"    {len(f0)} samples, {voiced:.0f}% voiced, "
                  f"{len(f0) * 8 / 1e6:.1f}MB -> {shard.split('.')[0]}  "
                  f"({time.time() - started:.0f}s)")
            if not args.no_verify:
                print("    " + _verify(repo, args.collection, job_id, f0,
                                       _TIME_STEP_S, t0))
            ok += 1
        except Exception as exc:
            print(f"    FAILED: {exc}")
            failed.append((job_id, str(exc)))
        finally:
            for path in cleanup:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        # Only pace the downloads — videos whose audio was already local cost
        # Drive nothing and should not be slowed down.
        if args.sleep and cleanup and i < len(todo):
            pause = args.sleep * random.uniform(0.5, 1.5)
            print(f"    pausing {pause:.0f}s before the next download")
            time.sleep(pause)

    print(f"\ndone: {ok} stored, {len(failed)} failed")
    for job_id, exc in failed:
        print(f"   {job_id}: {exc[:100]}")


if __name__ == "__main__":
    main()
