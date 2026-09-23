"""
scripts/backfill_verbal.py
--------------------------
Recompute the three verbal quantities that were computed wrongly, for videos
already shipped to MongoDB.

    # see what would change, touching nothing
    uv run python scripts/backfill_verbal.py --collection Yixi --dry-run

    # apply it
    uv run python scripts/backfill_verbal.py --collection Yixi

    # one video
    uv run python scripts/backfill_verbal.py --collection Yixi --job-id b56f7756

## What it fixes

1. **`verbal.word_count`** — was the number of raw ASR tokens. SenseVoice
   emits roughly one token per Han character, so the field meant "words" in
   English and "characters" in Chinese under one name (234k against 130k on
   Yixi). Chinese is now counted against spaCy's segmentation. English is
   left alone: there the ASR's tokens already are words, and re-counting them
   against spaCy would only split contractions and count punctuation.

2. **`prosody.speech_rate_syl_per_s`** — was `word_count / duration * 1.5`,
   where 1.5 is an assumed syllables-per-word ratio for English. Applied to a
   count that was already syllabic, it inflated Mandarin by about 1.66x
   (stored median 7.2 syl/s against a counted 4.4). Now Chinese counts its Han
   characters and everything else keeps the English estimate.

3. **`artifacts.wordlist`** — recorded the part-of-speech tag of a word's
   *first* occurrence in a video and filed every later occurrence under it, so
   a tag was a sample of one token. The symptom: `to` came out PART in 30
   videos and ADP in 9, with no video showing both, although nearly every talk
   uses both. Counts were never wrong, only their attribution. Now the list is
   keyed by (word, POS), so one word can have several rows.

The same fixes are in workers/verbal_worker.py and core/fusion_engine.py, so
anything processed from now on is already correct. This script exists for the
corpus that was processed before them — and running it matters most where a
collection will be *added to*, since a corpus half-corrected is worse than one
consistently wrong: every per-corpus statistic would mix two definitions.

## Why no audio is needed

Everything is recomputed from what MongoDB already holds. Each fused window
stores `verbal.tokens` — the ASR's own output with word-level timestamps — so
the transcript can be reassembled exactly as the worker first saw it and put
back through spaCy. Nothing is re-transcribed, so results are deterministic
and no video is downloaded.

## Cost

spaCy over a whole transcript, once per video: a few seconds each, dominated
by the MongoDB round trips. Yixi's 26 videos take a couple of minutes.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from unittest.mock import MagicMock  # noqa: E402

from core.models import WordToken  # noqa: E402
from core.results_repository import ResultsRepository  # noqa: E402
from workers.verbal_worker import _LOGOGRAPHIC, VerbalWorker  # noqa: E402

# Same expression as core/fusion_engine._HAN_CHAR — duplicated rather than
# imported so that a change there is a deliberate change in both places.
_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")

# A transcript with at least this share of Han characters is Chinese. Language
# is inferred rather than read back, because the shipped documents do not
# record the code Whisper detected. The gap between a Mandarin talk (~99% Han)
# and an English one (0%) is wide enough that the exact threshold is not a
# judgement call.
_HAN_SHARE_FOR_ZH = 0.2


def _worker() -> VerbalWorker:
    """A VerbalWorker with spaCy but without Whisper.

    Constructing one normally loads a Whisper model, which this script never
    uses: it re-runs the spaCy half of the pipeline over stored tokens. Going
    through __new__ keeps the real _build_transcript_doc / _compute_corpus_stats
    — the point is to reproduce the worker's behaviour exactly, not to
    reimplement it here and let the two drift.
    """
    worker = VerbalWorker.__new__(VerbalWorker)
    worker.store = MagicMock()
    worker._nlp_cache = {}
    return worker


def _language_of(text: str) -> str:
    stripped = "".join(text.split())
    if not stripped:
        return "en"
    return "zh" if len(_HAN.findall(stripped)) / len(stripped) >= _HAN_SHARE_FOR_ZH else "en"


def _collect(repo: ResultsRepository, collection: str, job_id: str):
    """(all_tokens, windows) rebuilt from storage, in window order."""
    fused = repo.get_all_fused(collection, job_id)
    all_tokens: list[WordToken] = []
    windows = []
    for idx, w in enumerate(fused):
        start, end = w.window.start_s, w.window.end_s
        tokens = list(w.verbal.tokens) if w.verbal else []
        all_tokens.extend(tokens)
        windows.append({
            "idx": idx,
            "start": start,
            "end": end,
            "transcript": (w.verbal.transcript if w.verbal else "") or "",
            "old_word_count": w.verbal.word_count if w.verbal else None,
            "old_rate": w.prosody.speech_rate_syl_per_s if w.prosody else None,
            "has_prosody": w.prosody is not None,
        })
    all_tokens.sort(key=lambda t: t.start_s)
    return all_tokens, windows


def _recompute(worker: VerbalWorker, all_tokens, windows, lang):
    """New word counts, speech rates and word list. Raises if spaCy is absent."""
    doc, join_sep = worker._build_transcript_doc(all_tokens, lang)
    if doc is None:
        raise RuntimeError(
            "spaCy produced no doc — model missing, or the transcript exceeds "
            "max_length. Nothing written."
        )
    segmented = worker._segmented_or_empty(doc, all_tokens, join_sep)
    if not segmented:
        raise RuntimeError("token segmentation produced nothing. Nothing written.")

    # Mirrors process_job: only logographic scripts re-count words against
    # spaCy. In a space-separated language the ASR's tokens already are words,
    # and spaCy would merely split contractions and count punctuation.
    recount = lang in _LOGOGRAPHIC

    updates: dict[int, dict] = {}
    rates = []
    for w in windows:
        fields = {}
        if recount:
            count = sum(1 for s in segmented if w["start"] <= s["start_s"] < w["end"])
            if count != w["old_word_count"]:
                fields["verbal.word_count"] = count
        else:
            count = w["old_word_count"] or 0

        duration = w["end"] - w["start"]
        if w["has_prosody"] and duration > 0:
            han = len(_HAN.findall(w["transcript"]))
            rate = han / duration if han else count / duration * 1.5
            rates.append((w["old_rate"], rate))
            if w["old_rate"] is None or abs(rate - w["old_rate"]) > 1e-9:
                fields["prosody.speech_rate_syl_per_s"] = rate
        if fields:
            updates[w["idx"]] = fields

    wordlist, _, _ = worker._compute_corpus_stats(all_tokens, lang, doc)
    return updates, wordlist, len(segmented), rates, recount


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("##")[0].strip())
    ap.add_argument("--collection", required=True, help="corpus, e.g. Yixi")
    ap.add_argument("--job-id", action="append", default=[],
                    help="only these jobs; repeatable (default: the whole corpus)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N videos")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    repo = ResultsRepository()
    videos = repo.list_videos(args.collection)
    if args.job_id:
        wanted = set(args.job_id)
        videos = [v for v in videos if v["_id"] in wanted]
        missing = wanted - {v["_id"] for v in videos}
        if missing:
            raise SystemExit(f"not in {args.collection}: {', '.join(sorted(missing))}")
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        raise SystemExit(f"no videos matched in {args.collection}")

    worker = _worker()
    print(f"{args.collection}: {len(videos)} video(s)"
          + ("   [DRY RUN — nothing will be written]" if args.dry_run else "") + "\n")

    ok, failed = 0, []
    for i, doc_v in enumerate(videos, 1):
        job_id = doc_v["_id"]
        label = (doc_v.get("label") or job_id)[:44]
        print(f"[{i}/{len(videos)}] {job_id}  {label}")
        try:
            all_tokens, windows = _collect(repo, args.collection, job_id)
            if not all_tokens:
                print("    no stored tokens — skipped")
                continue
            lang = _language_of("".join(t.word for t in all_tokens))
            updates, wordlist, n_seg, rates, recount = _recompute(
                worker, all_tokens, windows, lang)

            old_total = sum(w["old_word_count"] or 0 for w in windows)
            print(f"    lang={lang}  ASR tokens {len(all_tokens)}, spaCy words {n_seg}"
                  f"   ({len(wordlist['words'])} word-POS rows)")

            # Medians over the same windows on both sides, so the comparison
            # is of one population before and after rather than of all windows
            # against only the ones that moved.
            paired = [(o, n) for o, n in rates if o is not None]
            if paired:
                med_old = sorted(o for o, _ in paired)[len(paired) // 2]
                med_new = sorted(n for _, n in paired)[len(paired) // 2]
                print(f"    speech rate median {med_old:.2f} -> {med_new:.2f} syl/s"
                      f"  ({len(paired)} windows)")
            if recount:
                print(f"    word_count total {old_total} -> {n_seg}")
            else:
                print(f"    word_count unchanged ({old_total}) — "
                      f"ASR tokens are already words in {lang}")
            print(f"    {len(updates)} of {len(windows)} windows change")

            if args.dry_run:
                ok += 1
                continue

            modified = repo.update_window_fields(args.collection, job_id, updates)
            repo.update_wordlist(args.collection, job_id, wordlist)
            print(f"    written: {modified} windows + word list")
            ok += 1
        except Exception as exc:
            print(f"    FAILED: {exc}")
            failed.append((job_id, str(exc)))

    print(f"\ndone: {ok} processed, {len(failed)} failed")
    for job_id, exc in failed:
        print(f"   {job_id}: {exc[:110]}")
    if not args.dry_run and ok:
        print("\nThe notebooks' corpus cache is keyed on job ids and window counts, "
              "neither of which changed — re-run load_corpus(..., refresh=True) "
              "once to pick these values up.")


if __name__ == "__main__":
    main()
