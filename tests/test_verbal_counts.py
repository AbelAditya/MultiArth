"""
tests/test_verbal_counts.py
---------------------------
Regression tests for three quantities that were wrong in a language-specific
way, and whose wrongness was invisible in English.

The common cause: the ASR's tokens are words in English but roughly characters
in Chinese, so anything that counted them meant two different things under one
name. Each test below pins one of the fixes, in both languages, because a fix
that only holds for English is how this happened in the first place.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.fusion_engine import FusionEngine
from core.models import (
    FusedWindow, ProsodyFeatures, TimeWindow, VerbalFeatures, WordToken,
)
from workers.verbal_worker import VerbalWorker

# A Mandarin sentence the recogniser would emit one character at a time, and
# which pkuseg segments into clearly fewer, real words.
ZH_CHARS = list("我们为什么要讨论女性的权利问题因为这个社会还不平等")
EN_WORDS = "we should ask why the workplace still treats women differently".split()


def _tokens(items, step):
    return [
        WordToken(word=w, start_s=i * step, end_s=(i + 1) * step, confidence=0.9)
        for i, w in enumerate(items)
    ]


@pytest.fixture(scope="module")
def worker():
    """A VerbalWorker without Whisper — every method under test needs only spaCy."""
    w = VerbalWorker.__new__(VerbalWorker)
    w.store = MagicMock()
    w._nlp_cache = {}
    return w


def _segmented(worker, tokens, lang):
    doc, sep = worker._build_transcript_doc(tokens, lang)
    return doc, worker._segmented_or_empty(doc, tokens, sep)


# ── word_count means "words", in every language ──────────────────────────

def test_chinese_word_count_is_words_not_characters(worker):
    tokens = _tokens(ZH_CHARS, 0.2)
    _, seg = _segmented(worker, tokens, "zh")

    features = worker._process_window(0.0, 10.0, tokens, seg)

    # The whole point: fewer words than characters, because pkuseg merges
    # characters into words. Counting the ASR's tokens gave len(ZH_CHARS).
    assert features.word_count == len(seg)
    assert features.word_count < len(tokens)


def test_english_word_count_stays_on_asr_tokens(worker):
    """English keeps the ASR's count; only logographic scripts re-count.

    spaCy splits contractions and hyphenated compounds and emits punctuation
    as separate tokens, so re-counting English against it would inflate the
    number without disagreeing about how many words were spoken.
    """
    tokens = _tokens(EN_WORDS, 0.4)

    features = worker._process_window(0.0, 10.0, tokens, segmented=None)

    assert features.word_count == len(EN_WORDS)


def test_english_punctuation_does_not_inflate_word_count(worker):
    """The case that decided it: realistic ASR output with punctuation glued on."""
    asr = ["So,", "I", "don't", "think", "well-known", "leaders", "are", "right."]
    tokens = _tokens(asr, 0.4)
    _, seg = _segmented(worker, tokens, "en")

    # spaCy would make this 13 — "So" "," "I" "do" "n't" ... "right" "."
    assert len(seg) > len(asr)

    features = worker._process_window(0.0, 10.0, tokens, segmented=None)
    assert features.word_count == len(asr)


def test_word_count_falls_back_to_asr_tokens_without_spacy(worker):
    """No spaCy model means the old, ambiguous count rather than zero."""
    tokens = _tokens(EN_WORDS, 0.4)

    features = worker._process_window(0.0, 10.0, tokens, segmented=[])

    assert features.word_count == len(EN_WORDS)


def test_word_count_respects_window_bounds(worker):
    """Words are assigned to windows by their own start time."""
    tokens = _tokens(ZH_CHARS, 1.0)          # one character per second
    _, seg = _segmented(worker, tokens, "zh")

    first = worker._process_window(0.0, 5.0, tokens[:5], seg)
    second = worker._process_window(5.0, 10.0, tokens[5:10], seg)

    assert first.word_count == sum(1 for s in seg if 0.0 <= s["start_s"] < 5.0)
    assert second.word_count == sum(1 for s in seg if 5.0 <= s["start_s"] < 10.0)
    assert first.word_count > 0 and second.word_count > 0


def test_only_logographic_languages_recount_against_spacy():
    """The language switch itself, at the point process_job applies it."""
    from workers.verbal_worker import _LOGOGRAPHIC

    assert "zh" in _LOGOGRAPHIC
    assert "en" not in _LOGOGRAPHIC


# ── the word list keys on (word, POS), not on word alone ─────────────────

def test_wordlist_separates_parts_of_speech(worker):
    """A word used two ways gets two rows, with the counts split correctly.

    Previously the tag came from the word's first occurrence and every later
    occurrence was filed under it, so this produced a single row claiming all
    three uses were whichever came first.
    """
    words = "we can watch the watch and then watch it again".split()
    tokens = _tokens(words, 0.4)
    doc, _ = _segmented(worker, tokens, "en")

    wordlist, _, _ = worker._compute_corpus_stats(tokens, "en", doc)
    rows = {(e["word"], e["pos"]): e["count"]
            for e in wordlist["words"] if e["word"] == "watch"}

    assert rows == {("watch", "VERB"): 2, ("watch", "NOUN"): 1}


def test_wordlist_counts_every_token_exactly_once(worker):
    """Splitting by POS must not change the totals, only their attribution."""
    words = "we can watch the watch and then watch it again".split()
    tokens = _tokens(words, 0.4)
    doc, _ = _segmented(worker, tokens, "en")

    wordlist, _, _ = worker._compute_corpus_stats(tokens, "en", doc)
    total = sum(e["count"] for e in wordlist["words"])

    assert total == sum(1 for t in doc if not t.is_punct and t.text.isalpha())


# ── speech rate: counted for Chinese, estimated for English ──────────────

def _rate(transcript: str, word_count: int, duration: float = 5.0) -> float:
    window = TimeWindow(start_s=0.0, end_s=duration)
    fused = FusedWindow(
        window=window,
        verbal=VerbalFeatures(window=window, transcript=transcript,
                              tokens=[], word_count=word_count),
        prosody=ProsodyFeatures(window=window, mean_f0=None, f0_range=None,
                                f0_std=None, mean_intensity_db=0.0,
                                intensity_range_db=0.0,
                                speech_rate_syl_per_s=None),
    )
    FusionEngine._enrich(FusionEngine.__new__(FusionEngine), fused)
    return fused.prosody.speech_rate_syl_per_s


def test_chinese_speech_rate_counts_han_characters():
    # Stored transcripts join the ASR's per-character tokens with spaces; the
    # count must not be thrown off by them.
    transcript = " ".join(ZH_CHARS)

    rate = _rate(transcript, word_count=14, duration=5.0)

    assert rate == pytest.approx(len(ZH_CHARS) / 5.0)
    # And specifically NOT the old word_count * 1.5, in either reading.
    assert rate != pytest.approx(14 / 5.0 * 1.5)
    assert rate != pytest.approx(len(ZH_CHARS) / 5.0 * 1.5)


def test_english_speech_rate_keeps_the_syllables_per_word_estimate():
    rate = _rate(" ".join(EN_WORDS), word_count=len(EN_WORDS), duration=5.0)

    assert rate == pytest.approx(len(EN_WORDS) / 5.0 * 1.5)


def test_chinese_speech_rate_is_independent_of_word_count():
    """Syllables come from the transcript's characters, never from word_count.

    The two fixes must stay decoupled: word_count now holds spaCy words for
    Chinese, which are far fewer than the syllables spoken, so a speech rate
    derived from it would be wrong in the opposite direction from the bug it
    replaced.
    """
    transcript = " ".join(ZH_CHARS)
    expected = len(ZH_CHARS) / 5.0

    # Same transcript, wildly different word counts — the rate must not move.
    assert _rate(transcript, word_count=14) == pytest.approx(expected)
    assert _rate(transcript, word_count=25) == pytest.approx(expected)
    assert _rate(transcript, word_count=0) == pytest.approx(expected)


def test_mixed_transcript_with_any_han_is_treated_as_chinese():
    """One Latin brand name in a Mandarin talk must not flip the method."""
    transcript = " ".join(ZH_CHARS) + " TED"

    rate = _rate(transcript, word_count=15, duration=5.0)

    assert rate == pytest.approx(len(ZH_CHARS) / 5.0)
