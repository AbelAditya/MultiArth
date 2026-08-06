"""
workers/verbal_worker.py
------------------------
Faster-Whisper ASR + spaCy NLP worker.

Transcribes the full audio once (word-level timestamps), then
for each time window assigns tokens and computes linguistic features.
Language is auto-detected by Whisper; the matching spaCy model is loaded
lazily and cached so multi-language sessions don't reload unnecessarily.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import spacy
from faster_whisper import WhisperModel
from loguru import logger
from spacy.tokens import Doc

from core import corpus_analysis
from core.feature_store import FeatureStore
from core.models import TimeWindow, VerbalFeatures, WordToken
from core.preprocessing import VideoMeta

# Maps Whisper language codes to spaCy model names.
# Add entries here to support additional languages.
_SPACY_MODELS: dict[str, str] = {
    "en": "en_core_web_sm",
    "zh": "zh_core_web_sm",
}

# _split_number_glue: a digit run immediately touching non-digit content,
# in either direction ("30岁"/"5apples" = digit-prefix, "第5" = digit-suffix).
_DIGIT_PREFIX_GLUE_RE = re.compile(r"^(\d+)([^\d]+)$")
_DIGIT_SUFFIX_GLUE_RE = re.compile(r"^([^\d]+)(\d+)$")

# English suffixes that conventionally stay glued to a leading number as one
# legitimate unit (ordinals, times, multipliers) — never split apart.
# Chinese has no equivalent: every numeral+character fusion pkuseg produces
# (e.g. "30岁", "2024年") is the exact ambiguity being corrected, so its
# _compute_corpus_stats call passes an empty keep-set instead of this one.
_EN_NUMBER_GLUE_KEEP = frozenset({"am", "pm", "st", "nd", "rd", "th", "x", "k", "m", "b"})


def _split_number_glue(word: str, keep_glued: frozenset) -> Optional[tuple[str, str]]:
    """
    Shared by every language's _build_doc call: detect a token that glues a
    digit run onto adjacent non-digit content with no boundary, and split it
    into two strings at that boundary. Returns None if *word* doesn't match
    this pattern, or if the non-digit part (case-insensitive) is in
    *keep_glued* — a language-specific set of forms that conventionally stay
    attached to a number as one legitimate unit.
    """
    m = _DIGIT_PREFIX_GLUE_RE.match(word)
    if m:
        digits, content = m.group(1), m.group(2)
        if content.lower() in keep_glued:
            return None
        return digits, content

    m = _DIGIT_SUFFIX_GLUE_RE.match(word)
    if m:
        content, digits = m.group(1), m.group(2)
        if content.lower() in keep_glued:
            return None
        return content, digits

    return None


class VerbalWorker:
    def __init__(
        self,
        store: FeatureStore,
        model_size: str = "small",
        device: str = "cpu",
    ):
        self.store = store
        logger.info(f"[verbal] Loading Whisper model '{model_size}' on {device}")
        self._whisper = WhisperModel(model_size, device=device, compute_type="int8")
        self._nlp_cache: dict[str, Optional[object]] = {}

    def _get_nlp(self, lang_code: str):
        """Return a cached spaCy model for *lang_code*, loading it on first use."""
        if lang_code in self._nlp_cache:
            return self._nlp_cache[lang_code]
        model_name = _SPACY_MODELS.get(lang_code)
        if not model_name:
            logger.warning(
                f"[verbal] No spaCy model configured for language '{lang_code}' — "
                "word list will use raw counts, collocations skipped"
            )
            self._nlp_cache[lang_code] = None
            return None
        try:
            nlp = spacy.load(model_name)
            logger.info(f"[verbal] Loaded spaCy model '{model_name}' for '{lang_code}'")
            self._nlp_cache[lang_code] = nlp
            return nlp
        except OSError:
            logger.warning(
                f"[verbal] spaCy model '{model_name}' not found — "
                f"run: python -m spacy download {model_name}"
            )
            self._nlp_cache[lang_code] = None
            return None

    def process_job(
        self,
        job_id: str,
        meta: VideoMeta,
        windows: list[tuple[float, float]],
    ) -> None:
        logger.info(f"[verbal] Transcribing {meta.audio_path}")
        all_tokens, lang_code = self._transcribe(meta.audio_path)
        logger.info(
            f"[verbal] Transcription complete — {len(all_tokens)} word tokens "
            f"(language: {lang_code})"
        )

        for idx, (start, end) in enumerate(windows):
            try:
                window_tokens = [t for t in all_tokens if start <= t.start_s < end]
                features = self._process_window(start, end, window_tokens)
                self.store.put_verbal(job_id, idx, features)
                logger.debug(f"[verbal] window {idx} done")
            except Exception as exc:
                logger.error(f"[verbal] Window {idx} failed: {exc}")

        try:
            wordlist, ngrams, collocations, segmented_tokens = self._compute_corpus_stats(
                all_tokens, lang_code
            )
            self.store.put_wordlist(job_id, wordlist)
            self.store.put_ngrams(job_id, ngrams)
            self.store.put_collocations(job_id, collocations)
            if segmented_tokens:
                self.store.put_segmented_tokens(job_id, segmented_tokens)
            logger.info(f"[verbal] Collocations stored: {len(collocations)} entries for job {job_id}")
        except Exception as exc:
            logger.error(f"[verbal] Corpus stats failed: {exc}")

        logger.info(f"[verbal] Job {job_id} complete")

    # ------------------------------------------------------------------

    def _transcribe(self, audio_path: str) -> tuple[list[WordToken], str]:
        """Transcribe audio and return (tokens, detected_language_code)."""
        segments, info = self._whisper.transcribe(
            audio_path,
            word_timestamps=True,
            vad_filter=True,
        )
        lang_code = info.language
        logger.info(
            f"[verbal] Detected language: '{lang_code}' "
            f"(confidence: {info.language_probability:.2f})"
        )
        tokens: list[WordToken] = []
        for seg in segments:
            if seg.words is None:
                continue
            for word in seg.words:
                tokens.append(WordToken(
                    word=word.word.strip(),
                    start_s=word.start,
                    end_s=word.end,
                    confidence=word.probability,
                ))
        return tokens, lang_code

    def _process_window(
        self,
        start_s: float,
        end_s: float,
        tokens: list[WordToken],
    ) -> VerbalFeatures:
        return VerbalFeatures(
            window=TimeWindow(start_s=start_s, end_s=end_s),
            transcript=" ".join(t.word for t in tokens),
            tokens=tokens,
            word_count=len(tokens),
        )

    @staticmethod
    def _segment_words(doc, all_tokens: list[WordToken], sep_len: int) -> list[dict]:
        """
        Map every spaCy doc token back onto the original Whisper WordToken(s)
        it derives from, so Concordance (and anything else that wants word-
        level timestamps) can search spaCy's own tokenization instead of
        Whisper's raw, un-reprocessed word list — keeping Concordance, Word
        List, Collocations, Word Sketch, and Distributional Thesaurus all
        reading from the exact same tokenization, for every language.

        This matters differently depending on how the transcript was joined
        for spaCy (see _compute_corpus_stats' join_sep):
        - CJK (sep_len=0, no separator): a single spaCy word can span
          *multiple* original per-character Whisper tokens, since pkuseg
          merges individual characters into real multi-character words. Uses
          the first/last covered token's start/end timestamp.
        - Space-separated languages (sep_len=1): spaCy's tokenizer only ever
          *splits* within a chunk, never merges across the space we joined
          with — so each spaCy token falls entirely within exactly one
          original Whisper token. Sub-tokens split off a single ASR word
          (e.g. "don't" -> "do"/"n't", "well-known" -> "well"/"-"/"known")
          simply share that one original token's timing, since Whisper
          doesn't give finer-grained timestamps within one ASR word anyway.

        *sep_len*: number of characters joining consecutive Whisper tokens in
        the string handed to spaCy (0 for CJK's no-separator join, 1 for a
        single-space join).
        """
        # (start_char, end_char) for each Whisper token, in the same
        # coordinate space as the string the spaCy doc was built from.
        spans = []
        offset = 0
        for tok in all_tokens:
            n = len(tok.word)
            spans.append((offset, offset + n, tok))
            offset += n + sep_len

        segmented: list[dict] = []
        span_i = 0
        for token in doc:
            if not token.text.strip():
                continue
            w_start, w_end = token.idx, token.idx + len(token.text)
            while span_i < len(spans) - 1 and spans[span_i][1] <= w_start:
                span_i += 1
            if span_i >= len(spans):
                break
            start_tok = spans[span_i][2]
            end_i = span_i
            while end_i < len(spans) - 1 and spans[end_i][1] < w_end:
                end_i += 1
            end_tok = spans[end_i][2]
            segmented.append({
                "word": token.text,
                "start_s": start_tok.start_s,
                "end_s": end_tok.end_s,
            })
            span_i = end_i

        return segmented

    @staticmethod
    def _build_doc(nlp, text: str, keep_glued: frozenset = frozenset()):
        """
        Tokenize *text* with *nlp* and run the rest of its pipeline (tagger,
        parser, ...) — same as calling nlp(text) directly — except a
        digit-glue correction pass runs on the tokens first: faster-whisper
        (English) and pkuseg (Chinese, via zh_core_web_sm) both occasionally
        produce a single token that glues a number onto adjacent content
        with no boundary ("5apples", "30岁") — see _split_number_glue. This
        has to happen *between* tokenizing and parsing, not after nlp(text)
        has already run end to end: the parser needs to see "apples"/"岁" as
        their own tokens to assign them a real POS tag and dependency edge,
        not inherit whatever the merged token got.

        Rebuilds a Doc from the corrected tokens, preserving each original
        token's own `.whitespace_` so the rebuilt doc's character offsets
        stay identical to *text* — critical since _segment_words maps
        timestamps back onto this doc purely by character offset. Split
        halves get no space between them (they were glued with none); the
        second half inherits the original token's own trailing space.

        Skips the rebuild (just calls nlp(text) normally) when nothing
        needed splitting, which is the common case.
        """
        tok_doc = nlp.tokenizer(text)
        words: list[str] = []
        spaces: list[bool] = []
        changed = False
        for tok in tok_doc:
            if not tok.text:
                continue
            split = _split_number_glue(tok.text, keep_glued)
            if split:
                changed = True
                first, second = split
                words.append(first)
                spaces.append(False)
                words.append(second)
                spaces.append(bool(tok.whitespace_))
            else:
                words.append(tok.text)
                spaces.append(bool(tok.whitespace_))

        if not changed:
            return nlp(text)

        doc = Doc(nlp.vocab, words=words, spaces=spaces)
        for _, proc in nlp.pipeline:
            doc = proc(doc)
        return doc

    def _compute_corpus_stats(
        self, all_tokens: list[WordToken], lang_code: str
    ) -> tuple[dict, dict, dict, list]:
        """
        Build word list, n-grams, collocations, and a spaCy-word-segmented
        token list (with timestamps mapped back from the original Whisper
        tokens — see _segment_words) from the full transcript. Uses the
        spaCy model for *lang_code* if available, tokenized via _build_doc
        (rather than calling nlp(text) directly) so a number glued onto
        adjacent content with no boundary ("5apples", "30年") — which
        neither faster-whisper nor spaCy's own tokenizer reliably splits on
        its own — gets corrected before tagging/parsing, for every language.
        Returns (wordlist, ngrams, collocations, segmented_tokens).
        """
        nlp = self._get_nlp(lang_code)
        # Chinese (and other logographic scripts) must be joined without spaces
        # so that spaCy's jieba-based tokenizer can segment words correctly.
        join_sep = "" if lang_code in ("zh", "ja", "ko") else " "
        text = join_sep.join(t.word for t in all_tokens)
        raw_words = [t.word.lower().strip(".,!?\"'") for t in all_tokens if t.word.strip()]

        # ── spaCy pass (word list) ───────────────────────────────────────
        doc = None
        if nlp and text.strip():
            nlp_text = text[: nlp.max_length]
            keep_glued = frozenset() if lang_code in ("zh", "ja", "ko") else _EN_NUMBER_GLUE_KEEP
            doc = self._build_doc(nlp, nlp_text, keep_glued)

            # Keyed by surface form, not lemma — "run"/"running"/"ran" get
            # separate rows with their own counts, consistent with the
            # surface-form keying used for collocations (core/corpus_analysis.py).
            # No stopword filtering at all — spaCy's is_stop is unreliable for
            # zh_core_web_sm (mis-tags real content words like 大/是), and for
            # consistency N-grams/Collocations never filtered stop words
            # either, so the word list doesn't filter them for any language.
            word_data: dict[str, dict] = {}
            for token in doc:
                if token.is_punct:
                    continue
                # Strip stray leading/trailing punctuation Whisper sometimes
                # glues onto a token (e.g. "-female" from "male-female")
                # before the alpha check — same normalization Collocations
                # uses (core/corpus_analysis._clean_word) — so a word doesn't
                # silently drop out of the list just because of an attached
                # punctuation character.
                word = corpus_analysis._clean_word(token.text.lower())
                if not word or not word.isalpha():
                    continue
                if word not in word_data:
                    word_data[word] = {"pos": token.pos_, "count": 0}
                word_data[word]["count"] += 1

            total = max(sum(v["count"] for v in word_data.values()), 1)
            word_entries = [
                {
                    "word":          word,
                    "pos":           d["pos"],
                    "count":         d["count"],
                    "freq_per_1000": round(d["count"] * 1000 / total, 2),
                }
                for word, d in sorted(word_data.items(), key=lambda x: -x[1]["count"])
            ]
        else:
            counts = Counter(w for w in raw_words if w.isalpha())
            total = max(sum(counts.values()), 1)
            word_entries = [
                {
                    "word":          w,
                    "pos":           "?",
                    "count":         c,
                    "freq_per_1000": round(c * 1000 / total, 2),
                }
                for w, c in counts.most_common()
            ]

        wordlist = {"words": word_entries, "total_tokens": len(raw_words)}

        # ── N-grams (including stop words for natural phrasing) ──────────
        # For logographic scripts the spaCy doc gives proper word tokens;
        # raw Whisper tokens are individual characters (not useful for n-grams).
        if doc is not None and lang_code in ("zh", "ja", "ko"):
            alpha = [t.text.lower() for t in doc if t.is_alpha]
        else:
            alpha = [w for w in raw_words if w.isalpha()]
        bigram_counts: Counter = Counter()
        trigram_counts: Counter = Counter()
        for i in range(len(alpha)):
            if i + 1 < len(alpha):
                bigram_counts[f"{alpha[i]} {alpha[i+1]}"] += 1
            if i + 2 < len(alpha):
                trigram_counts[f"{alpha[i]} {alpha[i+1]} {alpha[i+2]}"] += 1

        ngrams = {
            "bigrams":  [{"ngram": ng, "count": c} for ng, c in bigram_counts.most_common(50)],
            "trigrams": [{"ngram": ng, "count": c} for ng, c in trigram_counts.most_common(30)],
        }

        # ── Collocations (dep parse on same doc, isolated so failure
        #    can't affect wordlist/ngrams above) ─────────────────────────
        collocations: dict = {}
        if doc is not None:
            try:
                if lang_code in ("zh", "ja", "ko"):
                    collocations = corpus_analysis.extract_collocations_zh(doc)
                else:
                    collocations = corpus_analysis.extract_collocation_en(doc)
            except Exception as exc:
                logger.warning(f"[verbal] Collocations extraction failed: {exc}")

        # ── Word-segmented tokens for Concordance (all languages) ────────
        # Built from the same doc as Word List/Collocations/Word Sketch/
        # Distributional Thesaurus, rather than Whisper's raw, un-reprocessed
        # word list, so Concordance can never disagree with them about what
        # words exist in the transcript (e.g. English contractions/hyphenated
        # compounds that spaCy splits differently than Whisper's own word
        # boundaries, or CJK words pkuseg merges from multiple characters).
        segmented_tokens: list[dict] = []
        if doc is not None:
            try:
                segmented_tokens = self._segment_words(doc, all_tokens, len(join_sep))
            except Exception as exc:
                logger.warning(f"[verbal] Word segmentation failed: {exc}")

        return wordlist, ngrams, collocations, segmented_tokens
