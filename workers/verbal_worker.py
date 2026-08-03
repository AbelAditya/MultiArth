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

from collections import Counter
from typing import Optional

import spacy
from faster_whisper import WhisperModel
from loguru import logger

from core import corpus_analysis
from core.feature_store import FeatureStore
from core.stopwords import load_stopwords
from core.models import TimeWindow, VerbalFeatures, WordToken
from core.preprocessing import VideoMeta

# Maps Whisper language codes to spaCy model names.
# Add entries here to support additional languages.
_SPACY_MODELS: dict[str, str] = {
    "en": "en_core_web_sm",
    "zh": "zh_core_web_sm",
}


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

            custom_stops = load_stopwords(lang_code)
            if custom_stops:
                # Clear the model's built-in stop word list and apply ours exclusively.
                # zh_core_web_sm has no dedicated stop word list and its statistical
                # is_stop flags incorrectly mark content words like 人/好/大.
                for lex in nlp.vocab:
                    lex.is_stop = False
                for word in custom_stops:
                    nlp.vocab[word].is_stop = True
                logger.info(
                    f"[verbal] Applied custom stop word list for '{lang_code}' "
                    f"({len(custom_stops)} entries)"
                )

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
    def _segment_cjk_words(doc, all_tokens: list[WordToken]) -> list[dict]:
        """
        For CJK: raw Whisper tokens are individual characters (see the
        n-grams comment below), but spaCy's jieba-based tokenizer re-segments
        the concatenated (no-separator) transcript into proper multi-character
        words. Map each spaCy word's character span back to the original
        per-character Whisper tokens it came from, so Concordance can search
        these properly-segmented words (with real timestamps) instead of
        being limited to matching single Whisper characters.
        """
        # (start_char, end_char) for each Whisper token, in the same
        # concatenated-string coordinate space the spaCy doc was built from.
        spans = []
        offset = 0
        for tok in all_tokens:
            n = len(tok.word)
            spans.append((offset, offset + n, tok))
            offset += n

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

    def _compute_corpus_stats(
        self, all_tokens: list[WordToken], lang_code: str
    ) -> tuple[dict, dict, dict, list]:
        """
        Build word list, n-grams, collocations, and (for CJK) a properly
        word-segmented token list from the full transcript. Uses the spaCy
        model for *lang_code* if available.
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
            doc = nlp(nlp_text)

            # Keyed by surface form, not lemma — "run"/"running"/"ran" get
            # separate rows with their own counts, consistent with the
            # surface-form keying used for collocations (core/corpus_analysis.py).
            word_data: dict[str, dict] = {}
            for token in doc:
                if token.is_stop or token.is_punct or not token.is_alpha:
                    continue
                word = token.text.lower()
                if not word:
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

        # ── Word-segmented tokens for Concordance (CJK only — Whisper's own
        #    per-character tokens are already word-granular for other
        #    languages, so Concordance keeps using those there) ───────────
        segmented_tokens: list[dict] = []
        if doc is not None and lang_code in ("zh", "ja", "ko"):
            try:
                segmented_tokens = self._segment_cjk_words(doc, all_tokens)
            except Exception as exc:
                logger.warning(f"[verbal] CJK word segmentation failed: {exc}")

        return wordlist, ngrams, collocations, segmented_tokens
