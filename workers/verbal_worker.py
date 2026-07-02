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
        self.store.log_event(
            job_id, "verbal",
            f"transcription done: {len(all_tokens)} tokens, language={lang_code}"
        )

        for idx, (start, end) in enumerate(windows):
            try:
                window_tokens = [t for t in all_tokens if start <= t.start_s < end]
                features = self._process_window(start, end, window_tokens)
                self.store.put_verbal(job_id, idx, features)
                self.store.log_event(job_id, "verbal", f"window {idx} done")
            except Exception as exc:
                logger.error(f"[verbal] Window {idx} failed: {exc}")
                self.store.log_event(job_id, "verbal", f"window {idx} ERROR: {exc}")

        try:
            wordlist, ngrams, collocations = self._compute_corpus_stats(
                all_tokens, lang_code
            )
            self.store.put_wordlist(job_id, wordlist)
            self.store.put_ngrams(job_id, ngrams)
            self.store.put_collocations(job_id, collocations)
            logger.info(f"[verbal] Collocations stored: {len(collocations)} entries for job {job_id}")
            self.store.log_event(job_id, "verbal", "corpus stats done")
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

    def _compute_corpus_stats(
        self, all_tokens: list[WordToken], lang_code: str
    ) -> tuple[dict, dict, dict]:
        """
        Build word list, n-grams, and collocations from the full transcript.
        Uses the spaCy model for *lang_code* if available.
        Returns (wordlist, ngrams, collocations).
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

            lemma_data: dict[str, dict] = {}
            for token in doc:
                if token.is_stop or token.is_punct or not token.is_alpha:
                    continue
                # zh_core_web_sm returns empty lemma_ for all tokens; fall back to surface form
                lemma = (token.lemma_ or token.text).lower()
                if not lemma:
                    continue
                if lemma not in lemma_data:
                    lemma_data[lemma] = {"pos": token.pos_, "count": 0, "example": token.text}
                lemma_data[lemma]["count"] += 1

            total = max(sum(v["count"] for v in lemma_data.values()), 1)
            word_entries = [
                {
                    "lemma":         lemma,
                    "pos":           d["pos"],
                    "example":       d["example"],
                    "count":         d["count"],
                    "freq_per_1000": round(d["count"] * 1000 / total, 2),
                }
                for lemma, d in sorted(lemma_data.items(), key=lambda x: -x[1]["count"])[:200]
            ]
        else:
            counts = Counter(w for w in raw_words if w.isalpha())
            total = max(sum(counts.values()), 1)
            word_entries = [
                {
                    "lemma":         w,
                    "pos":           "?",
                    "example":       w,
                    "count":         c,
                    "freq_per_1000": round(c * 1000 / total, 2),
                }
                for w, c in counts.most_common(200)
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
                collocations = corpus_analysis.extract_collocations(doc)
            except Exception as exc:
                logger.warning(f"[verbal] Collocations extraction failed: {exc}")

        return wordlist, ngrams, collocations
