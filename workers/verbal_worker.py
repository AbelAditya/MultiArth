"""
workers/verbal_worker.py
------------------------
Faster-Whisper ASR + spaCy NLP worker.

Transcribes the full audio once (word-level timestamps), then
for each time window assigns tokens and computes linguistic features.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import spacy
from faster_whisper import WhisperModel
from loguru import logger

from core.feature_store import FeatureStore
from core.models import TimeWindow, VerbalFeatures, WordToken
from core.preprocessing import VideoMeta

# Common English filler words
_FILLERS = frozenset({
    "uh", "um", "erm", "er", "ah", "like", "basically", "literally",
    "actually", "right", "okay", "so", "you know", "i mean",
})

# Common hedging expressions (single-token approximation)
_HEDGES = frozenset({
    "maybe", "perhaps", "possibly", "probably", "might", "could",
    "somewhat", "kind", "sort", "roughly", "approximately",
})


class VerbalWorker:
    def __init__(
        self,
        store: FeatureStore,
        model_size: str = "base",
        device: str = "cpu",
        spacy_model: str = "en_core_web_sm",
    ):
        self.store = store
        logger.info(f"[verbal] Loading Whisper model '{model_size}' on {device}")
        self._whisper = WhisperModel(model_size, device=device, compute_type="int8")
        logger.info(f"[verbal] Loading spaCy model '{spacy_model}'")
        try:
            self._nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(f"spaCy model '{spacy_model}' not found — run: python -m spacy download {spacy_model}")
            self._nlp = None

    def process_job(
        self,
        job_id: str,
        meta: VideoMeta,
        windows: list[tuple[float, float]],
    ) -> None:
        logger.info(f"[verbal] Transcribing {meta.audio_path}")
        all_tokens = self._transcribe(meta.audio_path)
        logger.info(f"[verbal] Transcription complete — {len(all_tokens)} word tokens")
        self.store.log_event(job_id, "verbal", f"transcription done: {len(all_tokens)} tokens")

        for idx, (start, end) in enumerate(windows):
            try:
                window_tokens = [t for t in all_tokens if start <= t.start_s < end]
                features = self._process_window(start, end, window_tokens)
                self.store.put_verbal(job_id, idx, features)
                self.store.log_event(job_id, "verbal", f"window {idx} done")
            except Exception as exc:
                logger.error(f"[verbal] Window {idx} failed: {exc}")
                self.store.log_event(job_id, "verbal", f"window {idx} ERROR: {exc}")

        logger.info(f"[verbal] Job {job_id} complete")

    # ------------------------------------------------------------------

    def _transcribe(self, audio_path: str) -> list[WordToken]:
        segments, _ = self._whisper.transcribe(
            audio_path,
            word_timestamps=True,
            language="en",
            vad_filter=True,
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
        return tokens

    def _process_window(
        self,
        start_s: float,
        end_s: float,
        tokens: list[WordToken],
    ) -> VerbalFeatures:
        window = TimeWindow(start_s=start_s, end_s=end_s)
        transcript = " ".join(t.word for t in tokens)

        word_count = len(tokens)
        mean_conf = float(sum(t.confidence for t in tokens) / max(word_count, 1))

        # Filler detection (case-insensitive single-token)
        filler_count = sum(
            1 for t in tokens if t.word.lower().strip(".,!?") in _FILLERS
        )
        duration_min = (end_s - start_s) / 60.0
        filler_rate = filler_count / max(duration_min, 1e-6)

        # Hedge detection
        hedge_count = sum(
            1 for t in tokens if t.word.lower().strip(".,!?") in _HEDGES
        )

        # Pause detection (inter-word gap > 250 ms)
        pauses = []
        for i in range(1, len(tokens)):
            gap = tokens[i].start_s - tokens[i - 1].end_s
            if gap > 0.25:
                pauses.append(gap)
        pause_count = len(pauses)
        mean_pause = float(sum(pauses) / max(pause_count, 1))

        # Sentence-level stats via spaCy
        sentence_count, mean_sent_len = self._sentence_stats(transcript)

        # Type-token ratio (vocabulary richness)
        words_lower = [t.word.lower().strip(".,!?\"'") for t in tokens if t.word.strip()]
        ttr = len(set(words_lower)) / max(len(words_lower), 1)

        return VerbalFeatures(
            window=window,
            transcript=transcript,
            tokens=tokens,
            word_count=word_count,
            filler_word_count=filler_count,
            filler_word_rate=filler_rate,
            mean_word_confidence=mean_conf,
            sentence_count=sentence_count,
            mean_sentence_length=mean_sent_len,
            type_token_ratio=ttr,
            hedge_count=hedge_count,
            pause_count=pause_count,
            mean_pause_duration_s=mean_pause,
        )

    def _sentence_stats(self, text: str) -> tuple[int, float]:
        if not text.strip():
            return 0, 0.0
        if self._nlp:
            doc = self._nlp(text)
            sents = list(doc.sents)
            if not sents:
                return 0, 0.0
            lengths = [len([t for t in s if not t.is_space]) for s in sents]
            return len(sents), float(sum(lengths) / max(len(sents), 1))
        else:
            # Fallback: split on sentence-ending punctuation
            sents = re.split(r"[.!?]+", text)
            sents = [s.strip() for s in sents if s.strip()]
            if not sents:
                return 0, 0.0
            lengths = [len(s.split()) for s in sents]
            return len(sents), float(sum(lengths) / len(sents))
