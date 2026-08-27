"""
workers/verbal_worker.py
------------------------
Faster-Whisper ASR + spaCy NLP worker.

Transcribes the full audio once (word-level timestamps), then
for each time window assigns tokens and computes linguistic features.
Language is auto-detected by Whisper; the matching spaCy model is loaded
lazily and cached so multi-language sessions don't reload unnecessarily.

SenseVoice (see _ALT_ASR_LANGS/_transcribe_alt below) can optionally run
remotely instead of loading funasr+torch locally — see
`colab/sensevoice_server.ipynb` and the SENSEVOICE_REMOTE_URL /
SENSEVOICE_API_KEY env vars below. Whichever mode is active, the rest of
this file (windowing, corpus stats, everything downstream of _transcribe)
is unaffected — it only ever sees a list[WordToken], same as from Whisper.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Optional

import requests
import soundfile as sf
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

# Languages transcribed with SenseVoice (via funasr) instead of Whisper —
# meaningfully fewer homophone-confusion errors on Mandarin in side-by-side
# testing, and ~5x faster on CPU. Whisper still does the (cheap) language
# detection pass for every job; only the actual transcription is routed
# elsewhere for languages in this set. Add a code here (and a branch in
# _transcribe_alt) to route another language the same way.
_ALT_ASR_LANGS = frozenset({"zh"})

# If set, _transcribe_alt calls this URL instead of loading SenseVoice
# (funasr + torch) into this process at all — see colab/sensevoice_server.ipynb.
# SENSEVOICE_API_KEY must match whatever API_KEY that notebook was given.
# Empty/unset means "load and run SenseVoice locally", the original behaviour.
_REMOTE_URL_ENV = "SENSEVOICE_REMOTE_URL"
_REMOTE_API_KEY_ENV = "SENSEVOICE_API_KEY"
_SENSEVOICE_MODEL = "iic/SenseVoiceSmall"

# Remote transcription is chunked rather than sent as one whole-file request
# — see _transcribe_alt_remote's docstring for why (it's not just about
# request size). _REMOTE_CHUNKS_PER_REQUEST chunks are bundled into each
# request as a list, so 5 x 60s = 300s of audio per request, matching the
# server's own batch_size_s=300 so FunASR actually batches them together.
_REMOTE_CHUNK_S = 60
_REMOTE_CHUNKS_PER_REQUEST = 5
_REMOTE_CHUNK_TIMEOUT_S = 180  # generous for one ~300s-of-audio batch request
_REMOTE_CHUNK_RETRIES = 2      # extra attempts per batch before giving up on
# remote entirely and letting _transcribe's existing fallback hand the whole
# job to Whisper — retrying a batch here means one flaky request doesn't
# throw away every chunk that already succeeded.
_REMOTE_CHUNK_RETRY_BACKOFF_S = 3

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
        self._device = device  # also used to place SenseVoice, see _get_sensevoice
        logger.info(f"[verbal] Loading Whisper model '{model_size}' on {device}")
        self._whisper = WhisperModel(model_size, device=device, compute_type="int8")
        self._nlp_cache: dict[str, Optional[object]] = {}
        # SenseVoice is loaded lazily (see _get_sensevoice) — most sessions
        # never touch it, and its torch-backed load is much heavier than
        # Whisper's, so it shouldn't be paid for English-only jobs.
        self._sensevoice = None
        # If set, SenseVoice never gets loaded in this process at all —
        # _transcribe_alt calls a remote instance instead (see
        # colab/sensevoice_server.ipynb, and _REMOTE_URL_ENV's docstring above).
        self._remote_url = os.environ.get(_REMOTE_URL_ENV) or None
        self._remote_api_key = os.environ.get(_REMOTE_API_KEY_ENV) or None
        if self._remote_url:
            logger.info(f"[verbal] SenseVoice will run remotely at {self._remote_url}")

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

    def _get_sensevoice(self):
        """
        Return the cached SenseVoice (funasr) model, loading it on first use.

        Placed on the same device as Whisper (self._device, from the
        constructor's `device` param — already threaded through from the CLI's
        `--device cpu|cuda` flag via Orchestrator/VerbalWorker). funasr itself
        falls back to CPU automatically if "cuda" is requested but unavailable
        (see funasr.auto.auto_model.AutoModel.build_model), so this is safe to
        pass through as-is.
        """
        if self._sensevoice is not None:
            return self._sensevoice
        from funasr import AutoModel

        logger.info(f"[verbal] Loading SenseVoice model '{_SENSEVOICE_MODEL}' on {self._device}")
        self._sensevoice = AutoModel(
            model=_SENSEVOICE_MODEL,
            trust_remote_code=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=self._device,
            disable_update=True,
        )
        logger.info("[verbal] SenseVoice model loaded")
        return self._sensevoice

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
        """
        Transcribe audio and return (tokens, detected_language_code).

        Whisper always runs first, purely for language detection — that part
        is cheap (~1-2s: an encoder pass over the first 30s + the language-ID
        head), well under the actual autoregressive decoding cost, and
        crucially the `segments` generator below is only consumed (paying
        that decoding cost) if the detected language isn't one of
        _ALT_ASR_LANGS. For those, transcription is handed off to SenseVoice
        instead — Whisper's own segments are simply left unconsumed.
        """
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

        if lang_code in _ALT_ASR_LANGS:
            try:
                return self._transcribe_alt(audio_path, lang_code), lang_code
            except Exception as exc:
                logger.warning(
                    f"[verbal] Alternate ASR failed for '{lang_code}' ({exc}) — "
                    "falling back to Whisper for this job"
                )
                # Fall through to Whisper's own (already in-flight) segments.

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

    def _transcribe_alt(self, audio_path: str, lang_code: str) -> list[WordToken]:
        """
        Transcribe *audio_path* with SenseVoice instead of Whisper — used for
        languages in _ALT_ASR_LANGS. SenseVoice doesn't report a per-word
        confidence the way Whisper does, so WordToken.confidence is set to a
        flat 1.0 (unused downstream for anything but display).

        Note: like Whisper's own CJK output, SenseVoice's Chinese "words" are
        still per-character, not per multi-character word — pkuseg (via
        spaCy, in _build_doc/_compute_corpus_stats) still does the real word
        segmentation on top, unchanged by which ASR engine produced the raw
        transcript.

        Runs locally (loading funasr+torch into this process) unless
        _remote_url is set, in which case _transcribe_alt_remote handles it
        instead and this process never imports funasr at all — see
        colab/sensevoice_server.ipynb.
        """
        if self._remote_url:
            return self._transcribe_alt_remote(audio_path, lang_code)

        model = self._get_sensevoice()
        results = model.generate(
            input=audio_path,
            cache={},
            language=lang_code,
            use_itn=True,
            batch_size_s=300,
            output_timestamp=True,
        )
        tokens: list[WordToken] = []
        for item in results:
            words = item.get("words") or []
            timestamps = item.get("timestamp") or []
            for word, (start_ms, end_ms) in zip(words, timestamps):
                tokens.append(WordToken(
                    word=word,
                    start_s=start_ms / 1000.0,
                    end_s=end_ms / 1000.0,
                    confidence=1.0,
                ))
        return tokens

    def _transcribe_alt_remote(self, audio_path: str, lang_code: str) -> list[WordToken]:
        """
        Same contract as the local branch of _transcribe_alt above, but via
        HTTP against a colab/sensevoice_server.ipynb instance instead of a
        local funasr call — see _REMOTE_URL_ENV's module-level docstring.

        The audio is split into _REMOTE_CHUNK_S-second pieces and sent
        _REMOTE_CHUNKS_PER_REQUEST at a time, as a list, in one HTTP request
        per batch, instead of the whole file in one request (the original
        design). That bundling is doing two distinct jobs, not one:

        - It's what actually lets the server's own batch_size_s=300 FunASR
          setting do anything — a single whole-file request only ever gave
          FunASR one item to batch, so that machinery sat idle. Sending
          5 x 60s chunks per request (300s total, matching batch_size_s)
          lets FunASR batch them together on the GPU instead of decoding
          one long sequence serially — a real compute-side speedup, not
          just a smaller request.
        - Each request's duration is now bounded by _REMOTE_CHUNK_S x
          _REMOTE_CHUNKS_PER_REQUEST regardless of the video's total length,
          instead of scaling with it (the whole-file design's request
          latency scaled with video length, which is what silently blew
          past a fixed client timeout on long videos). One long silent wait
          becomes a series of logged checkpoints instead (see the progress
          log below), so a long transcription's progress is actually
          visible while it runs.

        Each batch retries up to _REMOTE_CHUNK_RETRIES times (network
        errors, timeouts, bad/short responses) before this raises — same as
        before, _transcribe's own try/except around _transcribe_alt then
        falls back to Whisper for the whole job. Tokens from batches that
        *did* succeed before a later batch exhausts its retries are
        discarded rather than mixed with Whisper's output, so a job's
        transcript always comes from a single engine end to end.
        """
        audio, sample_rate = sf.read(audio_path, dtype="int16")
        total_s = len(audio) / sample_rate
        chunk_frames = int(_REMOTE_CHUNK_S * sample_rate)
        chunk_frame_starts = list(range(0, len(audio), chunk_frames))
        logger.info(
            f"[verbal] Sending {len(chunk_frame_starts)} chunks "
            f"({_REMOTE_CHUNK_S}s each, {total_s:.0f}s total) to remote "
            f"SenseVoice, {_REMOTE_CHUNKS_PER_REQUEST} chunks per request"
        )

        tokens: list[WordToken] = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for batch_start in range(0, len(chunk_frame_starts), _REMOTE_CHUNKS_PER_REQUEST):
                batch_frame_starts = chunk_frame_starts[batch_start:batch_start + _REMOTE_CHUNKS_PER_REQUEST]
                batch: list[tuple[Path, float]] = []
                for i, frame_start in enumerate(batch_frame_starts):
                    frame_end = min(frame_start + chunk_frames, len(audio))
                    chunk_path = Path(tmp_dir) / f"chunk_{batch_start + i}.wav"
                    sf.write(str(chunk_path), audio[frame_start:frame_end], sample_rate)
                    batch.append((chunk_path, frame_start / sample_rate))

                tokens.extend(self._send_chunk_batch(batch, lang_code))

                done_s = min((batch_start + len(batch)) * _REMOTE_CHUNK_S, total_s)
                pct = (done_s / total_s * 100) if total_s else 100
                logger.info(
                    f"[verbal] Remote SenseVoice progress: {done_s:.0f}s / "
                    f"{total_s:.0f}s ({pct:.0f}%) — {len(tokens)} tokens so far"
                )

        return tokens

    def _send_chunk_batch(
        self, batch: list[tuple[Path, float]], lang_code: str
    ) -> list[WordToken]:
        """
        POST one batch of chunk files (as a list, so the server batches them
        — see _transcribe_alt_remote's docstring) with retries, then offset
        each chunk's returned tokens back onto the full audio's timeline.
        Raises (after _REMOTE_CHUNK_RETRIES retries) if every attempt fails.
        """
        last_exc: Optional[Exception] = None
        attempts = _REMOTE_CHUNK_RETRIES + 1
        for attempt in range(1, attempts + 1):
            try:
                with ExitStack() as stack:
                    files = [
                        ("files", (path.name, stack.enter_context(open(path, "rb")), "audio/wav"))
                        for path, _ in batch
                    ]
                    response = requests.post(
                        self._remote_url,
                        files=files,
                        data={"lang_code": lang_code},
                        headers={"X-API-Key": self._remote_api_key or ""},
                        timeout=_REMOTE_CHUNK_TIMEOUT_S,
                    )
                response.raise_for_status()
                chunk_results = response.json()["chunks"]
                if len(chunk_results) != len(batch):
                    raise RuntimeError(
                        f"Remote server returned {len(chunk_results)} chunk "
                        f"results, expected {len(batch)}"
                    )

                tokens: list[WordToken] = []
                for (_, chunk_offset_s), result in zip(batch, chunk_results):
                    for tok in result["tokens"]:
                        tokens.append(WordToken(
                            word=tok["word"],
                            start_s=tok["start_s"] + chunk_offset_s,
                            end_s=tok["end_s"] + chunk_offset_s,
                            confidence=tok["confidence"],
                        ))
                return tokens
            except Exception as exc:
                last_exc = exc
                if attempt <= _REMOTE_CHUNK_RETRIES:
                    logger.warning(
                        f"[verbal] Remote SenseVoice batch failed "
                        f"(attempt {attempt}/{attempts}): {exc} — retrying "
                        f"in {_REMOTE_CHUNK_RETRY_BACKOFF_S}s"
                    )
                    time.sleep(_REMOTE_CHUNK_RETRY_BACKOFF_S)
        raise RuntimeError(
            f"Remote SenseVoice batch failed after {attempts} attempts"
        ) from last_exc

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
