# Verbal Worker

Source: [`workers/verbal_worker.py`](../workers/verbal_worker.py)
Dashboard section: **Verbal Language**
Feature model: [`VerbalFeatures`](../core/models.py) (in `core/models.py`)

## What it does

`VerbalWorker` transcribes the full audio **once** with **faster-whisper**,
then distributes word tokens into time windows and builds corpus-wide
lexical statistics with **spaCy**.

1. **Transcription** (`_transcribe`) — runs Whisper with
   `word_timestamps=True` and `vad_filter=True` (voice-activity filtering),
   auto-detecting the spoken language and returning a flat list of
   `WordToken`s (word, start/end time, confidence).
2. **Per-window features** (`_process_window`) — tokens are bucketed into
   each window by start time; a `VerbalFeatures` record (transcript text,
   token list, word count) is stored per window.
3. **Corpus statistics** (`_compute_corpus_stats`), computed once for the
   whole transcript and stored separately:
   - **Word list** — lemma frequency table (top 200), using the spaCy model
     matched to the detected language when available, otherwise raw
     lower-cased word counts.
   - **N-grams** — bigrams/trigrams (including stop words, for natural
     phrasing) over alphabetic tokens.
   - **Collocations** — dependency-parse-based collocations
     ([`core/corpus_analysis.py`](../core/corpus_analysis.py)), with a
     dedicated Chinese/Japanese/Korean path (`extract_collocations_zh`).

## Multi-language support

- Whisper auto-detects the spoken language; the matching spaCy model is
  looked up in `_SPACY_MODELS` (currently `en → en_core_web_sm`,
  `zh → zh_core_web_sm`) and loaded lazily, then cached per-language so
  repeated jobs don't reload the model.
- For logographic scripts (`zh`/`ja`/`ko`), tokens are joined **without**
  spaces before being handed to spaCy so its tokenizer can re-segment words
  correctly, and n-grams are built from the spaCy doc's word tokens rather
  than Whisper's per-character tokens.
- `zh_core_web_sm` ships no dedicated stop-word list and its statistical
  `is_stop` flags mis-tag common content words (e.g. 人/好/大). To work
  around this, the worker clears the model's built-in stop-word flags and
  applies a curated list instead, loaded via
  [`core/stopwords`](../core/stopwords) (`load_stopwords`).
- If no spaCy model is configured/installed for a detected language, the
  worker falls back to raw word counts and skips collocation extraction
  rather than failing the job.

## Package documentation

| Package | Role | Docs |
|---|---|---|
| faster-whisper | Speech-to-text transcription with word-level timestamps | https://github.com/SYSTRAN/faster-whisper#readme |
| spaCy | Tokenization, lemmatization, POS tagging, dependency parsing | https://spacy.io/api |
| loguru | Transcription/job logging | https://loguru.readthedocs.io/en/stable/ |
| Pydantic | `VerbalFeatures` / `WordToken` models | https://docs.pydantic.dev/latest/ |

See also [Home](Home.md) for the full dependency list.
</content>
