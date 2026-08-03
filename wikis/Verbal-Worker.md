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
   - **Word list** — surface-form frequency table for every content word
     (inflected forms like "run"/"running" stay separate, not merged under
     one lemma), using the spaCy model matched to the detected language when
     available, otherwise raw lower-cased word counts.
   - **N-grams** — bigrams/trigrams (including stop words, for natural
     phrasing) over alphabetic tokens.
   - **Collocations** — dependency-parse-based collocations
     ([`core/corpus_analysis.py`](../core/corpus_analysis.py):
     `extract_collocation_en`), with a separate Chinese/Japanese/Korean
     function (`extract_collocations_zh`) that keeps its own positional scan
     (nearest neighbour by direction/POS, plus a small conjunction-particle
     list) alongside the *same* generic dependency-relation capture English
     uses, shared via an internal `_capture_dep_relations` helper so both
     languages get identical relation names for anything the dependency
     parser itself provides. Chinese applies no stopword filtering — every
     alphabetic, non-punctuation token is a valid target.
     `extract_collocation_en` keys words by **surface form, not lemma**, so
     inflected forms (`run`/`running`) get separate profiles — except
     copula/auxiliary forms of `be`/`will`/`have` (`is`/`'s`/`are`/`'re`/...),
     which are normalised to one shared lemma so a construction isn't split
     apart by contraction spelling. Relations covered generically (both
     languages): subject (incl. clausal subjects), object (incl. predicate
     nominals/adjectives, dative, object predicates), prepositional and
     passive-agent attachment, negation, modification (adjectives, compounds,
     numerics, determiners, possessives, relative clauses, adverbs),
     coordination, verb complements, auxiliaries, and subordinating
     conjunctions/markers — each as a symmetric pair (e.g. `subj_of`/
     `has_subj`) so either side of a relation is searchable, plus, for
     anything without an established name, an automatic `{dep}_of`/
     `has_{dep}` fallback using spaCy's raw dependency label.

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
