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
   whole transcript and stored separately. The spaCy doc it's all built from
   is produced by `_build_doc` rather than a plain `nlp(text)` call: both
   faster-whisper (English — see `split_tokens_on_spaces` in
   `faster_whisper/tokenizer.py`) and pkuseg (Chinese, via `zh_core_web_sm`)
   occasionally produce a single token that glues a number onto adjacent
   content with no boundary (`"5apples"`, `"30岁"`), and spaCy's own
   tokenizer doesn't reliably split these back apart either. `_build_doc`
   tokenizes first, runs a shared digit-boundary correction pass
   (`_split_number_glue`) over the tokens — skipping a small set of
   conventionally-glued forms (`am`/`pm`, ordinals, multipliers) for
   English, no exceptions for Chinese — then rebuilds the doc and runs the
   rest of the pipeline (tagger, parser, ...) on the corrected tokens, so
   the split-off piece (`"apples"`, `"岁"`) gets a proper POS tag and
   dependency edge instead of inheriting the merged token's. This runs
   before Word List/Collocations/Segmented-tokens are computed, so all
   three see the correction uniformly, for every language.
   - **Word list** — surface-form frequency table for every alphabetic,
     non-punctuation word (inflected forms like "run"/"running" stay
     separate, not merged under one lemma), using the spaCy model matched to
     the detected language when available, otherwise raw lower-cased word
     counts. No stopword filtering, for any language — see "Multi-language
     support" below. Strips stray leading/trailing punctuation a token might
     be glued to (`core/corpus_analysis._clean_word`; e.g. Whisper
     transcribing "male-female" as the raw tokens `"male"`/`"-female"` —
     spaCy's own tokenizer only splits a leading hyphen off *digits*, not
     letters) before the alpha check, so a word doesn't silently drop out
     just because of an attached punctuation character. The dashboard's
     Frequency tab (`kw-wordlist-table`) displays each word's spaCy
     Universal POS tag as a full label (`_POS_LABELS` in
     `dashboard/app.py` — e.g. `PROPN` → "Proper noun", `CCONJ` →
     "Coordinating conjunction") rather than the raw abbreviation; the
     dropdown filter and colour-coded row styling still key off the raw
     tags underneath, only the displayed value is expanded.
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
     `has_{dep}` fallback using spaCy's raw dependency label. Both
     languages' validity checks and keying (`_eff`) apply the same
     `_clean_word` stray-punctuation stripping the word list uses — without
     it, a word whose only captured relation points to a punctuation-glued
     neighbour (e.g. "male", whose head is the glued token "-female") would
     have that entire relation silently dropped, leaving it with zero
     relations and so absent from Word Sketch/Distributional Thesaurus
     entirely, even though Concordance (which has always done its own
     equivalent cleanup) would still find it. `_clean_word` deliberately
     leaves leading/trailing **apostrophes** untouched (only strips other
     punctuation like the stray hyphen above) — contraction fragments
     (`'s`/`'re`/`'m`/`'ll`/`'ve`) legitimately start with one, and `_eff`'s
     lemma-based expansion (below) depends on that apostrophe surviving to
     even recognise them as contractions; stripping it would silently skip
     that logic and key e.g. "they're" as the meaningless fragment "re"
     instead of "are".
   - **Segmented tokens** (`_segment_words`) — every spaCy doc token mapped
     back onto the timestamp(s) of the original Whisper `WordToken`(s) it
     came from, for **every** language. The dashboard's Concordance tab
     searches this instead of Whisper's raw, un-reprocessed word list, so it
     can never disagree with Word List/Collocations/Word Sketch/
     Distributional Thesaurus about what words exist in the transcript —
     e.g. Whisper's raw ASR output keeps English contractions (`don't`) and
     hyphenated compounds (`well-known`) glued together as one token, which
     spaCy splits apart (`do`/`n't`, `well`/`-`/`known`); for CJK, raw
     Whisper tokens are individual characters, which pkuseg merges into real
     multi-character words. Split-off sub-tokens (English) share their
     source token's timing; merged multi-character words (CJK) use the
     first/last covered token's start/end timestamp.

     Because spaCy splits a contraction into two separate tokens, searching
     Concordance for the literal typed string ("don't") would otherwise
     never match anything. `kw_search` (`dashboard/app.py`) detects this —
     a query with no whitespace that spaCy still tokenizes into two parts
     (deliberately narrower than "any 2-token query", so a genuine phrase
     search like "New York" is unaffected) — and instead searches
     `segmented_tokens` for the two raw split parts (`"do"`/`"n't"`)
     occurring adjacently. Each matching pair becomes one occurrence: the
     displayed word is the original typed contraction, the timestamp is the
     **first** token's, and the span extends to the second token's end.

## Multi-language support

- Whisper auto-detects the spoken language; the matching spaCy model is
  looked up in `_SPACY_MODELS` (currently `en → en_core_web_sm`,
  `zh → zh_core_web_sm`) and loaded lazily, then cached per-language so
  repeated jobs don't reload the model.
- For logographic scripts (`zh`/`ja`/`ko`), tokens are joined **without**
  spaces before being handed to spaCy so its tokenizer can re-segment words
  correctly, and n-grams are built from the spaCy doc's word tokens rather
  than Whisper's per-character tokens.
- No stopword filtering is applied anywhere (word list or collocations), for
  any language — `zh_core_web_sm`'s built-in `is_stop` flags mis-tag common
  Chinese content words (e.g. 人/好/大/是), and rather than maintain a curated
  override list for just Chinese, `is_stop` is ignored entirely so every
  language is treated consistently: every alphabetic, non-punctuation token
  is a valid word.
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
