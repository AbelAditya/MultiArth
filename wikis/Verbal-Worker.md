# Verbal Worker

Source: [`workers/verbal_worker.py`](../workers/verbal_worker.py)
Dashboard section: **Verbal Language**
Feature model: [`VerbalFeatures`](../core/models.py) (in `core/models.py`)

## What it does

`VerbalWorker` transcribes the full audio **once** — with **faster-whisper**
for most languages, **SenseVoice** (via `funasr`) for Chinese specifically —
then distributes word tokens into time windows and builds corpus-wide
lexical statistics with **spaCy**.

1. **Transcription** (`_transcribe`) — runs Whisper with
   `word_timestamps=True` and `vad_filter=True` (voice-activity filtering),
   auto-detecting the spoken language and returning a flat list of
   `WordToken`s (word, start/end time, confidence).

   Whisper *always* runs first, purely to detect the language — that part is
   cheap (an encoder pass over the first ~30s + the language-ID head, ~1-2s),
   well under the cost of the actual autoregressive decoding. If the
   detected language is in `_ALT_ASR_LANGS` (currently just `zh`),
   transcription is handed off to `_transcribe_alt` (SenseVoice) instead —
   Whisper's own `segments` generator is simply left unconsumed, so its
   decoding cost is never paid. If SenseVoice fails for any reason, it falls
   back to consuming Whisper's already-in-flight segments rather than
   failing the job outright.

   SenseVoice was adopted after a direct side-by-side accuracy comparison on
   a real Chinese sample: Whisper produced five homophone-confusion errors in
   a 46-second clip (e.g. "潜**意**默化" — not a real phrase — where SenseVoice
   correctly produced the actual idiom "潜**移**默化"), and ran ~5x slower on
   CPU. SenseVoice's Chinese `words` output is still per-character, same
   granularity as Whisper's, so pkuseg (via spaCy, in `_build_doc`/
   `_compute_corpus_stats`) still does the real word segmentation on top,
   unchanged by which engine produced the raw transcript. SenseVoice doesn't
   report a per-word confidence the way Whisper does, so its `WordToken`s
   get a flat `1.0` (unused downstream beyond display).

   Both engines are placed on the same device (`VerbalWorker`'s `device`
   param — already threaded from the CLI's `--device cpu|cuda` flag via
   `Orchestrator`; `funasr` falls back to CPU automatically if `cuda` is
   requested but unavailable). The dashboard's `Orchestrator(store=store)`
   call doesn't currently pass this through, so dashboard-driven jobs always
   run on CPU regardless of what's available — only the CLI exposes GPU
   selection today. SenseVoice is loaded lazily and cached (`_get_sensevoice`,
   mirroring `_get_nlp`'s per-language spaCy caching) — English-only sessions
   never load it at all.

   **Optional: SenseVoice can run remotely instead of locally.**
   `funasr`+`torch` is a genuinely heavy dependency to load into this
   process — set `SENSEVOICE_REMOTE_URL` (and `SENSEVOICE_API_KEY`) and
   `_transcribe_alt` calls that URL over HTTP instead
   (`_transcribe_alt_remote`), and this process never imports `funasr` at
   all. [`colab/sensevoice_server.ipynb`](../colab/sensevoice_server.ipynb)
   hosts the same model behind a small FastAPI endpoint, tunnelled out via
   ngrok, meant to be run in a free Colab session while you're actively
   analysing videos — see the notebook itself for setup (a free ngrok
   account, and a shared `API_KEY`). Session lifetime is Colab's own free-tier
   limits (disconnects after idling, ~12h hard cap) — this is "spin it up
   for a batch, then it's fine to lose it" infrastructure, not a permanent
   service; re-running the notebook gives a new URL each time, so
   `SENSEVOICE_REMOTE_URL` needs updating to match.

   A failed remote call (network down, tunnel expired, wrong API key) isn't
   a new failure mode — it's an exception out of `_transcribe_alt_remote`,
   which propagates through `_transcribe_alt` exactly like a local SenseVoice
   failure always has, hitting the same try/except in `_transcribe` that
   already falls back to Whisper's output for that job. Leaving
   `SENSEVOICE_REMOTE_URL` unset (the default) keeps everything local,
   unchanged from before this option existed.
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

## Values shown on the dashboard

The **Verbal Language** section has four tabs plus one KPI card, all driven
by this worker's output:

| Where | What it shows |
|---|---|
| KPI card: **Words** | Total word count for the whole video. |
| **Transcript** tab | The full transcript, with the line for whatever moment the video is currently playing highlighted. |
| **Concordance** tab | Every occurrence of a searched word, shown with the text around it (keyword-in-context), plus how often it occurs overall. |
| **Word Sketch & Thesaurus** tab | For a searched word: what it typically appears *with*, broken down by role (e.g. what it modifies, what modifies it, what verbs govern it); a **Distributional Thesaurus** listing other words used in similar contexts; and, if a second word is entered, a side-by-side comparison of the two words' usage patterns. |
| **Frequency** tab | A ranked word list for the whole transcript, filterable by part of speech (noun, verb, adjective, adverb). |

All four tabs operate on the same transcript and word-level timestamps — the
tab you're on changes how that data is sliced and displayed, not which
video's words you're looking at.

## Multi-language support

- Whisper auto-detects the spoken language; the matching spaCy model is
  looked up in `_SPACY_MODELS` (currently `en → en_core_web_sm`,
  `zh → zh_core_web_sm`) and loaded lazily, then cached per-language so
  repeated jobs don't reload the model.
- The **ASR engine itself** is also chosen by detected language —
  `_ALT_ASR_LANGS` (currently `{"zh"}`) routes to SenseVoice instead of
  Whisper; see `_transcribe`/`_transcribe_alt`. Add a code there (and a
  branch in `_transcribe_alt`) to route another language to a different
  engine the same way.
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

## Benchmark accuracy (published)

The two ASR engines this worker actually routes between — Whisper **small**
(the `model_size` default; both the `WhisperModel` used everywhere and the
`faster-whisper` reimplementation of it, which doesn't change accuracy)
and **SenseVoice-Small** — have a direct, published head-to-head comparison
from SenseVoice's own paper (Table 6, FunAudioLLM). This is the same
comparison that motivated adopting SenseVoice for Chinese specifically (see
"What it does" above), now with the actual published numbers behind the
qualitative side-by-side test that originally motivated it:

| Benchmark | Metric | SenseVoice-Small | Whisper-small |
|---|---|---|---|
| AISHELL-1 test | CER | **2.96%** | 10.04% |
| AISHELL-2 test_ios | CER | **3.80%** | 8.78% |
| WenetSpeech test_meeting | CER | **7.44%** | 25.62% |
| WenetSpeech test_net | CER | **7.84%** | 16.66% |
| Common Voice zh-CN | CER | **10.78%** | 19.60% |
| Common Voice en | WER | 14.71% | **14.85%** (~tied) |
| LibriSpeech test-clean | WER | 3.15% | **3.13%** (~tied) |

The pattern matches this worker's own routing logic exactly: SenseVoice
wins by a wide, consistent margin on every **Chinese** benchmark (roughly
2-3.5x lower error rate), while the two are essentially tied on the
**English** benchmarks (Common Voice `en`, LibriSpeech) — which is exactly
why `_ALT_ASR_LANGS` only routes `zh` to SenseVoice and leaves English (and
every other language) on Whisper: there's no published accuracy case for
switching English away from Whisper, only for Chinese.

Source: [FunAudioLLM: Voice Understanding and Generation Foundation Models
for Natural Interaction Between Humans and
LLMs](https://arxiv.org/html/2407.04051v1), Table 6.

## Package documentation

| Package | Role | Docs |
|---|---|---|
| faster-whisper | Speech-to-text transcription with word-level timestamps (all languages except `_ALT_ASR_LANGS`) | https://github.com/SYSTRAN/faster-whisper#readme |
| funasr | Runs SenseVoice, the Chinese-specific ASR engine | https://github.com/modelscope/FunASR#readme |
| torch / torchaudio | funasr/SenseVoice's inference backend (CPU or CUDA) | https://pytorch.org/docs/stable/index.html |
| requests | HTTP client for `_transcribe_alt_remote` (optional remote SenseVoice) | https://requests.readthedocs.io/en/latest/ |
| spaCy | Tokenization, lemmatization, POS tagging, dependency parsing | https://spacy.io/api |
| loguru | Transcription/job logging | https://loguru.readthedocs.io/en/stable/ |
| Pydantic | `VerbalFeatures` / `WordToken` models | https://docs.pydantic.dev/latest/ |

See also [Home](Home.md) for the full dependency list.
</content>
