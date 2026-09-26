# Analysis decisions

Choices that shape every number in these notebooks and are **not** recoverable
by reading the code, because the code records what was done and not why, nor
what was rejected.

Most entries here exist because a quantity was correct in English and silently
wrong in Chinese. That asymmetry is the single most common failure mode in this
pipeline, and the reason is always the same one described in the next section.

---

## 1. Two token streams

Every word-level number comes from one of two tokenisations, and they are not
interchangeable.

| | ASR tokens | spaCy tokens |
|---|---|---|
| produced by | Whisper (en) / SenseVoice (zh) | `en_core_web_sm` / `zh_core_web_sm` (pkuseg) |
| stored as | `verbal.tokens` per window | `artifacts.segmented_tokens` |
| carries | word, start, end, confidence | word, start, end (**no POS** — see §6) |
| owns | timings, the spoken surface | part of speech, lemma, dependencies |
| in English | ≈ words; splits nothing | **splits**: `don't` → `do` + `n't` |
| in Chinese | ≈ **characters** | **merges**: 女 + 性 → `女性` |

The two move in opposite directions by language, which is why one rule can
serve both: it undoes the English split and leaves the Chinese merge intact.

**`segmented_tokens` is the bridge.** `VerbalWorker._segment_words` maps each
spaCy token back onto the ASR token(s) it spans, so spaCy tokens inherit real
timings. In English each spaCy token lies inside exactly one ASR token and
takes its timings, which makes a shared `start_s` a reliable grouping key. In
Chinese one spaCy token spans several ASR tokens and takes the first one's
start and the last one's end.

### Measured scale of the difference

| corpus | spaCy words | ASR tokens | ratio |
|---|---|---|---|
| Ted (39 videos) | 66,604 | 66,956 | **1.005** |
| YiXi (29 videos) | 145,142 | 260,325 | **1.794** |

YiXi also holds 232,314 Han characters, so three different totals coexist there
and any "per 1,000 tokens" figure must say which it means.

### The governing rule

> **ASR for how much was said. spaCy for which words were said.**

*How much* — counts, rates, density; the recogniser's tokens are the speech.
*Which words* — frequency, keyness, concordance, collocates; these need POS and
lemma, and want `well-known` findable as `known`.

---

## 2. Where each number comes from

| field | English | Chinese |
|---|---|---|
| `verbal.transcript` | ASR | ASR |
| `verbal.tokens` | ASR | ASR |
| `verbal.word_count` | **ASR** | **spaCy** |
| `speech_rate_syl_per_s` | ASR count × 1.5 | **counted Han characters** |
| `n_tokens` (coverage only) | ASR | ASR — ≈ characters, see §7 |
| `artifacts.wordlist` | ASR surface + composite POS (§4) | spaCy (no-op regrouping) |
| `artifacts.segmented_tokens` | spaCy | spaCy |
| `artifacts.collocations` | spaCy dependency parse | spaCy dependency parse |
| `artifacts.ngrams` | **ASR — outlier, see §7** | spaCy |
| nb06 search index | ASR surface | spaCy (= ASR groups collapse to singletons) |
| flagging (`flag_verbal`) | spaCy, re-parsed live | spaCy, re-parsed live |

---

## 3. Decision log

### `word_count` means words, in both languages

`word_count` was the number of raw ASR tokens, so it meant *words* in English
and *characters* in Chinese under one column name — 234k against 130k on YiXi.

**Decided:** English keeps the ASR count; Chinese re-counts against spaCy.
Both now mean "words".

**Why not spaCy for English too:** spaCy splits contractions and emits
punctuation as tokens, turning 9 spoken words into 14. That is not a better
word count, only a different tokenisation. `_LOGOGRAPHIC` in
`workers/verbal_worker.py` gates the behaviour.

### Speech rate is counted for Chinese, estimated for English

`speech_rate_syl_per_s` was `word_count / duration * 1.5`, where 1.5 is an
assumed syllables-per-word ratio. Applied to Chinese it multiplied a count that
was *already syllabic* — YiXi's stored median was 7.2 syl/s against a counted
4.4, an inflation of **1.66×**.

**Decided:** Chinese counts Han characters (one character is one syllable);
everything else keeps the ×1.5 estimate. `core/fusion_engine._enrich` branches
on the script found in the transcript, not on a language label, because fusion
has no language code and the script is what decides which method is valid.

**Consequence:** the two corpora's speech rates come from **different
instruments** and must not be compared until English has a real syllable
counter (a pronunciation dictionary such as CMUdict, with an out-of-vocabulary
fallback). `notebooks/03_tedx_vs_yixi.ipynb` §8 records this as a non-goal.

### Frequency is normalised by segmented words, not ASR tokens

`corpus_wordlist` divided spaCy-derived counts by the artifact's
`total_tokens`, which is the **ASR** total. Harmless in English (1.005), a
1.8× understatement in Chinese.

**Decided:** the denominator is the sum of the same counts being divided.
Numerator and denominator must always be the same tokenisation.

`wordlist.total_tokens` still holds the ASR count and is deliberately unused.

### Part-of-speech is per occurrence, not per word type

The word list recorded the tag of a word's **first** occurrence in a video and
filed every later occurrence under it. Evidence: across 1202 Chinese and 626
English multi-tagged types, the per-tag video sets were **100% disjoint** — no
video ever showed a word under two tags, although nearly every talk uses `to`
both ways. Counts were never wrong, only their attribution.

**Decided:** the word list is keyed by `(word, POS)`. A word may have several
rows; its total is their sum.

**Consequence:** `dashboard/app.py` sums across rows when building a
word→frequency map. A plain dict comprehension silently kept only the
lowest-count tag.

### `mean_video_freq` is computed per video, then averaged

It averaged the artifact's pre-rounded per-row rates. Once a word holds several
tags inside one video, those rows each carry part of the video's rate and the
video is counted more than once — `watch` used twice as a verb and once as a
noun in a 10k-word talk averaged to 0.15 instead of its true 0.3.

**Decided:** sum each word's counts per video, compute the rate, then average
across videos. This also removes the dependency on the artifact's two-decimal
rounding.

Note it is a mean over **videos containing the word**, not over all videos.
海盗 reads 0.568 pooled and 19.020 as a mean — the second is "rate among
speakers who used it". Always read it beside `n_videos`.

### The search index uses spaCy tokens with ASR surfaces

nb06 first indexed `verbal.tokens`, so the YiXi index held 女 and 性 but never
女性 — **every Chinese query returned zero**, which reads as "this corpus does
not discuss marriage" rather than as a failure.

**Decided:** the index is built from `segmented_tokens`, regrouped to ASR
surfaces (§4), so words appear as spoken. A video with no stored segmented
tokens falls back to ASR tokens **and names itself in a warning**.

**Consequence:** searching `women` no longer matches `women's`, because they
are now distinct tokens. Concept lists in `flags/concepts_en_zh.yaml` must
name both forms where both are wanted.

### Rates share one denominator

The index also holds contraction fragments and numerals so concordance lines
read correctly, but they are excluded from rate denominators via an `is_word`
column, so a figure from nb06 sits on the same base as nb01 and nb03.

---

## 4. Composite part-of-speech tags

**Status: implemented.** `VerbalWorker._wordlist_from_segments` builds the word
list; `_segment_words` carries `pos` and `asr_idx`; both corpora rebuilt
through `scripts/backfill_verbal.py`.

### The problem

The word list counted spaCy tokens, so `don't` was recorded as `do` — inflating
the auxiliary — and `n't` was dropped entirely by the `isalpha()` filter.
`won't` and `can't` contributed the non-words **`wo`** and **`ca`**, which are
currently in the frequency table as vocabulary items.

### The scheme

Group spaCy tokens under the ASR token they came from; the surface is the
cleaned concatenation, the tag is the components joined by `+` in token order.

```
ASR "don't"   -> do(AUX) + n't(PART)   -> "don't"    AUX+PART
ASR "now,"    -> now(ADV) + ,(PUNCT)   -> "now"      ADV
ASR "women"   -> women(NOUN)           -> "women"    NOUN
```

Punctuation components are dropped from **both** surface and tag — without
this, 8,663 ordinary English words would acquire a composite tag. A group whose
surface cleans to empty is discarded.

**Backward compatible by construction:** a plain tag splits to a one-element
list, so consumers written for composites handle pre-backfill data unchanged.

### Measured inventory (6 Ted videos)

```
PRON+AUX  164   i'm, it's, we're, i've      VERB+PRON   7   let's
AUX+PART   51   don't, didn't, wasn't       ADV+AUX     2   here's
NOUN+PART  29   women's, men's              VERB+PART   2   gonna
PRON+VERB  12   there's                     PRON+PART   1   someone's
                                            ADJ+PART    1   other's
```

Three findings here rule out the cheaper alternatives that were considered.
`'s` takes **four different tags** by context (AUX, PART, VERB, PRON), so no
lookup table can resolve it. `gonna` has no apostrophe, so no string rule would
find it. And `let's` is **VERB+PRON** — its second component is *us*, a real
pronoun argument rather than a suffix — so storing only a head would discard
half of a genuine two-word token. Grouping by ASR token gets all three without
heuristics.

### Matching rules

| use | rule | implemented in |
|---|---|---|
| dashboard POS filter, search | **any component matches** | `_pos_parts` / `kw_wordlist_table` |
| nb01 content-word display cut | **first component matches** | `_aggregate_wordlist`'s `pos=` filter |

The dashboard's OTHER bucket takes any entry with a component outside the known
set, so a contraction appears under both its host's filter and OTHER. nb01
colours bars by the first component and hatches composites, so the legend stays
at five or six entries instead of fifteen.

"Any" is inclusive and right for exploration. It misclassifies `there's`
(PRON+VERB) as a content word, which is why the content cut uses the first
component instead: the host word carries the lexical class and the clitic does
not. `there's` → PRON, excluded. `women's` → NOUN, included.

**The first-component rule is the only heuristic in this scheme** — everything
else is derived. It is right for every case in the inventory above, but it is a
choice and should be stated in any write-up.

Because "any" makes filters overlap, **results across filters must not be
summed**.

### Why POS must be re-derived rather than looked up

Per-token POS is persisted nowhere. `segmented_tokens` has no tag, and the word
list is a type-level summary. A three-table join was measured against all 39
Ted videos:

| | with punctuation | punctuation dropped |
|---|---|---|
| exactly one POS row | 67.9% | 76.5% |
| more than one row (ambiguous) | 18.2% | **20.5%** |
| no row at all | 13.9% | 2.9% |

The clitics are absent from the word list **by construction** — `_clean_word`
deliberately preserves apostrophes, and the next check is `isalpha()`, which an
apostrophe fails. So the half of a contraction that needs a tag is exactly the
half the table excludes. And one token in five joins to several rows, requiring
a tie-break that would be a guess.

**Decided:** add `pos` (and `asr_idx`) to `_segment_words`' output and rebuild
the word lists through `scripts/backfill_verbal.py`. Storing the tag the worker
already computed is cheaper than approximating it in two places.

The backfill therefore rewrites **`artifacts.segmented_tokens`** as well as
the word list, because nb06 groups on `asr_idx` and cannot do so until that
field exists in storage. Videos processed before it was added fall through
ungrouped, one spaCy token per row, rather than failing.

**Rejected:** deriving the word list at read time in `_corpus.py`. It would
work, but the dashboard reads the stored artifact and would then disagree with
the notebooks about the corpus vocabulary.

---

## 5. Cross-corpus comparability

From `notebooks/03_tedx_vs_yixi.ipynb`. A measure crosses the language boundary
only if it is **a ratio whose units cancel**, **relative to the speaker's own
baseline**, **assigned by one language-blind classifier**, or **mapped through
an explicit bridge**.

| comparable | not comparable |
|---|---|
| `wrist_speed_mean` — shoulder-widths/s | `pitch_var_st` — Mandarin f0 carries lexical tone |
| `pitch_st` — semitones from own median | `speech_rate_syl_per_s` — different instruments |
| shot, angle, cuts — one classifier | `word_count` — pkuseg word ≠ English word |
| concept rates via `concepts_en_zh.yaml` | `mean_f0`, `mean_wrist_velocity` — voice type, pixels |

Tests are run at the **video level** (n = 39 and 29), not the window level.
Windows within a talk are strongly autocorrelated, and treating ~20,000 of them
as independent makes any difference significant.

---

## 6. Backfill scripts

None re-download or re-transcribe; all recompute from what MongoDB already
holds, so they are deterministic and cheap.

| script | recomputes | when to run |
|---|---|---|
| `backfill_verbal.py` | `word_count`, `speech_rate_syl_per_s`, `wordlist` | after any change to tokenisation or word-list construction |
| `backfill_dense_prosody.py` | f0 + intensity at 10 ms hop | needed for word-level pitch (nb06) and sub-second coupling (nb04 §9) |

**Run a backfill before adding videos, not after.** A corpus half-corrected is
worse than one consistently wrong: every per-corpus statistic then mixes two
definitions, and nothing announces it.

The notebook cache is keyed on job ids and window counts, **neither of which a
backfill changes** — so re-run `load_corpus(..., refresh=True)` once afterwards.

---

## 7. Known limitations

**n-grams are the last ASR-based artifact in English, and the filter is
destructive.** `alpha = [w for w in raw_words if w.isalpha()]` removes `don't`
from the sequence rather than splitting it, making its neighbours falsely
adjacent — *"I don't think"* produces the bigram **"i think"**. The stored
bigram lists contain no apostrophes anywhere, which confirms it. Chinese
already uses the spaCy doc here; English never got that branch.

**`n_tokens` is raw ASR tokens for both corpora**, so for YiXi it is roughly a
character count in a column called "tokens". Used only as a coverage indicator,
never as a statistic, but the name flatters the content.

**Multi-token lexicon entries can never match.** Entries are matched against
single segmented tokens, so 被压迫 — which pkuseg splits into 被 + 压迫 —
scores zero. This makes flag **F9 (women as patient) structurally unable to
fire on Chinese**, and excludes `passive_verbs` from nb03's keyness. An
adjacency matcher would fix this, phrase search, and English contraction search
in one change.

**English has no syllable counter**, so `speech_rate_syl_per_s` remains an
estimate there and the two corpora cannot be compared on speech rate.

**Talk genre is recorded nowhere**, so §6's confound list cannot be fully
checked in `04_multimodal_coupling.ipynb`.

**Gesture is magnitude only.** Deictic, iconic, metaphoric and beat gestures
are indistinguishable to this pipeline; no correlation among these columns can
speak to co-expressivity.
