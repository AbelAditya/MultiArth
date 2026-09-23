# Writing a flagging manifest

A manifest is a YAML file that defines **your** flags. The code knows nothing
about discursive functions — it only applies the rules you write here. Copy
`multiarth_cda.yaml`, edit, and bump `version`.

Every result stores your manifest's id, version and content hash, so any
number can be traced back to the exact rule that produced it.

## The rule in one sentence

A 5-second window earns a flag when **enough** of that flag's elements match
(`min_elements`, default 5) **and** at least one element from each required
macro-category matches (verbal, camera, acoustic, gesture). Elements you don't
list are not evaluated and cannot count.

## Smallest working manifest

```yaml
id: my-rules
version: 1

variables:
  pitch_level:    {source: pitch_st,          labels: [low, medium, high]}
  wrist_velocity: {source: wrist_speed_mean,  labels: [low, medium, high]}
  shot:           {source: dominant_shot_type, method: passthrough}

lexicons:
  first_singular: {pos: [PRON], en: [i, me, my], zh: [我]}
  cognition_verbs: {pos: [VERB], en: [know, think, believe], zh: [知道, 认为]}

elements:
  subject: {role: subject}
  verb:    {role: any}

macro_categories:
  verbal:   [subject, verb]
  camera:   [shot]
  acoustic: [pitch_level]
  gesture:  [wrist_velocity]

rule:
  min_elements: 3
  require_each_macro: [verbal, camera, acoustic, gesture]

flags:
  - id: F1
    name: Personal reflection
    criteria:
      subject: [first_singular]
      verb: [cognition_verbs]
      shot: [close_up]
      pitch_level: [low]
      wrist_velocity: [low, medium]
```

## `variables` — continuous numbers into labels

Each entry turns one stored measurement into the words your flags use.

```yaml
pitch_level:
  source: pitch_st        # any column in the window data
  scope: per_speaker      # per_speaker (default) | per_corpus
  method: tertiles        # tertiles (default) | quantiles | thresholds | passthrough
  labels: [low, medium, high]
```

| method | what it does | needs |
|---|---|---|
| `tertiles` | equal thirds of the data | `labels` (3 of them) |
| `quantiles` | your own cut points, as fractions | `edges: [0.5]` + one more label than edges |
| `thresholds` | your own cut points, in real units | `edges: [8.0]` + one more label than edges |
| `passthrough` | value is already a word | optional `map:` to rename values |

**Scope matters more than the method.** `per_speaker` (the default) means
"low pitch" is low *for that speaker*. With `per_corpus`, a deep-voiced speaker
would be "low" in every window — you would be measuring voice type, not
discourse style.

**Scope applies to quantiles only.** `tertiles` and `quantiles` measure their
cut points over a population, so they need to know which one. `thresholds`
edges are literal values in the source's own units and are the same for every
speaker, so no scope applies — use it when the unit is already comparable
across people (a dB range, a count of cuts).

Useful sources already available per window: `pitch_st`, `pitch_var_st`,
`mean_intensity_db`, `intensity_range_db`, `wrist_speed_mean`,
`wrist_speed_p90`, `speech_rate_syl_per_s`, `word_count`, `cut_count`,
`mean_face_bbox_area`, `dominant_shot_type`, `horizontal_angle`,
`vertical_angle`.

## `lexicons` and `elements` — the words

A lexicon is a word list plus the parts of speech it applies to:

```yaml
agency_verbs:
  pos: [VERB]          # empty list = any part of speech
  match: lemma         # lemma (default) | surface
  en: [fight, resist, demand]
  zh: [争取, 反抗, 要求]
```

`match: lemma` means *fights*, *fought* and *fighting* all match `fight`.
Use `surface` when you need the exact written form.

A lexicon with parts of speech but **no words** for a language matches on part
of speech alone — that is how "any proper noun" is expressed.

Elements say **where in the sentence** a hit must occur:

```yaml
elements:
  subject: {role: subject}   # nsubj, nsubjpass …
  object:  {role: object}    # obj, dobj, pobj …
  verb:    {role: any}
```

Role is what separates "women demand change" from "women are silenced": the
same word list, different grammatical position. Without it those two flags
would be indistinguishable.

## `macro_categories` and `rule`

```yaml
macro_categories:
  verbal:   [subject, object, verb]
  camera:   [shot, horizontal, vertical]
  acoustic: [pitch_level, pitch_variation]
  gesture:  [wrist_velocity]

rule:
  min_elements: 5
  require_each_macro: [verbal, camera, acoustic, gesture]
```

You may define any macro-categories you like — add `scene:` or split camera
into framing and angle. Only the ones listed in `require_each_macro` are
compulsory.

## `flags`

```yaml
- id: F8
  name: Women as agent
  criteria:
    subject: [women_terms]       # a list = any of these counts
    verb: [agency_verbs]
    shot: [close_up, medium]
    pitch_level: [medium, high]
    wrist_velocity: [high]
  min_elements: 4                # optional: override the default for this flag
```

A criterion lists **acceptable values**. Any one of them matching counts as
that element matched. Omit an element (or write `/`) to leave it unevaluated.

## Errors you will see

The manifest is checked at load time, because a mistyped rule that silently
never fires looks like a finding.

| message | cause |
|---|---|
| `criterion 'shott' is neither a variable nor an element` | typo in a criterion name |
| `pitch_level cannot be 'quiet' — its labels are low, medium, high` | value not in that variable's labels |
| `references undefined lexicon(s)` | a verbal criterion naming a lexicon you didn't define |
| `needs 5 matches but specifies only 4 criteria — it can never fire` | threshold above the number of criteria |
| `macro-category verbal lists undefined element(s)` | element in a macro-category that doesn't exist |

## Practical advice

**Start with a threshold you can defend, then look at the near misses.** Every
run reports flags that fell one short and *which element* blocked them. If one
element blocks most near misses, that element — usually a word list — is your
bottleneck, not the threshold.

**Check verbal coverage before reading any flag count.** The report says how
often `subject`, `object` and `verb` were filled at all. A flag requiring a
subject can never exceed the share of windows that have one.

**Chinese omits subjects.** Mandarin routinely drops the grammatical subject,
so a manifest requiring the verbal macro-category will flag fewer Chinese
windows for reasons of grammar, not of style. Either accept it and say so, or
relax `require_each_macro` for that corpus — but then the two corpora are not
measured the same way, which has to be stated.

**Word lists are the real work.** Everything else is a threshold on data that
already exists. Budget your time accordingly, and grow the lists from what the
near-miss report shows you are missing.

**Bump `version` on every change**, however small. Two results computed under
different rules are not comparable, and the version is what tells you.
