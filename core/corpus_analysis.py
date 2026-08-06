"""
core/corpus_analysis.py
-----------------------
Local corpus analysis using spaCy dependency parse on the video's own transcript.

All functions are pure: they take pre-built data structures and return results.
The heavy spaCy doc parsing is done once in verbal_worker and stored via FeatureStore.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict


def _clean_word(text: str) -> str:
    """
    Strip leading/trailing non-word characters — stray punctuation Whisper
    sometimes glues onto a token with no space (e.g. transcribing
    "male-female" as the two raw tokens "male"/"-female", leaving the
    hyphen stuck to the front of "female"). Same normalization the
    dashboard's Concordance search already applies when matching tokens
    (dashboard/app.py's kw_search); used here too so a token like "-female"
    doesn't fail an alpha/validity check purely because of an attached
    punctuation character — which would otherwise silently drop it, and
    anything that depends on it, from Word List/Collocations.
    """
    return re.sub(r"^[^\w]+|[^\w]+$", "", text)


# "'s"/"'re"/"'m" all lemmatise to "be", which would throw away real
# tense/person distinctions (is vs are vs am) if we just used the lemma —
# so these three get an explicit override in _eff() (shared by
# extract_collocation_en and the dashboard's keyword search) to preserve
# that. Every other contraction fragment ("'ll", "'ve", the
# "wo"/"ca"/"sha" stems of won't/can't/shan't, "n't", pronoun contractions
# like "let's" -> "us") doesn't have that problem: its lemma is already the
# correct, unambiguous expansion, so no table entry is needed for those.
_BE_CONTRACTION_EXPANSIONS = {
    "'s":  "is",
    "'re": "are",
    "'m":  "am",
}


def _eff(token) -> str:
    """
    Effective surface key for an English spaCy token in extract_collocation_en.

    Expands contraction fragments to their spelled-out form, so "he's happy"/
    "he is happy" key the same, "don't"/"do not" key the same, etc. — but
    tense/person stay distinct (is/are/am/was/were don't merge into one
    another). Content words keep surface-form keying (see
    extract_collocation_en's docstring); this only touches the closed set of
    apostrophe/truncated-stem fragments below.

    Also used by the dashboard's keyword search (dashboard/app.py) to resolve
    a typed contraction into its constituent words for the word-sketch-diff
    view, so both places normalise contractions identically.
    """
    text = _clean_word(token.text.lower())
    if "'" not in text and text not in ("wo", "ca", "sha"):
        return text
    lemma = token.lemma_.lower()
    if lemma == text:
        # Lemma adds no information — e.g. possessive "'s" ("John's"),
        # which spaCy leaves un-lemmatised, unlike copula/auxiliary "'s".
        return text
    if lemma == "be" and text in _BE_CONTRACTION_EXPANSIONS:
        return _BE_CONTRACTION_EXPANSIONS[text]
    return lemma

_REL_LABELS: dict[str, str] = {
    "subj_of":         "subject of",
    "obj_of":          "object of",
    "has_subj":        "has subject",
    "has_obj":         "has object",
    "pobj_of":         "object of preposition",
    "has_pobj":        "has object of preposition",
    "modified_by":     "modified by",
    "modifies":        "modifies",
    "and_or":          "coordinated with",
    "modified_by_adv": "modified by adverb",
    "verb_comp_of":    "verb complement of",
    "has_verb_comp":   "has verb complement",
    "takes_aux":       "takes auxiliary",
    "aux_of":          "auxiliary of",
    "negates":         "negates",
    "negated_by":      "negated by",
    "takes_prep":      "takes preposition",
    "prep_of":         "preposition of",
    "modifies_adv":    "modifies (adverb)",
    "mark_of":         "marks clause of",
    "takes_mark":      "takes marker",
}

_DISPLAY_ORDER = [
    "subj_of", "obj_of", "has_subj", "has_obj", "pobj_of", "has_pobj",
    "verb_comp_of", "has_verb_comp",
    "takes_aux", "aux_of", "negates", "negated_by",
    "modified_by", "modifies", "and_or", "modified_by_adv", "modifies_adv",
    "takes_prep", "prep_of", "mark_of", "takes_mark",
]

# extract_collocation_en (and, via the shared _capture_dep_relations helper,
# extract_collocations_zh too) captures every dependency relation generically
# (using spaCy's own dep_ labels) rather than hand-enumerating specific ones —
# see _capture_dep_relations' docstring. These three control that:
#   _EXCLUDED_DEPS      — purely structural/punctuation labels, never captured.
#   _SYMMETRIC_DEPS      — relations where both sides play the same role
#                          (recorded under one shared name both ways).
#   _DEP_RELATION_NAMES — (dependent-side name, head-side name) for the
#                          relations common enough to have an established
#                          readable name; anything else automatically falls
#                          back to "{dep}_of" / "has_{dep}" using the raw
#                          dependency label, so a new/uncommon relation still
#                          gets captured without needing an entry here.
_EXCLUDED_DEPS = frozenset({})

_SYMMETRIC_DEPS = frozenset({"conj"})

_DEP_RELATION_NAMES: dict[str, tuple[str, str]] = {
    "nsubj": ("subj_of", "has_subj"), "nsubjpass": ("subj_of", "has_subj"),
    "nsubj:pass": ("subj_of", "has_subj"),
    "csubj": ("subj_of", "has_subj"), "csubjpass": ("subj_of", "has_subj"),
    "obj": ("obj_of", "has_obj"), "dobj": ("obj_of", "has_obj"),
    "iobj": ("obj_of", "has_obj"), "attr": ("obj_of", "has_obj"),
    "acomp": ("obj_of", "has_obj"), "dative": ("obj_of", "has_obj"),
    "oprd": ("obj_of", "has_obj"),
    "pobj": ("pobj_of", "has_pobj"),
    "amod": ("modifies", "modified_by"), "nummod": ("modifies", "modified_by"),
    "quantmod": ("modifies", "modified_by"), "det": ("modifies", "modified_by"),
    "poss": ("modifies", "modified_by"), "relcl": ("modifies", "modified_by"),
    "nmod": ("modifies", "modified_by"), "compound": ("modifies", "modified_by"),
    "compound:nn": ("modifies", "modified_by"),
    "ccomp": ("verb_comp_of", "has_verb_comp"), "xcomp": ("verb_comp_of", "has_verb_comp"),
    "aux": ("aux_of", "takes_aux"), "aux:asp": ("aux_of", "takes_aux"),
    "aux:modal": ("aux_of", "takes_aux"),
    "neg": ("negates", "negated_by"),
    "prep": ("prep_of", "takes_prep"), "agent": ("prep_of", "takes_prep"),
    "advmod": ("modifies_adv", "modified_by_adv"),
    "npadvmod": ("modifies_adv", "modified_by_adv"),
    "mark": ("mark_of", "takes_mark"), "mark:clf": ("mark_of", "takes_mark"),
}

# ── Chinese: positional relations (own heuristics) + the same generic
#    dependency relations extract_collocation_en uses (shared names) ─────────

_DISPLAY_ORDER_ZH_POSITIONAL = [
    "next_left",  "next_right",
    "verb_left",  "verb_right",
    "noun_left",  "noun_right",
    "adj_left",   "adj_right",
    "adv_left",   "adv_right",
    "conj",
]

# Used only by get_word_sketch to detect whether a profile came from the
# Chinese extractor. Must stay exclusive to Chinese's own positional relation
# names — extract_collocations_zh's profiles now also carry the shared
# dependency-relation names (subj_of, modifies, ...) that English profiles
# have too, so including those here would misdetect English words as Chinese.
_ZH_KEY_SET = frozenset(_DISPLAY_ORDER_ZH_POSITIONAL)

_REL_LABELS_ZH: dict[str, str] = {
    **_REL_LABELS,
    "next_left":  "next left",
    "next_right": "next right",
    "verb_left":  "verb left",
    "verb_right": "verb right",
    "noun_left":  "noun left",
    "noun_right": "noun right",
    "adj_left":   "adjective left",
    "adj_right":  "adjective right",
    "adv_left":   "adverb left",
    "adv_right":  "adverb right",
    "conj":       "conjunction",
}

_DISPLAY_ORDER_ZH = _DISPLAY_ORDER_ZH_POSITIONAL + _DISPLAY_ORDER

_ZH_CONJUNCTIONS = frozenset({
    "和", "与", "及", "或", "但", "而",
    "但是", "然而", "不过", "可是", "还是",
    "而且", "并且", "或者",
})

_ZH_SCAN_DEPTH = 6


def _serialise(raw: dict[str, dict[str, Counter]]) -> dict[str, dict[str, list]]:
    """Shared by extract_collocation_en and extract_collocations_zh: Counter
    -> sorted [[word, count], ...] list, dropping any word left with no
    non-empty relations."""
    result: dict[str, dict[str, list]] = {}
    for word, rels in raw.items():
        rel_dict: dict[str, list] = {}
        for rel, counter in rels.items():
            pairs = [[w, c] for w, c in counter.most_common()]
            if pairs:
                rel_dict[rel] = pairs
        if rel_dict:
            result[word] = rel_dict
    return result


def _capture_dep_relations(doc, raw, eff, is_valid_word) -> None:
    """
    Shared by extract_collocation_en and extract_collocations_zh: walks every
    token's relation to its head generically, using spaCy's own dep_ labels,
    rather than hand-enumerating specific ones — so a new relation doesn't
    need to be added one at a time as gaps are found. Mutates *raw* in place
    so a caller can accumulate additional relations of its own (e.g.
    Chinese's positional ones) into the same structure.

    Both sides of every relation are recorded in one pass (dependent's view
    via _DEP_RELATION_NAMES' first name, head's view via its second) — a
    small exclude list (_EXCLUDED_DEPS) drops purely structural/punctuation
    labels, and anything not explicitly named in _DEP_RELATION_NAMES
    automatically falls back to "{dep}_of" / "has_{dep}" using the raw
    dependency label.

    This favours recall over precision on purpose: earlier versions gated
    specific relations on the head's or token's POS tag (e.g. "only count
    subj_of if the head is a VERB/AUX"), but that gating turned out to
    silently drop real relations whenever spaCy's POS tagger — not the
    dependency parser — got a word wrong (e.g. tagging "masculine" as NOUN
    instead of ADJ in predicate position). The only filter is *is_valid_word*
    (a real word, not punctuation).

    *eff*: surface-key function (contraction-normalising for English,
    plain lowercased text for Chinese).
    """
    for token in doc:
        if not is_valid_word(token):
            continue

        dep = token.dep_
        if dep in _EXCLUDED_DEPS or token.head is token or not is_valid_word(token.head):
            continue

        t = eff(token)
        h = eff(token.head)

        if dep in _SYMMETRIC_DEPS:
            raw[t]["and_or"][h] += 1
            raw[h]["and_or"][t] += 1
        else:
            dep_of, has_dep = _DEP_RELATION_NAMES.get(dep, (f"{dep}_of", f"has_{dep}"))
            raw[t][dep_of][h] += 1
            raw[h][has_dep][t] += 1


def extract_collocation_en(doc) -> dict[str, dict[str, list]]:
    """
    Build a collocations dict from an English spaCy Doc.

    Returns:
        { word: { relation_key: [[collocate_word, count], ...] } }

    Keyed by surface form (not lemma) — inflected forms of the same word
    (e.g. "run" / "running") get separate profiles. See _capture_dep_relations
    for how relations are captured.
    """
    raw: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    def _is_valid(token) -> bool:
        if token.is_punct:
            return False
        cleaned = _clean_word(token.text).replace("'", "")
        return bool(cleaned) and cleaned.isalpha()

    _capture_dep_relations(doc, raw, _eff, _is_valid)

    return _serialise(raw)


def extract_collocations_zh(doc) -> dict[str, dict[str, list]]:
    """
    Collocations for Chinese: this function's own positional relations
    (proximity-based — nearest neighbour in each direction, an approach
    originally used because zh_core_web_sm assigned the generic 'dep' label
    to most tokens) PLUS the same generic dependency-relation capture
    extract_collocation_en uses (_capture_dep_relations, shared relation
    names) — now that a properly word-segmented Chinese doc (see
    _segment_words in workers/verbal_worker.py, and the spacy-pkuseg
    dependency fix) produces real, usable dependency labels instead of that
    degenerate fallback.

    No stopword filtering — every alphabetic, non-punctuation token is
    eligible as a target, same as extract_collocation_en.
    """
    raw: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    def _eff(token) -> str:
        return _clean_word(token.text.lower())

    def _content(token) -> bool:
        if token.is_punct:
            return False
        cleaned = _clean_word(token.text)
        return bool(cleaned) and cleaned.isalpha()

    def _nearest(toks, origin, direction, pos_filter=None):
        """Return the first content token in *direction* whose POS is in
        *pos_filter* (any content token if None), within _ZH_SCAN_DEPTH steps."""
        n = len(toks)
        j = origin + direction
        content_seen = 0
        while 0 <= j < n:
            tok = toks[j]
            if tok.is_punct:
                j += direction
                continue
            if _content(tok):
                if pos_filter is None or tok.pos_ in pos_filter:
                    return _eff(tok)
                content_seen += 1
                if content_seen >= _ZH_SCAN_DEPTH:
                    break
            j += direction
        return None

    for sent in doc.sents:
        toks = list(sent)
        n = len(toks)

        for i, token in enumerate(toks):
            if not _content(token):
                continue
            t = _eff(token)

            for rel, direction in (("next_left", -1), ("next_right", 1)):
                hit = _nearest(toks, i, direction)
                if hit and hit != t:
                    raw[t][rel][hit] += 1

            for rel, direction in (("verb_left", -1), ("verb_right", 1)):
                hit = _nearest(toks, i, direction, {"VERB", "AUX"})
                if hit and hit != t:
                    raw[t][rel][hit] += 1

            for rel, direction in (("noun_left", -1), ("noun_right", 1)):
                hit = _nearest(toks, i, direction, {"NOUN", "PROPN"})
                if hit and hit != t:
                    raw[t][rel][hit] += 1

            for rel, direction in (("adj_left", -1), ("adj_right", 1)):
                hit = _nearest(toks, i, direction, {"ADJ"})
                if hit and hit != t:
                    raw[t][rel][hit] += 1

            for rel, direction in (("adv_left", -1), ("adv_right", 1)):
                hit = _nearest(toks, i, direction, {"ADV"})
                if hit and hit != t:
                    raw[t][rel][hit] += 1

            # conjunction: look right for a conjunction particle then a content word
            j = i + 1
            while j < n and j <= i + _ZH_SCAN_DEPTH:
                tok = toks[j]
                if tok.is_punct:
                    j += 1
                    continue
                if tok.text in _ZH_CONJUNCTIONS:
                    k = j + 1
                    while k < n and k <= j + 3:
                        right = toks[k]
                        if _content(right):
                            c = _eff(right)
                            if c != t:
                                raw[t]["conj"][c] += 1
                                raw[c]["conj"][t] += 1
                            break
                        k += 1
                    break
                if _content(tok):
                    break
                j += 1

    # Same generic dependency-relation capture extract_collocation_en uses,
    # accumulated into the same raw dict alongside the positional relations above.
    _capture_dep_relations(doc, raw, _eff, _content)

    return _serialise(raw)


def get_word_sketch(collocations: dict, word: str) -> dict:
    """
    Return the collocation profile for *word*.

    Returns:
        {
          "word":            str,
          "found":           bool,
          "total_relations": int,
          "relations": [{"key": str, "name": str, "words": [[word, count], ...]}, ...]
        }
    """

    key = word.lower().strip()
    profile = collocations.get(key, {})

    is_zh = bool(profile.keys() & _ZH_KEY_SET)
    display_order = _DISPLAY_ORDER_ZH if is_zh else _DISPLAY_ORDER
    label_map = _REL_LABELS_ZH if is_zh else _REL_LABELS

    relations = []
    seen = set()
    for rel_key in display_order:
        if rel_key in profile:
            relations.append({
                "key":   rel_key,
                "name":  label_map.get(rel_key, rel_key),
                "words": profile[rel_key][:12],
            })
            seen.add(rel_key)
    for rel_key, pairs in profile.items():
        if rel_key not in seen:
            relations.append({
                "key":   rel_key,
                "name":  label_map.get(rel_key, rel_key),
                "words": pairs[:12],
            })

    return {
        "word":            key,
        "found":           bool(profile),
        "total_relations": len(relations),
        "relations":       relations,
    }


def word_sketch_diff(collocations: dict, word1: str, word2: str) -> dict:
    """
    Compare the collocational profiles of two words.

    Returns:
        {
          "word1": str, "word2": str,
          "found1": bool, "found2": bool,
          "relations": {
            rel_key: {
              "name":   str,
              "shared": [[word, count1, count2], ...],
              "only1":  [[word, count], ...],
              "only2":  [[word, count], ...],
            }
          }
        }
    """
    k1, k2 = word1.lower().strip(), word2.lower().strip()
    p1 = collocations.get(k1, {})
    p2 = collocations.get(k2, {})

    all_rels = set(p1.keys()) | set(p2.keys())
    diff_rels: dict = {}

    ordered = _DISPLAY_ORDER + [r for r in all_rels if r not in _DISPLAY_ORDER]
    for rel in ordered:
        if rel not in all_rels:
            continue
        words1 = {w: c for w, c in p1.get(rel, [])}
        words2 = {w: c for w, c in p2.get(rel, [])}
        if not words1 and not words2:
            continue

        shared = sorted(
            [[w, words1[w], words2[w]] for w in words1 if w in words2],
            key=lambda x: -(x[1] + x[2])
        )[:8]
        only1 = sorted(
            [[w, c] for w, c in words1.items() if w not in words2],
            key=lambda x: -x[1]
        )[:8]
        only2 = sorted(
            [[w, c] for w, c in words2.items() if w not in words1],
            key=lambda x: -x[1]
        )[:8]

        if shared or only1 or only2:
            diff_rels[rel] = {
                "name":   _REL_LABELS.get(rel, rel),
                "shared": shared,
                "only1":  only1,
                "only2":  only2,
            }

    return {
        "word1":     k1,
        "word2":     k2,
        "found1":    bool(p1),
        "found2":    bool(p2),
        "relations": diff_rels,
    }


def distributional_thesaurus(
    collocations: dict,
    word: str,
    top_n: int = 12,
) -> list[dict]:
    """
    Find words with similar distributional behaviour using Jaccard similarity
    over (relation, collocate) context tuples derived from the local corpus.

    Returns list of {"word": str, "score": float, "shared": int}
    sorted by score descending.
    """
    key = word.lower().strip()
    profile = collocations.get(key, {})
    if not profile:
        return []

    target_ctx: set[tuple] = {
        (rel, word)
        for rel, pairs in profile.items()
        for word, _ in pairs
    }
    if not target_ctx:
        return []

    results = []
    for other_key, other_profile in collocations.items():
        if other_key == key:
            continue
        other_ctx: set[tuple] = {
            (rel, word)
            for rel, pairs in other_profile.items()
            for word, _ in pairs
        }
        if not other_ctx:
            continue

        intersection = len(target_ctx & other_ctx)
        if intersection == 0:
            continue
        score = intersection / len(target_ctx | other_ctx)

        results.append({
            "word":   other_key,
            "score":  round(score, 3),
            "shared": intersection,
        })

    results.sort(key=lambda x: -x["score"])
    return results[:top_n]
