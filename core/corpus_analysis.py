"""
core/corpus_analysis.py
-----------------------
Local corpus analysis using spaCy dependency parse on the video's own transcript.

All functions are pure: they take pre-built data structures and return results.
The heavy spaCy doc parsing is done once in verbal_worker and stored via FeatureStore.
"""

from __future__ import annotations

from collections import Counter, defaultdict


# "'s"/"'re"/"'m" all lemmatise to "be", which would throw away real
# tense/person distinctions (is vs are vs am) if we just used the lemma —
# so these three get an explicit override in extract_collocations' _eff()
# to preserve that. Every other contraction fragment ("'ll", "'ve", the
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
    Effective surface key for an English spaCy token in extract_collocations.

    Expands contraction fragments to their spelled-out form, so "he's happy"/
    "he is happy" key the same, "don't"/"do not" key the same, etc. — but
    tense/person stay distinct (is/are/am/was/were don't merge into one
    another). Content words keep surface-form keying (see extract_collocations'
    docstring); this only touches the closed set of apostrophe/truncated-stem
    fragments below.

    Also used by the dashboard's keyword search (dashboard/app.py) to resolve
    a typed contraction into its constituent words for the word-sketch-diff
    view, so both places normalise contractions identically.
    """
    text = token.text.lower()
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
}

_DISPLAY_ORDER = [
    "subj_of", "obj_of", "has_subj", "has_obj", "pobj_of",
    "verb_comp_of", "has_verb_comp",
    "takes_aux", "aux_of", "negates", "negated_by",
    "modified_by", "modifies", "and_or", "modified_by_adv",
    "takes_prep", "prep_of",
]

# ── Chinese positional / POS-based relations ──────────────────────────────────

_REL_LABELS_ZH: dict[str, str] = {
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

_DISPLAY_ORDER_ZH = [
    "next_left",  "next_right",
    "verb_left",  "verb_right",
    "noun_left",  "noun_right",
    "adj_left",   "adj_right",
    "adv_left",   "adv_right",
    "conj",
]

_ZH_KEY_SET = frozenset(_DISPLAY_ORDER_ZH)

_ZH_CONJUNCTIONS = frozenset({
    "和", "与", "及", "或", "但", "而",
    "但是", "然而", "不过", "可是", "还是",
    "而且", "并且", "或者",
})

_ZH_SCAN_DEPTH = 6


def extract_collocations(doc) -> dict[str, dict[str, list]]:
    """
    Build a collocations dict from a spaCy Doc.

    Returns:
        { word: { relation_key: [[collocate_word, count], ...] } }

    Keyed by surface form (not lemma) — inflected forms of the same word
    (e.g. "run" / "running") get separate profiles.

    Only content words (non-stop, alpha, length >= 2) appear as targets.
    Collocates are filtered to alpha-only but may include light function words
    when they are verbs/adjectives headwords.
    """
    raw: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    def _content(token) -> bool:
        return not token.is_punct and token.text.replace("'", "").isalpha()

    def _alpha(token) -> bool:
        return not token.is_punct and token.text.replace("'", "").isalpha()

    for token in doc:
        if not _content(token):
            continue

        t = _eff(token)
        h = _eff(token.head)
        dep = token.dep_

        # ── Token as dependent ──────────────────────────────────────────────
        if dep in ("nsubj", "nsubjpass", "nsubj:pass", "csubj", "csubjpass"):
            # csubj/csubjpass = clausal subject ("what she said surprised me")
            # — the clause's own verb stands in for the whole clause here.
            if token.head.pos_ in ("VERB", "AUX") and _alpha(token.head):
                raw[t]["subj_of"][h] += 1

        elif dep in ("obj", "dobj", "iobj", "attr", "acomp", "dative", "oprd"):
            # attr/acomp = predicate nominal/adjective of a copula ("it's an
            # accident", "it seems fine"); dative = indirect object; oprd =
            # object predicate ("painted the house red"). All treated like an
            # object since they complete the verb's meaning the same way.
            if token.head.pos_ in ("VERB", "AUX") and _alpha(token.head):
                raw[t]["obj_of"][h] += 1

        elif dep == "pobj":
            # Object of a preposition ("in an accident") — the preposition
            # itself becomes the collocate, e.g. accident -> pobj_of -> [in, by].
            # The preposition is always a stopword so it can never become a
            # target itself, only ever a collocate here.
            if token.head.pos_ == "ADP" and _alpha(token.head):
                raw[t]["pobj_of"][h] += 1

        elif dep in ("amod", "nummod", "quantmod", "det", "poss", "relcl"):
            # nummod/quantmod = numeric modifiers ("three years", "about five");
            # det = determiners ("the", "this" — incl. demonstratives, a real
            # mannerism signal); poss = possessives ("my book"); relcl = relative
            # clause modifier ("the vase that broke"). All are "X modifies Y".
            if _alpha(token.head):
                raw[t]["modifies"][h] += 1
                raw[h]["modified_by"][t] += 1

        elif dep in ("nmod", "compound", "compound:nn"):
            if token.head.pos_ in ("NOUN", "PROPN") and _alpha(token.head):
                raw[t]["modifies"][h] += 1
                raw[h]["modified_by"][t] += 1

        elif dep == "conj":
            if _alpha(token.head):
                raw[t]["and_or"][h] += 1
                raw[h]["and_or"][t] += 1

        elif dep in ("ccomp", "xcomp"):
            if token.pos_ in ("VERB", "AUX") and token.head.pos_ in ("VERB", "AUX") and _alpha(token.head):
                raw[t]["verb_comp_of"][h] += 1

        elif dep in ("aux", "aux:asp", "aux:modal"):
            if token.head.pos_ in ("VERB", "AUX") and _alpha(token.head):
                raw[t]["aux_of"][h] += 1

        elif dep == "neg":
            if token.head.pos_ in ("VERB", "ADJ", "AUX") and _alpha(token.head):
                raw[t]["negates"][h] += 1

        elif dep in ("prep", "agent"):
            # agent = the "by" marker in passives ("broken by him") — same
            # shape as a regular prep attachment.
            if _alpha(token.head):
                raw[t]["prep_of"][h] += 1

        # ── Token as head — inspect children ───────────────────────────────
        for child in token.children:
            if not _alpha(child):
                continue
            c = _eff(child)
            cdep = child.dep_

            if cdep in ("nsubj", "nsubjpass", "nsubj:pass", "csubj", "csubjpass") and token.pos_ in ("VERB", "AUX"):
                raw[t]["has_subj"][c] += 1
            elif cdep in ("obj", "dobj", "attr", "acomp", "dative", "oprd") and token.pos_ in ("VERB", "AUX"):
                raw[t]["has_obj"][c] += 1
            elif cdep in ("advmod", "npadvmod") and token.pos_ in ("VERB", "ADJ", "AUX"):
                raw[t]["modified_by_adv"][c] += 1
            elif cdep in ("ccomp", "xcomp") and child.pos_ in ("VERB", "AUX") and token.pos_ in ("VERB", "AUX"):
                raw[t]["has_verb_comp"][c] += 1
            elif cdep in ("aux", "aux:asp", "aux:modal") and token.pos_ in ("VERB", "AUX"):
                raw[t]["takes_aux"][c] += 1
            elif cdep == "neg" and token.pos_ in ("VERB", "ADJ", "AUX"):
                raw[t]["negated_by"][c] += 1
            elif cdep in ("prep", "agent"):
                raw[t]["takes_prep"][c] += 1

    # Serialise: Counter → sorted [[word, count], ...] list
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


def extract_collocations_zh(doc, stopwords: frozenset = frozenset()) -> dict[str, dict[str, list]]:
    """
    Positional + POS-filtered collocations for Chinese.

    Scans linearly within each sentence rather than using dependency labels,
    because zh_core_web_sm assigns the generic 'dep' label to most tokens,
    making dep-based extraction almost always empty for Chinese text.

    stopwords: explicit set to exclude — passed in rather than read from
    token.is_stop, which is unreliable for lazily-created spaCy lexemes.
    """
    raw: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    def _eff(token) -> str:
        return token.text.lower()

    def _content(token) -> bool:
        return token.text not in stopwords and token.is_alpha and not token.is_punct

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
