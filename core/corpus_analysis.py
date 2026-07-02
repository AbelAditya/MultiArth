"""
core/corpus_analysis.py
-----------------------
Local corpus analysis using spaCy dependency parse on the video's own transcript.

All functions are pure: they take pre-built data structures and return results.
The heavy spaCy doc parsing is done once in verbal_worker and stored via FeatureStore.
"""

from __future__ import annotations

from collections import Counter, defaultdict

_REL_LABELS: dict[str, str] = {
    "subj_of":         "subject of",
    "obj_of":          "object of",
    "has_subj":        "has subject",
    "has_obj":         "has object",
    "modified_by":     "modified by",
    "modifies":        "modifies",
    "and_or":          "coordinated with",
    "modified_by_adv": "modified by adverb",
    "verb_comp_of":    "verb complement of",
    "has_verb_comp":   "has verb complement",
    "takes_aux":       "takes auxiliary",
    "aux_of":          "auxiliary of",
}

_DISPLAY_ORDER = [
    "subj_of", "obj_of", "has_subj", "has_obj",
    "verb_comp_of", "has_verb_comp",
    "takes_aux", "aux_of",
    "modified_by", "modifies", "and_or", "modified_by_adv",
]


def extract_collocations(doc) -> dict[str, dict[str, list]]:
    """
    Build a collocations dict from a spaCy Doc.

    Returns:
        { lemma: { relation_key: [[collocate_lemma, count], ...] } }

    Only content words (non-stop, alpha, length >= 2) appear as targets.
    Collocates are filtered to alpha-only but may include light function words
    when they are verbs/adjectives headwords.
    """
    raw: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    def _eff(token) -> str:
        return (token.lemma_ or token.text).lower()

    def _content(token) -> bool:
        return not token.is_stop and token.is_alpha and not token.is_punct

    def _alpha(token) -> bool:
        return token.is_alpha and not token.is_punct

    for token in doc:
        if not _content(token):
            continue

        t = _eff(token)
        h = _eff(token.head)
        dep = token.dep_

        # ── Token as dependent ──────────────────────────────────────────────
        if dep in ("nsubj", "nsubjpass", "nsubj:pass"):
            if token.head.pos_ in ("VERB", "AUX") and _alpha(token.head):
                raw[t]["subj_of"][h] += 1

        elif dep in ("obj", "dobj", "iobj"):
            if token.head.pos_ in ("VERB", "AUX") and _alpha(token.head):
                raw[t]["obj_of"][h] += 1

        elif dep == "amod":
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

        # ── Token as head — inspect children ───────────────────────────────
        for child in token.children:
            if not _alpha(child):
                continue
            c = _eff(child)
            cdep = child.dep_

            if cdep in ("nsubj", "nsubjpass", "nsubj:pass") and token.pos_ in ("VERB", "AUX"):
                raw[t]["has_subj"][c] += 1
            elif cdep in ("obj", "dobj") and token.pos_ in ("VERB", "AUX"):
                raw[t]["has_obj"][c] += 1
            elif cdep == "advmod" and token.pos_ in ("VERB", "ADJ", "AUX"):
                raw[t]["modified_by_adv"][c] += 1
            elif cdep in ("ccomp", "xcomp") and child.pos_ in ("VERB", "AUX") and token.pos_ in ("VERB", "AUX"):
                raw[t]["has_verb_comp"][c] += 1
            elif cdep in ("aux", "aux:asp", "aux:modal") and token.pos_ in ("VERB", "AUX"):
                raw[t]["takes_aux"][c] += 1

    # Serialise: Counter → sorted [[word, count], ...] list
    result: dict[str, dict[str, list]] = {}
    for lemma, rels in raw.items():
        rel_dict: dict[str, list] = {}
        for rel, counter in rels.items():
            pairs = [[w, c] for w, c in counter.most_common()]
            if pairs:
                rel_dict[rel] = pairs
        if rel_dict:
            result[lemma] = rel_dict

    return result



def get_word_sketch(collocations: dict, lemma: str) -> dict:
    """
    Return the collocation profile for *lemma*.

    Returns:
        {
          "lemma":           str,
          "found":           bool,
          "total_relations": int,
          "relations": [{"key": str, "name": str, "words": [[word, count], ...]}, ...]
        }
    """
    key = lemma.lower().strip()
    profile = collocations.get(key, {})

    relations = []
    seen = set()
    for rel_key in _DISPLAY_ORDER:
        if rel_key in profile:
            relations.append({
                "key":   rel_key,
                "name":  _REL_LABELS.get(rel_key, rel_key),
                "words": profile[rel_key][:12],
            })
            seen.add(rel_key)
    for rel_key, pairs in profile.items():
        if rel_key not in seen:
            relations.append({
                "key":   rel_key,
                "name":  _REL_LABELS.get(rel_key, rel_key),
                "words": pairs[:12],
            })

    return {
        "lemma":           key,
        "found":           bool(profile),
        "total_relations": len(relations),
        "relations":       relations,
    }


def word_sketch_diff(collocations: dict, lemma1: str, lemma2: str) -> dict:
    """
    Compare the collocational profiles of two lemmas.

    Returns:
        {
          "lemma1": str, "lemma2": str,
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
    k1, k2 = lemma1.lower().strip(), lemma2.lower().strip()
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
        "lemma1":    k1,
        "lemma2":    k2,
        "found1":    bool(p1),
        "found2":    bool(p2),
        "relations": diff_rels,
    }


def distributional_thesaurus(
    collocations: dict,
    lemma: str,
    top_n: int = 12,
) -> list[dict]:
    """
    Find words with similar distributional behaviour using Jaccard similarity
    over (relation, collocate) context tuples derived from the local corpus.

    Returns list of {"word": str, "score": float, "shared": int}
    sorted by score descending.
    """
    key = lemma.lower().strip()
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
