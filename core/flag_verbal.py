"""
core/flag_verbal.py
--------------------
Fills the verbal slots of a window profile: which lexicons the window's words
match, and in which grammatical role.

This is the one part of the flagging system that needs a parse rather than a
threshold. Two of the manifest's flags — "women/feminism as agent" and
"women/feminism as patient" — use the *same* word list and differ only in
whether those words are the subject or the object of their clause. A lexicon
lookup alone cannot tell them apart; the dependency relation can.

## What a hit records

    {"lex": "women_terms", "role": "subject", "token": "women"}

`role` comes from the dependency label: nsubj/nsubjpass/csubj -> subject,
obj/dobj/iobj/pobj -> object, anything else -> other. A manifest element
declares which role it wants (`role: subject`), and only hits in that role
fill it. `role: any` accepts all three.

## Matching rules

A token matches a lexicon when **both** hold:

  - its part of speech is in the lexicon's `pos` list (an empty list accepts
    any POS), and
  - its lemma (or surface form, if `match: surface`) is in that language's
    word list.

A lexicon with a `pos` list but an **empty** word list for the language
matches on part of speech alone — that is how `proper_nouns` works, since
naming every proper noun in advance is impossible.

## Language

Chosen per window from the script of the transcript, so a corpus does not
have to be homogeneous: any CJK character routes the window to the Chinese
model. Windows in a language with no configured spaCy model are left
unannotated rather than parsed with the wrong grammar — an English parse of
Chinese text would produce confident, meaningless dependency labels.

## Pro-drop

Mandarin routinely omits the subject. A window with no explicit subject
cannot fill the `subject` element, so under a rule requiring one element from
each macro-category, that window cannot be flagged at all. This is a property
of the language, not of the speaker, and it will make a Chinese corpus look
sparser than an English one. `verbal_coverage` reports it so the asymmetry is
visible rather than inferred from the flag counts.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

import pandas as pd
from loguru import logger

from .flagging import Manifest

# Dependency labels, mapped to the three roles a manifest can ask for.
# spaCy's English and Chinese models use overlapping but not identical label
# sets, so both vocabularies appear here.
_SUBJECT_DEPS = {"nsubj", "nsubjpass", "nsubj:pass", "csubj", "csubjpass", "expl"}
_OBJECT_DEPS = {"obj", "dobj", "iobj", "pobj", "obl", "dative", "attr", "oprd"}

_CJK = re.compile(r"[一-鿿]")

_SPACY_MODELS = {"en": "en_core_web_sm", "zh": "zh_core_web_sm"}
_NLP_CACHE: dict[str, object] = {}


def _language_of(text: str) -> str:
    return "zh" if _CJK.search(text or "") else "en"


def _nlp(lang: str):
    """Load a spaCy model once per process, with NER disabled — the parse and
    the lemmas are all that is used, and NER is a third of the runtime."""
    if lang in _NLP_CACHE:
        return _NLP_CACHE[lang]
    model = _SPACY_MODELS.get(lang)
    if not model:
        _NLP_CACHE[lang] = None
        return None
    try:
        import spacy

        _NLP_CACHE[lang] = spacy.load(model, disable=["ner"])
    except OSError:
        logger.warning(
            f"[flags] spaCy model {model!r} is not installed — {lang} windows "
            f"will have no verbal annotation. Install with: "
            f"python -m spacy download {model}"
        )
        _NLP_CACHE[lang] = None
    return _NLP_CACHE[lang]


def _role_of(dep: str) -> str:
    if dep in _SUBJECT_DEPS:
        return "subject"
    if dep in _OBJECT_DEPS:
        return "object"
    return "other"


def _hits_for_doc(doc, manifest: Manifest, lang: str) -> list[dict]:
    """Every (lexicon, role, token) this window's text supports."""
    hits = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        role = _role_of(token.dep_)
        lemma = (token.lemma_ or token.text).lower()
        surface = token.text.lower()
        for name, lexicon in manifest.lexicons.items():
            if lexicon.pos and token.pos_ not in lexicon.pos:
                continue
            words = lexicon.words(lang)
            if words:
                value = surface if lexicon.match == "surface" else lemma
                if value not in words:
                    continue
            # else: POS-only lexicon (e.g. proper_nouns) — the POS test above
            # is the whole criterion.
            hits.append({"lex": name, "role": role, "token": token.text})
    return hits


def annotate_verbal(
    df: pd.DataFrame,
    manifest: Manifest,
    text_col: str = "transcript",
    batch_size: int = 128,
) -> pd.DataFrame:
    """Add one column per manifest element (subject, object, verb, …), each
    holding the lexicon hits that filled it.

    Windows whose text is empty, or whose language has no model, get empty
    lists — which match nothing, rather than matching by default.
    """
    out = df.copy()
    if text_col not in out.columns:
        raise KeyError(f"{text_col!r} not in the window frame")

    texts = out[text_col].fillna("").astype(str).tolist()
    langs = [_language_of(t) for t in texts]
    hits: list[list[dict]] = [[] for _ in texts]

    for lang in sorted(set(langs)):
        nlp = _nlp(lang)
        idx = [i for i, l in enumerate(langs) if l == lang and texts[i].strip()]
        if not nlp or not idx:
            continue
        docs = nlp.pipe([texts[i] for i in idx], batch_size=batch_size)
        for i, doc in zip(idx, docs):
            hits[i] = _hits_for_doc(doc, manifest, lang)

    for name, element in manifest.elements.items():
        out[name] = [
            [h for h in window_hits
             if element.role == "any" or h["role"] == element.role]
            for window_hits in hits
        ]
    out["_verbal_lang"] = langs
    return out


def verbal_coverage(annotated: pd.DataFrame, manifest: Manifest) -> dict:
    """How often each verbal element was filled at all.

    Read this before reading any flag count. If `subject` is filled in 20% of
    windows, then at most 20% can carry a flag that requires one — and in a
    pro-drop language that ceiling is linguistic, not behavioural.
    """
    out = {"n_windows": len(annotated)}
    for name in manifest.elements:
        if name not in annotated.columns:
            continue
        filled = annotated[name].map(lambda v: bool(v)).sum()
        out[name] = {"filled": int(filled),
                     "pct": round(100 * filled / max(len(annotated), 1), 1)}
    if "_verbal_lang" in annotated.columns:
        out["languages"] = annotated["_verbal_lang"].value_counts().to_dict()
    return out
