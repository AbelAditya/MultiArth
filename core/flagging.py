"""
core/flagging.py
-----------------
Manifest-driven multimodal flagging: mark each 5s window with the discursive
functions it realises, from rules the analyst writes rather than rules the
code hardcodes.

A *flag* is a discursive function (e.g. "Individual Intellectual authority").
A window earns it when enough of that function's specified elements co-occur
— the co-occurrence rule, from FLAGS SYS X ABEL.xlsx:

  1. at least `min_elements` of the elements the flag specifies match, and
  2. at least one element from each required macro-category matches
     (verbal, camera, acoustic, gesture).

Elements a flag does not specify are not evaluated and cannot count.

## Why a manifest and not code

The rules are a research instrument, not an implementation detail: different
analysts define different functions, and the same analyst revises them as the
corpus is read. So everything — which features become which categories, where
the category boundaries sit, which lexicon counts as "cognition verbs", the
threshold itself — lives in a versioned YAML file, and every stored result
records the manifest id, version and content hash that produced it. A number
in a thesis has to be traceable to the exact rule that made it.

Adding an element the spreadsheet never mentioned (handedness, face area,
speech rate) needs a `variables:` entry and a mention in a flag's criteria.
No code change.

## Categorisation

The rules speak in labels ("low pitch", "high velocity"); the pipeline stores
continuous values. `categorise` converts one to the other, defaulting to
**tertiles computed per speaker** — an absolute Hz threshold would mostly sort
speakers by voice type rather than by discourse style. A manifest can override
per variable with explicit quantiles or absolute thresholds.

The edges actually used are returned alongside the labels and stored with the
results: "low pitch" is a different frequency for every speaker, and without
the edges a stored flag cannot be reconstructed later.

## What a result keeps

Not just the verdict. For every flag on every window, which elements matched
and which did not, plus `near` — flags that fell one short. That makes a flag
interrogable ("why was this window flagged?"), makes the lexicons tunable, and
answers "what would a threshold of 4 do?" without recomputing anything.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from loguru import logger

# Methods a variable may use to turn a stored value into a label.
_METHODS = ("tertiles", "quantiles", "thresholds", "passthrough")
# Scope decides the *population* a quantile is measured over. It has no
# meaning for `thresholds`, whose edges are already literal values — so there
# is deliberately no "absolute" scope: writing one was a second way to say
# `method: thresholds`, and combined with `quantiles` it silently reinterpreted
# fractions as raw values (edges [0.5] meaning 0.5 semitones, not the median).
_SCOPES = ("per_speaker", "per_corpus")

# Roles a lexicon hit may be required to occupy. "Women as agent" and "women
# as patient" share a word list and differ only here, so the role is part of
# the match, not decoration.
_ROLES = ("subject", "object", "any")


class ManifestError(ValueError):
    """A manifest that would silently misbehave — a criterion naming an
    undefined element, a label no variable produces, a lexicon that does not
    exist. Raised at load time: a typo'd rule that simply never matches is
    the worst outcome, because it looks like a finding."""


@dataclass
class Variable:
    name: str
    source: str
    method: str = "tertiles"
    scope: str = "per_speaker"
    labels: list[str] = field(default_factory=list)
    edges: list[float] = field(default_factory=list)
    map: dict[str, str] = field(default_factory=dict)   # passthrough renaming


@dataclass
class Lexicon:
    name: str
    pos: list[str] = field(default_factory=list)
    match: str = "lemma"
    items: dict[str, list[str]] = field(default_factory=dict)   # lang -> words

    def words(self, lang: str) -> set[str]:
        return {w.strip().lower() for w in self.items.get(lang, []) if w}


@dataclass
class Element:
    """A verbal slot filled by lexicon hits rather than by a threshold."""
    name: str
    role: str = "any"


@dataclass
class Flag:
    id: str
    name: str
    criteria: dict[str, list[str]]
    min_elements: Optional[int] = None
    require_each_macro: Optional[list[str]] = None


@dataclass
class Manifest:
    id: str
    version: int
    window_s: float
    variables: dict[str, Variable]
    lexicons: dict[str, Lexicon]
    elements: dict[str, Element]
    macro_categories: dict[str, list[str]]
    min_elements: int
    require_each_macro: list[str]
    flags: list[Flag]
    sha256: str
    path: Optional[str] = None
    # Which flag pairs the analyst expects to co-occur, and why. Optional, and
    # purely for reporting: the engine never uses it to decide anything. It
    # exists so the overlap report can say whether an observed overlap was
    # designed, between related functions, or unforeseen — the analysis
    # FLAGS SYS X ABEL.xlsx does by hand.
    expected_overlaps: dict[str, list[frozenset]] = field(default_factory=dict)

    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Manifest":
        import yaml

        raw_text = Path(path).read_text()
        raw = yaml.safe_load(raw_text)
        manifest = cls._from_dict(raw, sha256=hashlib.sha256(raw_text.encode()).hexdigest())
        manifest.path = str(path)
        return manifest

    @classmethod
    def from_string(cls, text: str) -> "Manifest":
        """For a manifest pasted into the dashboard rather than kept on disk."""
        import yaml

        return cls._from_dict(yaml.safe_load(text),
                              sha256=hashlib.sha256(text.encode()).hexdigest())

    @classmethod
    def _from_dict(cls, raw: dict, sha256: str) -> "Manifest":
        if not isinstance(raw, dict):
            raise ManifestError("manifest must be a YAML mapping")

        variables = {
            name: Variable(
                name=name,
                source=spec.get("source", name),
                method=spec.get("method", "tertiles"),
                scope=spec.get("scope", "per_speaker"),
                labels=list(spec.get("labels", [])),
                edges=[float(e) for e in spec.get("edges", [])],
                map=dict(spec.get("map", {})),
            )
            for name, spec in (raw.get("variables") or {}).items()
        }
        lexicons = {
            name: Lexicon(
                name=name,
                pos=list(spec.get("pos", [])),
                match=spec.get("match", "lemma"),
                items={k: list(v) for k, v in spec.items()
                       if k not in ("pos", "match") and isinstance(v, list)},
            )
            for name, spec in (raw.get("lexicons") or {}).items()
        }
        elements = {
            name: Element(name=name, role=spec.get("role", "any"))
            for name, spec in (raw.get("elements") or {}).items()
        }
        rule = raw.get("rule") or {}
        flags = [
            Flag(
                id=str(f["id"]),
                name=f.get("name", str(f["id"])),
                criteria={k: (v if isinstance(v, list) else [v])
                          for k, v in (f.get("criteria") or {}).items()
                          if v is not None and v != "/"},
                min_elements=f.get("min_elements"),
                require_each_macro=f.get("require_each_macro"),
            )
            for f in (raw.get("flags") or [])
        ]
        manifest = cls(
            id=str(raw.get("id", "unnamed")),
            version=int(raw.get("version", 1)),
            window_s=float(raw.get("window_s", 5.0)),
            variables=variables,
            lexicons=lexicons,
            elements=elements,
            macro_categories={k: list(v) for k, v in (raw.get("macro_categories") or {}).items()},
            min_elements=int(rule.get("min_elements", 5)),
            require_each_macro=list(rule.get("require_each_macro", [])),
            flags=flags,
            sha256=sha256,
            expected_overlaps={
                kind: [frozenset(str(x) for x in pair) for pair in pairs or []]
                for kind, pairs in (raw.get("overlaps") or {}).items()
            },
        )
        manifest.validate()
        return manifest

    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Every way a manifest can be wrong *and still run* — checked here so
        it fails at load instead of producing a rule that never fires."""
        if not self.flags:
            raise ManifestError("manifest defines no flags")

        known = set(self.variables) | set(self.elements)
        for var in self.variables.values():
            if var.method not in _METHODS:
                raise ManifestError(f"variable {var.name}: unknown method {var.method!r}"
                                    f" (expected one of {', '.join(_METHODS)})")
            if var.scope not in _SCOPES:
                hint = (" — for fixed cut points in the source's own units use "
                        "method: thresholds, which applies to every speaker alike"
                        if var.scope == "absolute" else "")
                raise ManifestError(
                    f"variable {var.name}: unknown scope {var.scope!r} "
                    f"(expected {' or '.join(_SCOPES)}){hint}")
            if var.method == "tertiles" and not var.labels:
                var.labels = ["low", "medium", "high"]
            if var.method in ("quantiles", "thresholds"):
                if len(var.labels) != len(var.edges) + 1:
                    raise ManifestError(
                        f"variable {var.name}: {len(var.edges)} edges need "
                        f"{len(var.edges) + 1} labels, got {len(var.labels)}")
            if var.method == "quantiles" and not all(0 < e < 1 for e in var.edges):
                raise ManifestError(f"variable {var.name}: quantile edges must be in (0, 1)")

        for elem in self.elements.values():
            if elem.role not in _ROLES:
                raise ManifestError(f"element {elem.name}: unknown role {elem.role!r}"
                                    f" (expected one of {', '.join(_ROLES)})")

        for macro, members in self.macro_categories.items():
            unknown = set(members) - known
            if unknown:
                raise ManifestError(
                    f"macro-category {macro} lists undefined element(s): "
                    f"{', '.join(sorted(unknown))}")

        for macro in self.require_each_macro:
            if macro not in self.macro_categories:
                raise ManifestError(f"rule requires macro-category {macro!r}, which is "
                                    f"not defined")

        flag_ids = {f.id for f in self.flags}
        for kind, pairs in self.expected_overlaps.items():
            for pair in pairs:
                unknown = set(pair) - flag_ids
                if unknown:
                    raise ManifestError(
                        f"overlaps.{kind} names undefined flag(s): "
                        f"{', '.join(sorted(unknown))}")
                if len(pair) != 2:
                    raise ManifestError(
                        f"overlaps.{kind}: each entry must be a pair of flag ids")

        for flag in self.flags:
            if not flag.criteria:
                raise ManifestError(f"flag {flag.id} specifies no criteria")
            for element, values in flag.criteria.items():
                if element not in known:
                    raise ManifestError(
                        f"flag {flag.id}: criterion {element!r} is neither a variable "
                        f"nor an element. Defined: {', '.join(sorted(known))}")
                if element in self.variables:
                    var = self.variables[element]
                    # passthrough labels come from the data, so only
                    # derived variables can be checked here.
                    if var.method != "passthrough" and var.labels:
                        bad = set(values) - set(var.labels)
                        if bad:
                            raise ManifestError(
                                f"flag {flag.id}: {element} cannot be "
                                f"{', '.join(sorted(bad))} — its labels are "
                                f"{', '.join(var.labels)}")
                else:
                    bad = set(values) - set(self.lexicons)
                    if bad:
                        raise ManifestError(
                            f"flag {flag.id}: {element} references undefined lexicon(s) "
                            f"{', '.join(sorted(bad))}")
            threshold = flag.min_elements or self.min_elements
            if threshold > len(flag.criteria):
                raise ManifestError(
                    f"flag {flag.id} needs {threshold} matches but specifies only "
                    f"{len(flag.criteria)} criteria — it can never fire")

    def overlap_kind(self, a: str, b: str) -> str:
        """How the manifest classifies this pair — 'designed', 'related',
        whatever names it used, or 'unexpected' when it named none."""
        pair = frozenset((a, b))
        for kind, pairs in self.expected_overlaps.items():
            if pair in pairs:
                return kind
        return "unexpected"

    def macro_of(self, element: str) -> Optional[str]:
        for macro, members in self.macro_categories.items():
            if element in members:
                return macro
        return None


# ──────────────────────────────────────────────────────────────────────────
# Categorisation
# ──────────────────────────────────────────────────────────────────────────

def categorise(
    df: pd.DataFrame, manifest: Manifest, group_col: str = "job_id",
) -> tuple[pd.DataFrame, dict]:
    """Add one label column per manifest variable, and report the cut points.

    Returns (frame, thresholds). The frame gains a column per variable name;
    `thresholds` records the edges each variable actually used, per group where
    the scope is per_speaker — that is what makes a stored flag reconstructable
    once the corpus has moved on.

    Windows whose source value is missing get NaN, which matches nothing: a
    window with no pose cannot satisfy a gesture criterion, and silently
    treating that as "low velocity" would invent evidence.
    """
    out = df.copy()
    thresholds: dict[str, Any] = {}

    for name, var in manifest.variables.items():
        if var.source not in out.columns:
            # Loud, because the consequence is silent: a variable that cannot
            # be computed matches nothing, and if it belongs to a required
            # macro-category every flag fails while the run still "succeeds".
            logger.warning(
                f"[flags] variable {name!r} wants column {var.source!r}, which is "
                f"not in the window frame — it will match nothing. Gesture and "
                f"pitch columns come from _corpus.load_corpus(), not from "
                f"load_windows() alone.")
            out[name] = np.nan
            thresholds[name] = {"error": f"source column {var.source!r} not in the data"}
            continue

        if var.method == "passthrough":
            values = out[var.source].astype("object")
            if var.map:
                values = values.map(lambda v: var.map.get(str(v), v))
            out[name] = values.map(lambda v: np.nan if pd.isna(v) else str(v).lower())
            thresholds[name] = {"method": "passthrough", "source": var.source}
            continue

        if var.method == "thresholds":
            # Literal cut points in the source's own units — the same number
            # for every speaker, so no scope applies.
            labels, edges = _cut_absolute(out[var.source], var)
            out[name] = labels
            thresholds[name] = {"method": "thresholds", "scope": "absolute",
                                "edges": edges, "labels": var.labels}
            continue

        # per_speaker (default) or per_corpus quantile-based binning
        per_group: dict[str, list[float]] = {}
        if var.scope == "per_speaker" and group_col in out.columns:
            labelled = pd.Series(index=out.index, dtype="object")
            for key, chunk in out.groupby(group_col):
                lab, edges = _cut_quantile(chunk[var.source], var)
                labelled.loc[chunk.index] = lab
                per_group[str(key)] = edges
            out[name] = labelled
        else:
            lab, edges = _cut_quantile(out[var.source], var)
            out[name] = lab
            per_group["__corpus__"] = edges
        thresholds[name] = {"method": var.method, "scope": var.scope,
                            "labels": var.labels, "edges": per_group}

    _check_required_macros(manifest, thresholds)
    return out, thresholds


def _check_required_macros(manifest: Manifest, thresholds: dict) -> None:
    """Refuse a run whose rule cannot be satisfied.

    If every variable in a required macro-category failed to resolve, no window
    can ever meet the coverage condition — the run would report zero flags and
    look like a finding about the corpus rather than a missing column.
    """
    for macro in manifest.require_each_macro:
        members = manifest.macro_categories.get(macro, [])
        variables = [m for m in members if m in manifest.variables]
        if not variables:
            continue                      # verbal-style category, filled by lexicons
        if all("error" in thresholds.get(m, {}) for m in variables):
            missing = ", ".join(sorted(
                manifest.variables[m].source for m in variables))
            raise ManifestError(
                f"the rule requires an element from macro-category {macro!r}, but "
                f"none of its variables could be computed (missing column(s): "
                f"{missing}). Every flag would fail silently. Load the windows "
                f"with _corpus.load_corpus(), which merges the gesture and pitch "
                f"columns, or drop {macro!r} from require_each_macro.")


def _cut_quantile(series: pd.Series, var: Variable) -> tuple[pd.Series, list[float]]:
    quantiles = ([i / len(var.labels) for i in range(1, len(var.labels))]
                 if var.method == "tertiles" else list(var.edges))
    values = pd.to_numeric(series, errors="coerce")
    edges = [float(values.quantile(q)) for q in quantiles]
    return _apply_edges(values, edges, var.labels), edges


def _cut_absolute(series: pd.Series, var: Variable) -> tuple[pd.Series, list[float]]:
    values = pd.to_numeric(series, errors="coerce")
    return _apply_edges(values, list(var.edges), var.labels), list(var.edges)


def _apply_edges(values: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    """Bin with -inf/+inf outer edges so every real value lands somewhere.

    Duplicate edges (a speaker whose values are mostly identical) would make
    pd.cut raise; collapsing them keeps the window labelled rather than
    dropping the video from the analysis.
    """
    bins = [-np.inf, *edges, np.inf]
    unique_bins, unique_labels = [bins[0]], []
    for edge, label in zip(bins[1:], labels):
        if edge > unique_bins[-1]:
            unique_bins.append(edge)
            unique_labels.append(label)
    if len(unique_labels) < len(labels):
        unique_labels.append(labels[-1])
        unique_bins[-1] = np.inf
    if len(unique_bins) < 2:
        return pd.Series(np.nan, index=values.index, dtype="object")
    cut = pd.cut(values, bins=unique_bins, labels=unique_labels[:len(unique_bins) - 1],
                 include_lowest=True)
    return cut.astype("object")


# ──────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────

def _as_set(value: Any) -> set[str]:
    """Profile values are sets: a categorical variable contributes one label,
    a verbal element contributes every lexicon it matched."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    if isinstance(value, (list, tuple, set, frozenset)):
        out = set()
        for item in value:
            if isinstance(item, dict):           # {lex, role, token}
                out.add(str(item.get("lex")))
            elif item is not None:
                out.add(str(item))
        return out
    return {str(value)}


def evaluate_window(profile: dict, manifest: Manifest, near_margin: int = 1) -> dict:
    """Which flags this window earns, with the evidence for each.

    `near_margin` also records flags that fell that many matches short — the
    diagnostic that says whether a threshold is admitting too little, and
    which element is doing the blocking.
    """
    flags, near = [], []

    for flag in manifest.flags:
        threshold = flag.min_elements or manifest.min_elements
        required = flag.require_each_macro
        if required is None:
            required = manifest.require_each_macro

        matched, missed, macros = [], [], set()
        for element, accepted in flag.criteria.items():
            have = _as_set(profile.get(element))
            if have & set(accepted):
                matched.append(element)
                macro = manifest.macro_of(element)
                if macro:
                    macros.add(macro)
            else:
                missed.append(element)

        covered = all(macro in macros for macro in required)
        record = {"id": flag.id, "n": len(matched), "matched": matched, "missed": missed}
        if len(matched) >= threshold and covered:
            flags.append(record)
        elif len(matched) >= threshold - near_margin:
            # A flag blocked *only* by macro coverage is worth seeing too: in
            # a pro-drop language the verbal category fails structurally, not
            # substantively.
            record["blocked_by"] = ("macro_coverage" if len(matched) >= threshold
                                    else "count")
            near.append(record)

    return {"flags": flags, "near": near}


def evaluate(
    df: pd.DataFrame, manifest: Manifest, verbal_col: Optional[str] = None,
) -> pd.DataFrame:
    """Evaluate every window in `df` (already categorised). Returns one row per
    window with `flags` and `near` columns holding the records above."""
    elements = sorted(set(manifest.variables) | set(manifest.elements))
    records = []
    for _, row in df.iterrows():
        profile = {e: row.get(e) for e in elements if e in row.index}
        records.append(evaluate_window(profile, manifest))
    out = df.copy()
    # NOT "flags": pandas already defines Series.flags / DataFrame.flags, so a
    # column of that name is unreachable as `row.flags` and raises KeyError —
    # only row["flags"] would work. Renaming is cheaper than the confusion.
    out["flag_detail"] = [r["flags"] for r in records]
    out["near_miss"] = [r["near"] for r in records]
    out["flag_ids"] = [[f["id"] for f in r["flags"]] for r in records]
    return out


def summarise(evaluated: pd.DataFrame, manifest: Manifest) -> dict:
    """Per-video pooling: how many windows each flag claims, which flags
    co-occur, and what blocks the near misses.

    The co-occurrence counts are the empirical answer to the overlap analysis
    in FLAGS SYS X ABEL.xlsx, which predicts which pairs *can* overlap; this
    measures which pairs actually do.
    """
    n_windows = len(evaluated)
    by_flag: dict[str, dict] = {}
    pairs: dict[str, int] = {}
    near_miss: dict[str, dict] = {}

    for flag in manifest.flags:
        by_flag[flag.id] = {"name": flag.name, "n": 0, "pct": 0.0}

    for ids in evaluated["flag_ids"]:
        for fid in ids:
            by_flag.setdefault(fid, {"name": fid, "n": 0, "pct": 0.0})["n"] += 1
        for i, a in enumerate(sorted(ids)):
            for b in sorted(ids)[i + 1:]:
                pairs[f"{a}|{b}"] = pairs.get(f"{a}|{b}", 0) + 1

    for rec in by_flag.values():
        rec["pct"] = round(100 * rec["n"] / n_windows, 2) if n_windows else 0.0

    for near_list in evaluated["near_miss"]:
        for rec in near_list:
            entry = near_miss.setdefault(rec["id"], {"n": 0, "missing": {}})
            entry["n"] += 1
            for element in rec["missed"]:
                entry["missing"][element] = entry["missing"].get(element, 0) + 1

    return {
        "n_windows": n_windows,
        "flagged_windows": int((evaluated["flag_ids"].str.len() > 0).sum()),
        "by_flag": by_flag,
        "pairs": dict(sorted(pairs.items(), key=lambda kv: -kv[1])),
        "near_miss": near_miss,
    }
