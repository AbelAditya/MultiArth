"""
tests/test_flagging.py
-----------------------
The flagging engine, and above all the ways a manifest can be *wrong while
still running*.

A rule with a typo'd element name, an impossible threshold, or a label no
variable produces does not crash — it simply never matches, and a flag that
never fires is indistinguishable from a discursive function the corpus does
not contain. That is a research error, not a software one, which is why the
validator is tested harder than the evaluator.

The second cluster of tests pins the co-occurrence rule itself: the count
threshold, the macro-coverage requirement, and the fact that unspecified
elements neither help nor hinder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.flagging import (
    Manifest,
    ManifestError,
    categorise,
    evaluate,
    evaluate_window,
    summarise,
)

BASE = """
id: test
version: 1
variables:
  pitch_level:    {source: pitch_st, labels: [low, medium, high]}
  wrist_velocity: {source: wrist_speed_mean, labels: [low, medium, high]}
  shot:           {source: dominant_shot_type, method: passthrough}
lexicons:
  first_singular: {pos: [PRON], en: [i]}
  cognition:      {pos: [VERB], en: [think]}
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
    name: Test flag
    criteria:
      subject: [first_singular]
      verb: [cognition]
      shot: [close_up]
      pitch_level: [low]
      wrist_velocity: [low, medium]
"""


def manifest(**edits) -> Manifest:
    text = BASE
    for old, new in edits.items():
        text = text.replace(old.replace("__", " "), new)
    return Manifest.from_string(text)


# ── the validator: failures that would otherwise be silent ───────────────

def test_base_manifest_loads():
    m = manifest()
    assert (m.id, m.version, m.min_elements) == ("test", 1, 3)
    assert len(m.flags) == 1 and len(m.lexicons) == 2


def test_criterion_naming_an_undefined_element_is_rejected():
    with pytest.raises(ManifestError, match="neither a variable nor an element"):
        Manifest.from_string(BASE.replace("      shot: [close_up]", "      shott: [close_up]"))


def test_label_no_variable_produces_is_rejected():
    """The subtlest typo: the element exists, the value does not, and the flag
    would simply never fire on that criterion."""
    with pytest.raises(ManifestError, match="cannot be quiet"):
        Manifest.from_string(BASE.replace("pitch_level: [low]", "pitch_level: [quiet]"))


def test_threshold_above_criteria_count_is_rejected():
    with pytest.raises(ManifestError, match="can never fire"):
        Manifest.from_string(BASE.replace("min_elements: 3", "min_elements: 9"))


def test_undefined_lexicon_is_rejected():
    with pytest.raises(ManifestError, match="undefined lexicon"):
        Manifest.from_string(BASE.replace("subject: [first_singular]", "subject: [nonexistent]"))


def test_macro_category_with_unknown_member_is_rejected():
    with pytest.raises(ManifestError, match="undefined element"):
        Manifest.from_string(BASE.replace("  camera:   [shot]", "  camera:   [shot, zoom]"))


def test_required_macro_that_does_not_exist_is_rejected():
    with pytest.raises(ManifestError, match="not defined"):
        Manifest.from_string(
            BASE.replace("require_each_macro: [verbal, camera, acoustic, gesture]",
                         "require_each_macro: [verbal, lighting]"))


def test_bad_method_and_scope_are_rejected():
    with pytest.raises(ManifestError, match="unknown method"):
        Manifest.from_string(BASE.replace("{source: pitch_st, labels: [low, medium, high]}",
                                          "{source: pitch_st, method: magic, labels: [a, b]}"))
    with pytest.raises(ManifestError, match="unknown scope"):
        Manifest.from_string(BASE.replace("{source: pitch_st, labels: [low, medium, high]}",
                                          "{source: pitch_st, scope: per_planet, labels: [a,b,c]}"))


def test_edges_and_labels_must_agree():
    with pytest.raises(ManifestError, match="need 3 labels"):
        Manifest.from_string(BASE.replace(
            "{source: pitch_st, labels: [low, medium, high]}",
            "{source: pitch_st, method: quantiles, edges: [0.3, 0.6], labels: [low, high]}"))


def test_unknown_role_is_rejected():
    with pytest.raises(ManifestError, match="unknown role"):
        Manifest.from_string(BASE.replace("subject: {role: subject}", "subject: {role: agent}"))


# ── categorisation ───────────────────────────────────────────────────────

def _frame(n=30, job="v1"):
    return pd.DataFrame({
        "job_id": [job] * n,
        "window_idx": range(n),
        "pitch_st": np.linspace(-6, 6, n),
        "wrist_speed_mean": np.linspace(0, 3, n),
        "dominant_shot_type": ["ShotType.CLOSE_UP"] * n,
    })


def test_tertiles_split_into_three_roughly_equal_groups():
    out, thresholds = categorise(_frame(30), manifest())
    counts = out.pitch_level.value_counts()
    assert set(counts.index) == {"low", "medium", "high"}
    assert counts.min() >= 9                      # 30 rows, three bins
    assert len(thresholds["pitch_level"]["edges"]["v1"]) == 2


def test_thresholds_are_computed_per_speaker():
    """The whole point of per_speaker scope: one speaker's 'high' is another's
    'low', and pooling would sort speakers by voice rather than by style."""
    quiet = _frame(30, job="quiet")
    loud = _frame(30, job="loud")
    loud["pitch_st"] = np.linspace(20, 32, 30)
    out, thresholds = categorise(pd.concat([quiet, loud]), manifest())

    assert out[out.job_id == "loud"].pitch_level.value_counts().min() >= 9
    assert (thresholds["pitch_level"]["edges"]["loud"][0]
            > thresholds["pitch_level"]["edges"]["quiet"][0])


def test_passthrough_lowercases_and_maps():
    m = Manifest.from_string(BASE.replace(
        "  shot:           {source: dominant_shot_type, method: passthrough}",
        "  shot:           {source: dominant_shot_type, method: passthrough,"
        " map: {ShotType.CLOSE_UP: close_up}}"))
    out, _ = categorise(_frame(6), m)
    assert set(out["shot"]) == {"close_up"}


def test_missing_source_column_yields_nan_not_a_guess():
    """A window with no pose must not be labelled 'low velocity' by default —
    that would invent evidence for a gesture criterion.

    The macro-category guard is relaxed here so the NaN path itself is what is
    under test; the guard has its own test below."""
    lenient = Manifest.from_string(
        BASE.replace("require_each_macro: [verbal, camera, acoustic, gesture]",
                     "require_each_macro: [verbal, camera]"))
    frame = _frame(10).drop(columns=["wrist_speed_mean"])
    out, thresholds = categorise(frame, lenient)
    assert out["wrist_velocity"].isna().all()
    assert "error" in thresholds["wrist_velocity"]


def test_unsatisfiable_required_macro_raises_rather_than_flagging_nothing():
    """The silent-zero case: gesture is required, its only column is absent, so
    every flag fails and the run reports "no flags" — indistinguishable from a
    corpus that genuinely contains none. It must refuse instead."""
    frame = _frame(10).drop(columns=["wrist_speed_mean"])
    with pytest.raises(ManifestError, match="macro-category 'gesture'"):
        categorise(frame, manifest())


def test_constant_values_do_not_crash():
    """A speaker whose values are all identical makes every quantile edge
    equal; pd.cut would raise, and dropping the video would be worse."""
    frame = _frame(10)
    frame["pitch_st"] = 1.0
    out, _ = categorise(frame, manifest())
    assert len(out) == 10


# ── the co-occurrence rule ───────────────────────────────────────────────

def _profile(**overrides):
    base = {
        "subject": [{"lex": "first_singular", "role": "subject", "token": "I"}],
        "verb": [{"lex": "cognition", "role": "other", "token": "think"}],
        "shot": "close_up",
        "pitch_level": "low",
        "wrist_velocity": "medium",
    }
    base.update(overrides)
    return base


def test_full_match_fires():
    result = evaluate_window(_profile(), manifest())
    assert [f["id"] for f in result["flags"]] == ["F1"]
    assert result["flags"][0]["n"] == 5


def test_below_threshold_does_not_fire():
    result = evaluate_window(
        _profile(shot="long", pitch_level="high", wrist_velocity="high"), manifest())
    assert result["flags"] == []
    assert [f["id"] for f in result["near"]] == ["F1"]      # 2 of 3 -> near miss


def test_macro_coverage_is_required_even_above_threshold():
    """Four matches, threshold three — but nothing from the gesture category,
    so the rule refuses it. Without this test the second condition could be
    dropped and most windows would still flag."""
    result = evaluate_window(_profile(wrist_velocity="high"), manifest())
    assert result["flags"] == []
    assert result["near"][0]["blocked_by"] == "macro_coverage"


ROLE_MANIFEST = """
id: roles
version: 1
variables:
  shot: {source: dominant_shot_type, method: passthrough}
lexicons:
  women_terms: {pos: [NOUN], en: [women]}
  agency:      {pos: [VERB], en: [demand]}
  passives:    {pos: [VERB], en: [silence]}
elements:
  subject: {role: subject}
  object:  {role: object}
  verb:    {role: any}
macro_categories:
  verbal: [subject, object, verb]
  camera: [shot]
rule:
  min_elements: 2
  require_each_macro: [verbal]
flags:
  - id: AGENT
    name: Women as agent
    criteria: {subject: [women_terms], verb: [agency]}
  - id: PATIENT
    name: Women as patient
    criteria: {object: [women_terms], verb: [passives]}
"""


def test_role_separates_agent_from_patient():
    """Same word list, different grammatical position — the distinction the
    agent/patient pair rests on. The annotation layer files each hit into the
    element whose role it matched, so an object-role hit never reaches the
    subject element."""
    m = Manifest.from_string(ROLE_MANIFEST)

    agent = {"subject": [{"lex": "women_terms", "role": "subject", "token": "women"}],
             "object": [],
             "verb": [{"lex": "women_terms", "role": "subject", "token": "women"},
                      {"lex": "agency", "role": "other", "token": "demand"}]}
    patient = {"subject": [],
               "object": [{"lex": "women_terms", "role": "object", "token": "women"}],
               "verb": [{"lex": "women_terms", "role": "object", "token": "women"},
                        {"lex": "passives", "role": "other", "token": "silenced"}]}

    assert [f["id"] for f in evaluate_window(agent, m)["flags"]] == ["AGENT"]
    assert [f["id"] for f in evaluate_window(patient, m)["flags"]] == ["PATIENT"]


def test_unspecified_elements_neither_help_nor_hinder():
    profile = _profile()
    profile["intensity_variation"] = "varying"      # not in F1's criteria
    assert evaluate_window(profile, manifest())["flags"][0]["n"] == 5


def test_missing_value_matches_nothing():
    assert evaluate_window(_profile(pitch_level=np.nan), manifest())["flags"] == []


def test_per_flag_threshold_overrides_the_default():
    """A flag may be stricter than the manifest's default: here 5 of 5, so
    losing any single element stops it firing."""
    strict = BASE.replace(
        "    criteria:\n      subject: [first_singular]",
        "    min_elements: 5\n    criteria:\n      subject: [first_singular]")
    m = Manifest.from_string(strict)
    assert m.flags[0].min_elements == 5
    assert evaluate_window(_profile(), m)["flags"] != []
    assert evaluate_window(_profile(shot="long"), m)["flags"] == []


# ── summarising ──────────────────────────────────────────────────────────

def test_summary_counts_windows_pairs_and_near_misses():
    m = Manifest.from_string(BASE + """
  - id: F2
    name: Second flag
    criteria:
      subject: [first_singular]
      verb: [cognition]
      shot: [close_up]
      pitch_level: [low]
      wrist_velocity: [low, medium]
""")
    frame = pd.DataFrame([{
        "job_id": "v1", "window_idx": i, **_profile(),
    } for i in range(4)])
    summary = summarise(evaluate(frame, m), m)
    assert summary["n_windows"] == 4
    assert summary["flagged_windows"] == 4
    assert summary["by_flag"]["F1"]["n"] == 4
    assert summary["by_flag"]["F1"]["pct"] == 100.0
    assert summary["pairs"]["F1|F2"] == 4          # both fire on every window


def test_evaluate_does_not_shadow_a_pandas_attribute():
    """`flags` is a real DataFrame/Series attribute, so a column of that name
    is unreachable as row.flags — the columns are named flag_detail/near_miss."""
    frame = pd.DataFrame([{"job_id": "v1", "window_idx": 0, **_profile()}])
    out = evaluate(frame, manifest())
    assert {"flag_detail", "near_miss", "flag_ids"} <= set(out.columns)
    assert out.iloc[0].flag_detail[0]["id"] == "F1"
