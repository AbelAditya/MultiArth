"""
tests/test_gallery_centrality.py
---------------------------------
Unit tests for _gallery_match's centrality tie-break.

When two or more candidates clear the gallery threshold, the one nearest
the frame centre wins rather than the highest scorer. The motivating case
is live relay on TED-style stages: the projection behind the speaker is
the same person, matches the gallery legitimately, and — being a sharp
close-up — usually outscores the real speaker.

The test that matters most is test_non_passing_central_candidate_is_ignored.
Centrality was previously *removed* from this worker because, used as a
general selector, it picked audience members who happened to sit in the
middle of the frame. Its reintroduction is only safe because it is scoped
to gallery-confirmed candidates. If that scoping ever regresses, that test
is what catches it — and the failure it guards against (a confident gesture
track built on an audience member) would otherwise raise nothing.

Embeddings are faked so these run without OSNet or any video.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import workers.gesture_worker as gw

W, H = 1000, 500


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


# One gallery exemplar; a candidate's score is its cosine against it, so a
# score can be dialled in directly by choosing the embedding's angle.
GALLERY = np.array([_unit([1.0, 0.0])])


def _emb_for_score(score: float) -> np.ndarray:
    return _unit([score, np.sqrt(max(0.0, 1.0 - score ** 2))])


def _box_at(cx: float, cy: float, half: int = 20):
    """A box centred at frame-normalised (cx, cy)."""
    x, y = int(cx * W), int(cy * H)
    return (x - half, y - half, x + half, y + half)


def _match(candidates, monkeypatch):
    """candidates: list of (normalised_centre, score). Returns chosen index."""
    dets = [SimpleNamespace(box=_box_at(*c)) for c, _ in candidates]
    embs = {id(d): _emb_for_score(s) for d, (_, s) in zip(dets, candidates)}
    worker = gw.GestureWorker(store=None)
    monkeypatch.setattr(worker, "_embed_candidate", lambda rgb, det: embs[id(det)])
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    result = worker._gallery_match(rgb, dets, GALLERY)
    return None if result is None else result[0]


PASS = gw._GALLERY_MATCH_THRESHOLD + 0.05
PASS_HIGH = gw._GALLERY_MATCH_THRESHOLD + 0.15
FAIL = gw._GALLERY_MATCH_THRESHOLD - 0.20


def test_nobody_passes_returns_none(monkeypatch):
    assert _match([((0.5, 0.5), FAIL), ((0.2, 0.2), FAIL)], monkeypatch) is None


def test_single_passer_wins_wherever_it_is(monkeypatch):
    """One passer is chosen even far from centre — the tie-break must not
    fire, and must not prefer a non-passing central candidate."""
    assert _match([((0.1, 0.1), PASS), ((0.5, 0.5), FAIL)], monkeypatch) == 0


def test_two_passers_the_central_one_wins_over_higher_score(monkeypatch):
    """The live-relay case: the projection (top-left, high score) against
    the speaker (central, lower score). Centrality must override score."""
    projection = ((0.28, 0.21), PASS_HIGH)
    speaker = ((0.45, 0.50), PASS)
    assert _match([projection, speaker], monkeypatch) == 1


def test_order_of_detections_does_not_matter(monkeypatch):
    projection = ((0.28, 0.21), PASS_HIGH)
    speaker = ((0.45, 0.50), PASS)
    assert _match([speaker, projection], monkeypatch) == 0


def test_non_passing_central_candidate_is_ignored(monkeypatch):
    """The guarantee that makes reintroducing centrality safe. An audience
    member sitting dead centre fails the gallery and must never be chosen,
    however central — the off-centre passers decide between themselves."""
    audience = ((0.5, 0.5), FAIL)          # perfectly central, wrong person
    speaker = ((0.40, 0.55), PASS)
    projection = ((0.25, 0.20), PASS_HIGH)
    assert _match([audience, speaker, projection], monkeypatch) == 1


def test_three_passers_nearest_centre_wins(monkeypatch):
    assert _match([((0.1, 0.1), PASS), ((0.52, 0.48), PASS), ((0.8, 0.9), PASS_HIGH)],
                  monkeypatch) == 1


def test_exact_distance_tie_falls_back_to_score(monkeypatch):
    """Deterministic when two passers are equidistant from centre."""
    assert _match([((0.4, 0.5), PASS), ((0.6, 0.5), PASS_HIGH)], monkeypatch) == 1


def test_threshold_is_strict(monkeypatch):
    """A score exactly at the threshold does not pass — matching the
    original `score > threshold` behaviour. The score is injected directly:
    building an embedding whose cosine is exactly the threshold is not
    possible in float32 (0.8 comes back as 0.80000001 and passes)."""
    at = gw._GALLERY_MATCH_THRESHOLD
    monkeypatch.setattr(gw._reid, "max_similarity", lambda emb, gallery: at)
    worker = gw.GestureWorker(store=None)
    monkeypatch.setattr(worker, "_embed_candidate", lambda rgb, det: np.ones(2))
    det = SimpleNamespace(box=_box_at(0.5, 0.5))
    assert worker._gallery_match(np.zeros((H, W, 3), np.uint8), [det], GALLERY) is None


def test_returns_the_winners_embedding(monkeypatch):
    """The caller uses the returned embedding as the continuity reference,
    so it must be the winner's, not the top scorer's."""
    dets = [SimpleNamespace(box=_box_at(0.28, 0.21)), SimpleNamespace(box=_box_at(0.45, 0.5))]
    e_proj, e_spk = _emb_for_score(PASS_HIGH), _emb_for_score(PASS)
    table = {id(dets[0]): e_proj, id(dets[1]): e_spk}
    worker = gw.GestureWorker(store=None)
    monkeypatch.setattr(worker, "_embed_candidate", lambda rgb, det: table[id(det)])
    idx, emb = worker._gallery_match(np.zeros((H, W, 3), np.uint8), dets, GALLERY)
    assert idx == 1
    assert np.array_equal(emb, e_spk)
