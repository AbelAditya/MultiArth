"""
core/gallery_builder.py
------------------------
Interactive speaker-gallery construction, driven by the dashboard's Bulk
Upload tab (see wikis/Gesture-Worker.md's "Speaker re-identification").
Dashboard-bulk-only — single-file upload never builds a gallery and always
uses workers/gesture_worker.py's heuristic-only path. Everything built
here lives in Redis under the same job_id GestureWorker later reads from
(core/feature_store.py's add_gallery_entry/get_gallery) — no storage of
its own, and it's cleaned up by the existing job:{job_id}:* delete_job
sweep once that job finishes.

Person detection and segmentation come from workers/_detector.py
(YOLO11n-seg); MediaPipe is not used in this module at all. Gallery
building only ever needed people-plus-masks, never a landmark, so the pose
model here was pure overhead — and its person detector was the weak link
on wide stage shots. See detect_candidates for the fuller reasoning.

This module holds the pure sampling/scoring logic only; the dashboard
owns all UI state (which video, which frame is currently shown, click
handling) and calls into this module per confirmation. Kept separate from
core/bulk_orchestrator.py deliberately — that class is a clean, headless,
synchronous per-video processing loop; this is inherently a paused,
click-driven, multi-step flow with a different shape entirely, so mixing
them would make both harder to reason about.
"""

from __future__ import annotations

import base64
import random
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from core.feature_store import FeatureStore
from core.models import GalleryEntry
from workers import _reid
from workers._detector import detect_people

# Redundancy threshold (max-similarity — see workers/_reid.py's
# max_similarity, and workers/gesture_worker.py's module docstring for why
# this is deliberately NOT the same pooling as runtime matching's top-K
# mean): "is this confirmed frame basically a duplicate of a look we
# already have". Measured against real single-person footage (see
# wikis/Gesture-Worker.md's re-ID section for the actual experiment and
# numbers) — a real starting point, not a fully validated final value.
GALLERY_REDUNDANCY_GAMMA = 0.85

PLATEAU_STREAK = 20    # consecutive redundant confirmations before stopping
GALLERY_MIN_SIZE = 3  # floor — keep asking at least this many times regardless
GALLERY_MAX_SIZE = 40  # hard cap, regardless of streak — a UX ceiling, not
# an algorithmic one (every confirmation costs the researcher's actual
# attention — see wikis/Gesture-Worker.md).

_JPEG_QUALITY = 85

# --- Sampling schedule (scene-stratified, duration-weighted, no repeats) ---

_TARGET_SLOT_SECONDS = 5.0  # duration-weighting: roughly one scheduled
# draw per this many seconds of scene length — a scene shorter than this
# floor still always gets exactly 1 slot, never fewer. Not empirically
# tuned.
_MAX_SLOTS_PER_SCENE = 6    # bounds how much one very long scene can
# dominate a lap's schedule — without this, a single long shot next to
# several short cutaways would crowd out ever giving the short ones a
# fair turn within a realistic number of total confirmations.
_MAX_FRAME_RETRY = 8        # retries against a frame index already shown
# (state.drawn_frames) before falling back to accepting the repeat — a
# genuine last resort, not the routine path. See _draw_fresh_frame's
# docstring for the confirmed-real bug this replaces: a scene under 0.2s
# used to return the exact same timestamp every single time it recurred
# across laps.
_MAX_SLOT_DETECTION_RETRY = 4  # how many different timestamps to try
# WITHIN one schedule slot's own sub-interval, looking for one where a
# person is actually detected, before giving up on that specific slot —
# see draw_next_candidate's docstring for the confirmed-real bug this
# fixes: retrying by drawing a whole new schedule slot (a different
# scene's guaranteed turn) instead of retrying the same one.


@dataclass
class GalleryBuildState:
    """Per-video state for one interactive gallery-building session — the
    dashboard keeps one of these (serialised into a dcc.Store) per
    manifest entry currently being confirmed."""
    job_id: str
    video_path: str
    duration_s: float
    scene_cuts: list[float]              # from GestureWorker's own PySceneDetect pass
    fps: float                           # needed to convert timestamps <-> frame
    # indices for drawn_frames' no-repeat tracking below.
    schedule: list = field(default_factory=list)   # [(scene_idx, slot_idx), ...],
    # one lap's worth — see _build_schedule. list, not list[tuple], since a
    # dcc.Store JSON round-trip turns tuples into 2-element lists anyway;
    # code here treats each entry as a generic 2-item sequence.
    schedule_cursor: int = 0
    drawn_frames: list = field(default_factory=list)  # index i = frame
    # indices already shown from scene i — a list-per-scene-index rather
    # than a dict keyed by scene_idx deliberately: JSON object keys are
    # always strings, so a dict[int, ...] would silently come back with
    # string keys after the dcc.Store round-trip (confirmed — Python's own
    # json module does this), breaking int-keyed lookups. A list sidesteps
    # that entirely.
    streak: int = 0
    entries_confirmed: int = 0
    stop_reason: Optional[str] = None    # "plateau" | "cap" | "manual" | None


def _scene_bounds(scene_cuts: list[float], duration_s: float) -> list[tuple[float, float]]:
    """Turns a flat list of cut timestamps into (start, end) scene bounds
    — GestureWorker's own _detect_scene_cuts only returns cut *points*,
    not bounds, since that's all its per-frame ref_pos reset needs; this
    is the one place gallery-building needs the fuller shape."""
    bounds = []
    prev = 0.0
    for cut in scene_cuts:
        bounds.append((prev, cut))
        prev = cut
    bounds.append((prev, duration_s))
    return bounds


def _n_slots_for_scene(duration: float) -> int:
    """How many scheduled turns a scene gets per lap — duration-weighted
    (a longer scene gets more turns, each later mapped to a distinct,
    stratified sub-interval of its own span — see _build_schedule /
    _sub_interval_bounds — rather than the same uniform-random-anywhere
    draw every time), bounded so one very long scene can't crowd out the
    rest of the schedule."""
    return max(1, min(_MAX_SLOTS_PER_SCENE, round(duration / _TARGET_SLOT_SECONDS)))


def _sub_interval_bounds(bounds: tuple[float, float], n_slots: int, slot_idx: int) -> tuple[float, float]:
    start, end = bounds
    step = (end - start) / n_slots
    return (start + step * slot_idx, start + step * (slot_idx + 1))


def _build_schedule(bounds: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """
    One lap's worth of (scene_idx, sub_interval_idx) draws — duration
    -weighted and interleaved round-robin (a scene with several turns
    doesn't get them all back-to-back; each "sweep" below gives every
    still-eligible scene its next turn, shuffled fresh each sweep so which
    scene goes first isn't fixed lap after lap) rather than the old flat
    per-scene round-robin. See wikis/Gesture-Worker.md's worked examples
    for the reasoning this replaces/extends.
    """
    per_scene_slots = [_n_slots_for_scene(end - start) for start, end in bounds]
    max_slots = max(per_scene_slots)
    schedule: list[tuple[int, int]] = []
    for round_i in range(max_slots):
        sweep = [i for i in range(len(bounds)) if round_i < per_scene_slots[i]]
        random.shuffle(sweep)
        schedule.extend((scene_idx, round_i) for scene_idx in sweep)
    return schedule


def init_state(
    job_id: str, video_path: str, scene_cuts: list[float], duration_s: float, fps: float,
) -> GalleryBuildState:
    bounds = _scene_bounds(scene_cuts, duration_s)
    return GalleryBuildState(
        job_id=job_id, video_path=video_path, duration_s=duration_s,
        scene_cuts=scene_cuts, fps=fps,
        schedule=_build_schedule(bounds),
        drawn_frames=[[] for _ in bounds],
    )


def _draw_fresh_frame(interval_start: float, interval_end: float, fps: float, seen: list[int]) -> tuple[float, int]:
    """
    Draws a random timestamp within [interval_start, interval_end) whose
    frame index hasn't been shown before (per `seen`), retrying up to
    _MAX_FRAME_RETRY times before falling back to accepting a repeat.

    This is the fix for a confirmed real bug: the original flat
    round-robin picked a pure-random timestamp with no memory of past
    draws at all, so any scene under 0.2s returned its own start
    timestamp — the exact same frame — every single time it recurred
    across laps (verified directly, not assumed). Returns (ts, frame_idx)
    — the caller marks frame_idx into `seen` itself, once it knows
    whether this frame is being kept or discarded (see
    draw_next_candidate).
    """
    frame_idx = round(interval_start * fps)
    for _ in range(_MAX_FRAME_RETRY):
        if interval_end - interval_start < 1.0 / fps:
            ts = interval_start
        else:
            ts = random.uniform(interval_start, interval_end)
        frame_idx = round(ts * fps)
        if frame_idx not in seen:
            return ts, frame_idx
    # Exhausted — every retry landed on an already-seen frame. Accept the
    # repeat rather than looping forever; a last resort, not the routine
    # path.
    return frame_idx / fps, frame_idx


def draw_next_candidate(reid_model, detector, state: GalleryBuildState) -> dict:
    """
    Pops exactly one (scene, sub-interval) slot off the duration
    -weighted, stratified schedule (rebuilding a fresh lap once
    exhausted — see _build_schedule) and *commits* to attempting it: up
    to _MAX_SLOT_DETECTION_RETRY different (frame-deduped, via
    _draw_fresh_frame) timestamps within that slot's own sub-interval are
    tried, looking for one where PoseLandmarker actually detects at least
    one person, before giving up on this specific slot. Returns a dict
    shaped like detect_candidates' own return value plus "ts", or an
    empty one if every local retry came up empty.

    This is the fix for a confirmed real bug: the previous design (a
    plain next_candidate_timestamp that only ever picked a timestamp, with
    "nobody detected" retried by the *caller* calling it again) advanced
    the schedule cursor on every single retry — so a "nobody detected"
    retry didn't retry the *same* scene, it silently consumed a
    *different* scene's guaranteed slot instead, in whichever order the
    schedule happened to list them. A short scene whose one-and-only slot
    landed on an undetectable frame (occluded, off-camera, a cutaway)
    could lose that slot entirely without ever actually being shown to
    the researcher — confirmed directly against real test footage, not
    assumed — even though the schedule itself was built specifically to
    guarantee that scene a turn. The caller (dashboard/app.py's
    _draw_until_nonempty) is responsible for calling this again, which
    now correctly advances to the genuinely next *scheduled* slot only
    once this one's own local retries are truly exhausted.
    """
    bounds = _scene_bounds(state.scene_cuts, state.duration_s)
    if state.schedule_cursor >= len(state.schedule):
        state.schedule = _build_schedule(bounds)
        state.schedule_cursor = 0

    scene_idx, slot_idx = state.schedule[state.schedule_cursor]
    state.schedule_cursor += 1

    n_slots = _n_slots_for_scene(bounds[scene_idx][1] - bounds[scene_idx][0])
    interval_start, interval_end = _sub_interval_bounds(bounds[scene_idx], n_slots, slot_idx)
    seen = state.drawn_frames[scene_idx]

    ts = interval_start
    for _ in range(_MAX_SLOT_DETECTION_RETRY):
        ts, frame_idx = _draw_fresh_frame(interval_start, interval_end, state.fps, seen)
        seen.append(frame_idx)  # mark tried regardless of outcome, so a
        # later retry (within this slot or a future lap) doesn't waste an
        # attempt re-landing on a frame already confirmed empty.
        result = detect_candidates(reid_model, detector, state.video_path, ts, state.fps)
        if result["candidates"]:
            result["ts"] = ts
            return result
    return {"frame_jpeg_b64": None, "candidates": [], "ts": ts}


def _scene_idx_for_timestamp(state: GalleryBuildState, ts: float) -> int:
    bounds = _scene_bounds(state.scene_cuts, state.duration_s)
    for i, (start, end) in enumerate(bounds):
        if start <= ts < end:
            return i
    return len(bounds) - 1


def detect_candidates(
    reid_model, detector, video_path: str, ts: float, fps: float,
) -> dict:
    """
    Grabs the frame at `ts`, runs the YOLO11n-seg person detector on it,
    and returns {"frame_jpeg_b64": ..., "candidates": [{"crop_jpeg_b64":
    ...}, ...]} — the full frame is for UI context (so the researcher can
    see the broader scene, e.g. who's actually on stage vs. in the
    audience), the per-candidate crops are what get shown as clickable
    thumbnails and, for whichever one is clicked, later decoded and
    embedded (see embed_candidate_from_b64) — nothing here is embedded
    yet, to avoid paying that cost for candidates nobody selects.

    MediaPipe is deliberately absent from this module entirely. Gallery
    building only ever needed *people, segmented* — it never used a single
    landmark — so running a pose model here was paying for a 33-keypoint
    skeleton purely to reach the segmentation mask riding alongside it.
    The detector supplies masks directly, and supplies better ones on
    exactly the footage this project cares about: MediaPipe's own person
    detector routinely missed a small speaker on a wide stage, or returned
    only audience members, which on this path meant the researcher was
    never even *shown* the speaker as a candidate to confirm. See
    workers/_detector.py's module docstring.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(ts * fps)))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        return {"frame_jpeg_b64": None, "candidates": []}

    ok, frame_jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    frame_jpeg_b64 = base64.b64encode(frame_jpeg.tobytes()).decode("ascii") if ok else None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    candidates = []
    for det in detect_people(detector, rgb):
        crop = _reid.crop_via_mask(rgb, det.mask)
        if crop is None:
            continue
        ok, jpeg = cv2.imencode(
            ".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
        )
        if not ok:
            continue
        candidates.append({
            "crop_jpeg_b64": base64.b64encode(jpeg.tobytes()).decode("ascii"),
        })
    return {"frame_jpeg_b64": frame_jpeg_b64, "candidates": candidates}


def embed_candidate_from_b64(reid_model, crop_jpeg_b64: str) -> np.ndarray:
    """Decodes a candidate's stored thumbnail back into an array and
    embeds it — only called for whichever single candidate the user
    actually clicks, not every candidate shown (see module docstring)."""
    jpeg_bytes = base64.b64decode(crop_jpeg_b64)
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _reid.embed_crop(reid_model, rgb)


def record_confirmation(
    store: FeatureStore,
    state: GalleryBuildState,
    embedding: np.ndarray,
    ts: float,
    thumbnail_jpeg_b64: str,
) -> str:
    """
    Adds one confirmed exemplar, updates the plateau streak, and returns
    the verdict: "new" or "redundant" — see wikis/Gesture-Worker.md's
    worked examples for exactly this state transition.
    """
    existing = store.get_gallery(state.job_id)
    scene_idx = _scene_idx_for_timestamp(state, ts)
    entry = GalleryEntry(
        embedding=embedding.tolist(), timestamp_s=ts, scene_idx=scene_idx,
        thumbnail_jpeg_b64=thumbnail_jpeg_b64,
    )
    store.add_gallery_entry(state.job_id, entry)
    state.entries_confirmed += 1

    if not existing:
        state.streak = 0
        return "new"

    gallery_arr = np.array([e.embedding for e in existing], dtype=np.float32)
    if _reid.max_similarity(embedding, gallery_arr) >= GALLERY_REDUNDANCY_GAMMA:
        state.streak += 1
        return "redundant"
    state.streak = 0
    return "new"


def should_stop(state: GalleryBuildState) -> Optional[str]:
    """Returns the stop reason ("plateau"/"cap"), or None to keep going.
    Manual stop is set directly by the dashboard's own callback when the
    researcher clicks "I'm done" — it's a user action, not something
    derivable from the accumulated state, so it isn't decided here."""
    if state.entries_confirmed >= GALLERY_MAX_SIZE:
        return "cap"
    if state.entries_confirmed >= GALLERY_MIN_SIZE and state.streak >= PLATEAU_STREAK:
        return "plateau"
    return None
