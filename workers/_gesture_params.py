"""
workers/_gesture_params.py
---------------------------
Every tunable in the gesture pipeline, in one place, as a frozen dataclass.

These were module constants in workers/gesture_worker.py, each with a
comment explaining its value; those comments are still there, next to the
aliases that now read out of `DEFAULT_PARAMS`. What changed is that a value
can differ *per video*, which the constants made impossible.

## Why per-video at all

No single setting wins everywhere, and `conf_threshold` is the proof: 0.25
discards the speaker when her arms are down (measured 0.162), while 0.1
admits enough of a crowd that audience members reach the identity check.
Which failure you prefer depends on the video. See wikis/Gesture-Worker.md.

## The rule for tuning these

Tune against **identity correctness** — is the track on the right person —
never against a downstream gesture statistic. Choosing thresholds until
mean_wrist_velocity looks right would make the metric a product of the
tuning rather than a measurement of the speaker, and nothing downstream
could tell the difference. scripts/inspect_selection.py exists to make the
former cheap: it runs detection, re-ID and selection *without* pose, so a
parameter can be judged from boxes in seconds per scene rather than hours
per video.

## Provenance

Whatever a job actually ran with is written into its MongoDB video document
(`gesture_params`), because a corpus processed under per-video settings is
uninterpretable without them — "which threshold produced this number" has
to be answerable years later, from the data alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable

# Frame-normalised centre, for the centrality tie-break.
_FRAME_CENTER = (0.5, 0.5)

TieBreakKey = Callable[[tuple[float, float]], float]


def _centrality_key(centre: tuple[float, float]) -> float:
    """Distance from the frame centre — lower is better.

    The original tie-break. On a TED-style stage the projection screen sits
    above and beside the speaker, so she is the more central of the two.
    """
    cx, cy = centre
    return ((cx - _FRAME_CENTER[0]) ** 2 + (cy - _FRAME_CENTER[1]) ** 2) ** 0.5


def _vertical_key(centre: tuple[float, float]) -> float:
    """Height in frame — lower *in the frame* is better.

    Box coordinates are image coordinates, so y grows downwards and the
    candidate standing lowest in frame has the **largest** cy; negating
    keeps "smaller key wins" consistent with _centrality_key.

    The physical argument is stronger than centrality's: a projection is
    mounted above stage level, so the live speaker is below her own relayed
    image regardless of where either sits horizontally. It is the right
    choice for a video where the screen is centred — centrality has no
    purchase there, since the screen is exactly where centrality looks.

    It is *not* a general improvement. In an audience cutaway the front row
    is lower in frame than anyone, so this rule prefers the nearest
    audience member where centrality would prefer a central one. Both are
    comparative: neither can reject a candidate set containing no speaker.
    That is what the gallery threshold is for.
    """
    return -centre[1]


TIE_BREAKS: dict[str, TieBreakKey] = {
    "centrality": _centrality_key,
    "vertical": _vertical_key,
}


@dataclass(frozen=True)
class GestureParams:
    """One job's gesture-pipeline settings. Frozen so a worker cannot
    mutate what its siblings were given — pool children each receive a copy
    through _pool_init, and a divergence between them would be invisible in
    the output."""

    # --- Detection ------------------------------------------------------
    # Detector confidence floor. Deliberately low: detector confidence is
    # anti-correlated with correctness in the relay case (0.89 for the
    # screen against 0.162 for the speaker), so a high floor removes her
    # and keeps the screen. The cost is crowd shots, where it admits many
    # small, low-contrast candidates.
    conf_threshold: float = 0.1

    # Minimum confidence for MediaPipe's pose *detector* stage to accept
    # that the crop contains a person at all. 0.60 rather than MediaPipe's
    # own 0.5 default, carried over from the value the worker shipped with.
    #
    # Worth tuning per video, but do not expect it to reject malformed
    # poses: the 2:10 failure in "20 How to design gender bias out of your
    # workplace" produced a skeleton stitched across several seated people
    # whose hip and knee landmarks still reported 0.85-0.91 visibility.
    # MediaPipe is confident about anatomically impossible output, so this
    # threshold filters "is anyone here", not "is this a plausible body".
    pose_detection_confidence: float = 0.60

    # Minimum confidence for landmarks to be emitted once a pose is
    # detected. MediaPipe's own default; exposed alongside the above
    # because the two gate different stages and a video needing a stricter
    # pose gate usually wants both moved.
    pose_presence_confidence: float = 0.5

    # min_tracking_confidence is deliberately NOT here. It only applies in
    # VIDEO/LIVE_STREAM running mode, and this landmarker runs in IMAGE
    # mode (see gesture_worker's "IMAGE mode"), so a field for it would be
    # a knob that silently does nothing — exactly what this class exists to
    # prevent.

    # --- Identity -------------------------------------------------------
    # Minimum nearest-exemplar (max-pooled) similarity for a candidate to
    # be accepted as the speaker. Reuses gallery-building's redundancy
    # threshold: both sides ask the same max-pooled question. Calibrated
    # against the speaker-vs-projection problem, NOT against audience
    # crops — raising it is the first lever for a video where audience
    # members are being matched.
    gallery_match_threshold: float = 0.80

    # While Locked, a candidate is verified against the previous accepted
    # frame rather than the gallery.
    continuity_threshold: float = 0.80

    # How long a lock survives before it must re-anchor against the
    # gallery. ~1s at 30fps, which is about where tracked appearance has
    # meaningfully moved (similarity 0.897 median at 1s, 0.796 at 2s).
    lock_lease_frames: int = 30

    # Frame-fraction distance beyond which a "continuation" is treated as
    # track loss. Reject-only: it can invalidate a lock, never choose who
    # holds it.
    max_track_jump: float = 0.3

    # --- Tracklets ------------------------------------------------------
    # Minimum box overlap to continue a tracklet. Loose on purpose: a
    # speaker at 30fps barely moves between frames, and fragmenting her
    # into stubs is the failure that matters.
    track_iou_min: float = 0.3

    # Frames a tracklet survives with no detection before it is closed.
    # Load-bearing: her detectability swings with her pose, so she drops
    # out repeatedly; a short tolerance shatters her track while the static
    # projection stays whole.
    track_max_gap_frames: int = 30

    # Detections a tracklet needs before it may be selected. A floor
    # against single-frame noise, not a preference for long tracks —
    # raising it hands projection scenes to the screen.
    track_min_support: int = 10

    # Identity crops are sampled every Nth frame of a scene (each costs a
    # ~4.8ms mask build), and at most this many per tracklet are embedded.
    track_id_stride: int = 10
    track_id_samples: int = 5

    # --- Selection ------------------------------------------------------
    # Which geometric rule breaks a tie between candidates that have
    # *already* cleared the gallery. Applies in both places a tie is
    # broken: per-frame (_gallery_match) and per-scene (_select_tracklet).
    # See TIE_BREAKS above for what each rule assumes and where it fails.
    tie_break: str = "centrality"

    # --- Execution (not a model parameter; affects speed, not output) ---
    pool_processes: int = 4
    pool_threads_per_process: int = 1

    def __post_init__(self) -> None:
        if self.tie_break not in TIE_BREAKS:
            raise ValueError(
                f"tie_break={self.tie_break!r} — expected one of "
                f"{', '.join(sorted(TIE_BREAKS))}"
            )
        for name in ("gallery_match_threshold", "continuity_threshold", "conf_threshold",
                     "pose_detection_confidence", "pose_presence_confidence"):
            v = getattr(self, name)
            if not 0.0 < v <= 1.0:
                raise ValueError(f"{name}={v} — must be in (0, 1]")
        if self.track_min_support < 1 or self.track_id_samples < 1:
            raise ValueError("track_min_support and track_id_samples must be >= 1")

    @property
    def tie_break_key(self) -> TieBreakKey:
        """The sort key for this job's tie-break: smaller wins."""
        return TIE_BREAKS[self.tie_break]

    def to_dict(self) -> dict:
        """For storage alongside a job's results — see "Provenance"."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "GestureParams":
        """Build from a manifest entry's `gesture_params` block, ignoring
        nothing: an unknown key is a typo'd parameter that would otherwise
        silently have no effect, which is the whole failure mode this
        class exists to prevent."""
        if not data:
            return DEFAULT_PARAMS
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown gesture parameter(s): {', '.join(sorted(unknown))}. "
                f"Known: {', '.join(sorted(known))}"
            )
        return replace(DEFAULT_PARAMS, **data)


DEFAULT_PARAMS = GestureParams()
