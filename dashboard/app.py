"""
dashboard/app.py
----------------
Single-column dashboard with Dash-served video player.

Video is served via a Flask route so the browser can actually load it.
Clicking any chart seeks the video to that timestamp.
"""

from __future__ import annotations

import base64
import dataclasses
import os
import re
import tempfile
import threading
import uuid as _uuid_module
from datetime import datetime
from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc
import flask
import plotly.graph_objects as go
import spacy
from loguru import logger
from plotly.subplots import make_subplots
from dash import ALL, Input, Output, State, callback, clientside_callback, dash_table, dcc, html
from dotenv import load_dotenv

load_dotenv()

from core.logging_setup import setup_file_logging

setup_file_logging("dashboard")

from core import corpus_analysis
from core.bulk_orchestrator import BulkOrchestrator, _resolve_path, load_manifest
from core.drive_download import download_drive_file
from core.feature_store import FeatureStore
from core.gallery_builder import (
    PLATEAU_STREAK, GalleryBuildState, draw_next_candidate,
    embed_candidate_from_b64, init_state,
    record_confirmation, should_stop,
)
from core.models import FusedWindow, HorizontalAngle, VerticalAngle
from core.orchestrator import Orchestrator
from core.preprocessing import probe_video
from core.results_repository import ResultsRepository
from workers._detector import load_detector
from workers._reid import load_reid_model
from workers.gesture_worker import GestureWorker

# ─────────────────────────────────────────────────────────────────────────────
# App + video serving
# ─────────────────────────────────────────────────────────────────────────────

store = FeatureStore()
_orch = Orchestrator(store=store)

# Used only to detect/decompose contractions typed into the keyword search
# box (see kw_search) — English-specific, same as extract_collocation_en.
_nlp_en = spacy.load("en_core_web_sm")

# Browse Corpus reads from Mongo. Optional at dashboard-startup time — if
# MONGO_URI isn't configured or the cluster isn't reachable, the Browse
# Corpus tab just reports that rather than crashing the whole app (Live
# Analysis must keep working either way).
try:
    repo: ResultsRepository | None = ResultsRepository()
except Exception as exc:
    logger.warning(f"[dashboard] MongoDB unavailable, Browse Corpus disabled: {exc}")
    repo = None


server = flask.Flask(__name__)
server.config["MAX_CONTENT_LENGTH"] = None  # allow large video uploads
FONT_URL = "https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Fraunces:ital,wght@0,300;0,600;1,300&display=swap"

app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_URL],
    title="MultiArth",
    update_title=None,
    suppress_callback_exceptions=True,
)

# Store the video path globally so the Flask route can serve it
_VIDEO_PATH: dict = {"path": None}

# Bulk Upload tab's background-thread progress, polled by poll_bulk_progress.
# Guarded by a lock since progress_cb fires from the background thread while
# the poll callback reads it from Dash's own request-handling thread.
_BULK_LOCK = threading.Lock()
_BULK_STATE: dict = {"running": False, "last_event": None, "summary": None}

# Speaker-gallery confirmation (Bulk Upload only — see
# core/gallery_builder.py, workers/gesture_worker.py's "Speaker
# re-identification"). The YOLO11n-seg detector and OSNet model are each
# loaded once, lazily, and reused across every video's gallery-confirmation
# in a session — safe as module-level state under gunicorn's single-worker
# config (see Dockerfile), same assumption _BULK_STATE above already makes.
# No pose model is loaded here: gallery building needs people and masks,
# never landmarks (see core/gallery_builder.py's detect_candidates).
_GALLERY_DETECTOR = None
_GALLERY_REID_MODEL = None


def _ensure_gallery_models():
    global _GALLERY_DETECTOR, _GALLERY_REID_MODEL
    if _GALLERY_REID_MODEL is None:
        _GALLERY_REID_MODEL = load_reid_model()
    if _GALLERY_DETECTOR is None:
        _GALLERY_DETECTOR = load_detector()
    return _GALLERY_DETECTOR, _GALLERY_REID_MODEL


def _bulk_work_dir() -> str:
    return os.environ.get("WORK_DIR", "/tmp/mannerism")

# Browse Corpus videos live on Drive, not locally — but a same-origin
# <video> tag (needed for the pose overlay's currentTime/videoWidth access,
# which a cross-origin Drive iframe can never expose) means fetching the
# bytes ourselves first. Cached under a bounded, LRU-evicted directory so
# this doesn't re-accumulate the same unbounded-local-storage problem the
# upload cleanup elsewhere in this file was written to avoid.
_BROWSE_CACHE_DIR = Path(os.environ.get("WORK_DIR", "/tmp/mannerism")) / "browse_cache"
_BROWSE_CACHE_MAX_VIDEOS = 3
_BROWSE_CACHE_LOCKS: dict[str, threading.Lock] = {}
_BROWSE_CACHE_LOCKS_GUARD = threading.Lock()


def _browse_cache_lock(job_id: str) -> threading.Lock:
    # A lock per job_id (not one global lock) so concurrent requests for
    # *different* cached videos don't block each other — only concurrent
    # requests for the *same* not-yet-cached video need to serialise.
    with _BROWSE_CACHE_LOCKS_GUARD:
        return _BROWSE_CACHE_LOCKS.setdefault(job_id, threading.Lock())


def _evict_browse_cache(keep: Path) -> None:
    files = sorted(
        (p for p in _BROWSE_CACHE_DIR.iterdir() if p.is_file() and p != keep),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in files[_BROWSE_CACHE_MAX_VIDEOS - 1:]:
        try:
            stale.unlink()
        except OSError as exc:
            logger.warning(f"[dashboard] Could not evict browse cache file {stale}: {exc}")


def _serve_browse_video(job_id: str, collection: str) -> flask.Response:
    if repo is None:
        return flask.Response("MongoDB not configured", status=404)

    _BROWSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _BROWSE_CACHE_DIR / f"{job_id}.mp4"

    with _browse_cache_lock(job_id):
        if cache_path.exists():
            cache_path.touch()  # bump mtime so eviction treats this as recently used
        else:
            doc = repo.get_video_doc(collection, job_id)
            drive_url = (doc or {}).get("drive_url")
            if not drive_url:
                return flask.Response("No source video for this job", status=404)
            try:
                download_drive_file(drive_url, str(cache_path))
            except Exception as exc:
                logger.error(f"[dashboard] Could not fetch {drive_url}: {exc}")
                return flask.Response(f"Could not fetch video: {exc}", status=502)
            _evict_browse_cache(keep=cache_path)

    return flask.send_from_directory(str(_BROWSE_CACHE_DIR), cache_path.name, conditional=True)


@server.route("/video")
def serve_video():
    """Serve the analysis video file so the browser <video> tag can load it.

    Live Analysis videos are served straight from local disk; Browse Corpus
    videos (?job_id=&collection=) are downloaded from Drive on first request
    and cached — see _serve_browse_video."""
    job_id = flask.request.args.get("job_id")
    collection = flask.request.args.get("collection")
    if job_id and collection:
        return _serve_browse_video(job_id, collection)

    path = _VIDEO_PATH.get("path")
    if not path or not os.path.exists(path):
        return flask.Response("Video not found", status=404)
    directory = str(Path(path).parent)
    filename   = Path(path).name
    return flask.send_from_directory(directory, filename, conditional=True)


# ─────────────────────────────────────────────────────────────────────────────
# Redis-vs-Mongo artifact lookup (Live Analysis vs Browse Corpus)
# ─────────────────────────────────────────────────────────────────────────────

def _artifacts(job_id: str, data_source: str, collection: str | None = None) -> dict:
    if data_source == "mongo":
        return (repo.get_artifacts(collection, job_id) if repo and collection else None) or {}
    return {
        "spectrogram":      store.get_spectrogram(job_id),
        "waveform":         store.get_waveform(job_id),
        "collocations":     store.get_collocations(job_id),
        "wordlist":         store.get_wordlist(job_id),
        "ngrams":           store.get_ngrams(job_id),
        "segmented_tokens": store.get_segmented_tokens(job_id),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────

C = {
    "bg":      "#F7F5F0",
    "surface": "#FFFFFF",
    "border":  "#E8E4DC",
    "text":    "#1A1814",
    "muted":   "#8A8478",
    "gesture": "#C84B31",
    "prosody": "#2D6A4F",
    "verbal":  "#1B4F8A",
    "camera":  "#7B5EA7",
    "cursor":  "#E8A838",
    "corpus":  "#4361EE",
}

KW_COLOUR = "#00B4D8"  # cyan-teal — distinct from all section colours and cursor amber

# Concordance context (kw_search): CJK words are conventionally displayed
# with no space between them, unlike space-separated languages — detected
# from the words themselves rather than a stored per-job language code
# (segmented_tokens carries no language tag), by checking for any character
# in the common CJK Unicode blocks.
_CJK_CHAR_RE = re.compile(r"[一-鿿぀-ヿ가-힯]")

_POS_FILTER_SETS: dict[str, set[str]] = {
    "NOUN":  {"NOUN", "PROPN"},
    "VERB":  {"VERB", "AUX"},
    "ADJ":   {"ADJ"},
    "ADV":   {"ADV"},
}
_KNOWN_POS = {"NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV"}

POS_COLOUR = {
    "NOUN": C["verbal"],  "PROPN": C["verbal"],
    "VERB": C["prosody"], "AUX":   C["prosody"],
    "ADJ":  C["cursor"],  "ADV":   C["camera"],
}

# Full display labels for spaCy's Universal POS tags (kw-wordlist-table's
# "pos" column) — stopwords are no longer filtered out of the word list, so
# entries can carry any of these, not just the six content-word tags above.
# "?" is the fallback tag used when no spaCy model exists for a language.
_POS_LABELS: dict[str, str] = {
    "ADJ":   "Adjective",
    "ADP":   "Adposition (preposition/postposition)",
    "ADV":   "Adverb",
    "AUX":   "Auxiliary verb",
    "CCONJ": "Coordinating conjunction",
    "DET":   "Determiner",
    "INTJ":  "Interjection",
    "NOUN":  "Noun",
    "NUM":   "Numeral",
    "PART":  "Particle",
    "PRON":  "Pronoun",
    "PROPN": "Proper noun",
    "PUNCT": "Punctuation",
    "SCONJ": "Subordinating conjunction",
    "SYM":   "Symbol",
    "VERB":  "Verb",
    "X":     "Other (unknown/foreign)",
    "SPACE": "Space",
    "?":     "Unknown",
}

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color=C["text"], size=11),
    margin=dict(l=52, r=24, t=36, b=36),
    xaxis=dict(
        showgrid=True, gridcolor=C["border"], gridwidth=1,
        zeroline=False, tickfont=dict(size=10),
        title=dict(text="Time (s)", font=dict(size=10, color=C["muted"])),
    ),
    yaxis=dict(
        showgrid=True, gridcolor=C["border"], gridwidth=1,
        zeroline=False, tickfont=dict(size=10),
    ),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="left", x=0, font=dict(size=10),
        bgcolor="rgba(0,0,0,0)",
    ),
    hovermode="x unified",
)

SECTION_STYLE = {
    "backgroundColor": C["surface"],
    "border": f"1px solid {C['border']}",
    "borderRadius": "12px",
    "padding": "28px 32px",
    "marginBottom": "20px",
}

LABEL_STYLE = {
    "fontFamily": "DM Mono, monospace",
    "fontSize": "10px",
    "letterSpacing": "0.12em",
    "textTransform": "uppercase",
    "color": C["muted"],
    "marginBottom": "4px",
    "margin": "0",
}

LABEL_STYLE_C = {
    "fontFamily": "DM Mono, monospace",
    "fontSize": "10px",
    "letterSpacing": "0.12em",
    "textTransform": "uppercase",
    "color": C["text"],
    "marginBottom": "4px",
    "margin": "0",
}


CHART_CFG = {"displayModeBar": False}

# Per-tab base style; each tab overrides color + top-border when selected
_TAB_BASE = {
    "fontFamily": "DM Mono, monospace",
    "fontSize":   "11px",
    "letterSpacing": "0.08em",
    "color":         C["muted"],
    "backgroundColor": C["bg"],
    "border":     f"1px solid {C['border']}",
    "borderBottom": "none",
    "padding":    "10px 22px",
    "borderRadius": "6px 6px 0 0",
}

def _tab_sel(accent):
    return {**_TAB_BASE, "color": accent,
            "borderTop": f"2px solid {accent}",
            "backgroundColor": C["surface"]}

_TAB_PAD = {
    "backgroundColor": C["surface"],
    "border": f"1px solid {C['border']}",
    "borderTop": "none",
    "borderRadius": "0 0 12px 12px",
    "padding": "28px 32px",
    "minHeight": "300px",
}

# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────────────

def kpi_card(label, value, accent, delta=""):
    return html.Div([
        html.P(label, style=LABEL_STYLE),
        html.P(value, style={
            "fontFamily": "Fraunces, serif",
            "fontSize": "1.8rem",
            "fontWeight": "600",
            "color": accent,
            "margin": "4px 0 0 0",
            "lineHeight": "1",
        }),
        html.P(delta, style={
            "fontFamily": "DM Mono, monospace",
            "fontSize": "10px",
            "color": C["muted"],
            "margin": "4px 0 0 0",
        }) if delta else html.Div(),
    ], style={
        "backgroundColor": C["surface"],
        "border": f"1px solid {C['border']}",
        "borderTop": f"3px solid {accent}",
        "borderRadius": "8px",
        "padding": "18px 22px",
        "flex": "1",
        "minWidth": "130px",
    })


def section_header(title, accent, subtitle=""):
    return html.Div([
        html.Div(style={
            "width": "3px", "height": "22px",
            "backgroundColor": accent,
            "borderRadius": "2px",
            "marginRight": "12px",
            "flexShrink": "0",
        }),
        html.Div([
            html.H3(title, style={
                "fontFamily": "Fraunces, serif",
                "fontSize": "1.1rem",
                "fontWeight": "600",
                "color": C["text"],
                "margin": "0",
            }),
            html.P(subtitle, style={
                "fontFamily": "DM Mono, monospace",
                "fontSize": "10px",
                "color": C["muted"],
                "margin": "2px 0 0 0",
            }) if subtitle else html.Div(),
        ]),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "20px"})


def _analysis_progress(state: str, message: str, worker_progress: dict | None = None):
    """Inline status indicator shown below the upload zone."""
    if state == "idle":
        return html.Div()

    colour = {"running": C["camera"], "done": C["prosody"], "failed": C["gesture"]}.get(state, C["muted"])
    icon = (
        dbc.Spinner(size="sm", spinner_style={"width": "12px", "height": "12px", "borderWidth": "2px",
                                               "marginRight": "8px", "color": colour})
        if state == "running"
        else html.Span("✓ " if state == "done" else "✗ ", style={"marginRight": "4px", "fontWeight": "600"})
    )
    header = html.Div([icon, html.Span(message, style={
        "fontFamily": "DM Mono, monospace", "fontSize": "11px", "color": colour,
    })], style={"display": "flex", "alignItems": "center", "marginTop": "10px"})

    if not worker_progress or state != "running":
        return html.Div([header])

    total = worker_progress.get("total", 0)
    if not total:
        return html.Div([header])

    _WORKERS = [
        ("gesture", "Pose Est.", C["gesture"]),
        ("prosody", "Acoustic",  C["prosody"]),
        ("verbal",  "Verbal",    C["verbal"]),
        ("camera",  "Camera",    C["camera"]),
    ]

    rows = []
    for key, label, bar_colour in _WORKERS:
        done = worker_progress.get(key, 0)
        pct = min(100, round(done * 100 / total))
        is_complete = pct == 100
        is_transcribing = (
            key == "verbal" and done == 0
            and worker_progress.get("gesture", 0) > 0
        )

        suffix = (
            html.Span(" transcribing…", style={
                "fontFamily": "DM Mono, monospace", "fontSize": "9px",
                "color": C["muted"], "marginLeft": "4px",
            }) if is_transcribing else html.Span()
        )

        rows.append(html.Div([
            html.Span(label, style={
                "fontFamily": "DM Mono, monospace", "fontSize": "10px",
                "color": C["muted"], "width": "54px", "flexShrink": "0",
            }),
            html.Div(
                html.Div(style={
                    "width": f"{pct}%", "height": "100%",
                    "background": C["prosody"] if is_complete else bar_colour,
                    "borderRadius": "3px",
                    "transition": "width 0.5s ease",
                }),
                style={
                    "flex": "1", "height": "5px", "borderRadius": "3px",
                    "background": C["border"],
                },
            ),
            html.Span(f"{pct}%", style={
                "fontFamily": "DM Mono, monospace", "fontSize": "10px",
                "color": C["prosody"] if is_complete else C["muted"],
                "width": "30px", "textAlign": "right", "flexShrink": "0",
            }),
            suffix,
        ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "5px"}))

    return html.Div([header, html.Div(rows, style={"marginTop": "8px"})])


def empty_fig(title=""):
    fig = go.Figure()
    fig.update_layout(**PLOT_LAYOUT,
        title=dict(text=title, font=dict(size=11, color=C["muted"])))
    return fig


def add_cursor(fig, t, occ=None):
    if t:
        fig.add_vline(x=t, line=dict(color=C["cursor"], width=2.5, dash="dot"))
    if occ:
        for o in occ:
            fig.add_vline(x=o["start_s"], line=dict(color=KW_COLOUR, width=2), opacity=0.7)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart IDs for universal scrubber
# ─────────────────────────────────────────────────────────────────────────────

CHART_IDS = [
    "g-velocity", "g-handedness",
    "p-spectrogram", "p-waveform", "p-f0", "p-intensity",
    "c-shot", "c-h-angle", "c-v-angle", "c-cutrate", "c-facearea", "c-trend",
]

# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

app.layout = html.Div(style={"backgroundColor": C["bg"], "minHeight": "100vh"}, children=[

    # ── Top bar ──────────────────────────────────────────────────────────────
    html.Div(style={
        "backgroundColor": C["surface"],
        "borderBottom": f"1px solid {C['border']}",
        "padding": "0 40px",
        "height": "56px",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "space-between",
        "position": "sticky", "top": "0", "zIndex": "100",
        "boxShadow": "0 1px 0 rgba(0,0,0,0.04)",
    }, children=[
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "4px"}, children=[
            html.Img(
                src="/assets/MultiArth-removebg-preview.png",
                style={"height": "35px", "width": "auto"},
            ),
            html.Div([
                html.Span("MULTI", style={
                    "fontFamily": "DM Mono, monospace", "fontSize": "13px",
                    "fontWeight": "500", "letterSpacing": "0.18em", "color": C["text"],
                }),
                html.Span("ARTH", style={
                    "fontFamily": "DM Mono, monospace", "fontSize": "13px",
                    "fontWeight": "300", "letterSpacing": "0.18em", "color": C["muted"],
                }),
            ]),
        ]),
    ]),

    # ── Main ─────────────────────────────────────────────────────────────────
    html.Div(style={"maxWidth": "1200px", "margin": "0 auto", "padding": "32px 40px"}, children=[

        # KPI strip
        html.Div(id="kpi-strip", style={
            "display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "28px",
        }),

        # ── MODE TOGGLE ────────────────────────────────────────────────────
        dcc.Tabs(id="mode-tabs", value="live", style={"marginBottom": "20px"}, children=[
            dcc.Tab(label="Live Analysis", value="live",
                    style=_TAB_BASE, selected_style=_tab_sel(C["muted"])),
            dcc.Tab(label="Browse Corpus", value="browse",
                    style=_TAB_BASE, selected_style=_tab_sel(C["muted"])),
            dcc.Tab(label="Bulk Upload", value="bulk",
                    style=_TAB_BASE, selected_style=_tab_sel(C["muted"])),
        ]),

        # ── UPLOAD (Live Analysis) ───────────────────────────────────────
        html.Div(id="live-panel", children=[
            html.Div(style=SECTION_STYLE, children=[
                section_header("Video Upload", C["muted"], "Drop a video file to begin analysis"),

                dcc.Upload(
                    id="video-upload",
                    children=html.Div([
                        html.P("Drop a video file here, or click to browse", style={
                            "fontFamily": "DM Mono, monospace", "fontSize": "12px",
                            "color": C["muted"], "margin": "0 0 4px 0",
                        }),
                        html.P("MP4 · MOV · MKV · AVI · WEBM", style={
                            "fontFamily": "DM Mono, monospace", "fontSize": "10px",
                            "letterSpacing": "0.1em", "color": C["border"], "margin": "0",
                        }),
                    ], style={"textAlign": "center", "padding": "16px 0"}),
                    accept="video/*",
                    multiple=False,
                    max_size=-1,
                    style={
                        "width": "100%",
                        "border": f"1px dashed {C['border']}",
                        "borderRadius": "8px",
                        "cursor": "pointer",
                        "marginBottom": "12px",
                        "backgroundColor": C["bg"],
                    },
                ),
                # ── Speaker gallery confirmation (Live Analysis) ────────
                # Same flow as Bulk Upload's (core/gallery_builder.py,
                # workers/gesture_worker.py's "Speaker re-identification"),
                # just for exactly one video instead of a manifest — see
                # handle_upload/handle_live_gallery_interaction below.
                # Analysis itself doesn't start until this finishes.
                dcc.Loading(type="dot", color=C["muted"], children=[
                html.Div(id="live-gallery-build-panel", style={
                    "display": "none", "marginBottom": "16px",
                    "border": f"1px solid {C['border']}", "borderRadius": "8px",
                    "padding": "14px", "backgroundColor": C["bg"],
                }, children=[
                    html.P("SPEAKER GALLERY", style=LABEL_STYLE),
                    html.Div(id="live-gallery-build-status", style={
                        "fontFamily": "DM Mono, monospace", "fontSize": "11px",
                        "color": C["text"], "marginBottom": "10px",
                    }),
                    html.Img(id="live-gallery-build-frame", style={
                        "maxWidth": "100%", "maxHeight": "320px", "display": "block",
                        "borderRadius": "6px", "marginBottom": "10px",
                        "border": f"1px solid {C['border']}",
                    }),
                    html.P(
                        "Click whichever crop below is the speaker — or skip if "
                        "they're not visible in this frame.",
                        style={"fontFamily": "DM Mono, monospace", "fontSize": "10px",
                               "color": C["muted"], "marginBottom": "8px"},
                    ),
                    html.Div(id="live-gallery-candidate-row", style={
                        "display": "flex", "gap": "8px", "flexWrap": "wrap",
                        "marginBottom": "12px",
                    }),
                    html.Div(style={"display": "flex", "gap": "10px"}, children=[
                        dbc.Button("No speaker visible — skip", id="live-gallery-skip-btn",
                                   size="sm", outline=True, color="secondary",
                                   style={"fontFamily": "DM Mono, monospace", "fontSize": "11px"}),
                        dbc.Button("I'm done — start analysis", id="live-gallery-done-btn",
                                   size="sm", color="primary",
                                   style={"fontFamily": "DM Mono, monospace", "fontSize": "11px"}),
                    ]),
                ]),
                ]),  # end dcc.Loading

                html.Div(id="analysis-status"),
            ]),
        ]),

        # ── BROWSE CORPUS ────────────────────────────────────────────────
        html.Div(id="browse-panel", style={"display": "none"}, children=[
            html.Div(style=SECTION_STYLE, children=[
                section_header("Browse Corpus", C["muted"],
                               "Previously-processed videos, shipped to MongoDB"),
                html.Div(id="browse-collection-buttons", style={
                    "display": "flex", "gap": "10px", "marginBottom": "16px", "flexWrap": "wrap",
                }),
                dcc.Dropdown(
                    id="browse-video-dropdown",
                    placeholder="Select a video…",
                    style={"fontFamily": "DM Mono, monospace", "fontSize": "12px",
                           "marginBottom": "16px"},
                ),
                html.Div(id="browse-video-info"),
            ]),
        ]),

        # ── BULK UPLOAD ──────────────────────────────────────────────────
        html.Div(id="bulk-panel", style={"display": "none"}, children=[
            html.Div(style=SECTION_STYLE, children=[
                section_header("Bulk Upload", C["muted"],
                               "Process a manifest of videos and ship them to MongoDB"),

                dcc.Upload(
                    id="manifest-upload",
                    children=html.Div([
                        html.P("Drop a manifest file here, or click to browse", style={
                            "fontFamily": "DM Mono, monospace", "fontSize": "12px",
                            "color": C["muted"], "margin": "0 0 4px 0",
                        }),
                        html.P("YAML · JSON — {collection, drive_url, label} per entry", style={
                            "fontFamily": "DM Mono, monospace", "fontSize": "10px",
                            "letterSpacing": "0.1em", "color": C["border"], "margin": "0",
                        }),
                    ], style={"textAlign": "center", "padding": "16px 0"}),
                    accept=".yml,.yaml,.json",
                    multiple=False,
                    style={
                        "width": "100%",
                        "border": f"1px dashed {C['border']}",
                        "borderRadius": "8px",
                        "cursor": "pointer",
                        "marginBottom": "12px",
                        "backgroundColor": C["bg"],
                    },
                ),

                # dcc.Loading shows its spinner whenever a running callback
                # targets an Output inside it — covers both the manifest
                # -upload callback's now-real work (resolve/download,
                # probe, scene-cut detection, load OSNet+detector, draw
                # the first candidate — all synchronous, several seconds
                # the first time) and each gallery click's own embed
                # -and-draw-next-candidate work, neither of which had any
                # visual feedback before — see core/gallery_builder.py.
                dcc.Loading(type="dot", color=C["muted"], children=[
                html.Div(id="bulk-manifest-summary", style={
                    "fontFamily": "DM Mono, monospace", "fontSize": "11px",
                    "color": C["muted"], "marginBottom": "12px",
                }),

                # ── Speaker gallery confirmation ────────────────────────
                # Front-loaded: every manifest video's gallery is built
                # interactively here before "Start" (below) becomes
                # available — see core/gallery_builder.py,
                # workers/gesture_worker.py's "Speaker re-identification".
                # Dashboard bulk-upload only; single-file upload never
                # shows this and always uses the heuristic-only path.
                html.Div(id="gallery-build-panel", style={
                    "display": "none", "marginBottom": "16px",
                    "border": f"1px solid {C['border']}", "borderRadius": "8px",
                    "padding": "14px", "backgroundColor": C["bg"],
                }, children=[
                    html.P("SPEAKER GALLERY", style=LABEL_STYLE),
                    html.Div(id="gallery-build-status", style={
                        "fontFamily": "DM Mono, monospace", "fontSize": "11px",
                        "color": C["text"], "marginBottom": "10px",
                    }),
                    html.Img(id="gallery-build-frame", style={
                        "maxWidth": "100%", "maxHeight": "320px", "display": "block",
                        "borderRadius": "6px", "marginBottom": "10px",
                        "border": f"1px solid {C['border']}",
                    }),
                    html.P(
                        "Click whichever crop below is the speaker — or skip if "
                        "they're not visible in this frame.",
                        style={"fontFamily": "DM Mono, monospace", "fontSize": "10px",
                               "color": C["muted"], "marginBottom": "8px"},
                    ),
                    html.Div(id="gallery-candidate-row", style={
                        "display": "flex", "gap": "8px", "flexWrap": "wrap",
                        "marginBottom": "12px",
                    }),
                    html.Div(style={"display": "flex", "gap": "10px"}, children=[
                        dbc.Button("No speaker visible — skip", id="gallery-skip-btn",
                                   size="sm", outline=True, color="secondary",
                                   style={"fontFamily": "DM Mono, monospace", "fontSize": "11px"}),
                        dbc.Button("I'm done with this video", id="gallery-done-btn",
                                   size="sm", outline=True, color="secondary",
                                   style={"fontFamily": "DM Mono, monospace", "fontSize": "11px"}),
                    ]),
                ]),
                ]),  # end dcc.Loading — deliberately stops here, before
                # bulk-start-btn: that button's `disabled` prop is also an
                # Output of poll_bulk_progress, which fires every 1.5s via
                # bulk-poll-interval for the *entire* bulk run, not just
                # gallery-building — including it inside dcc.Loading made
                # the spinner flash on every poll tick throughout
                # processing, not just during actual gallery setup.

                html.Div(style={"display": "flex", "gap": "10px", "alignItems": "center",
                                 "marginBottom": "16px"}, children=[
                    dbc.Checkbox(id="bulk-force", value=False, label="Reprocess already-shipped videos",
                                 style={"fontFamily": "DM Mono, monospace", "fontSize": "11px"}),
                    dbc.Button("Start", id="bulk-start-btn", size="sm", disabled=True,
                               style={"backgroundColor": C["muted"], "border": "none",
                                      "fontFamily": "DM Mono, monospace", "fontSize": "11px"}),
                ]),

                dcc.Interval(id="bulk-poll-interval", interval=1500, disabled=True),
                html.Div(id="bulk-status"),
            ]),
        ]),

        # ── CORPUS ANALYSIS ───────────────────────────────────────────────
        # Per-video windowed analysis — irrelevant while bulk-processing many
        # videos at once, so hidden for that tab (see toggle_mode).
        html.Div(id="analysis-charts-section", children=[
        html.Div(style=SECTION_STYLE, children=[
            section_header("Verbal Language", C["corpus"],
                           "spaCy dep parse · local corpus"),

            # ── Search inputs (always visible above corpus tabs) ──────────
            html.Div(style={"display": "flex", "gap": "10px", "marginBottom": "8px",
                            "alignItems": "center"}, children=[
                dcc.Input(
                    id="kw-input", type="text",
                    placeholder="Enter a word…",
                    debounce=False, n_submit=0,
                    style={
                        "fontFamily": "DM Mono, monospace", "fontSize": "12px",
                        "flex": "1", "padding": "8px 12px",
                        "border": f"1px solid {C['border']}", "borderRadius": "6px",
                        "backgroundColor": C["bg"], "color": C["text"], "outline": "none",
                    },
                ),
                dbc.Button("Search", id="kw-search-btn", size="sm",
                           style={"backgroundColor": C["corpus"], "border": "none",
                                  "fontFamily": "DM Mono, monospace", "fontSize": "11px"}),
            ]),
            html.Div(style={"display": "flex", "gap": "10px", "marginBottom": "20px",
                            "alignItems": "center"}, children=[
                html.Span("vs.", style={
                    "fontFamily": "DM Mono, monospace", "fontSize": "11px",
                    "color": C["muted"], "whiteSpace": "nowrap",
                }),
                dcc.Input(
                    id="kw-input2", type="text",
                    placeholder="Second keyword for Word Sketch Difference (optional)…",
                    debounce=False, n_submit=0,
                    style={
                        "fontFamily": "DM Mono, monospace", "fontSize": "12px",
                        "flex": "1", "padding": "8px 12px",
                        "border": f"1px solid {C['border']}", "borderRadius": "6px",
                        "backgroundColor": C["bg"], "color": C["text"], "outline": "none",
                    },
                ),
            ]),

            # ── Corpus tabs ───────────────────────────────────────────────
            dcc.Tabs(id="corpus-tabs", value="corpus-transcript", style={
                "marginBottom": "0",
            }, children=[

                # Tab: Transcript ──────────────────────────────────────
                dcc.Tab(label="Transcript", value="corpus-transcript",
                        style=_TAB_BASE, selected_style=_tab_sel(C["corpus"]),
                        children=[html.Div(style={"paddingTop": "20px"}, children=[
                    html.P("FULL TRANSCRIPT", style=LABEL_STYLE_C),
                    html.P("Current scene is highlighted as the video plays.",
                           style={"fontFamily": "DM Mono, monospace", "fontSize": "10px",
                                  "color": C["muted"], "marginTop": "2px",
                                  "marginBottom": "14px"}),
                    html.Div(
                        id="transcript-view",
                        style={
                            "overflowY": "auto",
                            "maxHeight": "560px",
                            "border": f"1px solid {C['border']}",
                            "borderRadius": "6px",
                            "padding": "4px 0",
                        },
                    ),
                ])]),

                # Tab: Concordance ─────────────────────────────────────
                dcc.Tab(label="Concordance", value="corpus-concordance",
                        style=_TAB_BASE, selected_style=_tab_sel(C["corpus"]),
                        children=[html.Div(style={"paddingTop": "20px"}, children=[
                    html.Div(id="kw-stats", style={
                        "fontFamily": "DM Mono, monospace", "fontSize": "11px",
                        "color": C["muted"], "marginBottom": "10px",
                    }),
                    html.Div(id="kw-concordance",
                             style={"marginTop": "16px"}),
                ])]),

                # Tab: Word Sketch & Thesaurus ─────────────────────────
                dcc.Tab(label="Word Sketch & Thesaurus", value="corpus-wordsketch",
                        style=_TAB_BASE, selected_style=_tab_sel(C["corpus"]),
                        children=[html.Div(style={"paddingTop": "20px"}, children=[
                    html.P("WORD SKETCH", style=LABEL_STYLE_C),
                    html.Div(id="kw-sketch-panel",
                             style={"marginTop": "8px", "minHeight": "60px",
                                    "marginBottom": "28px"}),
                    html.Hr(style={"borderColor": C["border"], "margin": "0 0 20px 0"}),
                    html.P("DISTRIBUTIONAL THESAURUS", style=LABEL_STYLE_C),
                    html.Div(id="kw-thesaurus-panel",
                             style={"marginTop": "8px", "minHeight": "60px",
                                    "marginBottom": "28px"}),
                    html.Hr(style={"borderColor": C["border"], "margin": "0 0 20px 0"}),
                    html.Div(id="kw-diff-panel"),
                ])]),

                # Tab: Frequency ───────────────────────────────────────
                dcc.Tab(label="Frequency", value="corpus-frequency",
                        style=_TAB_BASE, selected_style=_tab_sel(C["corpus"]),
                        children=[html.Div(style={"paddingTop": "20px"}, children=[
                    html.Div(style={"display": "flex", "alignItems": "center",
                                    "gap": "16px", "marginBottom": "10px"}, children=[
                        html.P("WORD LIST", style={**LABEL_STYLE, "marginBottom": "0"}),
                        dcc.Dropdown(
                            id="pos-filter",
                            options=[
                                {"label": "All words",          "value": "ALL"},
                                {"label": "Noun (NOUN/PROPN)",  "value": "NOUN"},
                                {"label": "Verb (VERB/AUX)",    "value": "VERB"},
                                {"label": "Adjective (ADJ)",    "value": "ADJ"},
                                {"label": "Adverb (ADV)",       "value": "ADV"},
                                {"label": "Other",              "value": "OTHER"},
                            ],
                            value="ALL",
                            clearable=False,
                            style={"width": "200px",
                                   "fontFamily": "DM Mono, monospace",
                                   "fontSize": "11px"},
                        ),
                    ]),
                    dash_table.DataTable(
                        id="kw-wordlist-table",
                        columns=[
                            {"name": "Word",       "id": "word"},
                            {"name": "POS",        "id": "pos"},
                            {"name": "Count",      "id": "count",         "type": "numeric"},
                            {"name": "Freq/1000",  "id": "freq_per_1000", "type": "numeric"},
                        ],
                        data=[],
                        sort_action="native",
                        filter_action="native",
                        page_action="native",
                        page_size=25,
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": C["surface"], "color": C["muted"],
                            "fontFamily": "DM Mono, monospace", "fontSize": "10px",
                            "fontWeight": "600", "border": f"1px solid {C['border']}",
                            "textTransform": "uppercase", "letterSpacing": "0.05em",
                        },
                        style_cell={
                            "backgroundColor": C["bg"], "color": C["text"],
                            "fontFamily": "DM Mono, monospace", "fontSize": "11px",
                            "border": f"1px solid {C['border']}",
                            "padding": "6px 10px", "textAlign": "left",
                        },
                        style_filter={
                            "backgroundColor": C["bg"], "color": C["text"],
                            "fontFamily": "DM Mono, monospace", "fontSize": "10px",
                        },
                        style_data_conditional=[
                            {
                                "if": {"filter_query": f'{{pos}} = "{_POS_LABELS.get(pos, pos)}"'},
                                "borderLeft": f"3px solid {colour}",
                            }
                            for pos, colour in POS_COLOUR.items()
                        ],
                    ),
                    html.P("N-GRAMS", style={**LABEL_STYLE, "marginTop": "28px"}),
                    dcc.Graph(id="kw-ngrams-chart",
                              style={"height": "380px"}, config=CHART_CFG),
                ])]),
            ]),
        ]),

        # ── GESTURE ──────────────────────────────────────────────────────
        html.Div(style=SECTION_STYLE, children=[
            section_header("Pose Estimation", C["gesture"], "MediaPipe · Kinematic features"),

            html.Div(style={
                "marginBottom": "28px", "paddingBottom": "24px",
                "borderBottom": f"1px solid {C['border']}",
            }, children=[
                html.Div(style={
                    "display": "flex", "alignItems": "flex-end",
                    "justifyContent": "space-between", "flexWrap": "wrap",
                    "gap": "12px", "marginBottom": "14px",
                }, children=[
                    html.Div([
                        html.P("BODY SEGMENT", style=LABEL_STYLE),
                        html.Div([
                            dbc.Button("Head / Face", id="seg-head",  size="sm",
                                       outline=True, color="primary", className="me-2 mt-1"),
                            dbc.Button("Arms",        id="seg-arms",  size="sm",
                                       outline=True, color="warning", className="me-2 mt-1"),
                            dbc.Button("Hands",       id="seg-hands", size="sm",
                                       outline=True, color="danger",  className="me-2 mt-1"),
                            dbc.Button("Torso",       id="seg-torso", size="sm",
                                       outline=True, color="success", className="me-2 mt-1"),
                            dbc.Button("Gaze",        id="seg-gaze",  size="sm",
                                       outline=True, color="pink", className="mt-1"),
                        ], style={"marginTop": "6px", "display": "flex", "flexWrap": "wrap"}),
                    ]),
                    html.Div([
                        html.P("POSE OVERLAY", style=LABEL_STYLE),
                        dbc.Button("Landmarks  ON", id="landmark-toggle", size="sm",
                                   color="success", className="mt-1", disabled=True),
                    ]),
                ]),

                html.Div(style={"position": "relative", "lineHeight": "0"}, children=[
                    html.Video(
                        id="video-player",
                        controls=True,
                        style={
                            "width": "100%", "maxHeight": "520px",
                            "display": "block", "borderRadius": "8px",
                            "backgroundColor": "#000",
                        },
                    ),
                    html.Canvas(id="pose-canvas", style={
                        "position": "absolute", "top": "0", "left": "0",
                        "width": "100%", "height": "100%",
                        "pointerEvents": "none", "borderRadius": "8px",
                    }),
                ]),

                html.Div(style={
                    "display": "flex", "gap": "32px", "alignItems": "flex-end",
                    "marginTop": "12px", "flexWrap": "wrap",
                }, children=[
                    html.Div([
                        html.P("CURRENT TIME", style=LABEL_STYLE),
                        html.P(id="current-time-display", children="—", style={
                            "fontFamily": "Fraunces, serif", "fontSize": "2rem",
                            "color": C["cursor"], "margin": "4px 0 0 0", "lineHeight": "1",
                        }),
                    ]),
                    html.Div(style={"flex": "1", "minWidth": "200px"}, children=[
                        html.P("CURRENT WINDOW", style=LABEL_STYLE),
                        html.P(id="current-window-display", children="—", style={
                            "fontFamily": "DM Mono, monospace", "fontSize": "11px",
                            "color": C["text"], "margin": "4px 0 0 0",
                        }),
                    ]),
                ]),

                dcc.Graph(id="g-handedness", style={"height": "160px", "marginTop": "20px"},
                          config=CHART_CFG),
            ]),

            dcc.Graph(id="g-velocity", style={"height": "200px"}, config=CHART_CFG),
        ]),

        # ── ACOUSTIC ──────────────────────────────────────────────────────
        html.Div(style=SECTION_STYLE, children=[
            section_header("Acoustic Properties", C["prosody"], "Parselmouth · Praat algorithms"),
            dcc.Graph(id="p-spectrogram", style={"height": "280px"}, config=CHART_CFG),
            dcc.Graph(id="p-waveform",   style={"height": "220px", "marginTop": "4px"}, config=CHART_CFG),
            dcc.Graph(id="p-f0",        style={"height": "200px"}, config=CHART_CFG),
            dcc.Graph(id="p-intensity", style={"height": "200px"}, config=CHART_CFG),
        ]),

        # ── CAMERA ────────────────────────────────────────────────────────
        html.Div(style=SECTION_STYLE, children=[
            section_header("Camera", C["camera"], "PySceneDetect · Haar cascade"),
            dcc.Graph(id="c-shot",     style={"height": "200px"}, config=CHART_CFG),
            dcc.Graph(id="c-h-angle",  style={"height": "200px"}, config=CHART_CFG),
            dcc.Graph(id="c-v-angle",  style={"height": "200px"}, config=CHART_CFG),
            dcc.Graph(id="c-cutrate",  style={"height": "200px"}, config=CHART_CFG),
            dcc.Graph(id="c-facearea", style={"height": "200px"}, config=CHART_CFG),
            dcc.Graph(id="c-trend",    style={"height": "200px"}, config=CHART_CFG),
        ]),
        ]),  # /analysis-charts-section
    ]),

    # ── Hidden stores ─────────────────────────────────────────────────────────
    dcc.Store(id="fused-data"),
    dcc.Store(id="current-time", data=0.0),
    dcc.Store(id="active-job-id"),
    dcc.Store(id="data-source", data="redis"),
    dcc.Store(id="browse-collection", data=None),
    dcc.Store(id="bulk-manifest-entries", data=None),
    dcc.Store(id="gallery-manifest-idx", data=None),
    dcc.Store(id="gallery-build-state", data=None),
    dcc.Store(id="gallery-candidates", data=None),
    dcc.Store(id="live-gallery-build-state", data=None),
    dcc.Store(id="live-gallery-candidates", data=None),
    dcc.Store(id="active-drive-url", data=None),
    dcc.Store(id="active-collection", data=None),
    dcc.Store(id="pose-segment", data=[]),
    dcc.Store(id="pose-timeline", data=None),
    dcc.Store(id="pose-render-dummy"),
    dcc.Store(id="landmarks-visible", data=True),
    dcc.Store(id="keyword-occurrences", data=[]),
    dcc.Store(id="active-keyword", data=None),
    dcc.Store(id="second-keyword", data=None),
    dcc.Store(id="search-display-keyword", data=None),
    dcc.Store(id="window-times", data=[]),
    dcc.Store(id="transcript-hl-dummy"),
    dcc.Store(id="seek-to", data=None),
    dcc.Interval(id="poll-interval", interval=2000, n_intervals=0, disabled=True),
])

# ─────────────────────────────────────────────────────────────────────────────
# Clientside: chart click → seek video
# ─────────────────────────────────────────────────────────────────────────────

clientside_callback(
    """
    function(clickDataList) {
        for (var i = 0; i < clickDataList.length; i++) {
            var cd = clickDataList[i];
            if (cd && cd.points && cd.points.length > 0) {
                var t = cd.points[0].x;
                if (t !== undefined && t !== null) {
                    var video = document.getElementById('video-player');
                    if (video && video.src) {
                        video.currentTime = parseFloat(t);
                    }
                    return parseFloat(t);
                }
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("current-time", "data"),
    [Input(cid, "clickData") for cid in CHART_IDS],
)

clientside_callback(
    """
    function(t) {
        if (t === null || t === undefined) return window.dash_clientside.no_update;
        return t.toFixed(1) + 's';
    }
    """,
    Output("current-time-display", "children"),
    Input("current-time", "data"),
)

# seek-to store → seek video + propagate to current-time (for prev/next buttons)
clientside_callback(
    """
    function(t) {
        if (t === null || t === undefined) return window.dash_clientside.no_update;
        var video = document.getElementById('video-player');
        if (video && video.src) { video.currentTime = parseFloat(t); }
        return parseFloat(t);
    }
    """,
    Output("current-time", "data", allow_duplicate=True),
    Input("seek-to", "data"),
    prevent_initial_call=True,
)

# Set up a native browser setInterval when window-times loads.
# Runs entirely outside Dash so it never triggers page scroll.
clientside_callback(
    """
    function(times) {
        if (window._transcriptHlInterval) {
            clearInterval(window._transcriptHlInterval);
            window._transcriptHlInterval = null;
        }
        if (!times || !times.length) return '';
        window._transcriptHlInterval = setInterval(function() {
            var video = document.getElementById('video-player');
            if (!video) return;
            var t = video.currentTime || 0;
            var activeIdx = -1;
            for (var i = 0; i < times.length; i++) {
                if (t >= times[i].start && t < times[i].end) { activeIdx = i; break; }
            }
            var prev = document.querySelector('[data-ts-active="1"]');
            if (prev) {
                var prevIdx = parseInt(prev.getAttribute('data-ts-active-idx'), 10);
                if (prevIdx === activeIdx) return;
                prev.style.backgroundColor = '';
                prev.removeAttribute('data-ts-active');
                prev.removeAttribute('data-ts-active-idx');
            }
            if (activeIdx === -1) return;
            var el = document.getElementById('ts-seg-' + activeIdx);
            if (el) {
                el.style.backgroundColor = 'rgba(67, 97, 238, 0.12)';
                el.setAttribute('data-ts-active', '1');
                el.setAttribute('data-ts-active-idx', activeIdx);
            }
        }, 500);
        return '';
    }
    """,
    Output("transcript-hl-dummy", "data"),
    Input("window-times", "data"),
    prevent_initial_call=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Upload video → save to disk → launch analysis in background thread
# ─────────────────────────────────────────────────────────────────────────────

_UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/tmp/mannerism/uploads"))


def _cleanup_previous_upload() -> None:
    """Delete the previously uploaded video and its extracted-audio cache.

    The dashboard only ever plays back one active video at a time (`_VIDEO_PATH`
    is a single global slot), so by the time a new upload arrives the old one
    is guaranteed to no longer be needed — nothing else in the codebase ever
    cleans these up for the live-analysis (non-bulk) path, so they'd otherwise
    accumulate on disk with every upload.
    """
    old_path = _VIDEO_PATH.get("path")
    if not old_path:
        return
    audio_path = _orch.work_dir / (Path(old_path).stem + "_audio.wav")
    for p in (Path(old_path), audio_path):
        try:
            if p.exists():
                p.unlink()
        except OSError as exc:
            logger.warning(f"[dashboard] Could not delete {p}: {exc}")


@callback(
    Output("active-job-id", "data"),
    Output("poll-interval", "disabled"),
    Output("video-player", "src"),
    Output("analysis-status", "children"),
    Output("landmark-toggle", "disabled"),
    Output("data-source", "data"),
    Output("active-drive-url", "data"),
    Output("active-collection", "data"),
    Output("live-gallery-build-panel", "style"),
    Output("live-gallery-build-status", "children"),
    Output("live-gallery-build-frame", "src"),
    Output("live-gallery-candidate-row", "children"),
    Output("live-gallery-candidates", "data"),
    Output("live-gallery-build-state", "data"),
    Input("video-upload", "contents"),
    State("video-upload", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    no_change = (dash.no_update,) * 14
    if not contents:
        return no_change

    _cleanup_previous_upload()

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _, b64 = contents.split(",", 1)
    video_path = str(_UPLOAD_DIR / filename)
    with open(video_path, "wb") as fh:
        fh.write(base64.b64decode(b64))

    _VIDEO_PATH["path"] = video_path
    video_src = f"/video?t={os.path.getmtime(video_path)}"

    # Analysis itself doesn't start yet — first the speaker gallery gets
    # built interactively (see _setup_live_gallery /
    # handle_live_gallery_interaction), mirroring Bulk Upload's
    # front-loaded gallery phase for exactly one video. GestureWorker
    # picks up whatever gallery ends up in Redis under this same job_id
    # automatically — no orchestrator/worker changes needed for this.
    setup = _setup_live_gallery(video_path)
    job_id = setup["job_id"]

    if setup["failed"]:
        # Couldn't prepare a gallery at all (probe/scene-detect failure)
        # — fall back to analyzing with no gallery, exactly like this
        # upload path always worked before this feature existed.
        # GestureWorker treats "no gallery for this job_id" as "use
        # heuristics", automatically.
        threading.Thread(
            target=_orch.analyze, args=(video_path,),
            kwargs={"job_id": job_id}, daemon=True,
        ).start()
        return (
            job_id, False, video_src, _analysis_progress("running", f"Analysing {filename}…"),
            True, "redis", None, None,
            _gallery_panel_style(False), "", None, [], None, None,
        )

    state, frame_data = setup["state"], setup["frame_data"]
    return (
        job_id, True, video_src,
        _analysis_progress("running", "Confirm the speaker before analysis starts…"),
        True, "redis", None, None,
        _gallery_panel_style(True), _live_gallery_status_text(state),
        _frame_src(frame_data["frame_jpeg_b64"]), _render_candidate_row(frame_data["candidates"], "live-gallery-candidate-btn"),
        _candidates_store_payload(frame_data), dataclasses.asdict(state),
    )


@callback(
    Output("active-job-id", "data", allow_duplicate=True),
    Output("poll-interval", "disabled", allow_duplicate=True),
    Output("analysis-status", "children", allow_duplicate=True),
    Output("live-gallery-build-panel", "style", allow_duplicate=True),
    Output("live-gallery-build-status", "children", allow_duplicate=True),
    Output("live-gallery-build-frame", "src", allow_duplicate=True),
    Output("live-gallery-candidate-row", "children", allow_duplicate=True),
    Output("live-gallery-candidates", "data", allow_duplicate=True),
    Output("live-gallery-build-state", "data", allow_duplicate=True),
    Input({"type": "live-gallery-candidate-btn", "index": ALL}, "n_clicks"),
    Input("live-gallery-skip-btn", "n_clicks"),
    Input("live-gallery-done-btn", "n_clicks"),
    State("live-gallery-build-state", "data"),
    State("live-gallery-candidates", "data"),
    prevent_initial_call=True,
)
def handle_live_gallery_interaction(_candidate_clicks, _skip_clicks, _done_clicks,
                                     state_data, candidates_data):
    ctx = dash.callback_context
    if not ctx.triggered_id or state_data is None:
        return (dash.no_update,) * 9

    triggered = ctx.triggered_id
    state = GalleryBuildState(**state_data)

    # "I'm done — start analysis" — manual override, independent of the
    # plateau/cap algorithmic stop (see wikis/Gesture-Worker.md). A click
    # with zero confirmations so far means an empty Redis gallery for this
    # job_id — GestureWorker's own "no gallery -> heuristics" fallback
    # handles that automatically, so this doubles as "skip the gallery
    # entirely" with no separate button needed for that.
    if triggered == "live-gallery-done-btn":
        return _start_live_analysis(state)

    detector, reid_model = _ensure_gallery_models()

    # "No speaker visible — skip" — draws a fresh candidate without
    # touching the gallery or the plateau streak at all.
    if triggered == "live-gallery-skip-btn":
        frame_data = _draw_until_nonempty(detector, reid_model, state)
        return (
            dash.no_update, dash.no_update, dash.no_update,
            _gallery_panel_style(True), _live_gallery_status_text(state),
            _frame_src(frame_data["frame_jpeg_b64"]), _render_candidate_row(frame_data["candidates"], "live-gallery-candidate-btn"),
            _candidates_store_payload(frame_data), dataclasses.asdict(state),
        )

    # A candidate thumbnail was clicked — triggered is the pattern-matched
    # id dict {"type": "live-gallery-candidate-btn", "index": i}.
    clicked_idx = triggered["index"]
    candidates = (candidates_data or {}).get("candidates") or []
    if clicked_idx >= len(candidates):
        return (dash.no_update,) * 9
    crop_b64 = candidates[clicked_idx]["crop_jpeg_b64"]
    ts = candidates_data["ts"]

    embedding = embed_candidate_from_b64(reid_model, crop_b64)
    record_confirmation(store, state, embedding, ts, crop_b64)

    stop_reason = should_stop(state)
    if stop_reason:
        state.stop_reason = stop_reason
        logger.info(
            f"[dashboard] Live gallery for job {state.job_id} stopped ({stop_reason}) "
            f"with {state.entries_confirmed} entries"
        )
        return _start_live_analysis(state)

    frame_data = _draw_until_nonempty(detector, reid_model, state)
    return (
        dash.no_update, dash.no_update, dash.no_update,
        _gallery_panel_style(True), _live_gallery_status_text(state),
        _frame_src(frame_data["frame_jpeg_b64"]), _render_candidate_row(frame_data["candidates"], "live-gallery-candidate-btn"),
        _candidates_store_payload(frame_data), dataclasses.asdict(state),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bulk Upload — manifest upload, background-threaded BulkOrchestrator run
# ─────────────────────────────────────────────────────────────────────────────

def _run_bulk_manifest(entries: list[dict], force: bool) -> None:
    """Runs in a background thread; publishes progress into _BULK_STATE for
    poll_bulk_progress to read, mirroring how Live Analysis's job status is
    polled via Redis instead of a shared in-memory dict — bulk runs don't
    have a Redis-backed job of their own to poll, so this fills that role."""
    with _BULK_LOCK:
        _BULK_STATE["running"] = True
        _BULK_STATE["last_event"] = None
        _BULK_STATE["summary"] = None

    def _progress_cb(evt: dict) -> None:
        with _BULK_LOCK:
            _BULK_STATE["last_event"] = evt

    try:
        bulk_orch = BulkOrchestrator(
            feature_store=store, repo=repo,
            # No persistent_gesture flag needed on this branch — see
            # core/orchestrator.py; BulkOrchestrator already reuses one
            # Orchestrator instance for the whole manifest, keeping every
            # lazy worker (gesture included, now) warm across every video.
            orchestrator_kwargs={},
        )
        summary = bulk_orch.run(entries, force=force, progress_cb=_progress_cb)
    except Exception as exc:
        logger.error(f"[dashboard] Bulk run failed: {exc}")
        summary = {"succeeded": [], "skipped": [],
                   "failed": [{"path": "(manifest)", "error": str(exc)}]}

    with _BULK_LOCK:
        _BULK_STATE["summary"] = summary
        _BULK_STATE["running"] = False



# ── Speaker gallery confirmation (Bulk Upload only) ─────────────────────
# Front-loaded: every manifest video gets its gallery built interactively
# here, one at a time, before "Start" (the actual, unattended
# BulkOrchestrator.run() pass) becomes available — see
# core/gallery_builder.py and workers/gesture_worker.py's "Speaker
# re-identification" for the design this implements.

_GALLERY_DRAW_RETRIES = 5  # give up on this video's current attempt after
# this many consecutive empty draws (e.g. a long stretch of slides with
# nobody visible) rather than looping forever — the UI just shows nothing
# found and lets the researcher skip or mark the video done manually.


def _gallery_panel_style(visible: bool) -> dict:
    return {
        "display": "block" if visible else "none", "marginBottom": "16px",
        "border": f"1px solid {C['border']}", "borderRadius": "8px",
        "padding": "14px", "backgroundColor": C["bg"],
    }


def _frame_src(frame_jpeg_b64: Optional[str]) -> Optional[str]:
    """Data-URI-prefixes a frame's base64 JPEG for use as an <img src> —
    unlike _render_candidate_row's thumbnails, the raw base64 string was
    being handed to gallery-build-frame's src directly with no `data:
    image/jpeg;base64,` prefix, which a browser can't interpret as an
    image at all (renders as a broken-image icon)."""
    return f"data:image/jpeg;base64,{frame_jpeg_b64}" if frame_jpeg_b64 else None


def _candidates_store_payload(frame_data: dict) -> dict:
    """What goes into the gallery-candidates Store — bundles the drawn
    frame's timestamp alongside its candidate crops, since the click
    handler needs both (the timestamp to record the confirmation against)
    and Dash Stores don't share state across callback invocations any
    other way."""
    return {"ts": frame_data["ts"], "candidates": frame_data["candidates"]}


def _render_candidate_row(candidates: list[dict], btn_type: str = "gallery-candidate-btn"):
    """btn_type must match whichever callback's pattern-matched Input is
    meant to receive these clicks — bulk's handle_gallery_interaction
    listens for "gallery-candidate-btn", Live Analysis's
    handle_live_gallery_interaction for "live-gallery-candidate-btn". Two
    different flows, two different id types, so a click in one panel
    can't accidentally fire the other's callback."""
    return [
        html.Button(
            html.Img(src=f"data:image/jpeg;base64,{c['crop_jpeg_b64']}", style={
                "height": "140px", "borderRadius": "4px", "display": "block",
            }),
            id={"type": btn_type, "index": i},
            n_clicks=0,
            style={
                "padding": "0", "border": f"2px solid {C['border']}",
                "borderRadius": "6px", "backgroundColor": "transparent", "cursor": "pointer",
            },
        )
        for i, c in enumerate(candidates)
    ]


def _draw_until_nonempty(detector, reid_model, state: GalleryBuildState) -> dict:
    """Keeps advancing to successive *schedule slots* (see
    core/gallery_builder.py's draw_next_candidate — each call there
    already retries several different timestamps within one slot before
    giving up on it) until one produces at least one detected candidate,
    or gives up after _GALLERY_DRAW_RETRIES consecutive empty slots.
    state already carries video_path/fps, so nothing else needs passing."""
    result = {"frame_jpeg_b64": None, "candidates": [], "ts": 0.0}
    for _ in range(_GALLERY_DRAW_RETRIES):
        result = draw_next_candidate(reid_model, detector, state)
        if result["candidates"]:
            return result
    return result


def _setup_gallery_for_entry(entries: list[dict], idx: int) -> Optional[dict]:
    """
    Prepares interactive gallery-building for entries[idx]: resolves and
    (if only a drive_url is given) downloads the video, probes it, runs
    scene-cut detection, mints a job_id, and draws the first candidate
    frame. Returns None once idx has run past the end of entries — the
    caller takes that as "every gallery is done, reveal Start".

    Mutates entries[idx] in place to attach the minted job_id — the exact
    id BulkOrchestrator later reuses via its own preassigned-job_id
    handling (see core/bulk_orchestrator.py's _analyze_with_progress), so
    GestureWorker finds this video's gallery in Redis when it actually
    processes this entry.

    A video that can't be prepared at all (missing with no drive_url,
    download failure, probe failure) is skipped straight to the next
    entry rather than blocking the whole front-loaded phase — the actual
    bulk run will hit and report the same failure for it later.
    """
    if idx >= len(entries):
        return None

    entry = entries[idx]
    path = _resolve_path(entry, _bulk_work_dir())

    if not os.path.exists(path):
        drive_url = entry.get("drive_url")
        if not drive_url:
            logger.warning(f"[dashboard] Gallery phase: {path} not found, no drive_url — skipping")
            return _setup_gallery_for_entry(entries, idx + 1)
        try:
            download_drive_file(drive_url, path)
        except Exception as exc:
            logger.error(f"[dashboard] Gallery phase: download failed for {drive_url}: {exc}")
            return _setup_gallery_for_entry(entries, idx + 1)

    try:
        meta = probe_video(path, audio_path="")
        cuts = GestureWorker._detect_scene_cuts(path)
    except Exception as exc:
        logger.error(f"[dashboard] Gallery phase: could not prepare {path}: {exc}")
        return _setup_gallery_for_entry(entries, idx + 1)

    job_id = str(_uuid_module.uuid4())[:8]
    entry["job_id"] = job_id
    state = init_state(job_id, path, cuts, meta.duration_s, meta.fps)

    detector, reid_model = _ensure_gallery_models()
    frame_data = _draw_until_nonempty(detector, reid_model, state)

    label = entry.get("label") or os.path.basename(path)
    return {"entries": entries, "idx": idx, "state": state, "frame_data": frame_data, "label": label}


def _gallery_status_text(setup: dict) -> str:
    state: GalleryBuildState = setup["state"]
    n = len(setup["entries"])
    return (
        f'Video {setup["idx"] + 1} of {n}: {setup["label"]} — '
        f'gallery: {state.entries_confirmed} confirmed · streak {state.streak}/{PLATEAU_STREAK}'
    )


# ── Speaker gallery confirmation (Live Analysis) ────────────────────────
# Same underlying mechanics as the Bulk Upload flow above (this reuses
# _draw_until_nonempty, _render_candidate_row, _frame_src,
# _candidates_store_payload, _ensure_gallery_models, _gallery_panel_style,
# and core/gallery_builder.py's init_state/draw_next_candidate/
# record_confirmation/should_stop directly, unmodified) but for exactly
# one video with no manifest/"next entry" concept — see
# handle_upload/handle_live_gallery_interaction below. Kept as separate
# functions rather than reusing _setup_gallery_for_entry/_advance_gallery,
# since those are shaped around a list-of-manifest-entries abstraction
# (collection, drive_url, "move to the next one") that doesn't apply here.

def _live_gallery_status_text(state: GalleryBuildState) -> str:
    return f'Gallery: {state.entries_confirmed} confirmed · streak {state.streak}/{PLATEAU_STREAK}'


def _setup_live_gallery(video_path: str) -> dict:
    """Prepares interactive gallery-building for a freshly-uploaded Live
    Analysis video. Always returns a job_id (minted here, reused by the
    actual analysis run once the gallery session ends — see
    _start_live_analysis) — if scene-cut detection or probing itself
    fails, `failed=True` and the caller falls back to analyzing with no
    gallery at all (today's original behaviour, before this feature
    existed), rather than getting stuck with nothing to show."""
    job_id = str(_uuid_module.uuid4())[:8]
    try:
        meta = probe_video(video_path, audio_path="")
        cuts = GestureWorker._detect_scene_cuts(video_path)
    except Exception as exc:
        logger.error(f"[dashboard] Live gallery: could not prepare {video_path}: {exc}")
        return {"job_id": job_id, "failed": True}

    state = init_state(job_id, video_path, cuts, meta.duration_s, meta.fps)
    detector, reid_model = _ensure_gallery_models()
    frame_data = _draw_until_nonempty(detector, reid_model, state)
    return {"job_id": job_id, "failed": False, "state": state, "frame_data": frame_data}


def _start_live_analysis(state: GalleryBuildState):
    """Ends the gallery-confirmation phase and kicks off the real
    analysis run under the same job_id the gallery was built with — see
    handle_live_gallery_interaction, whose Output order this 9-tuple
    matches exactly: (active-job-id, poll-interval.disabled,
    analysis-status, live-gallery-build-panel.style,
    live-gallery-build-status, live-gallery-build-frame.src,
    live-gallery-candidate-row.children, live-gallery-candidates.data,
    live-gallery-build-state.data)."""
    threading.Thread(
        target=_orch.analyze, args=(state.video_path,),
        kwargs={"job_id": state.job_id}, daemon=True,
    ).start()
    return (
        state.job_id, False, _analysis_progress("running", "Analysing…"),
        _gallery_panel_style(False), "", None, [], None, None,
    )


def _advance_gallery(entries: list[dict], next_idx: int):
    """
    Moves on to the next manifest entry needing a gallery, or — once
    every entry has been through this — reveals "Start" and hides the
    gallery panel. Returns a 9-tuple in exactly the Output order
    handle_gallery_interaction declares below — (entries, bulk-start-btn
    disabled, gallery-manifest-idx, gallery-build-state, gallery-build
    -panel style, gallery-build-status, gallery-build-frame src,
    gallery-candidate-row children, gallery-candidates data) — since
    several of that callback's branches return this tuple directly rather
    than through handle_manifest_upload's own reordering/summary-insertion
    step, this order has to be the single source of truth for both.
    """
    setup = _setup_gallery_for_entry(entries, next_idx)
    if setup is None:
        # Every video's gallery is built (or skipped) — the actual bulk
        # run can begin.
        return (
            entries, False, None, None,
            _gallery_panel_style(False), "", None, [], None,
        )

    return (
        setup["entries"], True, setup["idx"], dataclasses.asdict(setup["state"]),
        _gallery_panel_style(True), _gallery_status_text(setup),
        _frame_src(setup["frame_data"]["frame_jpeg_b64"]),
        _render_candidate_row(setup["frame_data"]["candidates"]),
        _candidates_store_payload(setup["frame_data"]),
    )


@callback(
    Output("bulk-manifest-entries", "data"),
    Output("bulk-manifest-summary", "children"),
    Output("bulk-start-btn", "disabled"),
    Output("gallery-manifest-idx", "data"),
    Output("gallery-build-state", "data"),
    Output("gallery-build-panel", "style"),
    Output("gallery-build-status", "children"),
    Output("gallery-build-frame", "src"),
    Output("gallery-candidate-row", "children"),
    Output("gallery-candidates", "data"),
    Input("manifest-upload", "contents"),
    State("manifest-upload", "filename"),
    prevent_initial_call=True,
)
def handle_manifest_upload(contents, filename):
    no_change = (dash.no_update,) * 10
    if not contents:
        return no_change

    _, b64 = contents.split(",", 1)
    suffix = Path(filename).suffix.lower() or ".yml"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as fh:
            fh.write(base64.b64decode(b64))
            tmp_path = fh.name
        entries = load_manifest(tmp_path)
    except Exception as exc:
        return (
            None, f'Could not parse "{filename}": {exc}', True,
            None, None, _gallery_panel_style(False), "", None, [], None,
        )
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    collections = sorted({e["collection"] for e in entries})
    summary = f'Loaded {len(entries)} entries from "{filename}" — collections: {", ".join(collections)}'

    (
        entries, start_disabled, gallery_idx, gallery_state, panel_style,
        status_text, frame_src, candidate_row, candidates_data,
    ) = _advance_gallery(entries, 0)

    return (
        entries, summary, start_disabled,
        gallery_idx, gallery_state, panel_style, status_text,
        frame_src, candidate_row, candidates_data,
    )


@callback(
    Output("bulk-manifest-entries", "data", allow_duplicate=True),
    Output("bulk-start-btn", "disabled", allow_duplicate=True),
    Output("gallery-manifest-idx", "data", allow_duplicate=True),
    Output("gallery-build-state", "data", allow_duplicate=True),
    Output("gallery-build-panel", "style", allow_duplicate=True),
    Output("gallery-build-status", "children", allow_duplicate=True),
    Output("gallery-build-frame", "src", allow_duplicate=True),
    Output("gallery-candidate-row", "children", allow_duplicate=True),
    Output("gallery-candidates", "data", allow_duplicate=True),
    Input({"type": "gallery-candidate-btn", "index": ALL}, "n_clicks"),
    Input("gallery-skip-btn", "n_clicks"),
    Input("gallery-done-btn", "n_clicks"),
    State("bulk-manifest-entries", "data"),
    State("gallery-manifest-idx", "data"),
    State("gallery-build-state", "data"),
    State("gallery-candidates", "data"),
    prevent_initial_call=True,
)
def handle_gallery_interaction(_candidate_clicks, _skip_clicks, _done_clicks,
                                entries, idx, state_data, candidates_data):
    ctx = dash.callback_context
    if not ctx.triggered_id or entries is None or idx is None or state_data is None:
        return (dash.no_update,) * 9

    triggered = ctx.triggered_id

    # "I'm done with this video" — manual override, independent of the
    # plateau/cap algorithmic stop (see wikis/Gesture-Worker.md).
    if triggered == "gallery-done-btn":
        result = _advance_gallery(entries, idx + 1)
        return result

    state = GalleryBuildState(**state_data)
    detector, reid_model = _ensure_gallery_models()

    # "No speaker visible — skip" — draws a fresh candidate for the same
    # video without touching the gallery or the plateau streak at all.
    if triggered == "gallery-skip-btn":
        frame_data = _draw_until_nonempty(detector, reid_model, state)
        setup = {"entries": entries, "idx": idx, "state": state, "frame_data": frame_data,
                 "label": entries[idx].get("label") or os.path.basename(path)}
        return (
            entries, True, idx, dataclasses.asdict(state),
            _gallery_panel_style(True), _gallery_status_text(setup),
            _frame_src(frame_data["frame_jpeg_b64"]), _render_candidate_row(frame_data["candidates"]),
            _candidates_store_payload(frame_data),
        )

    # A candidate thumbnail was clicked — triggered is the pattern-matched
    # id dict {"type": "gallery-candidate-btn", "index": i}. candidates_data
    # bundles the timestamp that frame was drawn at alongside its crops
    # (see _candidates_store_payload) — needed to record the confirmation
    # against the right moment in the video.
    clicked_idx = triggered["index"]
    candidates = (candidates_data or {}).get("candidates") or []
    if clicked_idx >= len(candidates):
        return (dash.no_update,) * 9
    crop_b64 = candidates[clicked_idx]["crop_jpeg_b64"]
    ts = candidates_data["ts"]

    embedding = embed_candidate_from_b64(reid_model, crop_b64)
    record_confirmation(store, state, embedding, ts, crop_b64)

    stop_reason = should_stop(state)
    if stop_reason:
        state.stop_reason = stop_reason
        logger.info(
            f"[dashboard] Gallery for job {state.job_id} stopped ({stop_reason}) "
            f"with {state.entries_confirmed} entries"
        )
        return _advance_gallery(entries, idx + 1)

    frame_data = _draw_until_nonempty(detector, reid_model, state)
    setup = {"entries": entries, "idx": idx, "state": state, "frame_data": frame_data,
             "label": entries[idx].get("label") or os.path.basename(path)}
    return (
        entries, True, idx, dataclasses.asdict(state),
        _gallery_panel_style(True), _gallery_status_text(setup),
        _frame_src(frame_data["frame_jpeg_b64"]), _render_candidate_row(frame_data["candidates"]),
        _candidates_store_payload(frame_data),
    )


@callback(
    Output("bulk-poll-interval", "disabled"),
    Output("bulk-start-btn", "disabled", allow_duplicate=True),
    Input("bulk-start-btn", "n_clicks"),
    State("bulk-manifest-entries", "data"),
    State("bulk-force", "value"),
    prevent_initial_call=True,
)
def start_bulk_run(_, entries, force):
    if not entries or _BULK_STATE["running"]:
        return dash.no_update, dash.no_update
    if repo is None:
        return dash.no_update, dash.no_update

    threading.Thread(
        target=_run_bulk_manifest,
        args=(entries, bool(force)),
        daemon=True,
    ).start()
    return False, True


@callback(
    Output("bulk-status", "children"),
    Output("bulk-poll-interval", "disabled", allow_duplicate=True),
    Output("bulk-start-btn", "disabled", allow_duplicate=True),
    Input("bulk-poll-interval", "n_intervals"),
    prevent_initial_call=True,
)
def poll_bulk_progress(_):
    mono = {"fontFamily": "DM Mono, monospace", "fontSize": "11px"}
    with _BULK_LOCK:
        running = _BULK_STATE["running"]
        evt = _BULK_STATE["last_event"]
        summary = _BULK_STATE["summary"]

    if running:
        if not evt:
            return _analysis_progress("running", "Starting…"), False, True

        kind = {"download": "downloading", "start": "analysing",
                "tick": "analysing", "skip": "skipped", "done": "done"}.get(evt["type"], evt["type"])
        message = f'[{evt["index"]}/{evt["total"]}] {evt.get("label") or evt["path"]} — {kind}'

        # Same per-worker progress bars as Live Analysis — "tick" events
        # already carry the same total_windows/counts shape poll_analysis
        # builds worker_progress from.
        worker_progress = None
        if evt["type"] == "tick" and evt.get("total_windows"):
            worker_progress = {"total": evt["total_windows"], **(evt.get("counts") or {})}

        return _analysis_progress("running", message, worker_progress), False, True

    if summary is None:
        return dash.no_update, True, False

    children = [
        html.P([
            html.Span(f"✓ Succeeded: {len(summary['succeeded'])}   ",
                      style={**mono, "color": C['prosody']}),
            html.Span(f"− Skipped: {len(summary['skipped'])}   ",
                      style={**mono, "color": C['muted']}),
            html.Span(f"✗ Failed: {len(summary['failed'])}",
                      style={**mono, "color": C['gesture']}),
        ]),
    ]
    for failure in summary["failed"]:
        children.append(html.P(f"{failure['path']}: {failure['error']}",
                                style={**mono, "color": C["gesture"], "fontSize": "10px"}))
    return html.Div(children), True, False


# ─────────────────────────────────────────────────────────────────────────────
# Browse Corpus — flat list of videos already shipped to MongoDB
# ─────────────────────────────────────────────────────────────────────────────

def _format_duration(seconds) -> str:
    if not seconds:
        return "—"
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _format_timestamp(ts) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _build_video_info(collection: str | None, job_id: str | None, videos: list[dict] | None):
    if repo is None:
        return html.P(
            "MongoDB is not configured (set MONGO_URI) — Browse Corpus is unavailable.",
            style={"fontFamily": "DM Mono, monospace", "fontSize": "11px", "color": C["muted"]},
        )
    if not collection:
        return html.P(
            "Select a collection above to see its videos.",
            style={"fontFamily": "DM Mono, monospace", "fontSize": "11px", "color": C["muted"]},
        )
    if videos is not None and not videos:
        display_name = _COLLECTION_DISPLAY_NAMES.get(collection, collection)
        return html.P(
            f'No videos have been shipped to "{display_name}" yet — run `analyze bulk`.',
            style={"fontFamily": "DM Mono, monospace", "fontSize": "11px", "color": C["muted"]},
        )
    if not job_id:
        return html.P(
            "Select a video above to load its analysis.",
            style={"fontFamily": "DM Mono, monospace", "fontSize": "11px", "color": C["muted"]},
        )
    video = next((v for v in (videos or []) if v["_id"] == job_id), None)
    if not video:
        return html.Div()

    mono_sm = "DM Mono, monospace"
    fields = [
        ("Filename",  video.get("video_filename", "—")),
        ("Duration",  _format_duration(video.get("duration_s"))),
        ("Processed", _format_timestamp(video.get("completed_at") or video.get("shipped_at"))),
    ]
    spans = []
    for i, (name, value) in enumerate(fields):
        prefix = "" if i == 0 else "  ·  "
        spans.append(html.Span(f"{prefix}{name}: ",
                                style={"fontFamily": mono_sm, "fontSize": "11px", "color": C["muted"]}))
        spans.append(html.Span(value,
                                style={"fontFamily": mono_sm, "fontSize": "11px", "color": C["text"]}))
    return html.Div(spans)


@callback(
    Output("live-panel", "style"),
    Output("browse-panel", "style"),
    Output("bulk-panel", "style"),
    Output("kpi-strip", "style"),
    Output("analysis-charts-section", "style"),
    Input("mode-tabs", "value"),
)
def toggle_mode(mode):
    live_style = {"display": "block" if mode == "live" else "none"}
    browse_style = {"display": "block" if mode == "browse" else "none"}
    bulk_style = {"display": "block" if mode == "bulk" else "none"}
    # Per-video windowed analysis (KPIs + all the modality charts) is
    # irrelevant while bulk-processing many videos at once.
    charts_display = "none" if mode == "bulk" else "block"
    kpi_style = {
        "display": "flex" if mode != "bulk" else "none",
        "gap": "14px", "flexWrap": "wrap", "marginBottom": "28px",
    }
    charts_style = {"display": charts_display}
    return live_style, browse_style, bulk_style, kpi_style, charts_style


# Display-only labels for Browse Corpus collection buttons — the underlying
# collection name (manifest `collection:`, Mongo collection prefix, dedupe
# scope) is unchanged; only what's shown on the button differs. Collection
# names can't contain spaces (see _validate_collection in
# core/results_repository.py), so renames that need one go here rather than
# in the stored data.
_COLLECTION_DISPLAY_NAMES = {
    "TedX": "Ted Talks",
    "YiXi": "YiXi"
}


@callback(
    Output("browse-collection-buttons", "children"),
    Output("browse-collection", "data"),
    Input("mode-tabs", "value"),
    State("browse-collection", "data"),
)
def populate_browse_collections(mode, current_value):
    if mode != "browse" or repo is None:
        return dash.no_update, dash.no_update
    collections = repo.list_collections()
    value = current_value if current_value in collections else (collections[0] if collections else None)
    buttons = [
        dbc.Button(
            _COLLECTION_DISPLAY_NAMES.get(name, name),
            id={"type": "collection-btn", "index": name},
            size="sm", color="primary" if name == value else "secondary",
            outline=(name != value),
        )
        for name in collections
    ]
    return buttons, value


@callback(
    Output("browse-collection", "data", allow_duplicate=True),
    Input({"type": "collection-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_collection_button(_):
    ctx = dash.callback_context
    if not ctx.triggered_id:
        return dash.no_update
    return ctx.triggered_id["index"]


@callback(
    Output({"type": "collection-btn", "index": ALL}, "color"),
    Output({"type": "collection-btn", "index": ALL}, "outline"),
    Input("browse-collection", "data"),
    State({"type": "collection-btn", "index": ALL}, "id"),
)
def restyle_collection_buttons(selected, ids):
    if not ids:
        return [], []
    colors   = ["primary" if i["index"] == selected else "secondary" for i in ids]
    outlines = [i["index"] != selected for i in ids]
    return colors, outlines


@callback(
    Output("browse-video-dropdown", "options"),
    Output("browse-video-dropdown", "value"),
    Input("browse-collection", "data"),
)
def populate_browse_videos(collection):
    if repo is None or not collection:
        return [], None
    videos = repo.list_videos(collection)
    options = [
        {"label": v.get("label") or v.get("video_filename", v["_id"]), "value": v["_id"]}
        for v in videos
    ]
    return options, None


@callback(
    Output("browse-video-info", "children"),
    Input("browse-video-dropdown", "value"),
    State("browse-collection", "data"),
)
def show_browse_video_info(job_id, collection):
    if repo is None or not collection:
        return _build_video_info(collection, job_id, None)
    videos = repo.list_videos(collection)
    return _build_video_info(collection, job_id, videos)


@callback(
    Output("active-job-id", "data", allow_duplicate=True),
    Output("data-source", "data", allow_duplicate=True),
    Output("active-drive-url", "data", allow_duplicate=True),
    Output("active-collection", "data", allow_duplicate=True),
    Output("fused-data", "data", allow_duplicate=True),
    Output("analysis-status", "children", allow_duplicate=True),
    Output("landmark-toggle", "disabled", allow_duplicate=True),
    Output("poll-interval", "disabled", allow_duplicate=True),
    Output("video-player", "src", allow_duplicate=True),
    Input("browse-video-dropdown", "value"),
    State("browse-collection", "data"),
    prevent_initial_call=True,
)
def handle_browse_select(job_id, collection):
    if not job_id or repo is None or not collection:
        return (dash.no_update,) * 9

    videos = repo.list_videos(collection)
    row = next((v for v in videos if v["_id"] == job_id), None)
    drive_url = (row or {}).get("drive_url") or None

    windows = repo.get_all_fused(collection, job_id)
    serialised = [w.model_dump(mode="json") for w in windows]
    status_msg = _analysis_progress("done", f"Loaded from corpus — {len(windows)} windows")
    # video-player is served same-origin (via /video's Drive-download-and-cache
    # path) rather than a cross-origin Drive iframe, specifically so the pose
    # overlay's clientside draw loop can read currentTime/videoWidth — an
    # iframe from a different origin can never expose those to this page's JS.
    video_src = f"/video?job_id={job_id}&collection={collection}" if drive_url else dash.no_update
    return job_id, "mongo", drive_url, collection, serialised, status_msg, False, True, video_src


# ─────────────────────────────────────────────────────────────────────────────
# Poll analysis status every 2 s; load results when done
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("fused-data", "data", allow_duplicate=True),
    Output("analysis-status", "children", allow_duplicate=True),
    Output("poll-interval", "disabled", allow_duplicate=True),
    Output("landmark-toggle", "disabled", allow_duplicate=True),
    Input("poll-interval", "n_intervals"),
    State("active-job-id", "data"),
    prevent_initial_call=True,
)
def poll_analysis(_, job_id):
    if not job_id:
        return dash.no_update, dash.no_update, True, dash.no_update

    status = store.get_status(job_id)
    if status is None:
        return dash.no_update, dash.no_update, True, dash.no_update

    if status.value == "done":
        windows = store.get_all_fused(job_id)
        serialised = [w.model_dump(mode="json") for w in windows]
        return (
            serialised,
            _analysis_progress("done", f"Analysis complete — {len(windows)} windows"),
            True,
            False,  # enable landmark toggle
        )

    if status.value == "failed":
        return (
            dash.no_update,
            _analysis_progress("failed", "Analysis failed — check server logs"),
            True,
            dash.no_update,
        )

    # Still running — show per-worker progress bars
    job = store.get_job(job_id)
    total = job.total_windows if job and job.total_windows else 0
    worker_progress = None
    if total:
        worker_progress = {
            "total":   total,
            "gesture": store.count_windows(job_id, "gesture"),
            "prosody": store.count_windows(job_id, "prosody"),
            "verbal":  store.count_windows(job_id, "verbal"),
            "camera":  store.count_windows(job_id, "camera"),
        }

    return (
        dash.no_update,
        _analysis_progress("running", "Processing…", worker_progress),
        False,
        dash.no_update,
    )


# ─────────────────────────────────────────────────────────────────────────────
# KPI strip
# ─────────────────────────────────────────────────────────────────────────────

@callback(Output("kpi-strip", "children"), Input("fused-data", "data"))
def render_kpis(data):
    if not data:
        return []
    ws, _ = _parse(data)
    if not ws:
        return [html.Span("Failed to parse windows — check terminal for errors",
                          style={"fontFamily": "DM Mono, monospace", "fontSize": "11px", "color": C["gesture"]})]
    duration    = ws[-1].window.end_s if ws else 0
    total_words = sum(w.verbal.word_count for w in ws if w.verbal)
    mean_f0     = _mean([w.prosody.mean_f0 for w in ws if w.prosody and w.prosody.mean_f0])
    mean_vel    = _mean([w.gesture.mean_wrist_velocity for w in ws if w.gesture])
    total_cuts  = sum(w.camera.cut_count for w in ws if w.camera)
    return [
        kpi_card("Duration",   f"{duration:.0f}s",                      C["muted"]),
        kpi_card("Words",      str(total_words),                         C["verbal"]),
        kpi_card("Mean F0",    f"{mean_f0:.0f} Hz" if mean_f0 else "—", C["prosody"]),
        kpi_card("Wrist Vel.", f"{mean_vel:.0f} px/s" if mean_vel else "—", C["gesture"]),
        kpi_card("Cuts",       str(total_cuts),                          C["camera"]),
    ]

# ─────────────────────────────────────────────────────────────────────────────
# Current window display
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("current-window-display", "children"),
    Input("current-time", "data"),
    State("fused-data", "data"),
)
def update_window_display(t, data):
    if not data or t is None:
        return "—"
    ws, _ = _parse(data)
    for w in ws:
        if w.window.start_s <= t < w.window.end_s:
            transcript = w.verbal.transcript[:80] + "…" if w.verbal and len(w.verbal.transcript) > 80 else (w.verbal.transcript if w.verbal else "")
            return f"[{w.window.start_s:.0f}s – {w.window.end_s:.0f}s]  {transcript}"
    return f"{t:.1f}s"

# ─────────────────────────────────────────────────────────────────────────────
# Chart callbacks — Gesture
# ─────────────────────────────────────────────────────────────────────────────

@callback(Output("g-velocity", "figure"), Input("fused-data", "data"), Input("current-time", "data"), Input("keyword-occurrences", "data"))
def g_velocity(data, ct, occ):
    if not data: return empty_fig("Wrist Velocity")
    ws, t = _parse(data)
    fig = go.Figure(go.Scatter(
        x=t, y=[w.gesture.mean_wrist_velocity if w.gesture else None for w in ws],
        mode="lines", line=dict(color=C["gesture"], width=2),
        fill="tozeroy", fillcolor="rgba(200,75,49,0.08)", name="px/s",
    ))
    _style(fig, "Wrist Velocity", "px/s")
    return add_cursor(fig, ct, occ)



# ─────────────────────────────────────────────────────────────────────────────
# Pose viewer callbacks
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("pose-segment", "data"),
    Output("seg-head",  "outline"),
    Output("seg-arms",  "outline"),
    Output("seg-hands", "outline"),
    Output("seg-torso", "outline"),
    Output("seg-gaze",  "outline"),
    Input("seg-head",  "n_clicks"),
    Input("seg-arms",  "n_clicks"),
    Input("seg-hands", "n_clicks"),
    Input("seg-torso", "n_clicks"),
    Input("seg-gaze",  "n_clicks"),
    State("pose-segment", "data"),
    prevent_initial_call=True,
)
def select_segment(_h, _a, _ha, _t, _g, current_segs):
    ctx = dash.callback_context
    if not ctx.triggered:
        return [dash.no_update] * 6
    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
    seg = {"seg-head": "head", "seg-arms": "arms",
           "seg-hands": "hands", "seg-torso": "torso",
           "seg-gaze": "gaze"}.get(btn_id)
    active = list(current_segs or [])
    if seg in active:
        active.remove(seg)
    else:
        active.append(seg)
    outlines = {s: (s not in active) for s in ["head", "arms", "hands", "torso", "gaze"]}
    return (active,
            outlines["head"], outlines["arms"],
            outlines["hands"], outlines["torso"], outlines["gaze"])


@callback(
    Output("pose-timeline", "data"),
    Input("fused-data", "data"),
)
def build_pose_timeline(data):
    """Flatten all per-frame pose keyframes into one sorted timeline for clientside animation."""
    if not data:
        return None
    ws, _ = _parse(data)
    frames = []
    for w in ws:
        if not w.gesture:
            continue
        for kf in w.gesture.pose_keyframes:
            frames.append({"ts": kf.ts, "px": kf.pose_x, "py": kf.pose_y, "pv": kf.pose_vis})
    frames.sort(key=lambda f: f["ts"])
    return {"frames": frames} if frames else None


clientside_callback(
    """
    function(timeline, segment, visible) {
        // Sync Dash store values into a plain JS object so the RAF loop can
        // read them without going through the Dash callback system.
        if (!window._poseState) {
            window._poseState = { timeline: null, segment: [], visible: true };
        }
        window._poseState.timeline = timeline || null;
        window._poseState.segment  = Array.isArray(segment) ? segment : [];
        window._poseState.visible  = (visible !== false);

        // Bootstrap the RAF draw loop exactly once.
        if (window._poseRafRunning) return window.dash_clientside.no_update;
        window._poseRafRunning = true;

        // MediaPipe BlazePose 33-point topology (see
        // workers/gesture_worker.py's module docstring) — NOT MeTRAbs's old
        // coco_19 (19 points, different indices entirely; this file used to
        // be hardcoded for that and silently drew the wrong edges/highlights
        // once GestureFrame.pose switched to BlazePose's indices):
        // 0 nose, 1-3 left eye (inner/center/outer), 4-6 right eye
        // (inner/center/outer), 7 left ear, 8 right ear, 9-10 mouth
        // (left/right), 11/12 shoulders, 13/14 elbows, 15/16 wrists,
        // 17/18 pinky tips, 19/20 index tips, 21/22 thumb tips, 23/24 hips,
        // 25/26 knees, 27/28 ankles, 29/30 heels, 31/32 foot index.
        // "Left"/"right" are the subject's own, mirrored from the viewer's
        // perspective when facing the camera.
        //
        // These are the *pose* model's own approximate fingertip points
        // (17-22) — GestureFrame.left_hand/right_hand exist on the model
        // but are always empty (HandLandmarker was tried and removed
        // again; see gesture_worker.py's module docstring), so this pose
        // skeleton is the only hand-adjacent data there is to draw.
        // Each segment is a *list of groups* (each group its own list of
        // landmark indices), not a flat list — an edge only lights up if
        // both endpoints fall in the *same* group, not just the same
        // segment (see the lit check below). "arms" needs two groups
        // (left/right) specifically so the [11,12] shoulder-shoulder edge
        // — anatomically part of the torso, not an arm bone, but present
        // in official POSE_CONNECTIONS below — doesn't light up under the
        // "arms" highlight just because both its endpoints happen to be
        // *some* arm landmark. Every other segment only needs one group.
        var SEG_LMS = {
            head:  [[0,1,2,3,4,5,6,7,8,9,10]],
            arms:  [[11,13,15], [12,14,16]],
            hands: [[15,16,17,18,19,20,21,22]],
            torso: [[11,12,23,24]],
            gaze:  [[1,2,3,4,5,6]]
        };
        var SEG_COLORS = {
            head:  '#4A90D9', arms:  '#F0A500',
            hands: '#C84B31', torso: '#28a745', gaze: '#D63384'
        };
        var CONNECTIONS = [
            [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],
            [11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],
            [12,14],[14,16],[16,18],[16,20],[16,22],[18,20],
            [11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],
            [27,29],[28,30],[29,31],[30,32],[27,31],[28,32]
        ];

        function drawLoop() {
            requestAnimationFrame(drawLoop);

            var state   = window._poseState;
            var canvas  = document.getElementById('pose-canvas');
            var video   = document.getElementById('video-player');
            if (!canvas || !video) return;

            var el_w = video.offsetWidth, el_h = video.offsetHeight;
            if (!el_w || !el_h) return;

            // object-fit:contain frame rect
            var vid_w = video.videoWidth, vid_h = video.videoHeight;
            var frame_w, frame_h, frame_x, frame_y;
            if (vid_w && vid_h) {
                var scale = Math.min(el_w / vid_w, el_h / vid_h);
                frame_w = Math.round(vid_w * scale);
                frame_h = Math.round(vid_h * scale);
                frame_x = Math.round((el_w - frame_w) / 2);
                frame_y = Math.round((el_h - frame_h) / 2);
            } else {
                frame_w = el_w; frame_h = el_h; frame_x = 0; frame_y = 0;
            }

            if (canvas.width !== frame_w || canvas.height !== frame_h) {
                canvas.width = frame_w; canvas.height = frame_h;
            }
            canvas.style.left   = frame_x + 'px';
            canvas.style.top    = frame_y + 'px';
            canvas.style.width  = frame_w + 'px';
            canvas.style.height = frame_h + 'px';

            var ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, frame_w, frame_h);

            var timeline = state.timeline;
            if (!state.visible || !timeline || !timeline.frames || !timeline.frames.length) return;

            // Binary-search for the most recent keyframe at-or-before
            // video.currentTime (never the next one) — a pose is only ever
            // shown once it's actually happened in the video's own
            // timeline, never slightly early. Standard upper_bound search:
            // find the first index whose ts is *past* t, then step back
            // one to land on the last index at-or-before it.
            var t = video.currentTime;
            var frames = timeline.frames;
            var lo = 0, hi = frames.length;
            while (lo < hi) {
                var mid = (lo + hi) >> 1;
                if (frames[mid].ts <= t) lo = mid + 1; else hi = mid;
            }
            lo--;
            if (lo < 0) return;  // nothing at or before this moment yet
            // (e.g. before the first real detection of the video/window).

            // Reject the match if it's too stale — without this, a span
            // with no real detections at all (opening-graphics intro, a
            // slide cutaway) still finds the last real keyframe before it
            // and holds it, frozen, for the whole span, however long ago
            // it actually was. At step=2 keyframe density (see
            // gesture_worker.py's _extract_keyframes), real gaps during
            // active tracking are on the order of two native frames'
            // duration — comfortably under this cap — so it only ever
            // suppresses genuinely empty spans, not normal tracking.
            var MAX_KEYFRAME_GAP_S = 0.3;
            if (t - frames[lo].ts > MAX_KEYFRAME_GAP_S) return;

            var f = frames[lo];
            var px = f.px, py = f.py, pv = f.pv, nKp = px.length;
            if (nKp < 33) return;  // BlazePose keypoint count (was 19 under MeTRAbs's coco_19)

            var cx_arr = new Float32Array(nKp), cy_arr = new Float32Array(nKp);
            for (var m = 0; m < nKp; m++) {
                cx_arr[m] = px[m] * frame_w;
                cy_arr[m] = (1 - py[m]) * frame_h;
            }

            var activeSegs = state.segment;
            var segColor = {};  // landmark idx -> highlight color
            var segGroup = {};  // landmark idx -> "segname-groupindex", for the
            // same-group check below — two landmarks can share a color (same
            // segment) without counting as connectable (different group).
            activeSegs.forEach(function(seg) {
                var groups = SEG_LMS[seg];
                if (!groups) return;
                groups.forEach(function(group, gi) {
                    group.forEach(function(i) {
                        segColor[i] = SEG_COLORS[seg];
                        segGroup[i] = seg + '-' + gi;
                    });
                });
            });
            var anyActive = activeSegs.length > 0;
            var normalEdge = anyActive ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.60)';

            for (var e = 0; e < CONNECTIONS.length; e++) {
                var i = CONNECTIONS[e][0], j = CONNECTIONS[e][1];
                if (i >= nKp || j >= nKp || pv[i] < 0.15 || pv[j] < 0.15) continue;
                var ci = segColor[i], cj = segColor[j];
                // Same group, not just same segment — see SEG_LMS comment.
                var lit = ci !== undefined && cj !== undefined && segGroup[i] === segGroup[j];
                ctx.beginPath();
                ctx.moveTo(cx_arr[i], cy_arr[i]);
                ctx.lineTo(cx_arr[j], cy_arr[j]);
                ctx.strokeStyle = lit ? ci : normalEdge;
                ctx.lineWidth   = lit ? 2.5 : 1.5;
                ctx.stroke();
            }

            for (var k = 0; k < nKp; k++) {
                if (pv[k] < 0.15) continue;
                var kc = segColor[k];
                var litJ = kc !== undefined;
                ctx.beginPath();
                ctx.arc(cx_arr[k], cy_arr[k], litJ ? 5 : 3, 0, 2 * Math.PI);
                ctx.fillStyle = litJ ? kc : (anyActive ? 'rgba(255,255,255,0.40)' : 'rgba(255,255,255,0.80)');
                ctx.fill();
                if (litJ) { ctx.strokeStyle = 'white'; ctx.lineWidth = 1.5; ctx.stroke(); }
            }
        }

        requestAnimationFrame(drawLoop);
        return window.dash_clientside.no_update;
    }
    """,
    Output("pose-render-dummy", "data"),
    Input("pose-timeline", "data"),
    Input("pose-segment", "data"),
    Input("landmarks-visible", "data"),
)


clientside_callback(
    """
    function(n, visible) {
        var nu = window.dash_clientside.no_update;
        if (n === null || n === undefined) return [nu, nu, nu];
        var on = !visible;
        return [on, on ? 'Landmarks  ON' : 'Landmarks  OFF', on ? 'success' : 'danger'];
    }
    """,
    Output("landmarks-visible", "data"),
    Output("landmark-toggle", "children"),
    Output("landmark-toggle", "color"),
    Input("landmark-toggle", "n_clicks"),
    State("landmarks-visible", "data"),
)


@callback(Output("g-handedness", "figure"), Input("fused-data", "data"), Input("current-time", "data"), Input("keyword-occurrences", "data"))
def g_handedness(data, ct, occ):
    if not data:
        return empty_fig("Handedness  (R−L) / (R+L)")
    ws, t = _parse(data)
    y = [w.gesture.handedness_ratio if w.gesture else None for w in ws]
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=C["muted"], width=1, dash="dash"))
    fig.add_trace(go.Scatter(
        x=t, y=y, mode="lines+markers",
        line=dict(color=C["gesture"], width=2),
        marker=dict(size=5),
        name="(R−L)/(R+L)",
    ))
    layout = {**PLOT_LAYOUT}
    layout["yaxis"] = {**PLOT_LAYOUT["yaxis"],
                       "range": [-1.05, 1.05],
                       "title": dict(text="← Left  |  Right →", font=dict(size=9, color=C["muted"])),
                       "tickvals": [-1, -0.5, 0, 0.5, 1],
                       "ticktext": ["-1", "-0.5", "0", "0.5", "1"]}
    layout["title"] = dict(text="Handedness  (R−L) / (R+L)", font=dict(size=11, color=C["muted"]))
    fig.update_layout(**layout)
    return add_cursor(fig, ct, occ)


# ─────────────────────────────────────────────────────────────────────────────
# Chart callbacks — Acoustic
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("p-spectrogram", "figure"),
    Input("fused-data", "data"),
    Input("keyword-occurrences", "data"),
    State("active-job-id", "data"),
    State("data-source", "data"),
    State("active-collection", "data"),
)
def p_spectrogram(data, occ, job_id, data_source, collection):
    if not data or not job_id:
        return empty_fig("Spectrogram")
    sg = _artifacts(job_id, data_source, collection).get("spectrogram")
    if not sg:
        return empty_fig("Spectrogram")
    fig = go.Figure(go.Heatmap(
        x=sg["times"],
        y=sg["freqs"],
        z=sg["data"],
        colorscale="greys",
        showscale=True,
        colorbar=dict(
            title=dict(text="dB", side="right",
                       font=dict(size=9, color=C["muted"])),
            tickfont=dict(size=8, color=C["muted"]),
            thickness=10,
            len=0.9,
        ),
        hovertemplate="<b>Time: %{x:.2f}s<br>Freq: %{y:.2f} kHz<br>%{z:.1f} dB</b><extra></extra>",
    ))
    if sg.get("f0_times"):
        fig.add_trace(go.Scatter(
            x=sg["f0_times"],
            y=sg["f0_values"],
            mode="lines",
            line=dict(color="#FFD700", width=2),
            name="F0",
            connectgaps=False,
            hovertemplate="<b>%{x:.2f}s  F0: %{y:.3f} kHz</b><extra></extra>",
        ))
    layout = {k: v for k, v in PLOT_LAYOUT.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(
        **layout,
        hoverlabel=dict(
            bgcolor=C["text"],
            bordercolor=C["muted"],
            font=dict(color="#FFFFFF", size=11, family="DM Mono, monospace"),
        ),
        title=dict(text="Spectrogram  (narrow-band) + F0 contour", font=dict(size=11, color=C["muted"])),
        xaxis=dict(
            **PLOT_LAYOUT["xaxis"],
            showspikes=False,
        ),
        yaxis=dict(
            title=dict(text="Frequency (kHz)", font=dict(size=9, color=C["muted"])),
            showgrid=False, zeroline=False, showspikes=False,
        ),
        showlegend=False,
    )
    if occ:
        for o in occ:
            fig.add_vline(x=o["start_s"], line=dict(color=KW_COLOUR, width=2), opacity=0.7)
    return fig


@callback(
    Output("p-waveform", "figure"),
    Input("fused-data", "data"),
    Input("keyword-occurrences", "data"),
    State("active-job-id", "data"),
    State("data-source", "data"),
    State("active-collection", "data"),
)
def p_waveform(data, occ, job_id, data_source, collection):
    if not data or not job_id:
        return empty_fig("Waveform")
    wf = _artifacts(job_id, data_source, collection).get("waveform")
    if not wf:
        return empty_fig("Waveform")
    times = wf["times"]
    amps  = wf["amplitudes"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=amps,
        mode="lines",
        line=dict(color=C["prosody"], width=1),
        fill="tozeroy",
        fillcolor="rgba(45,106,79,0.15)",
        hovertemplate="<b>%{x:.2f}s</b>  %{y:.3f}<extra></extra>",
        name="Amplitude",
    ))
    layout = {k: v for k, v in PLOT_LAYOUT.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(
        **layout,
        title=dict(text="Waveform  (amplitude)", font=dict(size=11, color=C["muted"])),
        xaxis=dict(**PLOT_LAYOUT["xaxis"], showspikes=False),
        yaxis=dict(
            title=dict(text="Amplitude", font=dict(size=9, color=C["muted"])),
            range=[-1.05, 1.05],
            showgrid=False, zeroline=True,
            zerolinecolor=C["border"], zerolinewidth=1,
            showspikes=False,
        ),
    )
    if occ:
        for o in occ:
            fig.add_vline(x=o["start_s"], line=dict(color=KW_COLOUR, width=2), opacity=0.7)
    return fig


@callback(Output("p-f0", "figure"), Input("fused-data", "data"), Input("current-time", "data"), Input("keyword-occurrences", "data"))
def p_f0(data, ct, occ):
    if not data: return empty_fig("Pitch (F0)")
    ws, t = _parse(data)
    mean_f0 = [w.prosody.mean_f0 if w.prosody and w.prosody.mean_f0 else None for w in ws]
    f0_std  = [w.prosody.f0_std  if w.prosody and w.prosody.f0_std  else None for w in ws]
    upper = [m + s if m and s else None for m, s in zip(mean_f0, f0_std)]
    lower = [m - s if m and s else None for m, s in zip(mean_f0, f0_std)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t + t[::-1], y=upper + lower[::-1],
        fill="toself", fillcolor="rgba(45,106,79,0.1)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=t, y=mean_f0, mode="lines",
        line=dict(color=C["prosody"], width=2), name="F0 Hz",
    ))
    _style(fig, "Pitch (F0) ± std", "Hz")
    return add_cursor(fig, ct, occ)


@callback(Output("p-intensity", "figure"), Input("fused-data", "data"), Input("current-time", "data"), Input("keyword-occurrences", "data"))
def p_intensity(data, ct, occ):
    if not data: return empty_fig("Intensity")
    ws, t = _parse(data)
    fig = go.Figure(go.Scatter(
        x=t, y=[w.prosody.mean_intensity_db if w.prosody else None for w in ws],
        mode="lines", line=dict(color=C["prosody"], width=2),
        fill="tozeroy", fillcolor="rgba(45,106,79,0.08)", name="dB",
    ))
    _style(fig, "Intensity", "dB")
    return add_cursor(fig, ct, occ)




# ─────────────────────────────────────────────────────────────────────────────
# Chart callbacks — Verbal
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Chart callbacks — Camera
# ─────────────────────────────────────────────────────────────────────────────

SHOT_COLOURS = {
    "extreme_close_up": "#9B2335",
    "close_up":         C["gesture"],
    "medium_close":     "#7B5EA7",
    "medium":           C["camera"],
    "medium_long":      "#2D6A4F",
    "long":             "#1B4F8A",
    "very_long":        "#5C6970",
    "unknown":          C["muted"],
}

@callback(Output("c-shot", "figure"), Input("fused-data", "data"), Input("current-time", "data"), Input("keyword-occurrences", "data"))
def c_shot(data, ct, occ):
    if not data: return empty_fig("Shot Type")
    ws, t = _parse(data)
    shot_types = [w.camera.dominant_shot_type.value if w.camera else "unknown" for w in ws]
    fig = go.Figure(go.Bar(
        x=t, y=[1] * len(t),
        marker_color=[SHOT_COLOURS.get(s, C["muted"]) for s in shot_types],
        text=shot_types, textposition="inside",
        textfont=dict(size=9, family="DM Mono, monospace"),
        showlegend=False,
        hovertemplate="%{text}<extra></extra>",
    ))
    layout = {k: v for k, v in PLOT_LAYOUT.items() if k != "yaxis"}
    fig.update_layout(**layout,
        title=dict(text="Dominant Shot Type", font=dict(size=11, color=C["muted"])),
        yaxis=dict(showticklabels=False, showgrid=False),
        bargap=0.05,
    )
    return add_cursor(fig, ct, occ)


H_ANGLE_COLOURS = {
    HorizontalAngle.FRONTAL: C["prosody"],
    HorizontalAngle.OBLIQUE: C["cursor"],
    HorizontalAngle.UNKNOWN: C["muted"],
}

V_ANGLE_COLOURS = {
    VerticalAngle.HIGH:      C["gesture"],
    VerticalAngle.EYE_LEVEL: C["camera"],
    VerticalAngle.LOW:       C["verbal"],
    VerticalAngle.UNKNOWN:   C["muted"],
}


@callback(Output("c-h-angle", "figure"), Input("fused-data", "data"), Input("current-time", "data"), Input("keyword-occurrences", "data"))
def c_h_angle(data, ct, occ):
    if not data: return empty_fig("Horizontal Angle")
    ws, t = _parse(data)
    yaws = [w.camera.mean_shoulder_yaw_deg if w.camera and w.camera.mean_shoulder_yaw_deg is not None else None for w in ws]
    classes = [w.camera.horizontal_angle if w.camera else HorizontalAngle.UNKNOWN for w in ws]
    labels = [c.value for c in classes]
    fig = go.Figure(go.Bar(
        x=t,
        y=[abs(v) if v is not None else 0 for v in yaws],
        marker_color=[H_ANGLE_COLOURS.get(c, C["muted"]) for c in classes],
        text=labels,
        textposition="inside",
        textfont=dict(size=9, family="DM Mono, monospace"),
        customdata=[[round(v, 1)] if v is not None else [None] for v in yaws],
        hovertemplate="%{text}  |  %{customdata[0]}°<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=10, line=dict(color=C["muted"], width=1, dash="dot"))
    layout = {k: v for k, v in PLOT_LAYOUT.items() if k not in ("yaxis",)}
    fig.update_layout(**layout,
        title=dict(text="Horizontal Angle  (shoulder yaw)", font=dict(size=11, color=C["muted"])),
        yaxis=dict(title=dict(text="|yaw| °", font=dict(size=10, color=C["muted"])),
                   range=[0, 95], showgrid=True, gridcolor=C["border"],
                   zeroline=False, tickfont=dict(size=10)),
        bargap=0.05,
    )
    return add_cursor(fig, ct, occ)


@callback(Output("c-v-angle", "figure"), Input("fused-data", "data"), Input("current-time", "data"), Input("keyword-occurrences", "data"))
def c_v_angle(data, ct, occ):
    if not data: return empty_fig("Vertical Angle")
    ws, t = _parse(data)
    pitches = [w.camera.mean_face_pitch_deg if w.camera and w.camera.mean_face_pitch_deg is not None else None for w in ws]
    classes = [w.camera.vertical_angle if w.camera else VerticalAngle.UNKNOWN for w in ws]
    labels = [c.value.replace("_", " ") for c in classes]
    fig = go.Figure(go.Bar(
        x=t,
        y=[v if v is not None else 0 for v in pitches],
        marker_color=[V_ANGLE_COLOURS.get(c, C["muted"]) for c in classes],
        text=labels,
        textposition="inside",
        textfont=dict(size=9, family="DM Mono, monospace"),
        customdata=[[round(v, 1)] if v is not None else [None] for v in pitches],
        hovertemplate="%{text}  |  %{customdata[0]}°<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=10,  line=dict(color=C["muted"], width=1, dash="dot"))
    fig.add_hline(y=-10, line=dict(color=C["muted"], width=1, dash="dot"))
    fig.add_hline(y=0,   line=dict(color=C["muted"], width=0.5))
    layout = {k: v for k, v in PLOT_LAYOUT.items() if k not in ("yaxis",)}
    fig.update_layout(**layout,
        title=dict(text="Vertical Angle  (face pitch)", font=dict(size=11, color=C["muted"])),
        yaxis=dict(title=dict(text="pitch °", font=dict(size=10, color=C["muted"])),
                   showgrid=True, gridcolor=C["border"],
                   zeroline=False, tickfont=dict(size=10)),
        bargap=0.05,
    )
    return add_cursor(fig, ct, occ)


@callback(Output("c-cutrate", "figure"), Input("fused-data", "data"), Input("current-time", "data"), Input("keyword-occurrences", "data"))
def c_cutrate(data, ct, occ):
    if not data: return empty_fig("Scene Cuts")
    ws, t = _parse(data)
    fig = go.Figure(go.Bar(
        x=t, y=[w.camera.cut_count if w.camera else None for w in ws],
        marker_color=C["camera"], marker_opacity=0.7, name="cuts",
    ))
    _style(fig, "Scene Cuts", "cuts")
    return add_cursor(fig, ct, occ)


@callback(Output("c-facearea", "figure"), Input("fused-data", "data"), Input("current-time", "data"), Input("keyword-occurrences", "data"))
def c_facearea(data, ct, occ):
    if not data: return empty_fig("Face Area")
    ws, t = _parse(data)
    vals = [w.camera.mean_face_bbox_area * 100 if w.camera and w.camera.mean_face_bbox_area else None for w in ws]
    fig = go.Figure(go.Scatter(
        x=t, y=vals, mode="lines",
        line=dict(color=C["camera"], width=2),
        fill="tozeroy", fillcolor="rgba(123,94,167,0.08)", name="%",
    ))
    _style(fig, "Mean Face Area (zoom proxy)", "% of frame")
    return add_cursor(fig, ct, occ)


@callback(Output("c-trend", "figure"), Input("fused-data", "data"), Input("current-time", "data"), Input("keyword-occurrences", "data"))
def c_trend(data, ct, occ):
    if not data: return empty_fig("Zoom Trend")
    ws, t = _parse(data)
    vals = [w.camera.face_bbox_trend if w.camera and w.camera.face_bbox_trend is not None else None for w in ws]
    fig = go.Figure(go.Bar(
        x=t, y=vals,
        marker_color=[C["gesture"] if v and v > 0 else C["prosody"] if v else C["muted"] for v in vals],
        name="slope",
    ))
    fig.add_hline(y=0, line=dict(color=C["muted"], width=1))
    _style(fig, "Zoom Trend  (+ = zoom in)", "slope")
    return add_cursor(fig, ct, occ)

# ─────────────────────────────────────────────────────────────────────────────
# Corpus Analysis callbacks
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("keyword-occurrences", "data"),
    Output("active-keyword", "data"),
    Output("second-keyword", "data"),
    Output("search-display-keyword", "data"),
    Input("kw-search-btn", "n_clicks"),
    Input("kw-input", "n_submit"),
    State("kw-input", "value"),
    State("kw-input2", "value"),
    State("fused-data", "data"),
    State("active-job-id", "data"),
    State("data-source", "data"),
    State("active-collection", "data"),
    prevent_initial_call=True,
)
def kw_search(_, __, keyword, keyword2, data, job_id, data_source, collection):
    if not keyword or not keyword.strip() or not data:
        return [], None, None, None
    kw = keyword.strip().lower()
    kw2 = keyword2.strip().lower() if keyword2 and keyword2.strip() else None

    # A contraction typed as-is ("don't", "it's") never matches a collocations
    # key, but its two constituent words might. If the user hasn't manually
    # filled in a second keyword for comparison, detect a contraction via
    # spaCy's own tokenizer and route to the word-sketch-diff view for its
    # two parts instead of a single (empty) word sketch.
    #
    # Concordance also can't match the literal typed string anymore, now that
    # it reads segmented_tokens: spaCy splits "don't" into "do"/"n't" there,
    # so no single token ever equals "don't". contraction_parts captures the
    # two RAW (un-normalised) token texts — not active_kw/second_kw's _eff()
    # form, since segmented_tokens holds literal spaCy token text — so
    # Concordance can search for them occurring adjacently instead. Scoped to
    # single-word queries with no whitespace (not just "any 2-token query",
    # which a genuine phrase search like "New York" would also produce) so a
    # real two-word phrase search still gets today's literal-match behaviour.
    active_kw, second_kw = keyword, kw2
    contraction_parts = None
    if not kw2:
        query_toks = [t for t in _nlp_en(keyword.strip()) if not t.is_space and not t.is_punct]
        if len(query_toks) == 2:
            active_kw = corpus_analysis._eff(query_toks[0])
            second_kw = corpus_analysis._eff(query_toks[1])
            if " " not in keyword.strip():
                contraction_parts = (
                    re.sub(r"^[^\w]+|[^\w]+$", "", query_toks[0].text, flags=re.UNICODE).lower(),
                    re.sub(r"^[^\w]+|[^\w]+$", "", query_toks[1].text, flags=re.UNICODE).lower(),
                )

    occurrences = []

    # Concordance reads the spaCy-word-segmented token list (built once
    # during processing — see VerbalWorker._segment_words) rather than raw
    # Whisper tokens, for every language: this keeps it from ever disagreeing
    # with Word List/Collocations/Word Sketch/Distributional Thesaurus about
    # what words exist in the transcript — e.g. Whisper's raw ASR tokens
    # keep English contractions ("don't") and hyphenated compounds
    # ("well-known") glued together as one token, and CJK's raw tokens are
    # individual characters, so a multi-character search would never match
    # a single one — either way, spaCy's own tokenization is what the other
    # features already use, so Concordance now does too. Falls back to raw
    # per-window Whisper tokens only if segmentation wasn't available for
    # this job (e.g. no spaCy model for the detected language).
    segmented = None
    if job_id:
        segmented = _artifacts(job_id, data_source, collection).get("segmented_tokens")

    if segmented:
        # CJK words conventionally read with no space between them.
        sep = "" if _CJK_CHAR_RE.search(segmented[0]["word"]) else " "

        if contraction_parts:
            part1, part2 = contraction_parts
            for i in range(len(segmented) - 1):
                w1 = re.sub(r"^[^\w]+|[^\w]+$", "", segmented[i]["word"], flags=re.UNICODE).lower()
                w2 = re.sub(r"^[^\w]+|[^\w]+$", "", segmented[i + 1]["word"], flags=re.UNICODE).lower()
                if w1 == part1 and w2 == part2:
                    left  = sep.join(t["word"] for t in segmented[max(0, i - 4):i])
                    right = sep.join(t["word"] for t in segmented[i + 2:i + 6])
                    occurrences.append({
                        "word":          keyword,
                        "start_s":       segmented[i]["start_s"],
                        "end_s":         segmented[i + 1]["end_s"],
                        "context_left":  left,
                        "context_right": right,
                        "window_start":  segmented[i]["start_s"],
                    })
            return occurrences, active_kw, second_kw, keyword

        for i, tok in enumerate(segmented):
            tok_clean = re.sub(r"^[^\w]+|[^\w]+$", "", tok["word"], flags=re.UNICODE).lower()
            if tok_clean == kw:
                left  = sep.join(t["word"] for t in segmented[max(0, i - 4):i])
                right = sep.join(t["word"] for t in segmented[i + 1:i + 5])
                occurrences.append({
                    "word":          tok["word"],
                    "start_s":       tok["start_s"],
                    "end_s":         tok["end_s"],
                    "context_left":  left,
                    "context_right": right,
                    "window_start":  tok["start_s"],
                })
        return occurrences, active_kw, second_kw, keyword

    ws, _ = _parse(data)
    for w in ws:
        if not w.verbal or not w.verbal.tokens:
            continue
        tokens = w.verbal.tokens
        for i, tok in enumerate(tokens):
            tok_clean = re.sub(r"^[^\w]+|[^\w]+$", "", tok.word, flags=re.UNICODE).lower()
            if tok_clean == kw:
                left  = " ".join(t.word for t in tokens[max(0, i - 4):i])
                right = " ".join(t.word for t in tokens[i + 1:i + 5])
                occurrences.append({
                    "word":          tok.word,
                    "start_s":       tok.start_s,
                    "end_s":         tok.end_s,
                    "context_left":  left,
                    "context_right": right,
                    "window_start":  w.window.start_s,
                })
    return occurrences, active_kw, second_kw, keyword


@callback(
    Output("kw-stats", "children"),
    Input("keyword-occurrences", "data"),
    Input("search-display-keyword", "data"),
)
def kw_stats(occ, keyword):
    if not occ or not keyword:
        return ""
    return f'"{keyword}"  —  {len(occ)} occurrence{"s" if len(occ) != 1 else ""}'


@callback(
    Output("kw-concordance", "children"),
    Input("keyword-occurrences", "data"),
    Input("active-keyword", "data"),
)
def kw_concordance(occ, keyword):
    if not keyword:
        return html.Div()
    if not occ:
        return html.P(
            f'No occurrences of "{keyword}" found in the transcript.',
            style={"fontFamily": "DM Mono, monospace", "fontSize": "11px",
                   "color": C["muted"], "marginTop": "8px"},
        )

    mono = {"fontFamily": "DM Mono, monospace", "fontSize": "11px"}
    rows = []
    for i, o in enumerate(occ):
        m, s = divmod(int(o["start_s"]), 60)
        ts_label = f"{m}:{s:02d}"
        rows.append(html.Div(style={
            "display": "flex", "alignItems": "baseline", "gap": "12px",
            "padding": "5px 0",
            "borderBottom": f"1px solid {C['border']}",
        }, children=[
            dbc.Button(ts_label, id={"type": "conc-seek", "index": i},
                       size="sm", color="link",
                       style={**mono, "color": KW_COLOUR, "padding": "0",
                              "minWidth": "36px", "textAlign": "right"}),
            html.Span(o["context_left"] + " ", style={**mono, "color": C["muted"],
                                                       "textAlign": "right", "flex": "1"}),
            html.Span(o["word"].upper(),
                      style={**mono, "color": KW_COLOUR, "fontWeight": "600",
                             "whiteSpace": "nowrap"}),
            html.Span(" " + o["context_right"], style={**mono, "color": C["muted"], "flex": "1"}),
        ]))

    return html.Div([
        html.P("CONCORDANCE", style={**LABEL_STYLE, "marginBottom": "8px"}),
        html.Div(rows),
    ])


@callback(
    Output("seek-to", "data"),
    Input({"type": "conc-seek", "index": ALL}, "n_clicks"),
    State("keyword-occurrences", "data"),
    prevent_initial_call=True,
)
def conc_seek(_, occurrences):
    ctx = dash.callback_context
    if not ctx.triggered_id or not occurrences:
        return dash.no_update
    idx = ctx.triggered_id.get("index", -1)
    if 0 <= idx < len(occurrences):
        return occurrences[idx]["start_s"]
    return dash.no_update



@callback(
    Output("kw-sketch-panel", "children"),
    Input("active-keyword", "data"),
    State("active-job-id", "data"),
    State("data-source", "data"),
    State("active-collection", "data"),
    prevent_initial_call=True,
)
def kw_sketch(keyword, job_id, data_source, collection):
    if not keyword:
        return html.Div()
    if not job_id:
        return html.P("No active job — upload a video and run analysis first.",
                      style={"fontFamily": "DM Mono, monospace", "fontSize": "10px", "color": C["muted"]})
    collocations = _artifacts(job_id, data_source, collection).get("collocations")
    if not collocations:
        return html.P("No collocations data — run analysis first.",
                      style={"fontFamily": "DM Mono, monospace", "fontSize": "10px",
                             "color": C["muted"]})
    sketch = corpus_analysis.get_word_sketch(collocations, keyword)
    if not sketch["found"]:
        return html.P(f'"{keyword}" not found in transcript collocations.',
                      style={"fontFamily": "DM Mono, monospace", "fontSize": "10px",
                             "color": C["muted"]})
    mono = {"fontFamily": "DM Mono, monospace", "fontSize": "10px"}
    sections = []
    for rel in sketch["relations"]:
        words = rel["words"]
        if not words:
            continue
        sections.append(html.Div([
            html.P(rel["name"].upper(),
                   style={**LABEL_STYLE, "marginBottom": "4px", "marginTop": "10px"}),
            html.Div([
                html.Span(
                    f"{w[0]} ({w[1]})",
                    style={**mono, "color": C["corpus"],
                           "marginRight": "12px", "display": "inline-block"},
                )
                for w in words
            ]),
        ]))
    return html.Div(sections) if sections else html.P(
        "No collocational relations found.",
        style={"fontFamily": "DM Mono, monospace", "fontSize": "10px", "color": C["muted"]})


@callback(
    Output("kw-thesaurus-panel", "children"),
    Input("active-keyword", "data"),
    State("active-job-id", "data"),
    State("data-source", "data"),
    State("active-collection", "data"),
    prevent_initial_call=True,
)
def kw_thesaurus(keyword, job_id, data_source, collection):
    if not keyword or not job_id:
        return html.Div()
    artifacts = _artifacts(job_id, data_source, collection)
    collocations = artifacts.get("collocations")
    if not collocations:
        return html.P("No collocations data — run analysis first.",
                      style={"fontFamily": "DM Mono, monospace", "fontSize": "10px",
                             "color": C["muted"]})
    similar = corpus_analysis.distributional_thesaurus(collocations, keyword, top_n=100)
    if not similar:
        return html.P(
            f'No distributionally similar words found for "{keyword}". '
            "The transcript may be too short for reliable similarity.",
            style={"fontFamily": "DM Mono, monospace", "fontSize": "10px", "color": C["muted"]})

    wl = artifacts.get("wordlist")
    freq_map: dict[str, int] = {}
    if wl and wl.get("words"):
        freq_map = {e["word"]: e["count"] for e in wl["words"]}

    rows = [
        {
            "rank":     i + 1,
            "word":     s["word"],
            "freq":     freq_map.get(s["word"], 0),
            "jaccard":  s["score"],
            "shared":   s["shared"],
        }
        for i, s in enumerate(similar)
    ]

    mono_sm = "DM Mono, monospace"
    return html.Div([
        html.P(f"Top {len(similar)} similar words · Jaccard similarity",
               style={**LABEL_STYLE, "marginBottom": "10px"}),
        dash_table.DataTable(
            data=rows,
            columns=[
                {"name": "#",                  "id": "rank"},
                {"name": "Word",               "id": "word"},
                {"name": "Frequency",          "id": "freq"},
                {"name": "Jaccard Similarity", "id": "jaccard"},
                {"name": "Shared Contexts",    "id": "shared"},
            ],
            sort_action="native",
            page_size=25,
            style_table={
                "overflowY": "auto",
                "maxHeight": "520px",
                "border": f"1px solid {C['border']}",
                "borderRadius": "4px",
            },
            style_cell={
                "fontFamily":       mono_sm,
                "fontSize":         "11px",
                "color":            C["text"],
                "backgroundColor":  C["surface"],
                "border":           f"1px solid {C['border']}",
                "padding":          "6px 14px",
                "textAlign":        "left",
                "whiteSpace":       "normal",
            },
            style_header={
                "fontFamily":      mono_sm,
                "fontSize":        "10px",
                "letterSpacing":   "0.07em",
                "textTransform":   "uppercase",
                "color":           C["muted"],
                "backgroundColor": C["bg"],
                "border":          f"1px solid {C['border']}",
                "padding":         "8px 14px",
                "fontWeight":      "normal",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": C["bg"]},
            ],
            style_cell_conditional=[
                {"if": {"column_id": "rank"},    "width": "44px",  "textAlign": "right"},
                {"if": {"column_id": "freq"},    "width": "100px", "textAlign": "right"},
                {"if": {"column_id": "jaccard"}, "width": "140px", "textAlign": "right"},
                {"if": {"column_id": "shared"},  "width": "130px", "textAlign": "right"},
            ],
        ),
    ])


@callback(
    Output("kw-diff-panel", "children"),
    Input("active-keyword", "data"),
    Input("second-keyword", "data"),
    State("active-job-id", "data"),
    State("data-source", "data"),
    State("active-collection", "data"),
    prevent_initial_call=True,
)
def kw_diff(kw1, kw2, job_id, data_source, collection):
    if not kw1 or not kw2 or not job_id:
        return html.Div()
    collocations = _artifacts(job_id, data_source, collection).get("collocations")
    if not collocations:
        return html.Div()
    diff = corpus_analysis.word_sketch_diff(collocations, kw1, kw2)
    if not diff["relations"]:
        return html.P(
            f'No comparable collocations between "{kw1}" and "{kw2}".',
            style={"fontFamily": "DM Mono, monospace", "fontSize": "10px", "color": C["muted"]})

    mono = {"fontFamily": "DM Mono, monospace", "fontSize": "10px"}
    col1_style = {**mono, "color": C["gesture"],  "display": "inline-block",
                  "marginRight": "8px", "marginBottom": "3px"}
    col2_style = {**mono, "color": C["prosody"],  "display": "inline-block",
                  "marginRight": "8px", "marginBottom": "3px"}
    shared_style = {**mono, "color": C["corpus"], "display": "inline-block",
                    "marginRight": "8px", "marginBottom": "3px"}

    blocks = [
        html.P("WORD SKETCH DIFFERENCE", style={**LABEL_STYLE, "marginBottom": "10px"}),
        html.Div(style={"display": "flex", "gap": "28px", "marginBottom": "8px"}, children=[
            html.Span(f"● {kw1}", style={**mono, "color": C["gesture"], "fontWeight": "500"}),
            html.Span(f"● {kw2}", style={**mono, "color": C["prosody"], "fontWeight": "500"}),
            html.Span("● shared", style={**mono, "color": C["corpus"], "fontWeight": "500"}),
        ]),
    ]

    for rel_key, rel_data in diff["relations"].items():
        cells = []
        for w, c in rel_data["only1"]:
            cells.append(html.Span(f"{w}({c})", style=col1_style))
        for w, c1, c2 in rel_data["shared"]:
            cells.append(html.Span(f"{w}({c1}/{c2})", style=shared_style))
        for w, c in rel_data["only2"]:
            cells.append(html.Span(f"{w}({c})", style=col2_style))
        if cells:
            blocks.append(html.Div([
                html.P(rel_data["name"].upper(),
                       style={**LABEL_STYLE, "marginTop": "10px", "marginBottom": "4px"}),
                html.Div(cells),
            ]))

    return html.Div(blocks, style={
        "borderTop": f"1px solid {C['border']}", "paddingTop": "20px",
    })


@callback(
    Output("transcript-view", "children"),
    Output("window-times", "data"),
    Input("fused-data", "data"),
    prevent_initial_call=True,
)
def load_transcript(fused_raw):
    if not fused_raw:
        return html.Div(), []
    ws, _ = _parse(fused_raw)
    segs = []
    times = []
    mono = "DM Mono, monospace"
    for idx, w in enumerate(ws):
        start = w.window.start_s
        end = w.window.end_s
        text = (w.verbal.transcript if w.verbal else "") or ""
        times.append({"start": start, "end": end})
        ts_label = f"{int(start)//60}:{int(start)%60:02d} – {int(end)//60}:{int(end)%60:02d}"
        segs.append(html.Div(
            id=f"ts-seg-{idx}",
            children=[
                html.Span(ts_label, style={
                    "fontFamily": mono, "fontSize": "9px",
                    "color": C["muted"], "display": "block",
                    "marginBottom": "5px",
                }),
                html.Span(text or "—", style={
                    "fontFamily": "'Inter', 'DM Sans', sans-serif",
                    "fontSize": "13px", "color": C["text"], "lineHeight": "1.75",
                }),
            ],
            style={
                "padding": "12px 16px",
                "borderBottom": f"1px solid {C['border']}",
                "transition": "background-color 0.25s ease",
                "cursor": "default",
            },
        ))
    return segs, times


@callback(
    Output("kw-wordlist-table", "data"),
    Input("fused-data", "data"),
    Input("pos-filter", "value"),
    State("active-job-id", "data"),
    State("data-source", "data"),
    State("active-collection", "data"),
)
def kw_wordlist_table(data, pos_filter, job_id, data_source, collection):
    if not data or not job_id:
        return []
    wl = _artifacts(job_id, data_source, collection).get("wordlist")
    if not wl or not wl.get("words"):
        return []

    pos_filter = pos_filter or "ALL"
    all_entries = wl["words"]
    if pos_filter == "ALL":
        entries = all_entries
    elif pos_filter == "OTHER":
        entries = [e for e in all_entries if e["pos"] not in _KNOWN_POS]
    else:
        allowed = _POS_FILTER_SETS.get(pos_filter, set())
        entries = [e for e in all_entries if e["pos"] in allowed]

    # Filtering above uses spaCy's raw tag abbreviations (matching
    # _POS_FILTER_SETS/_KNOWN_POS); only the displayed value is expanded to
    # a full label, here at the end, so filtering logic is unaffected.
    return [{**e, "pos": _POS_LABELS.get(e["pos"], e["pos"])} for e in entries]


@callback(
    Output("kw-ngrams-chart", "figure"),
    Input("fused-data", "data"),
    State("active-job-id", "data"),
    State("data-source", "data"),
    State("active-collection", "data"),
)
def kw_ngrams_chart(data, job_id, data_source, collection):
    if not data or not job_id:
        return empty_fig("N-grams")
    ng = _artifacts(job_id, data_source, collection).get("ngrams")
    if not ng:
        return empty_fig("N-grams")

    bigrams  = ng.get("bigrams",  [])[:15]
    trigrams = ng.get("trigrams", [])[:10]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Top Bigrams", "Top Trigrams"),
        horizontal_spacing=0.12,
    )

    if bigrams:
        bg_rev = list(reversed(bigrams))
        fig.add_trace(go.Bar(
            x=[b["count"] for b in bg_rev],
            y=[b["ngram"] for b in bg_rev],
            orientation="h",
            marker_color=C["verbal"],
            showlegend=False,
            hovertemplate="%{y}  |  %{x}<extra></extra>",
        ), row=1, col=1)

    if trigrams:
        tg_rev = list(reversed(trigrams))
        fig.add_trace(go.Bar(
            x=[t["count"] for t in tg_rev],
            y=[t["ngram"] for t in tg_rev],
            orientation="h",
            marker_color=C["camera"],
            showlegend=False,
            hovertemplate="%{y}  |  %{x}<extra></extra>",
        ), row=1, col=2)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color=C["text"], size=11),
        margin=dict(l=140, r=24, t=48, b=36),
        hovermode="y",
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=C["border"], zeroline=False,
        title_text="count", title_font=dict(size=9, color=C["muted"]),
        tickfont=dict(size=9),
    )
    fig.update_yaxes(
        showgrid=False, zeroline=False, tickfont=dict(size=9), automargin=True,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse(data):
    ws = []
    for i, d in enumerate(data):
        try:
            ws.append(FusedWindow.model_validate(d))
        except Exception as e:
            import traceback
            print(f"[dashboard] _parse failed on window {i}: {e}\n{traceback.format_exc()}")
    return ws, [w.window.midpoint for w in ws]

def _mean(vals):
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None

def _style(fig, title, yaxis_title):
    layout = {**PLOT_LAYOUT}
    layout["yaxis"] = {**PLOT_LAYOUT["yaxis"],
                       "title": dict(text=yaxis_title, font=dict(size=9, color=C["muted"]))}
    layout["title"] = dict(text=title, font=dict(size=11, color=C["muted"]))
    fig.update_layout(**layout)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)