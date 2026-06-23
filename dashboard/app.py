"""
dashboard/app.py
----------------
Single-column dashboard with Dash-served video player.

Video is served via a Flask route so the browser can actually load it.
Clicking any chart seeks the video to that timestamp.
"""

from __future__ import annotations

import base64
import os
import threading
import uuid as _uuid_module
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import flask
import plotly.graph_objects as go
from dash import Input, Output, State, callback, clientside_callback, dcc, html

from core.feature_store import FeatureStore
from core.models import FusedWindow
from core.orchestrator import Orchestrator

# ─────────────────────────────────────────────────────────────────────────────
# App + video serving
# ─────────────────────────────────────────────────────────────────────────────

store = FeatureStore()
_orch = Orchestrator(store=store)

server = flask.Flask(__name__)
server.config["MAX_CONTENT_LENGTH"] = None  # allow large video uploads
FONT_URL = "https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Fraunces:ital,wght@0,300;0,600;1,300&display=swap"

app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_URL],
    title="Mannerism Analyzer",
    suppress_callback_exceptions=True,
)

# Store the video path globally so the Flask route can serve it
_VIDEO_PATH: dict = {"path": None}


@server.route("/video")
def serve_video():
    """Serve the analysis video file so the browser <video> tag can load it."""
    path = _VIDEO_PATH.get("path")
    if not path or not os.path.exists(path):
        return flask.Response("Video not found", status=404)
    directory = str(Path(path).parent)
    filename   = Path(path).name
    return flask.send_from_directory(directory, filename, conditional=True)


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

CHART_CFG = {"displayModeBar": False}

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


def _analysis_progress(state: str, message: str):
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
    return html.Div([icon, html.Span(message, style={
        "fontFamily": "DM Mono, monospace", "fontSize": "11px", "color": colour,
    })], style={"display": "flex", "alignItems": "center", "marginTop": "10px"})


def empty_fig(title=""):
    fig = go.Figure()
    fig.update_layout(**PLOT_LAYOUT,
        title=dict(text=title, font=dict(size=11, color=C["muted"])))
    return fig


def add_cursor(fig, t):
    if t:
        fig.add_vline(x=t, line=dict(color=C["cursor"], width=1.5, dash="dot"))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart IDs for universal scrubber
# ─────────────────────────────────────────────────────────────────────────────

CHART_IDS = [
    "g-velocity", "g-handedness",
    "p-f0", "p-intensity", "p-voiced", "p-jitter", "p-shimmer", "p-hnr",
    "v-filler", "v-ttr", "v-pauses", "v-hedge", "v-sent-len", "v-confidence",
    "c-shot", "c-cutrate", "c-facearea", "c-trend",
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
        html.Div([
            html.Span("MANNERISM", style={
                "fontFamily": "DM Mono, monospace", "fontSize": "13px",
                "fontWeight": "500", "letterSpacing": "0.18em", "color": C["text"],
            }),
            html.Span(" ANALYZER", style={
                "fontFamily": "DM Mono, monospace", "fontSize": "13px",
                "fontWeight": "300", "letterSpacing": "0.18em", "color": C["muted"],
            }),
        ]),
    ]),

    # ── Main ─────────────────────────────────────────────────────────────────
    html.Div(style={"maxWidth": "1200px", "margin": "0 auto", "padding": "32px 40px"}, children=[

        # KPI strip
        html.Div(id="kpi-strip", style={
            "display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "28px",
        }),

        # ── GESTURE ──────────────────────────────────────────────────────
        html.Div(style=SECTION_STYLE, children=[
            section_header("Gesture", C["gesture"], "MediaPipe Holistic · Kinematic features"),

            # Upload drop zone
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
            html.Div(id="analysis-status", style={"marginBottom": "16px"}),

            # ── Pose Viewer ───────────────────────────────────────────
            html.Div(style={
                "marginBottom": "28px",
                "paddingBottom": "24px",
                "borderBottom": f"1px solid {C['border']}",
            }, children=[

                # Controls row: segment selector + landmark toggle
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
                                       outline=True, color="info",    className="mt-1"),
                        ], style={"marginTop": "6px", "display": "flex", "flexWrap": "wrap"}),
                    ]),
                    html.Div([
                        html.P("POSE OVERLAY", style=LABEL_STYLE),
                        dbc.Button("Landmarks  ON", id="landmark-toggle", size="sm",
                                   color="success", className="mt-1", disabled=True),
                    ]),
                ]),

                # Video with transparent landmark canvas overlay
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

                # Time / window info below the video
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

                # Handedness time-series
                dcc.Graph(id="g-handedness", style={"height": "160px", "marginTop": "20px"},
                          config=CHART_CFG),
            ]),

            # ── Kinematic time-series charts ──────────────────────────
            dcc.Graph(id="g-velocity", style={"height": "200px"}, config=CHART_CFG),
        ]),

        # ── PROSODY ───────────────────────────────────────────────────────
        html.Div(style=SECTION_STYLE, children=[
            section_header("Prosody", C["prosody"], "Parselmouth · Praat algorithms"),
            dcc.Graph(id="p-f0",        style={"height": "200px"}, config=CHART_CFG),
            dcc.Graph(id="p-intensity", style={"height": "200px"}, config=CHART_CFG),
            dcc.Graph(id="p-voiced",    style={"height": "200px"}, config=CHART_CFG),
            dcc.Graph(id="p-jitter",    style={"height": "200px"}, config=CHART_CFG),
            dcc.Graph(id="p-shimmer",   style={"height": "200px"}, config=CHART_CFG),
            dcc.Graph(id="p-hnr",       style={"height": "200px"}, config=CHART_CFG),
        ]),

        # ── VERBAL ────────────────────────────────────────────────────────
        html.Div(style=SECTION_STYLE, children=[
            section_header("Verbal", C["verbal"], "faster-whisper · spaCy"),
            dcc.Graph(id="v-filler",     style={"height": "200px"}, config=CHART_CFG),
            dcc.Graph(id="v-ttr",        style={"height": "200px"}, config=CHART_CFG),
            dcc.Graph(id="v-pauses",     style={"height": "200px"}, config=CHART_CFG),
            dcc.Graph(id="v-hedge",      style={"height": "200px"}, config=CHART_CFG),
            dcc.Graph(id="v-sent-len",   style={"height": "200px"}, config=CHART_CFG),
            dcc.Graph(id="v-confidence", style={"height": "200px"}, config=CHART_CFG),
        ]),

        # ── CAMERA ────────────────────────────────────────────────────────
        html.Div(style=SECTION_STYLE, children=[
            section_header("Camera", C["camera"], "PySceneDetect · Haar cascade"),
            dcc.Graph(id="c-shot",     style={"height": "200px"}, config=CHART_CFG),
            dcc.Graph(id="c-cutrate",  style={"height": "200px"}, config=CHART_CFG),
            dcc.Graph(id="c-facearea", style={"height": "200px"}, config=CHART_CFG),
            dcc.Graph(id="c-trend",    style={"height": "200px"}, config=CHART_CFG),
        ]),
    ]),

    # ── Hidden stores ─────────────────────────────────────────────────────────
    dcc.Store(id="fused-data"),
    dcc.Store(id="current-time", data=0.0),
    dcc.Store(id="active-job-id"),
    dcc.Store(id="pose-segment", data=[]),
    dcc.Store(id="pose-timeline", data=None),
    dcc.Store(id="pose-render-dummy"),
    dcc.Store(id="landmarks-visible", data=True),
    dcc.Interval(id="poll-interval", interval=2000, n_intervals=0, disabled=True),
    dcc.Interval(id="pose-ticker", interval=67, n_intervals=0),  # ~15 fps animation
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

# ─────────────────────────────────────────────────────────────────────────────
# Upload video → save to disk → launch analysis in background thread
# ─────────────────────────────────────────────────────────────────────────────

_UPLOAD_DIR = Path("/tmp/mannerism/uploads")

@callback(
    Output("active-job-id", "data"),
    Output("poll-interval", "disabled"),
    Output("video-player", "src"),
    Output("analysis-status", "children"),
    Output("landmark-toggle", "disabled"),
    Input("video-upload", "contents"),
    State("video-upload", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if not contents:
        return dash.no_update, True, dash.no_update, dash.no_update, dash.no_update

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _, b64 = contents.split(",", 1)
    video_path = str(_UPLOAD_DIR / filename)
    with open(video_path, "wb") as fh:
        fh.write(base64.b64decode(b64))

    _VIDEO_PATH["path"] = video_path

    job_id = str(_uuid_module.uuid4())[:8]
    threading.Thread(
        target=_orch.analyze,
        args=(video_path,),
        kwargs={"job_id": job_id},
        daemon=True,
    ).start()

    video_src = f"/video?t={os.path.getmtime(video_path)}"
    return job_id, False, video_src, _analysis_progress("running", f"Analysing {filename}…"), True


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

    # Still running — show last log event as progress hint
    events = store.read_events(job_id)
    msg = f"[{events[-1]['worker']}] {events[-1]['msg']}" if events else "Processing…"
    return (
        dash.no_update,
        _analysis_progress("running", msg),
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
    fillers     = sum(w.verbal.filler_word_count for w in ws if w.verbal)
    mean_f0     = _mean([w.prosody.mean_f0 for w in ws if w.prosody and w.prosody.mean_f0])
    mean_vel    = _mean([w.gesture.mean_wrist_velocity for w in ws if w.gesture])
    total_cuts  = sum(w.camera.cut_count for w in ws if w.camera)
    mean_hnr    = _mean([w.prosody.hnr_db for w in ws if w.prosody and w.prosody.hnr_db])
    return [
        kpi_card("Duration",   f"{duration:.0f}s",                      C["muted"]),
        kpi_card("Words",      str(total_words),                         C["verbal"]),
        kpi_card("Fillers",    str(fillers), C["gesture"],
                 f"{fillers/max(total_words,1)*100:.1f}% of words"),
        kpi_card("Mean F0",    f"{mean_f0:.0f} Hz" if mean_f0 else "—", C["prosody"]),
        kpi_card("Wrist Vel.", f"{mean_vel:.0f} px/s" if mean_vel else "—", C["gesture"]),
        kpi_card("Cuts",       str(total_cuts),                          C["camera"]),
        kpi_card("Mean HNR",   f"{mean_hnr:.1f} dB" if mean_hnr else "—", C["prosody"]),
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

@callback(Output("g-velocity", "figure"), Input("fused-data", "data"), Input("current-time", "data"))
def g_velocity(data, ct):
    if not data: return empty_fig("Wrist Velocity")
    ws, t = _parse(data)
    fig = go.Figure(go.Scatter(
        x=t, y=[w.gesture.mean_wrist_velocity if w.gesture else None for w in ws],
        mode="lines", line=dict(color=C["gesture"], width=2),
        fill="tozeroy", fillcolor="rgba(200,75,49,0.08)", name="px/s",
    ))
    _style(fig, "Wrist Velocity", "px/s")
    return add_cursor(fig, ct)



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
        active.remove(seg)   # click active segment → deselect
    else:
        active.append(seg)   # click inactive segment → add
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
    function(n, timeline, segment, visible) {
        var canvas = document.getElementById('pose-canvas');
        var video  = document.getElementById('video-player');
        if (!canvas || !video || !video.src) return window.dash_clientside.no_update;

        var el_w = video.offsetWidth, el_h = video.offsetHeight;
        if (!el_w || !el_h) return window.dash_clientside.no_update;

        // Compute the actual rendered frame rect inside the element box.
        // The browser fits the video with object-fit:contain semantics (aspect-ratio
        // preserved, centred, letterboxed/pillarboxed as needed by maxHeight etc.).
        var vid_w = video.videoWidth, vid_h = video.videoHeight;
        var frame_w, frame_h, frame_x, frame_y;
        if (vid_w && vid_h) {
            var scale = Math.min(el_w / vid_w, el_h / vid_h);
            frame_w = Math.round(vid_w * scale);
            frame_h = Math.round(vid_h * scale);
            frame_x = Math.round((el_w - frame_w) / 2);
            frame_y = Math.round((el_h - frame_h) / 2);
        } else {
            // Metadata not loaded yet — use full element, will correct next tick.
            frame_w = el_w; frame_h = el_h; frame_x = 0; frame_y = 0;
        }

        // Position the canvas to exactly cover the rendered frame (not the full element).
        if (canvas.width !== frame_w || canvas.height !== frame_h) {
            canvas.width  = frame_w;
            canvas.height = frame_h;
        }
        canvas.style.left   = frame_x + 'px';
        canvas.style.top    = frame_y + 'px';
        canvas.style.width  = frame_w + 'px';
        canvas.style.height = frame_h + 'px';

        var ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, frame_w, frame_h);

        if (!visible || !timeline || !timeline.frames || timeline.frames.length === 0)
            return window.dash_clientside.no_update;

        // Binary-search nearest keyframe by timestamp
        var t = video.currentTime;
        var frames = timeline.frames;
        var lo = 0, hi = frames.length - 1;
        while (lo < hi) {
            var mid = (lo + hi) >> 1;
            if (frames[mid].ts < t) lo = mid + 1; else hi = mid;
        }
        if (lo > 0 && Math.abs(frames[lo-1].ts - t) < Math.abs(frames[lo].ts - t)) lo--;
        var f = frames[lo];
        var px = f.px, py = f.py, pv = f.pv, n33 = px.length;
        if (n33 < 33) return window.dash_clientside.no_update;

        // Convert normalised coords → canvas pixels
        // py is stored y-flipped (1 − raw_y); canvas y=0 is top, so un-flip
        var cx_arr = new Float32Array(n33), cy_arr = new Float32Array(n33);
        for (var m = 0; m < n33; m++) {
            cx_arr[m] = px[m] * frame_w;
            cy_arr[m] = (1 - py[m]) * frame_h;
        }

        var SEG_LMS = {
            head:  [0,1,2,3,4,5,6,7,8,9,10],
            arms:  [11,12,13,14,15,16,17,18,19,20,21,22],
            hands: [15,16,17,18,19,20,21,22],
            torso: [11,12,23,24],
            gaze:  [1,2,3,4,5,6]
        };
        var SEG_COLORS = {
            head:  '#4A90D9',
            arms:  '#F0A500',
            hands: '#C84B31',
            torso: '#28a745',
            gaze:  '#20c2d4'
        };
        var CONNECTIONS = [
            [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],
            [11,12],[11,23],[12,24],[23,24],
            [11,13],[13,15],[15,17],[15,19],[15,21],[17,19],
            [12,14],[14,16],[16,18],[16,20],[16,22],[18,20],
            [23,25],[25,27],[27,29],[27,31],[29,31],
            [24,26],[26,28],[28,30],[28,32],[30,32]
        ];

        // Build landmark → color map from all active segments.
        // Segments applied in order: later entries overwrite for shared landmarks
        // (e.g. 'hands' colours hand landmarks on top of 'arms').
        var activeSegs = Array.isArray(segment) ? segment : (segment ? [segment] : []);
        var segColor = {};  // landmark_idx → hex color string
        activeSegs.forEach(function(seg) {
            if (SEG_LMS[seg]) {
                SEG_LMS[seg].forEach(function(i) { segColor[i] = SEG_COLORS[seg]; });
            }
        });
        var anyActive = activeSegs.length > 0;
        var normalEdge = anyActive ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.60)';

        // Draw edges
        for (var e = 0; e < CONNECTIONS.length; e++) {
            var i = CONNECTIONS[e][0], j = CONNECTIONS[e][1];
            if (i >= n33 || j >= n33 || pv[i] < 0.15 || pv[j] < 0.15) continue;
            var ci = segColor[i], cj = segColor[j];
            var lit = ci !== undefined && cj !== undefined;
            ctx.beginPath();
            ctx.moveTo(cx_arr[i], cy_arr[i]);
            ctx.lineTo(cx_arr[j], cy_arr[j]);
            ctx.strokeStyle = lit ? (ci || cj) : normalEdge;
            ctx.lineWidth   = lit ? 2.5 : 1.5;
            ctx.stroke();
        }

        // Draw joints
        for (var k = 0; k < n33; k++) {
            if (pv[k] < 0.15) continue;
            var kc = segColor[k];
            var litJ = kc !== undefined;
            ctx.beginPath();
            ctx.arc(cx_arr[k], cy_arr[k], litJ ? 5 : 3, 0, 2 * Math.PI);
            ctx.fillStyle = litJ ? kc : (anyActive ? 'rgba(255,255,255,0.40)' : 'rgba(255,255,255,0.80)');
            ctx.fill();
            if (litJ) {
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        }

        return window.dash_clientside.no_update;
    }
    """,
    Output("pose-render-dummy", "data"),
    Input("pose-ticker", "n_intervals"),
    State("pose-timeline", "data"),
    State("pose-segment", "data"),
    State("landmarks-visible", "data"),
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


@callback(Output("g-handedness", "figure"), Input("fused-data", "data"), Input("current-time", "data"))
def g_handedness(data, ct):
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
    return add_cursor(fig, ct)


# ─────────────────────────────────────────────────────────────────────────────
# Chart callbacks — Prosody
# ─────────────────────────────────────────────────────────────────────────────

@callback(Output("p-f0", "figure"), Input("fused-data", "data"), Input("current-time", "data"))
def p_f0(data, ct):
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
    return add_cursor(fig, ct)


@callback(Output("p-intensity", "figure"), Input("fused-data", "data"), Input("current-time", "data"))
def p_intensity(data, ct):
    if not data: return empty_fig("Intensity")
    ws, t = _parse(data)
    fig = go.Figure(go.Scatter(
        x=t, y=[w.prosody.mean_intensity_db if w.prosody else None for w in ws],
        mode="lines", line=dict(color=C["prosody"], width=2),
        fill="tozeroy", fillcolor="rgba(45,106,79,0.08)", name="dB",
    ))
    _style(fig, "Intensity", "dB")
    return add_cursor(fig, ct)


@callback(Output("p-voiced", "figure"), Input("fused-data", "data"), Input("current-time", "data"))
def p_voiced(data, ct):
    if not data: return empty_fig("Voiced Fraction")
    ws, t = _parse(data)
    fig = go.Figure(go.Bar(
        x=t, y=[w.prosody.voiced_fraction if w.prosody else None for w in ws],
        marker_color=C["prosody"], marker_opacity=0.7, name="fraction",
    ))
    _style(fig, "Voiced Fraction", "0–1")
    return add_cursor(fig, ct)


@callback(Output("p-jitter", "figure"), Input("fused-data", "data"), Input("current-time", "data"))
def p_jitter(data, ct):
    if not data: return empty_fig("Jitter")
    ws, t = _parse(data)
    fig = go.Figure(go.Scatter(
        x=t, y=[w.prosody.jitter_local if w.prosody and w.prosody.jitter_local else None for w in ws],
        mode="lines+markers", line=dict(color=C["prosody"], width=1.5),
        marker=dict(size=3), name="local",
    ))
    _style(fig, "Jitter (local)", "ratio")
    return add_cursor(fig, ct)


@callback(Output("p-shimmer", "figure"), Input("fused-data", "data"), Input("current-time", "data"))
def p_shimmer(data, ct):
    if not data: return empty_fig("Shimmer")
    ws, t = _parse(data)
    fig = go.Figure(go.Scatter(
        x=t, y=[w.prosody.shimmer_local if w.prosody and w.prosody.shimmer_local else None for w in ws],
        mode="lines+markers", line=dict(color=C["prosody"], width=1.5),
        marker=dict(size=3), name="local",
    ))
    _style(fig, "Shimmer (local)", "ratio")
    return add_cursor(fig, ct)


@callback(Output("p-hnr", "figure"), Input("fused-data", "data"), Input("current-time", "data"))
def p_hnr(data, ct):
    if not data: return empty_fig("HNR")
    ws, t = _parse(data)
    fig = go.Figure(go.Scatter(
        x=t, y=[w.prosody.hnr_db if w.prosody and w.prosody.hnr_db else None for w in ws],
        mode="lines", line=dict(color=C["prosody"], width=2),
        fill="tozeroy", fillcolor="rgba(45,106,79,0.08)", name="dB",
    ))
    _style(fig, "Harmonics-to-Noise Ratio", "dB")
    return add_cursor(fig, ct)

# ─────────────────────────────────────────────────────────────────────────────
# Chart callbacks — Verbal
# ─────────────────────────────────────────────────────────────────────────────

@callback(Output("v-filler", "figure"), Input("fused-data", "data"), Input("current-time", "data"))
def v_filler(data, ct):
    if not data: return empty_fig("Filler Rate")
    ws, t = _parse(data)
    fig = go.Figure(go.Bar(
        x=t, y=[w.verbal.filler_word_rate if w.verbal else None for w in ws],
        marker_color=C["verbal"], marker_opacity=0.7, name="per min",
    ))
    _style(fig, "Filler Word Rate", "per minute")
    return add_cursor(fig, ct)


@callback(Output("v-ttr", "figure"), Input("fused-data", "data"), Input("current-time", "data"))
def v_ttr(data, ct):
    if not data: return empty_fig("Type-Token Ratio")
    ws, t = _parse(data)
    fig = go.Figure(go.Scatter(
        x=t, y=[w.verbal.type_token_ratio if w.verbal else None for w in ws],
        mode="lines", line=dict(color=C["verbal"], width=2),
        fill="tozeroy", fillcolor="rgba(27,79,138,0.08)", name="TTR",
    ))
    fig.add_hline(y=0.5, line=dict(color=C["muted"], width=1, dash="dash"))
    _style(fig, "Type-Token Ratio", "0–1")
    return add_cursor(fig, ct)


@callback(Output("v-pauses", "figure"), Input("fused-data", "data"), Input("current-time", "data"))
def v_pauses(data, ct):
    if not data: return empty_fig("Pauses")
    ws, t = _parse(data)
    fig = go.Figure(go.Bar(
        x=t, y=[w.verbal.pause_count if w.verbal else None for w in ws],
        marker_color=C["verbal"], marker_opacity=0.6, name="count",
    ))
    _style(fig, "Pause Count", "count")
    return add_cursor(fig, ct)


@callback(Output("v-hedge", "figure"), Input("fused-data", "data"), Input("current-time", "data"))
def v_hedge(data, ct):
    if not data: return empty_fig("Hedge Words")
    ws, t = _parse(data)
    fig = go.Figure(go.Bar(
        x=t, y=[w.verbal.hedge_count if w.verbal else None for w in ws],
        marker_color=C["verbal"], marker_opacity=0.6, name="count",
    ))
    _style(fig, "Hedge Word Count", "count")
    return add_cursor(fig, ct)


@callback(Output("v-sent-len", "figure"), Input("fused-data", "data"), Input("current-time", "data"))
def v_sent_len(data, ct):
    if not data: return empty_fig("Sentence Length")
    ws, t = _parse(data)
    fig = go.Figure(go.Scatter(
        x=t, y=[w.verbal.mean_sentence_length if w.verbal else None for w in ws],
        mode="lines+markers", line=dict(color=C["verbal"], width=1.5),
        marker=dict(size=3), name="words",
    ))
    _style(fig, "Mean Sentence Length", "words")
    return add_cursor(fig, ct)


@callback(Output("v-confidence", "figure"), Input("fused-data", "data"), Input("current-time", "data"))
def v_confidence(data, ct):
    if not data: return empty_fig("Word Confidence")
    ws, t = _parse(data)
    fig = go.Figure(go.Scatter(
        x=t, y=[w.verbal.mean_word_confidence if w.verbal else None for w in ws],
        mode="lines", line=dict(color=C["verbal"], width=2),
        fill="tozeroy", fillcolor="rgba(27,79,138,0.08)", name="conf",
    ))
    fig.add_hline(y=0.8, line=dict(color=C["muted"], width=1, dash="dash"))
    _style(fig, "ASR Word Confidence", "0–1")
    return add_cursor(fig, ct)

# ─────────────────────────────────────────────────────────────────────────────
# Chart callbacks — Camera
# ─────────────────────────────────────────────────────────────────────────────

SHOT_COLOURS = {
    "close_up": C["gesture"], "medium": C["camera"],
    "wide": C["verbal"], "cutaway": C["prosody"], "unknown": C["muted"],
}

@callback(Output("c-shot", "figure"), Input("fused-data", "data"), Input("current-time", "data"))
def c_shot(data, ct):
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
    return add_cursor(fig, ct)


@callback(Output("c-cutrate", "figure"), Input("fused-data", "data"), Input("current-time", "data"))
def c_cutrate(data, ct):
    if not data: return empty_fig("Cut Rate")
    ws, t = _parse(data)
    fig = go.Figure(go.Bar(
        x=t, y=[w.camera.cut_rate if w.camera else None for w in ws],
        marker_color=C["camera"], marker_opacity=0.7, name="per min",
    ))
    _style(fig, "Scene Cut Rate", "cuts/min")
    return add_cursor(fig, ct)


@callback(Output("c-facearea", "figure"), Input("fused-data", "data"), Input("current-time", "data"))
def c_facearea(data, ct):
    if not data: return empty_fig("Face Area")
    ws, t = _parse(data)
    vals = [w.camera.mean_face_bbox_area * 100 if w.camera and w.camera.mean_face_bbox_area else None for w in ws]
    fig = go.Figure(go.Scatter(
        x=t, y=vals, mode="lines",
        line=dict(color=C["camera"], width=2),
        fill="tozeroy", fillcolor="rgba(123,94,167,0.08)", name="%",
    ))
    _style(fig, "Mean Face Area (zoom proxy)", "% of frame")
    return add_cursor(fig, ct)


@callback(Output("c-trend", "figure"), Input("fused-data", "data"), Input("current-time", "data"))
def c_trend(data, ct):
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
    return add_cursor(fig, ct)

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