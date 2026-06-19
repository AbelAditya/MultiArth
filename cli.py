"""
cli.py
------
Command-line interface for the mannerism analyzer.

Usage:
    # Run full analysis
    uv run analyze run path/to/talk.mp4

    # Check job status
    uv run analyze status <job_id>

    # Export fused results to JSON
    uv run analyze export <job_id> --out results.json

    # Launch dashboard
    uv run analyze dashboard
"""

from __future__ import annotations

import json
import sys

import click
from loguru import logger

from core.feature_store import FeatureStore
from core.models import JobStatus
from core.orchestrator import Orchestrator


@click.group()
@click.option("--redis-host", default="localhost", show_default=True)
@click.option("--redis-port", default=6379, show_default=True)
@click.pass_context
def main(ctx, redis_host, redis_port):
    """Multimodal linguistic mannerism analyzer."""
    ctx.ensure_object(dict)
    ctx.obj["store"] = FeatureStore(host=redis_host, port=redis_port)


@main.command()
@click.argument("video_path")
@click.option("--window", default=5.0, show_default=True, help="Window size in seconds")
@click.option("--whisper-model", default="base", show_default=True, help="Whisper model size")
@click.option("--device", default="cpu", show_default=True, help="cpu or cuda")
@click.option("--sequential", is_flag=True, default=False, help="Disable parallel workers")
@click.option("--work-dir", default="/tmp/mannerism", show_default=True)
@click.pass_context
def run(ctx, video_path, window, whisper_model, device, sequential, work_dir):
    """Analyze a video file."""
    store = ctx.obj["store"]
    orchestrator = Orchestrator(
        store=store,
        work_dir=work_dir,
        window_size_s=window,
        whisper_model=whisper_model,
        whisper_device=device,
        parallel=not sequential,
    )
    try:
        job_id = orchestrator.analyze(video_path)
        click.echo(f"\n✓ Analysis complete. Job ID: {click.style(job_id, fg='cyan', bold=True)}")
        click.echo(f"  View results: uv run analyze dashboard  (then enter job ID: {job_id})")
    except Exception as exc:
        logger.error(f"Analysis failed: {exc}")
        sys.exit(1)
    finally:
        orchestrator.close()


@main.command()
@click.argument("job_id")
@click.pass_context
def status(ctx, job_id):
    """Check the status of an analysis job."""
    store = ctx.obj["store"]
    s = store.get_status(job_id)
    if s is None:
        click.echo(f"Job '{job_id}' not found.")
        sys.exit(1)

    colour = {
        JobStatus.DONE: "green",
        JobStatus.RUNNING: "yellow",
        JobStatus.FAILED: "red",
        JobStatus.PENDING: "blue",
    }.get(s, "white")

    click.echo(f"Job {job_id}: {click.style(s.value.upper(), fg=colour, bold=True)}")

    events = store.read_events(job_id)
    if events:
        click.echo(f"\nLast {min(10, len(events))} events:")
        for e in events[-10:]:
            click.echo(f"  [{e.get('worker','?'):10s}] {e.get('msg','')}")


@main.command()
@click.argument("job_id")
@click.option("--out", default=None, help="Output JSON path (default: stdout)")
@click.pass_context
def export(ctx, job_id, out):
    """Export fused results for a job to JSON."""
    store = ctx.obj["store"]
    windows = store.get_all_fused(job_id)
    if not windows:
        click.echo(f"No fused data found for job '{job_id}'.")
        sys.exit(1)

    payload = [w.model_dump() for w in windows]
    text = json.dumps(payload, indent=2, default=str)

    if out:
        with open(out, "w") as f:
            f.write(text)
        click.echo(f"Exported {len(windows)} windows to {out}")
    else:
        click.echo(text)


@main.command()
@click.option("--port", default=8050, show_default=True)
def dashboard(port):
    """Launch the Dash visualisation dashboard."""
    from dashboard.app import app
    click.echo(f"Dashboard running at http://localhost:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
