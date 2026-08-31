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
import os
import sys

import click
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

load_dotenv()

from core.logging_setup import setup_file_logging

setup_file_logging("cli")

from core.bulk_orchestrator import BulkOrchestrator, load_manifest
from core.feature_store import FeatureStore
from core.models import JobStatus
from core.orchestrator import Orchestrator
from core.results_repository import ResultsRepository


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
@click.option("--whisper-model", default="small", show_default=True, help="Whisper model size")
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


def _make_bulk_progress_reporter():
    """
    Renders BulkOrchestrator's progress events as two tqdm bars: an outer
    one advancing once per video (skipped/succeeded/failed all count), and
    an inner single-line status showing live per-modality window counts
    while the current video is being processed. Returns (on_progress, close).
    """
    outer = None
    inner = tqdm(total=0, bar_format="{desc}", leave=False)

    def on_progress(evt: dict) -> None:
        nonlocal outer
        if outer is None:
            outer = tqdm(total=evt["total"], desc="Bulk processing", unit="video", position=0)

        index, total = evt["index"], evt["total"]
        name = evt.get("label") or os.path.basename(evt["path"])
        tag = f"[{evt['collection']}][{index}/{total}] {name}"
        etype = evt["type"]

        if etype == "start":
            inner.set_description_str(f"{tag}: starting…")
        elif etype == "tick":
            counts, total_windows = evt.get("counts"), evt.get("total_windows")
            if counts and total_windows:
                inner.set_description_str(
                    f"{tag}: gesture {counts['gesture']}/{total_windows} · "
                    f"prosody {counts['prosody']}/{total_windows} · "
                    f"verbal {counts['verbal']}/{total_windows} · "
                    f"camera {counts['camera']}/{total_windows}"
                )
            else:
                inner.set_description_str(f"{tag}: processing…")
        elif etype == "skip":
            tqdm.write(f"  − {tag}: already shipped, skipped")
            outer.update(1)
        elif etype == "done":
            symbol = "✓" if evt["result"] == "succeeded" else "✗"
            colour = "green" if evt["result"] == "succeeded" else "red"
            msg = f"  {symbol} {tag}: {evt['result']}"
            if evt.get("error"):
                msg += f" — {evt['error']}"
            tqdm.write(click.style(msg, fg=colour))
            outer.update(1)

    def close():
        inner.close()
        if outer:
            outer.close()

    return on_progress, close


@main.command()
@click.argument("manifest_path")
@click.option("--force", is_flag=True, default=False, help="Reprocess even if already shipped to Mongo")
@click.option("--mongo-uri", default=None, help="MongoDB Atlas connection string (default: $MONGO_URI)")
@click.option("--mongo-db", default=None, help="MongoDB database name (default: $MONGO_DB or 'multiarth')")
@click.option("--window", default=5.0, show_default=True, help="Window size in seconds")
@click.option("--whisper-model", default="small", show_default=True, help="Whisper model size")
@click.option("--device", default="cpu", show_default=True, help="cpu or cuda")
@click.option("--work-dir", default="/tmp/mannerism", show_default=True)
@click.pass_context
def bulk(ctx, manifest_path, force, mongo_uri, mongo_db, window, whisper_model, device, work_dir):
    """
    Sequentially process every video listed in a manifest JSON file and ship
    each result to MongoDB. Manifest format: a JSON list of
    {"path": "...", "collection": "...", "drive_url": "...", "label": "..."}
    objects. "collection" names the corpus (e.g. "TedX", "Yixi") a video's
    results are shipped into — see core/results_repository.py.
    """
    store = ctx.obj["store"]
    try:
        repo = ResultsRepository(uri=mongo_uri, db_name=mongo_db)
    except Exception as exc:
        click.echo(click.style(f"Could not connect to MongoDB: {exc}", fg="red"), err=True)
        sys.exit(1)

    runner = BulkOrchestrator(
        feature_store=store,
        repo=repo,
        orchestrator_kwargs={
            "work_dir": work_dir,
            "window_size_s": window,
            "whisper_model": whisper_model,
            "whisper_device": device,
            # No persistent_gesture flag needed on this branch — gesture is
            # just another lazy in-process worker now (see
            # core/orchestrator.py), so BulkOrchestrator reusing one
            # Orchestrator instance for the whole manifest already keeps it
            # warm across every video, same as prosody/verbal/camera.
        },
    )
    on_progress, close_progress = _make_bulk_progress_reporter()
    try:
        entries = load_manifest(manifest_path)
        click.echo(f"Loaded {len(entries)} entries from {manifest_path}\n")
        summary = runner.run(entries, force=force, progress_cb=on_progress)
    finally:
        close_progress()
        repo.close()

    click.echo("")
    click.echo(click.style(f"✓ Succeeded: {len(summary['succeeded'])}", fg="green", bold=True))
    click.echo(click.style(f"− Skipped:   {len(summary['skipped'])}", fg="blue"))
    click.echo(click.style(f"✗ Failed:    {len(summary['failed'])}", fg="red"))
    for failure in summary["failed"]:
        click.echo(f"    {failure['path']}: {failure['error']}")

    if summary["failed"]:
        sys.exit(1)


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
