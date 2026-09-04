"""
scripts/export_yolo11n_seg.py
------------------------------
One-off export of Ultralytics' YOLO11n-seg weights to ONNX, producing the
`models/yolo11n-seg.onnx` that workers/_detector.py loads at runtime.

Why export rather than download a ready-made .onnx: Ultralytics publishes
only `.pt` weights (confirmed — the `.onnx` path in their HuggingFace repo
404s). Pre-exported ONNX files do exist in third-party HuggingFace repos,
but for a research tool whose outputs end up in a thesis, a model file
nobody can trace back to an official release is a provenance problem, and
those repos can vanish or change contents silently. Exporting from the
official `.pt` ourselves is reproducible and pinned.

Why this isn't part of the runtime path: `ultralytics` is a heavy
dependency (it pulls torchvision, polars, matplotlib and more) and its
license is AGPL-3.0. Runtime needs neither — workers/_detector.py runs the
exported graph through `onnxruntime`, which this project already depends
on. So ultralytics is an *export-time only* tool: installed to run this
script, absent from the runtime image. See the Dockerfile's builder stage,
which runs this same script so the image doesn't ship ultralytics either.

    uv pip install ultralytics onnx onnxslim
    python scripts/export_yolo11n_seg.py

Note that `uv sync` prunes anything not in uv.lock, so an ultralytics
installed ad-hoc for this will be removed by the next sync. That's fine
and expected — the export only needs to happen when the pinned model
version changes, and the .onnx it produces is what matters afterwards.
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _ROOT / "models"

# Pinned to a specific Ultralytics assets release rather than "latest" so
# re-running this doesn't silently swap in different weights than the ones
# every threshold in this project was calibrated against.
_PT_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt"
_PT_PATH = _MODELS_DIR / "yolo11n-seg.pt"
_ONNX_PATH = _MODELS_DIR / "yolo11n-seg.onnx"

# Must match workers/_detector.py's _INPUT_SIZE — the exported graph has a
# fixed input resolution baked in, so these two cannot drift apart.
_IMGSZ = 640


def main() -> int:
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not _PT_PATH.exists():
        print(f"Downloading {_PT_URL} -> {_PT_PATH}")
        urllib.request.urlretrieve(_PT_URL, str(_PT_PATH))
    else:
        print(f"Using existing {_PT_PATH}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ultralytics is not installed — it is an export-time-only "
            "dependency (see this file's docstring):\n"
            "    uv pip install ultralytics onnx onnxslim",
            file=sys.stderr,
        )
        return 1

    model = YOLO(str(_PT_PATH))
    # opset 12 is comfortably within onnxruntime 1.28's supported range and
    # avoids newer-opset operators that some ORT builds lack kernels for.
    # simplify=True folds the export's constant subgraphs, which measurably
    # shrinks first-inference latency.
    out = model.export(format="onnx", imgsz=_IMGSZ, opset=12, simplify=True)

    produced = Path(out)
    if produced.resolve() != _ONNX_PATH.resolve():
        shutil.move(str(produced), str(_ONNX_PATH))

    # The .pt is only an export input; keeping it would leave a second,
    # unused copy of the weights in the image/models dir.
    _PT_PATH.unlink(missing_ok=True)

    print(f"\nWrote {_ONNX_PATH} ({_ONNX_PATH.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
