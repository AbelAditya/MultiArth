# Acoustic (Prosody) Worker

Source: [`workers/prosody_worker.py`](../workers/prosody_worker.py)
Dashboard section: **Acoustic Properties**
Feature model: [`ProsodyFeatures`](../core/models.py) (in `core/models.py`)

> Naming note: this worker's module, class (`ProsodyWorker`), feature model
> (`ProsodyFeatures`) and Redis keys (`job:{id}:prosody:{w}`) all still use
> the original **"prosody"** name. Only the dashboard-facing label was
> renamed to **"Acoustic"** — see [Home](Home.md) and the main
> [README](../README.md) for the full rebrand history.

## What it does

`ProsodyWorker` loads the extracted audio once per job with **parselmouth**
(the Python binding for **Praat**) and produces both per-window features and
two full-clip visual aids:

1. **Per-window features** (`_process_window`) — for each time window it
   extracts a Praat `Sound` segment and computes:
   - `mean_f0` / `f0_range` / `f0_std` — pitch (F0) statistics over voiced
     frames only (pitch floor 75 Hz, ceiling 500 Hz).
   - `mean_intensity_db` / `intensity_range_db` — loudness statistics.
   - `speech_rate_syl_per_s` — left `None` here; filled in later by the
     `FusionEngine` once ASR word timings are available.
2. **Spectrogram** (`_compute_spectrogram`) — a downsampled narrow-band
   spectrogram (≤1000 time steps × ≤200 frequency bins, clipped to a 60 dB
   range around the 99th percentile) plus an aligned F0 contour, stored once
   for the whole clip and rendered by the dashboard's spectrogram chart.
3. **Waveform** (`_compute_waveform`) — the raw audio downsampled to ~4000
   points and peak-normalised to `[-1, 1]`, used for the amplitude-vs-time
   plot.

All three outputs are written to the [`FeatureStore`](../core/feature_store.py)
(`put_prosody`, `put_spectrogram`, `put_waveform`).

## Implementation notes

- Segment extraction uses `parselmouth.WindowShape.RECTANGULAR` with
  `preserve_times=True` so window timestamps line up exactly with the parent
  clip.
- F0 samples of `0` from Praat mean "unvoiced" and are excluded from mean/
  range/std calculations, not treated as `0 Hz`.
- Spectrogram/waveform computation failures are caught independently so a
  failure in one doesn't prevent the other (or per-window features) from
  being stored.

## Package documentation

| Package | Role | Docs |
|---|---|---|
| praat-parselmouth | Pitch, intensity, spectrogram extraction (Praat bindings) | https://parselmouth.readthedocs.io/en/stable/ |
| Praat (underlying algorithms) | Acoustic analysis theory/manual | https://www.fon.hum.uva.nl/praat/manual/Intro.html |
| soundfile | Audio I/O used during preprocessing/extraction | https://python-soundfile.readthedocs.io/en/latest/ |
| NumPy | Percentile clipping, log/dB conversion, downsampling | https://numpy.org/doc/stable/ |
| Pydantic | `ProsodyFeatures` / `PitchFrame` models | https://docs.pydantic.dev/latest/ |
| loguru | Per-window/job logging | https://loguru.readthedocs.io/en/stable/ |

See also [Home](Home.md) for the full dependency list.
</content>
