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

## Values shown on the dashboard

The **Acoustic Properties** section has one KPI card and four charts:

| Where | What it shows |
|---|---|
| KPI card: **Mean F0** | Average pitch across the whole video, in Hz. |
| **Spectrogram** chart | A visual "fingerprint" of the audio's frequency content over time — brighter means more energy at that frequency at that moment — with the pitch (F0) contour drawn on top. |
| **Waveform** chart | The raw amplitude of the audio over time — the up-and-down shape most people picture when they think "audio waveform." |
| **F0** (pitch) chart | Average pitch per time window, with a shaded band showing how much it varies within that window — a rough proxy for vocal expressiveness (a flat, narrow band reads as monotone; a wide, moving band reads as more animated). |
| **Intensity** chart | Loudness (in dB) per time window. |

`f0_range` (max − min pitch per window) and `speech_rate_syl_per_s` (filled
in later by `FusionEngine` once transcript timing is available) are
computed but aren't currently charted anywhere in the dashboard UI.

## Implementation notes

- Segment extraction uses `parselmouth.WindowShape.RECTANGULAR` with
  `preserve_times=True` so window timestamps line up exactly with the parent
  clip.
- F0 samples of `0` from Praat mean "unvoiced" and are excluded from mean/
  range/std calculations, not treated as `0 Hz`.
- Spectrogram/waveform computation failures are caught independently so a
  failure in one doesn't prevent the other (or per-window features) from
  being stored.

## Benchmark accuracy (published)

Praat's pitch tracker (an autocorrelation method) doesn't have a single
universally-cited "accuracy" figure the way a benchmark-trained model
does — F0-extraction accuracy studies vary in which error metric they
report (gross pitch error, frame F0 error, cents accuracy, voicing-decision
F1) and which algorithms they compare against, and results shift with
recording conditions. The general, consistent picture across the
literature checked:

- A recent (2025) comparative study under **clean** recording conditions
  found Praat's autocorrelation method achieved the **highest F1 score
  (93.63%)** among the algorithms tested for voicing decisions, competitive
  with newer methods on raw accuracy — but noted it "makes more octave
  errors" (mistaking a pitch for double/half its true value) than some
  alternatives; a newer algorithm in the same study (SwiftF0) reported
  slightly higher cents-accuracy (94.98%) and gross-error accuracy (97.04%).
- Older comparative work found Praat's autocorrelation method performing
  **slightly worse than YIN and RAPT** (two other established pitch-
  tracking algorithms) in some evaluations, while a separate, more recent
  evaluation found it among the better performers in its test set (~5.05%
  average Frame F0 Error across 20 speakers).

Net picture: Praat's F0 tracker is a well-established, competitive (not
clearly best-in-class) choice under clean speech conditions — reasonable
for this project's use case (windowed pitch/intensity statistics over
whole video clips, not fine-grained per-sample F0 curves used for anything
downstream-sensitive to occasional octave errors).

Sources: ["SwiftF0: Fast and Accurate Monophonic Pitch
Detection"](https://arxiv.org/pdf/2508.18440) (2025 comparative benchmark);
["Today's Most Frequently Used F0 Estimation Methods, and Their Accuracy in
Estimating Male and Female Pitch in Clean
Speech"](https://www.researchgate.net/publication/307888988) (YIN/RAPT
comparison).

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
