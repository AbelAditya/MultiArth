# MultiArth Wiki

MultiArth (package `multiarth`, repo directory `mannerism_analyzer`) runs four
independent analysis workers over a video, coordinated by an
[`Orchestrator`](../core/orchestrator.py) and merged by a
[`FusionEngine`](../core/fusion_engine.py). Each worker owns one modality and
writes its output to a shared Redis [`FeatureStore`](../core/feature_store.py).

| Worker | Modality | Dashboard label | Wiki page |
|---|---|---|---|
| [`GestureWorker`](../workers/gesture_worker.py) | Body/hand pose | Pose Estimation | [Gesture-Worker.md](Gesture-Worker.md) |
| [`ProsodyWorker`](../workers/prosody_worker.py) | Pitch, intensity, spectrogram | Acoustic Properties | [Acoustic-Prosody-Worker.md](Acoustic-Prosody-Worker.md) |
| [`VerbalWorker`](../workers/verbal_worker.py) | Transcript, lexical stats | Verbal Language | [Verbal-Worker.md](Verbal-Worker.md) |
| [`CameraWorker`](../workers/camera_worker.py) | Scene cuts, framing | Camera | [Camera-Worker.md](Camera-Worker.md) |

> Naming note: the `ProsodyWorker` module/class/Redis-key names were kept as
> `prosody` when the dashboard label was rebranded to **"Acoustic"** — see the
> main [README](../README.md) for the full history of the **MultiArth**
> rebrand.

## Full dependency list

All package versions are pinned in [`pyproject.toml`](../pyproject.toml).

| Package | Used by | Docs |
|---|---|---|
| [mediapipe](https://pypi.org/project/mediapipe/) | Gesture | https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker |
| [opencv-python](https://pypi.org/project/opencv-python/) | Gesture, Camera | https://docs.opencv.org/4.x/ |
| [praat-parselmouth](https://pypi.org/project/praat-parselmouth/) | Acoustic/Prosody | https://parselmouth.readthedocs.io/en/stable/ |
| [soundfile](https://pypi.org/project/soundfile/) | Acoustic/Prosody, preprocessing | https://python-soundfile.readthedocs.io/en/latest/ |
| [faster-whisper](https://pypi.org/project/faster-whisper/) | Verbal | https://github.com/SYSTRAN/faster-whisper#readme |
| [spacy](https://pypi.org/project/spacy/) | Verbal | https://spacy.io/api |
| [scenedetect](https://pypi.org/project/scenedetect/) | Camera | https://www.scenedetect.com/docs/latest/api.html |
| [redis](https://pypi.org/project/redis/) | Feature store (all workers) | https://redis-py.readthedocs.io/en/stable/ |
| [numpy](https://pypi.org/project/numpy/) | Gesture, Acoustic/Prosody, Camera | https://numpy.org/doc/stable/ |
| [pandas](https://pypi.org/project/pandas/) | Fusion / dashboard | https://pandas.pydata.org/docs/ |
| [scipy](https://pypi.org/project/scipy/) | Fusion / signal helpers | https://docs.scipy.org/doc/scipy/ |
| [dash](https://pypi.org/project/dash/) | Dashboard | https://dash.plotly.com/ |
| [plotly](https://pypi.org/project/plotly/) | Dashboard | https://plotly.com/python/ |
| [dash-bootstrap-components](https://pypi.org/project/dash-bootstrap-components/) | Dashboard | https://dash-bootstrap-components.opensource.faculty.ai/ |
| [pydantic](https://pypi.org/project/pydantic/) | Shared data models | https://docs.pydantic.dev/latest/ |
| [click](https://pypi.org/project/click/) | CLI | https://click.palletsprojects.com/ |
| [loguru](https://pypi.org/project/loguru/) | Logging (all workers) | https://loguru.readthedocs.io/en/stable/ |
| [tqdm](https://pypi.org/project/tqdm/) | Progress indicators | https://tqdm.github.io/ |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Config loading | https://saurabh-kumar.com/python-dotenv/ |
| [gunicorn](https://pypi.org/project/gunicorn/) | Docker/production dashboard server | https://docs.gunicorn.org/en/stable/ |
</content>
