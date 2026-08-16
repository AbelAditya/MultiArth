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

## Guides

| Guide | Covers |
|---|---|
| [Bulk-Upload.md](Bulk-Upload.md) | Using the dashboard's Bulk Upload tab to process a manifest of videos |

> Naming note: the `ProsodyWorker` module/class/Redis-key names were kept as
> `prosody` when the dashboard label was rebranded to **"Acoustic"** — see the
> main [README](../README.md) for the full history of the **MultiArth**
> rebrand.

## Remote workers (optional)

Two workers can offload their heaviest model to a free Colab GPU instead of
running it on the machine hosting this project — see each worker's own wiki
page ("Remote MeTRAbs, local fallback" / the SenseVoice remote section) for
the full design:

| Notebook | Hosts | Env vars |
|---|---|---|
| [`colab/gesture_server.ipynb`](../colab/gesture_server.ipynb) | MeTRAbs (pose) | `GESTURE_REMOTE_URL`, `GESTURE_API_KEY` |
| [`colab/sensevoice_server.ipynb`](../colab/sensevoice_server.ipynb) | SenseVoice (Chinese ASR) | `SENSEVOICE_REMOTE_URL`, `SENSEVOICE_API_KEY` |

## Full dependency list

All package versions are pinned in [`pyproject.toml`](../pyproject.toml).

| Package | Used by | Docs |
|---|---|---|
| [tensorflow](https://pypi.org/project/tensorflow/) | Gesture (MeTRAbs) | https://www.tensorflow.org/api_docs |
| [fastapi](https://pypi.org/project/fastapi/) / [uvicorn](https://pypi.org/project/uvicorn/) | Gesture (`gesture_server.py`, bulk runs) | https://fastapi.tiangolo.com/ · https://www.uvicorn.org/ |
| [opencv-contrib-python](https://pypi.org/project/opencv-contrib-python/) | Gesture, Camera | https://docs.opencv.org/4.x/ |
| [praat-parselmouth](https://pypi.org/project/praat-parselmouth/) | Acoustic/Prosody | https://parselmouth.readthedocs.io/en/stable/ |
| [soundfile](https://pypi.org/project/soundfile/) | Acoustic/Prosody, preprocessing | https://python-soundfile.readthedocs.io/en/latest/ |
| [faster-whisper](https://pypi.org/project/faster-whisper/) | Verbal | https://github.com/SYSTRAN/faster-whisper#readme |
| [funasr](https://pypi.org/project/funasr/) | Verbal (SenseVoice, Chinese) | https://github.com/modelscope/FunASR#readme |
| [torch](https://pypi.org/project/torch/) / [torchaudio](https://pypi.org/project/torchaudio/) | Verbal (funasr/SenseVoice backend) | https://pytorch.org/docs/stable/index.html |
| [modelscope](https://pypi.org/project/modelscope/) | Verbal (SenseVoice model hub) | https://www.modelscope.cn/docs |
| [requests](https://pypi.org/project/requests/) | Gesture + Verbal remote HTTP clients | https://requests.readthedocs.io/en/latest/ |
| [spacy](https://pypi.org/project/spacy/) | Verbal | https://spacy.io/api |
| [scenedetect](https://pypi.org/project/scenedetect/) | Camera | https://www.scenedetect.com/docs/latest/api.html |
| [redis](https://pypi.org/project/redis/) | Feature store (all workers) | https://redis-py.readthedocs.io/en/stable/ |
| [pymongo](https://pypi.org/project/pymongo/) | Durable results store (Browse Corpus, bulk runs) | https://pymongo.readthedocs.io/en/stable/ |
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
