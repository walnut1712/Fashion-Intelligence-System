# Fashion Intelligence System

RMIT COSC2753 Assignment 2. Four models over a catalogue of ~38,600 low-resolution
fashion product photographs, plus a FastAPI + vanilla-JS application that answers all
four questions from a single uploaded image.

| Task | Question | Notebook | Headline |
|---|---|---|---|
| 1 | What type of item is this? (`articleType`, 92 classes) | `notebooks/02_task1_item_type.ipynb` | 87.1 weighted-F1 |
| 2 | Which season is it for? (4 classes) | `notebooks/03_task2_season_pytorch.ipynb` | 67.5% accuracy |
| 3 | Who is it for, and for what occasion? (`gender` x `usage`) | `notebooks/04_task3_cnn_architectures.ipynb` | 90.1% / 91.2% |
| 4 | Which catalogue items look like this? (top-K retrieval) | `notebooks/05`–`07` | P@10 80.2 |

The notebooks are the deliverable. `src/`, `app/` and `artifacts/` exist so the trained
models can be reused and served without re-training.

## Setup

Python 3.13 on a plain python.org interpreter or a virtual environment — **not Anaconda**.
Anaconda's MKL and PyTorch both ship `libiomp5md.dll`, and loading both crashes the kernel
with `OMP: Error #15`. If a notebook dies on `import torch`, check the interpreter first.

```bash
python -m venv .venv
.venv/Scripts/activate                 # Windows;  source .venv/bin/activate elsewhere

# torch is not on PyPI - match the index to your driver (see `nvidia-smi`)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r requirements.txt           # notebooks
python -m pip install -r requirements-backend.txt   # API only
```

## Data

The dataset is not in this repository. Unpack it so the tree looks like this:

```
A2_FashionDataset/
  FashionDataset/
    train/{images_train, styles_train.csv}
    test/{images_test, styles_prediction_template.csv}
  processed/          # written by 01_eda.ipynb
  input_images/       # 31 real-world photos used for out-of-domain evaluation
```

Two caches under `processed/` are large and are rebuilt on first use rather than shipped:
`image_cache_task1_60x80.npy` and `search_cache_60x80.npy` (~555 MB each). Expect several
minutes of disk-bound work the first time notebooks 02 or 05 run.

## Running the notebooks

They are run in **VS Code**, not the Jupyter web UI, so `Path.cwd()` is the project root
and paths are anchored on an explicit `PROJECT_DIR`. Run them in order — `01` produces the
cleaned metadata every other notebook reads.

```
01_eda.ipynb                          shared cleaning, splits, prediction metadata
02_task1_item_type.ipynb              Task 1
03_task2_season_pytorch.ipynb         Task 2
04_task3_cnn_architectures.ipynb      Task 3
05_task4_visual_search.ipynb          Task 4 - method comparison and encoder selection
06_task4_background_augmentation.ipynb Task 4 - robustness to real photographs
07_task4_clustering.ipynb             Task 4 - structure of the embedding space
08_ultimate_judgement.ipynb           cross-task judgement and deployment policy
```

`08` reads only the artefacts the other notebooks wrote - no models are loaded - so it runs in
seconds and can be re-run any time the other numbers change.

Notebook `06` reads the clean encoder produced by `05` and writes the deployed one, so run
them in that order. `07` clusters whatever index is current, so re-run it after `06`
promotes a new encoder.

## The application

```bash
python -m uvicorn app.backend.main:app --reload      # http://127.0.0.1:8000
```

`main.py` mounts `app/frontend` at `/`, so the UI, the docs (`/docs`) and the API share one
process. Upload an image and all four models answer it.

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | all four tasks on one upload |
| `POST /api/task4/search` | retrieval only, with `?k=` and `?mode=` |
| `POST /api/task4/regions` | per-garment retrieval: one result group per proposed region |
| `GET /api/health` | per-task `loaded` / `error`, plus live model cards |
| `GET /api/catalogue/{id}/image` | a catalogue thumbnail |

`?mode=` selects how an upload is coerced to 60x80: `letterbox` (pad to aspect), `crop`
(centre crop), or `nobg` (segment the subject onto white — the default, and the only one
that survives a cluttered photograph).

If the API is unreachable the frontend falls back to synthetic demo data and shows a
banner. A plausible-looking result grid is therefore not proof the backend is up — check
`/api/health`.

## Batch outputs

```bash
# Task 1 predictions over the test set, and the graded four-column submission
python predict.py --images A2_FashionDataset/FashionDataset/test/images_test \
                  --out outputs/task1_item_type_predictions.csv --submission
python scripts/build_submission.py

# Task 4's own deliverable: top-K retrieval + cluster assignment for all 5,829 test images
python scripts/build_task4_outputs.py
```

Task 4 contributes nothing to the graded classification CSV — the assignment defines no
submission format for retrieval — so its evidence is produced deliberately by that last
script rather than as a side effect.

## Tests

```bash
python -m pytest tests/ -q
```

They mainly guard that the shipped checkpoints still load into the shipped architectures.
That is not hypothetical: a duplicated network definition once left a trained checkpoint
the API could not load, and a background-augmented encoder was mistaken for the clean
baseline for three rounds of experiments. Both now fail a test instead of a report.

`tests/test_prediction.py` runs full inference and takes several minutes; skip it with
`--ignore=tests/test_prediction.py` during development.

## A note on pretrained components

The assignment forbids "pre-trained systems which are trained on other datasets". Every model
that produces a prediction here is trained from scratch on the supplied data only - the four
task models and the Task 4 encoder included.

One optional dependency deserves stating plainly. Upload ingestion (`src/data/user_image.py`)
segments the subject with a tiered ladder, and its highest tier uses **rembg (u2netp)**, which
is a pretrained matting network. It is not installed by either requirements file and it is not
used to produce any reported result: across all 5,829 test images the segmentation tier was
`border-model` or a decline to centre-crop, never rembg (see `ingest_method` in
`outputs/task4_test_retrieval.csv`). The ladder degrades to GrabCut and then to a numpy border
colour model when rembg is absent.

If you install it for convenience, the graded run should be repeated without it so that no
reported number depends on a network trained elsewhere.

## Repository layout

```
notebooks/     the deliverable
src/
  data/        upload ingestion, synthetic backgrounds and compositing
  models/      architectures the services import
  features/    hand-built descriptors and embedding fusion (Task 4)
  evaluation/  metrics, retrieval protocol, OOD benchmark
  visual_search/  search and cluster engines
app/backend/   FastAPI service, one module per task
app/frontend/  vanilla JS, no build step
artifacts/     trained checkpoints and indexes, per task
outputs/       predictions and evaluation tables
scripts/       submission and artefact builders
tests/
```
