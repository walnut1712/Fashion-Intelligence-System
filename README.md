# Fashion Intelligence System

RMIT COSC2753 Assignment 2. Four models over a catalogue of ~38,600 low-resolution
fashion product photographs, plus a FastAPI + vanilla-JS application that answers all
four questions from a single uploaded image.

| Task | Question | Notebook | Catalogue tiles | Real backgrounds |
|---|---|---|---|---|
| 1 | What type of item is this? (`articleType`, 92 classes) | `notebooks/02_task1_item_type.ipynb` | 87.14 weighted-F1 | 57.96 |
| 2 | Which season is it for? (4 classes) | `notebooks/03_task2_season.ipynb` | 63.06 macro-F1 | 31.26 |
| 3 | Who is it for, and for what occasion? (`gender` x `usage`) | `notebooks/04_task3_gender_usage.ipynb` | 77.23 / 82.67 macro-F1 | 46.36 / 64.12 |
| 4 | Which catalogue items look like this? (top-K retrieval) | `notebooks/05`-`07` | P@10 76.19 | P@10 55.78 |

Two columns rather than one, because a single catalogue number is the most misleading
thing this project could report. Every training photograph is one garment, centred, on
white, and a model trained only on that does not degrade gracefully when the background
changes - it collapses. Task 1's catalogue-only checkpoint scores 89.68 weighted-F1 on
tiles and **16.84** through held-out photographic scenes.

**Tasks 1 and 3 therefore ship background-adapted models**, trained with photographic
backdrops composited behind the garment. Task 1 pays for it: weighted-F1 89.68 -> 87.14
and macro-F1 73.11 -> 63.88, the second being the larger loss and concentrated in the
rare tail. Task 3 pays nothing measurable - it improves on both axes. Task 2 keeps its
baseline, where the trade is only about 2.6:1 on the project's weakest model. Task 4
ships the mixed-backdrop encoder (arm D). The right-hand column is what the choice buys;
the left is what it costs. Both are measured, and the per-task notebooks carry the
matched controls that attribute the gain to the backdrops rather than to extra training.

Both columns of a row use the **same metric**, which matters most for Task 3. Its accuracy
would read 90.20% / 90.33% on tiles and 77.4% / 83.9% on backgrounds, which looks like a
model that barely degrades; `Casual` is 77.3% of the usage split, so a model collapsing
toward the majority class still scores well on accuracy. Macro-F1 on the identical frames
is the row above. Never quote Task 3 robustness as accuracy.

The real-background column is measured on **composited** frames and is an upper bound:
on 23 genuine photographs Task 4 scores 16.52 where its composited benchmark says 55.78.

The notebooks are the deliverable. `src/`, `app/` and `artifacts/` exist so the trained
models can be reused and served without re-training.

## Setup

Python 3.13 on a plain python.org interpreter or a virtual environment - **not Anaconda**.
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
and paths are anchored on an explicit `PROJECT_DIR`. Run them in order - `01` produces the
cleaned metadata every other notebook reads.

```
01_eda.ipynb                          shared cleaning, splits, prediction metadata
02_task1_item_type.ipynb              Task 1
03_task2_season.ipynb         Task 2
04_task3_gender_usage.ipynb      Task 3
05_task4_visual_search.ipynb        Task 4 - the encoder and the background comparison
07_ultimate_judgement.ipynb           cross-task judgement and deployment policy
```

Task 4 is one notebook holding two models: the same encoder trained on the catalogue as it
ships, and trained again with photographic backdrops behind the garments.
`06_task4_clustering.ipynb` was removed; the numbering keeps a gap where it was.

`07` reads only the artefacts the other notebooks wrote - no models are loaded - so it runs in
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
(centre crop), or `nobg` (segment the subject onto white - the default, and the only one
that survives a cluttered photograph).

If the API is unreachable the frontend falls back to synthetic demo data and shows a
banner. A plausible-looking result grid is therefore not proof the backend is up - check
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

Task 4 contributes nothing to the graded classification CSV - the assignment defines no
submission format for retrieval - so its evidence is produced deliberately by that last
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
is a pretrained matting network. The ladder degrades to GrabCut and then to a numpy border
colour model when rembg is absent.

**As of 2026-09-11 rembg is installed and is listed in both requirements files.** What it is and
is not used for matters, so both halves are stated here.

It is **not** used for anything graded. The graded deliverable is
`styles_prediction_template.csv`, 5,829 catalogue tiles. Task 1 reaches them through
`predict.py`, whose `--ingest` default is `squash`, a plain resize that never calls
`foreground_mask`. Task 2 and Task 3 reach them through their services, which route on
`looks_like_catalogue`; every one of the 5,829 tiles takes the catalogue branch, which is
`letterbox` and likewise never segments. That is asserted rather than assumed - see
`tests/test_graded_path_never_segments.py`, which walks all 5,829 and fails if any one of them
routes to the photograph branch.

That test exists because the property did not hold when rembg was installed. Four tiles
(52166, 56624, 59593, 59606) are 53x80 or 54x80 rather than 60x80, and `looks_like_catalogue`
applied its aspect-ratio gate before its size test, so those four were classified as
photographs and segmented. The size test is now absolute (a dataset tile is at most 100px on
its long side, against the smallest real upload's 225px) and runs first, so the aspect gate only
ever sees images too large to be tiles.

It **is** used for one reported result, and this is a genuine qualification on it: the
real-photograph tables in notebooks 02, 05 and 07, measured on the 31 personal photographs in
`A2_FashionDataset/input_images/`. Those are an out-of-domain analysis of the deployed models,
not a graded prediction and not a training signal - no model was trained, fine-tuned or selected
using rembg. The figures there were remeasured on 2026-09-11 and the earlier GrabCut values are
recorded beside them in `CLAUDE.md`.

To reproduce with no pretrained component present at all, uninstall rembg: the ladder falls back
to GrabCut, every graded number is bit-for-bit identical because the graded path never reaches
the ladder, and the real-photograph tables return to their GrabCut values, which are less stable
(Task 1's 7-repeat spread goes from 0.00 back to +/-1.64) but depend on nothing trained
elsewhere.

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
