# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

RMIT **COSC2753 Assignment 2 — Fashion Intelligence System**. Four modelling tasks over a
catalogue of ~38k low-resolution fashion product photos, plus a FastAPI + vanilla-JS app that
serves all four models from one uploaded image.

The **notebooks are the deliverable**; `src/`, `app/` and `artifacts/` exist so the trained
models can be reused outside the notebook without re-training.

| Task | Question | Notebook | Artifacts |
|---|---|---|---|
| 1 | `articleType` (92 classes) | `notebooks/02_task1_item_type.ipynb` | `artifacts/task1/task1_cnn.pt` |
| 2 | `season` (4 classes) | `notebooks/03_task2_season_pytorch.ipynb` | `artifacts/task2/task2_season_best_pytorch.pth` |
| 3 | `gender` (5) + `usage` (4), one multi-task CNN | `notebooks/04_task3_cnn_architectures.ipynb` | `artifacts/task3/task3_cnn_model.pt` |
| 4 | visual search — top-K similar items | `notebooks/05_task4_visual_search.ipynb`, `06_task4_background_augmentation.ipynb`, `07_task4_clustering.ipynb` | `artifacts/task4/` |

`notebooks/01_eda.ipynb` produces the shared cleaned metadata every task reads.
`notebooks/08_ultimate_judgement.ipynb` is cross-task comparison.

## Environment

Python 3.13, plain python.org interpreter or `.venv` — **not Anaconda**. Anaconda's MKL and
PyTorch both ship `libiomp5md.dll` and loading both crashes the kernel with `OMP: Error #15`.
This has already cost hours once; if a notebook kernel dies on `import torch`, check the
interpreter first.

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt          # notebooks
python -m pip install -r requirements-backend.txt  # API
```

Notebooks are run in **VS Code**, not the Jupyter web UI. Consequence: `Path.cwd()` is the
project root, not `notebooks/`, so paths in notebooks are anchored on an explicit
`PROJECT_DIR` rather than relative to the file.

## Commands

```bash
# API + frontend (from the project root; main.py mounts app/frontend at /)
python -m uvicorn app.backend.main:app --reload      # http://127.0.0.1:8000  /docs  /api/health

# tests — mainly guard that checkpoints still load into the shipped architectures
python -m pytest tests/ -q

# Task 1 batch inference / submission CSV
python predict.py --images A2_FashionDataset/FashionDataset/test/images_test \
                  --out outputs/task1_item_type_predictions.csv --submission

# Task 4 at 120x160 - caches first (~13 min each), then either arm. Both arms
# train from scratch: ~6 h at 60x80 and ~24 h at 120x160 on CPU, and --resume
# survives a killed run. Notebook 06 section 16 reads what they write.
python scripts/build_task4_cache.py --resolution 120x160
python -m src.training.train_task4_120x160 --resolution 60x80   --seed 42
python -m src.training.train_task4_120x160 --resolution 120x160 --seed 42

# Rebuild the served index after promoting a Task 4 encoder
python scripts/build_task4_outputs.py --rebuild-index
```

`/api/health` reports per-task `loaded` + `error`; a failed checkpoint does not take the API
down. `POST /api/analyze` runs all four tasks on one upload.

## Data

```
A2_FashionDataset/
  FashionDataset/train/{images_train, styles_train.csv}   # gitignored, local only
  FashionDataset/test/{images_test, styles_prediction_template.csv}
  processed/           # written by 01_eda.ipynb — clean_train_metadata.csv, prediction_metadata.csv, splits
  input_images/        # 31 real-world user photos: mixed formats, backgrounds, multi-garment, non-clothing
```

Facts that matter:

- **The assignment images are 60×80**, and for Tasks 1-3 that is still the hard ceiling on
  accuracy — fine print, fabric and subtle colour are not representable at that size.
- **Task 4 no longer trains at 60×80.** `A2_FashionDataset/processed/images_train_120x160/`
  holds the training catalogue rebuilt from the high-resolution source (git LFS, on `main`
  since PR #5). The files carry real detail rather than interpolation — 6.1× the spectral
  energy above the 60×80 Nyquist limit that a bicubic upscale has — and cover training ids
  only, so they do not touch the assignment's 5,829 held-out test ids.
- `styles_train.csv` has **unquoted commas inside `productDisplayName`** (21 rows), which spill
  into `Unnamed: 10` / `Unnamed: 11`. Re-join those columns *then* drop them. Dropping them
  first silently truncates product names.
- Splits are **grouped by product name** so near-duplicate photos of one item cannot straddle
  train/val/test, and stratified on the target(s).
- The Kaggle high-res set is the **same items at 900× the pixels** — 0 new rows, but it
  **contains labels for the assignment's held-out test ids**. Never train on or score against
  those ids. The 120×160 catalogue above was derived from it and is scoped to training ids;
  going back to the raw set for anything else needs the teacher's approval again.
- Task 4's gallery is the **38,571** rows `train_metadata_120x160_supervised.csv` flags
  `use_for_supervised`, not all 38,612. The 41 excluded rows carry conflicting task labels.
  Dropping them is not cosmetic: `RetrievalProtocol` shuffles *products* to pick its holdout,
  so the reduced gallery shares only **6%** of its held-out queries with the split every
  pre-120×160 Task 4 number was measured on. Those numbers are not a baseline for it, and
  neither is the shipped encoder, which trained under the old split.

## Model state

- **Task 1** — `CNN_weights_none_full`, 92 classes after dropping 32 rare ones; TTA deployed.
  Test weighted-F1 87.13, macro-F1 73.09. Summary: `artifacts/task1/task1_summary.json`.
- **Task 2** — PyTorch season CNN, test accuracy 67.5%, macro-F1 64.3 (baseline 49.6% / 16.6).
- **Task 3** — VGG-style CNN with **early branching** (shared blocks, then a per-attribute conv
  pathway and head). All three configurations (`base`, `balanced`, `balanced_aug`) train under one
  60-epoch cap with early stopping, three seeds each; selection ranks on seed-averaged validation
  macro-F1, and every tie-break must clear the same noise threshold or it falls through to the
  simplest pipeline. `base` (seed 42) is retained — nothing beat it by a measurable margin.
  Threshold adjustment was evaluated and **removed** — `class_weights` in the checkpoint are ones,
  so the service takes plain argmax. Test: gender 90.08% / 77.08 macro-F1, usage 91.17% / 84.28,
  exact match 82.09%.
- **Task 4** — deployed: `Improved+TTA+bgaug`, 128-dim, 38,612-item served index, trained at
  60×80. Clean P@10 80.2; on the disjoint out-of-domain bank 60.6, recorded as
  `hard_metrics_disjoint` in the manifest. The older `hard_metrics` 52.8 was measured against
  the encoder's own training backgrounds and is circular — do not publish it.
  `?mode=` selects ingestion (`nobg` default); the confidence gate is advisory and is now
  surfaced in the UI rather than discarded.
  Augmentation models the camera and the serve path as well as the backdrop (`degrade`,
  `simulate_ingestion`), which adds a third benchmark, `wild`, beside `clean` and `hard`.
  `colour@10` is always reported beside `colourfam@10`, which merges lexical colour variants
  only (`Navy Blue` → Blue): naming explains 3.04 of the 26-point gap to `P@10`, the other 23
  are real.
  **Not run yet**, and nothing is promoted until they are: the 120×160 candidate
  (`artifacts/task4_120x160/`), its 60×80 counterpart, the degradation study (notebook 06
  §12) and the colour branch (§15). `ImprovedEncoderV2` warm-starts from the clean encoder
  carrying 97.8% of its parameters, so the colour branch is a fine-tune, not a from-scratch
  ladder.

### Task 3 label policy (do not silently change)

`usage`: 8 raw classes → 4. `Smart Casual` (55) and `Travel` (25, all bags) merge into `Casual`;
`Party` (13) merges into `Formal`; `Home` (1, a cushion cover) and 72 missing-label rows are
dropped — 73 rows total, 0.19%. Pooling the rare ones into an `Other` class was tried and failed
(F1 = 0.000, ~15 points off usage macro-F1). `CONFIG["rare_strategy"]` in the EDA notebook still
supports `merge` / `drop` / `group`.

`gender`: 5 classes. `Girls` and `Unisex` are the weak ones. Makeup items are systematically
predicted `Men` — 70 makeup rows against a Personal Care category that is majority Men, and at
60×80 a lipstick and a deodorant are the same object. That is a data limit, not a bug to fix.

## Conventions

- **One definition per architecture.** Services import from `src/models/` (e.g. Task 1's
  `ItemTypeCNN`); they must never redeclare a network. A duplicated definition previously left a
  trained checkpoint the API could not load — `tests/test_models.py` exists because of it.
- Task 3's checkpoint **records its own architecture** and the service rebuilds from that, so
  changing widths in the notebook does not break serving.
- Artifacts go in `artifacts/task{1,2,3,4}/` — flat, no `task3_cnn/`-style variants. Notebook
  `ARTIFACT_DIR` and the service path must agree.
- Every model comparison is judged against a stated **noise floor**; differences inside it are
  reported as ties, not as results. Run-to-run variance on identical code has been measured at
  >1 point of macro-F1, so single-run comparisons are not evidence.
- Negative results stay in the notebook (thresholds, balanced sampling, `Other` pooling) —
  written up as evaluated-and-declined rather than deleted.
- Notebook prose reads as a normal research project: **no references to class notes, teacher,
  rubric or report requirements** in markdown or comments. Declarative headings, not questions.

## Working with the user

They are a student learning this material, not a practitioner. Explain what a change does and
why in plain terms, name the trade-off explicitly, and prefer training fewer models with better
justification over broad sweeps — full Task 3 runs take hours on their machine.

Typical loop: Claude edits the notebook → the user runs it in VS Code and saves → "check the
output". Before editing a notebook the user has open, be aware VS Code holds an in-memory copy;
if they hit "content is newer", the fix is to close the tab **without saving** and reopen, never
to overwrite.
