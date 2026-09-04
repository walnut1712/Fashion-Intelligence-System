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
| 4 | visual search — top-K similar items | `notebooks/05_task4_visual_search.ipynb`, `06_task4_clustering.ipynb` | `artifacts/task4/` |

`notebooks/01_eda.ipynb` produces the shared cleaned metadata every task reads.
`notebooks/07_ultimate_judgement.ipynb` is cross-task comparison.

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
```

`/api/health` reports per-task `loaded` + `error`; a failed checkpoint does not take the API
down. `POST /api/analyze` runs all four tasks on one upload.

## Data

```
A2_FashionDataset/
  FashionDataset/train/{images_train, styles_train.csv}   # gitignored, local only
  FashionDataset/test/{images_test, styles_prediction_template.csv}
  processed/           # written by 01_eda.ipynb — clean_train_metadata.csv, prediction_metadata.csv, splits
  processed/images_train_120x160/   # 38,612 train ids re-exported at 120×160 (git lfs)
  input_images/        # 31 real-world user photos: mixed formats, backgrounds, multi-garment, non-clothing
```

Facts that matter:

- **Images are 60×80** for Tasks 1, 2 and 4. `processed/images_train_120x160/` holds the same
  38,612 train ids at 120×160 — genuinely higher resolution, not upscaled (downsizing them back
  reproduces the 60×80 originals to MAE ≈ 1/255; they carry ~36% more high-frequency energy than
  a bicubic upscale). No held-out test id appears in that folder.
- **Resolution was not the bottleneck for Task 3.** Retrained at 120×160 with a fifth conv block,
  test scores landed within ±0.2 of the 60×80 model on every metric. Do not assume more pixels
  will move the other tasks either without measuring it.
- `styles_train.csv` has **unquoted commas inside `productDisplayName`** (21 rows), which spill
  into `Unnamed: 10` / `Unnamed: 11`. Re-join those columns *then* drop them. Dropping them
  first silently truncates product names.
- Splits are **grouped by product name** so near-duplicate photos of one item cannot straddle
  train/val/test, and stratified on the target(s).
- **Product name is not a sufficient group.** Hashing at full resolution finds 1,399 byte-identical
  images in 636 families, and **218 families span different product names** ("Idee Men Black
  Sunglasses" / "IDEE Men Black Sunglasses" is one photograph) — 470 rows that name-grouping alone
  leaves on both sides of a split. `processed/splits_120x160.csv` groups on `split_group`, which
  merges names *and* duplicate families; 41 further rows whose duplicates carry conflicting
  gender/usage labels are marked `use_for_supervised=False`. Prefer these shared splits for any
  new work; they also make the four tasks comparable on one held-out set.
- **Even those splits are not leakage-free** — they are duplicate-free, which is not the same
  thing. Product variants shot in the same frame (compact powder shade 01 vs 07, MAE 0.1/255) are
  near-identical, not byte-identical, and differently named, so both guards miss them. At least
  1.5% of Task 3's held-out rows have such a twin in train; gender accuracy there is 98.8% against
  89.9% elsewhere. Net effect on the headline numbers is ≈ +0.13 gender / −0.20 usage — inside the
  noise threshold, but say "no duplicate images", never "no leakage".
- `A2_FashionDataset/fashion-product-images-dataset.zip` (Kaggle, high-res) is the **same items
  at 900× the pixels** — 0 new rows, but it **contains labels for the assignment's held-out test
  ids**. Do not train on or score against those ids. Using the high-res images at all needs the
  teacher's approval; treat it as blocked until the user says otherwise. Note the exception
  already in the repo: `processed/images_train_120x160/` is a downsized export covering **only
  the 38,612 train ids** — no held-out id is present, so that folder is safe to use, but do not
  widen it to the test ids.

## Model state

- **Task 1** — `CNN_weights_none_full`, 92 classes after dropping 32 rare ones; TTA deployed.
  Test weighted-F1 87.13, macro-F1 73.09. Summary: `artifacts/task1/task1_summary.json`.
- **Task 2** — PyTorch season CNN, test accuracy 67.5%, macro-F1 64.3 (baseline 49.6% / 16.6).
- **Task 3** — VGG-style CNN with **early branching** (shared blocks, then a per-attribute conv
  pathway and head), now at **120×160** with five conv blocks per branch (`branch_widths
  (128, 256, 512)`, 13.3M parameters) so the classifier still sees a 5×3 map. Trained with mixed
  precision on the shared `splits_120x160.csv`. All three configurations (`base`, `balanced`,
  `balanced_aug`) train under one 50-epoch cap with early stopping, **one seed each**; selection
  ranks on validation macro-F1 and every tie-break must clear the same noise threshold or it falls
  through to the simplest pipeline. **`balanced` (seed 42) is retained.** Threshold adjustment was
  evaluated and **removed** — `class_weights` in the checkpoint are ones, so the service takes
  plain argmax. Test: gender 90.07% / 77.19 macro-F1, usage 91.02% / 84.10, exact match 82.00%.

  Two qualifiers travel with that result and must not be dropped from the writeup. **The noise
  threshold (±1.17) is carried over** from the earlier three-seed run at 60×80, not measured here —
  one seed leaves no spread to measure, and without the carried figure the sampling error (±0.67)
  alone would decide the ranking. **And `base` is excluded from the tie by 1.47 against that
  1.17**, a margin of 0.30, so the flip from `base` to `balanced` is provisional. Separately, the
  50-epoch cap binds: `base` early-stopped at 34, but `balanced` and `balanced_aug` peaked at
  epochs 46 and 45 and were still improving, so their scores are floors. That truncation works
  *against* the selected model, which is why the result stands as reported.
- **Task 4** — `Improved+TTA+bgaug` encoder, 128-dim, 32,837-item index. Deployment P@10 0.80;
  on the harder benchmark 0.53. Search modes are exposed via `?mode=` (`nobg` default).

### Task 3 label policy (do not silently change)

`usage`: 8 raw classes → 4. `Smart Casual` (55) and `Travel` (25, all bags) merge into `Casual`;
`Party` (13) merges into `Formal`; `Home` (1, a cushion cover) and 72 missing-label rows are
dropped — 73 rows total, 0.19%. The 120×160 pipeline drops a further **41 rows** whose duplicate
images carry conflicting gender/usage labels (`use_for_supervised=False`), leaving 38,498.
Pooling the rare ones into an `Other` class was tried and failed (F1 = 0.000, ~15 points off
usage macro-F1). `CONFIG["rare_strategy"]` in the EDA notebook still
supports `merge` / `drop` / `group`.

`gender`: 5 classes. `Unisex` (F1 0.566) and `Girls` (0.680) are the weak ones. An earlier note
that makeup is systematically predicted `Men` **did not reproduce** — cosmetics score 78% and
fragrance 100% on the held-out set. The real limitation is **Accessories: 38.5% of all gender
errors** from watches, bags and belts that are near-identical across the men's and women's ranges,
and 120×160 did not fix it. If makeup errors still show in the app, that is on user photos
(out-of-distribution), not catalogue data.

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
