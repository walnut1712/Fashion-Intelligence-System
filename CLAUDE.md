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
| 1 | `articleType` (92 classes) | `notebooks/02_task1_item_type.ipynb` | `artifacts/task1_120x160/task1_120x160_background_adapted.pt` (promoted 2026-09-11) |
| 2 | `season` (4 classes) | `notebooks/03_task2_season.ipynb` | `artifacts/task2/task2_season_best_pytorch.pth` |
| 3 | `gender` (5) + `usage` (4), one multi-task CNN | `notebooks/04_task3_gender_usage.ipynb` | `artifacts/task3/task3_cnn_model.pt` (background-adapted since 2026-09-11) |
| 4 | visual search — top-K similar items | `notebooks/05_task4_visual_search.ipynb` (triplet CNN encoder, two background arms) | `artifacts/task4/` |

`notebooks/01_eda.ipynb` produces the shared cleaned metadata every task reads.
`notebooks/07_ultimate_judgement.ipynb` is cross-task comparison. Rewritten and
**executed for the first time on 2026-09-09** (22 cells, 8 code, execution counts 1-8, no
errors). It recomputes nothing - every figure is read from an artefact one of the four task
notebooks or a script in `scripts/` already wrote, so it cannot disagree with them and it runs
in seconds. Three things in it are worth knowing before editing any task:

- **Task 1 has its own version of Task 4's arm C vs arm D**, and nothing had connected the two.
  The `webphoto` recipe in `src/training/train_item_type.py` is the same `ItemTypeCNN` trained
  with background randomisation. Against the shipped checkpoint it costs 2.79 points of clean
  accuracy (`squash`) and buys 22.28 averaged over the mild/moderate/severe corruptions - an
  8.0:1 trade against Task 4's 8.1:1. Under `nobg` it costs nothing at all and still gains
  20.38. Two tasks, two architectures, two corruption families, same conclusion.
  `candidate_webphoto.pt` is **not in the repo**; the rows survive in
  `outputs/evaluation/task1_ood_results.csv`.

  **Task 1's deployed model is the BACKGROUND-ADAPTED checkpoint since 2026-09-11.**
  `artifacts/task1_120x160/task1_120x160_background_adapted.pt` is what `predict.py`,
  `Task1Service` and `scripts/refresh_task1_fixture.py` all load by default. The catalogue-only
  `task1_120x160_onecycle_best.pt` is still on disk, still git-tracked, and is still the better
  model on catalogue tiles; it is now historical.

  | | weighted-F1 | macro-F1 | balanced acc | held-out Places365 weighted-F1 |
  |---|---|---|---|---|
  | catalogue only (historical) | **89.68** | **73.11** | **72.84** | 16.84 |
  | background-adapted (deployed) | 87.14 | 63.88 | 62.59 | **57.96** |

  **Why, and it is a deliberate reversal of the earlier decision.** The earlier reasoning was
  that the graded deliverable is 5,829 catalogue tiles, so in-domain cost is real and
  out-of-domain gain buys nothing graded. That was correct about the tiles and wrong about the
  brief: the system is also served to real uploads, and the assignment credits solving the
  dataset's limitations rather than only scoring its test split. Against held-out Places365
  scenes the catalogue model does not degrade, it collapses - 16.84 weighted-F1, against a
  89.68 clean - and the adapted model holds 57.96.

  **Two runs of this arm exist and their figures must not be mixed.** The teammate's committed
  run is the one `outputs/evaluation/task1_bgadapt_comparison.csv` records (clean 87.06, OOD
  57.80, -2.62 / +40.99, 15.7:1) and **its weights are gitignored and were never on this
  machine**. What is deployed is a local reproduction, `--tag _local`, recorded in
  `..._comparison_local.csv`: clean **87.14**, OOD **57.96**, **-2.53 / +41.11, 16.2:1**. Every
  figure describing the *deployed* model therefore comes from the local run, and that is the row
  above. The two agree to within 0.16 on every metric, which is what makes the reproduction
  usable; an earlier version of this file quoted the local clean number beside the committed
  out-of-domain number, which is a row that describes no model that exists.

  **The in-domain cost is larger than the weighted-F1 line suggests, and both numbers must be
  quoted.** Weighted-F1 falls 2.53; **macro-F1 falls 9.23** (73.11 -> 63.88) and balanced
  accuracy 10.25. On the graded set the adapted model predicts **66 distinct classes against
  76**, so the loss is concentrated in the rare tail - exactly the part of Task 1 that was
  already weakest. Never report the -2.53 on its own.

  **Submission effect**: `articleType` changed on **1,454 of 5,829 rows (24.94%)**. Regenerated
  with `python predict.py --images ... --submission` then `python scripts/build_submission.py`,
  and `scripts/refresh_task1_fixture.py` was re-run so
  `artifacts/task1_120x160/task1_predictions_120x160.csv` replays the deployed model.

  **What the promotion is worth on real photographs is much less clear than the composite says.**
  Holding ingestion at `nobg` so only the weights differ, top-1 on the 23 labelled photographs
  goes **13.04 -> 17.39** - one extra photograph, well inside the +/-10 sampling error at n=23.
  The 2,000-row composited benchmark is the evidence for this promotion; the 23 photographs
  neither confirm nor refute it.

  **`tests/test_prediction.py` no longer hardcodes the deployed checkpoint.** It reads
  `service.model_path`, because what that test pins is that the module and the API run identical
  maths on identical pixels - a statement about whichever checkpoint ships. Hardcoding turned
  this promotion into a test failure that said nothing about agreement.
- **The two tasks then made opposite deployment decisions from that same evidence, and both are
  right.** Task 1's deliverable is 5,829 catalogue tiles, so in-domain cost is real and
  out-of-domain gain buys nothing graded. Task 4's deliverable is a search box that accepts an
  upload. Do not "fix" the inconsistency.
- **"Resolution is the ceiling" is retired.** It was the closing recommendation of the old
  version of `07` and it is contradicted by three measurements (Task 3 within +/-0.2, Task 1
  +0.39 with the CI straddling zero, Task 4 tied on 3 of 5 benchmarks). The lever is the
  training distribution. Task 4's colour metrics are the only measured resolution win.

**Task 4 is one notebook and two models.** `06_task4_clustering.ipynb` was folded into `05`
as Part 9 and then Part 9 itself was removed, both on 2026-09-09 at the user's request, so
there is no clustering in Task 4 at all - see the note further down before re-adding any. The
notebook numbering keeps a gap where `06` was; `07` was not renamed.

The two models are the **two arms of the background comparison**: the same `ImprovedEncoder`
(four conv blocks, 128-d projection, batch-hard triplet loss plus auxiliary `articleType` and
`baseColour` heads) trained on the catalogue **as it ships**, against the same network trained
**with photographic backdrops composited behind the garments**. Arms C and D of the background
grid in `05` Part 6b, whose third arm P is a table rather than a checkpoint. The
second is deployed; the first is a control and **must never be promoted**.

**Clustering has been removed from Task 4 entirely (2026-09-09), at the user's explicit
request, after the trade-off below was put to them.** Do not re-add it without being asked.

The trade-off, recorded so nobody has to rediscover it: the encoder is a *supervised* CNN. The
triplet loss needs `articleType` to know which pairs belong together, and the auxiliary heads
are ordinary classification. Tasks 1-3 are supervised CNNs too. **So the project now contains
no unsupervised learning at all**. The last of it was the k-Means inside the Part 6b
classical arms, and those were removed on 2026-09-10 (see the Task 4 entry). The user was told
this before deciding, both times.

What was deleted with it, in case any of it is ever wanted back (it is in git history at
`18ae8690e`): the k sweep with elbow and silhouette, k=120 validated against labels it never
saw (purity 0.723, ARI 0.225, NMI 0.684), k-Means against DBSCAN and agglomerative ward
(DBSCAN refuses 91.6% of the sample as noise; scored as a router must use it, ARI 0.020), the
router trade-off (3 probes keeps 96.9% of the exact answer for 3.2% of the catalogue, 8.3x
faster), and cluster stability under a background swap (59.8% keep their cluster).

**The clustering code and artefacts were deleted on 2026-09-09**, completing the removal the
notebook change started. Gone: `src/visual_search/cluster_engine.py`, the clustering pass and
`--no-clusters` flag in `scripts/build_task4_outputs.py`, and eleven output files
(`kmeans_centroids.npy`, `cluster_assignments.csv`, `cluster_summary.csv`, `kmeans_k_sweep.csv`,
`cluster_k_choice.csv`, `cluster_model.json`, `cluster_domain_comparison.csv`,
`cluster_search_tradeoff.csv`, `clustering_comparison.csv`, `clustering_summary.json`,
`task4_test_clusters.csv`). All recoverable from git.

An earlier note here claimed the module's tests still passed. **There were none** - the only
mentions of `ClusterEngine` in `tests/` were two comments. The suite is 140 pass / 6 skip both
before and after the deletion, which is what confirmed it.

Four earlier methods (Classical HSV+gradient histograms, the Task 3 CNN reused as a
feature extractor, a convolutional autoencoder, a plain triplet network) were built, measured
and removed; their scores survive as a table in `05` section 5 and as CSVs in
`artifacts/task4/superseded/`, their checkpoints do not. Do not reintroduce them **as
retrieval methods**. `classical_features` was briefly readmitted as the substrate for the
unsupervised arms of the background grid in `05` Part 6b; **those arms were removed on
2026-09-10** and it is now scored nowhere. Its 128-bin HSV block survives as `colour_histogram`
and *is* still scored, as the `Colour histogram` baseline in the `05` Part 5 method table
(P@10 28.67 against the encoder's 76.19). Notebook 05 cell 24 explains the hand-written rule
and points at that baseline.

## What to run, in order

Everything below is already built and committed, so **nothing has to be re-run to use the
project**. This is the order to rebuild Task 4 from scratch.

| # | Run | Where | Cost | Needed when |
|---|---|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | VS Code | minutes | once, and only if `processed/` is missing |
| 2 | `python scripts/build_task4_cache.py --resolution 120x160` | terminal | 2.4 min | once; writes the image + mask caches |
| 3 | `python -m src.training.train_task4_120x160 --backgrounds mixed --seed 42` | terminal | ~100 min (GPU) | to retrain the encoder |
| 4 | `python scripts/promote_task4_encoder.py --encoder artifacts/task4_120x160/task4_encoder_mixed_seed42.pt` | terminal | ~2 min | after step 3, to serve the new encoder |
| 5 | `notebooks/05_task4_visual_search.ipynb` | VS Code | ~10 min | reads steps 2-4 and writes the figures. Part 8b rebuilds both encoder indexes, ~3 min of it |

The background grid in `05` Part 6b is produced by these commands, none of which the
notebook re-runs:

| Run | Cost | Produces |
|---|---|---|
| `python -m src.training.train_task4_120x160 --backgrounds none --seed 42` | 78 min | arm C, the catalogue-only control |
| `python -m src.training.train_task4_120x160 --backgrounds procedural --seed 42` | 104 min | arm P. **Only needed for an interval** - the run's scores are already in `outputs/task4_resolution_comparison.csv`, and the checkpoint is deleted, not in git |
| `python scripts/compare_task4_background_arms.py` | 3 min | `outputs/evaluation/task4_background_arms{,_significance}.csv` |
| `python scripts/eval_task4_classical_clustering.py [--views N]` | 20 min | `outputs/evaluation/task4_classical_clustering_arms_v{N}.csv`. **Nothing reads this any more** - the classical arms left notebook 05 on 2026-09-10. Kept as the evidence behind the recorded rows |
| `python scripts/eval_task4_real_photos.py` | 3 min | `outputs/evaluation/task4_real_photo_arms{,_per_image}.csv` - both encoders on the 23 labelled real photographs. Notebook 07 reads it |

Training happens in **step 3, a script, not a notebook**: a run is ~100 minutes and is
checkpointed every epoch so `--resume` survives a killed kernel. Notebook `05` loads what it
produced and plots it, so the notebook itself runs in minutes.

**The state right now**: steps 1-5 are done. Both the encoder and the clustering were re-run
against the promoted 120x160 encoder, and notebook `05` stores output for every cell.

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

**`rembg` is installed as of 2026-09-11, and pip cannot resolve it cleanly.** It declares
`numpy>=2.3.0` while `opencv-python` declares `numpy<2.3.0`, so the two conflict. Install rembg
first, then pin numpy back for OpenCV:

```bash
python -m pip install "rembg[cpu]"
python -m pip install "numpy<2.3"
```

rembg's floor is a packaging artifact, not a real ABI requirement - verified working at numpy
2.2.6, as were `cv2.grabCut`, `cv2.resize` and the torch array bridge. pip still prints the
conflict; it is expected. It also pulls `onnxruntime`, `numba`, `llvmlite`, `scikit-image` and
`pymatting`, and bumps pillow to 12.3.0. First use downloads `u2netp.onnx` (4.6 MB) to
`~/.rembg`, so it needs network once.

**Why it is installed**: it is the deterministic top tier of `foreground_mask`. Below it sits
`cv2.grabCut`, which seeds its GMM from OpenCV's process-global RNG, so every real-photo number
measured before this date rode on that RNG. With rembg the harnesses reproduce exactly - Task 1's
7-repeat spread went from +/-1.64 to **0.00** and two runs of `eval_task4_real_photos.py` are
byte-identical. **It is pretrained**, so what it may and may not touch is a constraint rather
than a preference: see the README's pretrained-components note and
`tests/test_graded_path_never_segments.py`. Nothing graded reaches it; the 31-photograph tables
do.

Notebooks are run in **VS Code**, not the Jupyter web UI. `Path.cwd()` in a notebook is
`notebooks/` — the notebook's own directory, not the project root — so every notebook sets
`PROJECT_DIR = Path.cwd().parent` and anchors its paths on that rather than writing them
relative to the file. (`03_task2_season.ipynb` probes for `A2_FashionDataset/`
instead and works either way.) An earlier version of this file claimed the opposite; the
stored output of `01_eda.ipynb` cell 1 prints the real value.

## Commands

```bash
# API + frontend (from the project root; main.py mounts app/frontend at /)
python -m uvicorn app.backend.main:app --reload      # http://127.0.0.1:8000  /docs  /api/health

# tests — mainly guard that checkpoints still load into the shipped architectures.
# pytest lives in .venv, not in the global interpreter the notebooks use, so call it
# through the venv or you get "No module named pytest".
.venv/Scripts/python.exe -m pytest tests/ -q

# Task 1's image cache, which tests/test_splits.py needs and notebook 02 only
# builds as a side effect of training. 21 s. --check verifies without rebuilding.
python scripts/build_task1_cache.py

# Contact sheet for hand-labelling the 31 real-world photographs. Renders every
# photo with Task 1's top-5 as a SPELLING hint only - the template is left blank
# on purpose, because rubber-stamping the model's own guesses would turn Task 4's
# real-photo P@10 into a measure of agreement with Task 1. `none` marks a photo
# that is not one catalogue garment (stickers, flat-lays) and excludes it.
python scripts/make_label_contact_sheet.py            # renders outputs/label_contact_sheet.png
python scripts/make_label_contact_sheet.py --check    # validates a filled-in template

# Task 1 batch inference / submission CSV
python predict.py --images A2_FashionDataset/FashionDataset/test/images_test \
                  --out outputs/task1_item_type_predictions.csv --submission

# Task 1 resolution comparison. The 120x160 arm is already trained and committed;
# `--resolution 60x80` is the missing control that makes it interpretable. Same
# script, same splits_120x160.csv, same recipe, same dropout - the image folder is
# the only difference. ~4 h on CPU (369 s/epoch), much less on a GPU.
python -m src.training.train_task1_120x160 --resolution 60x80 --seed 42

# Task 4 - cache (2.4 min, not the 13 an earlier note claimed), then train.
# This machine HAS a CUDA device (RTX 3060 Laptop), so the "~24 h on CPU" figures
# in the script docstring are upper bounds: a 42-epoch 120x160 run is ~100 min.
# --resume survives a killed run. Notebook 05 reads what the run writes.
python scripts/build_task4_cache.py --resolution 120x160
python -m src.training.train_task4_120x160 --backgrounds mixed --seed 42

# Promote a trained encoder to the served artefacts. Rebuilds the index, the
# metadata and the ids together - they must move as one, because this gallery is
# 38,571 rows against the 38,612 an older index held.
python scripts/promote_task4_encoder.py     --encoder artifacts/task4_120x160/task4_encoder_mixed_seed42.pt
```

`/api/health` reports per-task `loaded` + `error`; a failed checkpoint does not take the API
down. `POST /api/analyze` runs all four tasks on one upload.

**Two interpreters, on purpose.** The global python.org install has `torch 2.11.0+cu128` and is
what the notebooks train on; `.venv` has `torch 2.13.0+cpu` and pytest. Tests only load
checkpoints and run a forward pass, so CPU is fine, but don't train from `.venv`.

**144 collected: 139 pass, 5 skip, ~71 s** (measured 2026-09-11, after the Task 1
routing and Task 3 promotion). Nothing fails.
The count moved from 140/6 when `ImprovedEncoderV2` and its nine tests were deleted
(-9) and `test_shipped_predictions_are_reproducible` was added (+1), then to 144/5
when `tests/test_graded_path_never_segments.py` was added (+6) with the router fix
below.

**That new test is the only one that anchors a stored result to the shipped model.**
Every other submission check compares one file to another - the CSV against the template,
the task1 column against the CSV, its sha256 against the recipe manifest - and all of them
pass when the CSV and the manifest are stale *together*, which is exactly what a retrain
produces. That is not hypothetical: on 2026-09-10 the shipped submission was found at 65%
agreement on `articleType`, 83% on `season` and 89% on `gender`, with the whole suite green.
It replays `artifacts/task1_120x160/task1_predictions_120x160.csv` through the deployed
checkpoint; rebuild it with `python scripts/refresh_task1_fixture.py` only after confirming
the change was intended, then rerun `predict.py --submission` and `build_submission.py`.
The script's `--pair legacy` handles the older 60x80 fixture, which pairs with
`task1_cnn.pt` and must not be rebuilt against the shipped model.

The five `tests/test_splits.py` failures this note used to record are fixed. They wanted
`processed/image_cache_task1_60x80{,_ids}.npy`, which was simply not on disk: notebook `02`
builds it as a side effect of training, so a checkout that has not run that notebook fails
those tests with `FileNotFoundError`. `scripts/build_task1_cache.py` now builds it in **21 s**
without a training run, and `--check` verifies an existing one by re-decoding a sample.

Do not substitute Task 3's `image_cache_60x80.npy` for it. The two hold **different row
subsets** — 38,539 rows against Task 1's 38,491, which are the rows surviving the
`MIN_CLASS_SIZE` floor — and neither is a superset of the other.

Both caches are gitignored (`A2_FashionDataset/processed/*.npy`), so a fresh clone has to build
them. A test run additionally writes `image_cache_task1_60x80_extra{,_ids}.npy`, the 121 tiles
`_extend_cache` decodes for the dropped-class merge; that is expected, not drift.

Every skip is a data-presence guard — the tests that need gitignored images or a built
submission skip rather than fail, so the count varies with what is on disk.

## Data

```
A2_FashionDataset/
  FashionDataset/train/{images_train, styles_train.csv}   # gitignored, local only
  FashionDataset/test/{images_test, styles_prediction_template.csv}
  processed/           # written by 01_eda.ipynb — clean_train_metadata.csv, prediction_metadata.csv, splits
  processed/images_train_120x160/   # 38,612 train ids re-exported at 120×160 (git lfs)
  input_images/        # 31 real-world user photos: mixed formats, backgrounds, multi-garment, non-clothing
  external_data/places365/  # 36,500 Places365 scenes, the Task 4 background corpus
  external_data/sfs/        # SFS_metadata.csv, used by the Task 2B/3B scripts
```

Facts that matter:

- **Task 1 final uses 120×160; Tasks 2-3 use 60×80; Task 4 uses 120×160.** `processed/images_train_120x160/` holds the same
  38,612 train ids at 120×160 — genuinely higher resolution, not upscaled (downsizing them back
  reproduces the 60×80 originals to MAE ≈ 1/255; they carry ~36% more high-frequency energy than
  a bicubic upscale). No held-out test id appears in that folder.
- **Resolution was not the bottleneck for Task 3.** Retrained at 120×160 with a fifth conv block,
  test scores landed within ±0.2 of the 60×80 model on every metric, at four times the compute.
  Task 3 has since been returned to 60×80 on that evidence. Do not assume more pixels will move
  the other tasks either without measuring it.
- **Task 1's final model is the 120×160 BACKGROUND-ADAPTED checkpoint** (promoted 2026-09-11),
  `artifacts/task1_120x160/task1_120x160_background_adapted.pt`: test weighted-F1 **87.14%**,
  macro-F1 **63.88%**, balanced accuracy **62.59%**, and **57.96%** weighted-F1 through held-out
  Places365 scenes, where the checkpoint below scores 16.84. See the Task 1 entry under **Model
  state** for the trade and why it was taken.

  **Everything in the rest of this bullet describes `task1_120x160_onecycle_best.pt`**, the
  catalogue-only checkpoint it was fine-tuned from. Those figures are **historical**: that model
  is no longer submitted or served, though it remains the arm every resolution comparison here
  was measured on. It
  had test accuracy **90.23%**, weighted-F1 **89.68%**,
  macro-F1 **73.11%**, and balanced accuracy **72.84%**. Earlier 60×80 results remain historical
  development and ablation evidence. The resolution comparison was measured on rows neither
  model trained on (3,248 of them; the fully clean test-only intersection is 884 and agrees):

  | arm | acc | weighted-F1 | macro-F1 | vs shipped (paired bootstrap) |
  |---|---|---|---|---|
  | 120×160 @ true 120×160 | 90.06 | 89.59 | 70.58 | **+0.39** [−0.55, +1.35], P(better) 79% |
  | 120×160 @ upscaled 60×80 | 85.65 | 85.23 | 62.77 | **−3.97** [−5.13, −2.83], P(better) **0%** |
  | 60×80 shipped + flip TTA | 89.75 | 89.20 | 71.48 | — |

  How to read that table, since the headline and the middle row measure different things. The
  **89.68** quoted above is this model on `splits_120x160.csv`, which is the shared partition
  Task 3 also uses, so it is the right number for comparing Task 1 against the other tasks. The
  rows above compare the two *checkpoints* on ground neither trained on, and there the 120×160
  model is a **tie** rather than a win: +0.39 with the CI straddling zero, and 0.9 worse on
  macro-F1.

  **The serving cost is real and is accepted.** No graded id has a 120×160 version (the export
  covers train ids 1163–51999; the graded set is 52003–60000, verified: 5,829 files, zero
  overlap), so `predict.py` upscales every graded image to the checkpoint's input contract. That
  is the −3.97 row, at P(better) = 0%. The team has chosen the 120×160 checkpoint as Task 1's
  final model with that cost known; do not revert it, and do not substitute the historical
  checkpoint or the blocked high-resolution labelled zip.

  What would remove the cost rather than accept it is 120×160 versions of the graded ids, which
  needs the teacher's approval for the Kaggle zip. A matched retrain settles the resolution
  question separately: `train_task1_120x160.py --resolution 60x80` trains exactly that arm (same
  script, split, recipe and dropout 0.4), so one run is needed, not both.
- **The assignment's own images are not perfectly uniform.** 17 of 38,612 train images and 6 of
  5,829 graded images are not 60×80 — 60×77, 60×76, 60×75, 60×60, 53×80. Every path that reads
  them resizes (`load_image_array`, `predict.py`, and now `CandidateDataset`), so this is
  invisible until something batches them raw: a `DataLoader` cannot collate frames of different
  shapes, which is how it first surfaced. The 120×160 re-export *is* uniform, so the 120×160 arm
  never hit it.
- **Task 4 trains at 120×160**, on the evidence above being a Task 3 result rather than a
  general one — it has not been measured for retrieval yet, and the Task 3 null is a reason to
  measure rather than to assume. Both arms of that comparison are trained from scratch by
  `src/training/train_task4_120x160.py`; see the Task 4 entry for why the shipped encoder
  cannot serve as the 60×80 reference.
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
- **Task 4 has not adopted `split_group` yet, and it costs it.** `RetrievalProtocol` still
  groups on `productDisplayName`, which catches 413 of the 636 duplicate families;
  `split_group` catches all 636. Measured on the Task 4 gallery, **23 of 2,000 held-out queries
  have a byte-identical twin in the catalogue** — for those, retrieval is finding the same file
  rather than a similar item. Fix before spending compute on either resolution arm, since it
  changes which items are held out.
- Task 4's gallery is the **38,571** rows flagged `use_for_supervised`, not all 38,612. Dropping
  the 41 is not cosmetic: the holdout is drawn by shuffling *products*, so the reduced gallery
  shares only **6%** of its held-out queries with the split every pre-120×160 Task 4 number was
  measured on. Those numbers are not a baseline for it, and neither is the shipped encoder,
  which trained under the old split.
- **`A2_FashionDataset/external_data/places365/` holds 36,500 scene photographs** (the Places365 validation set,
  100 per category × 365 categories), used as real photographic backdrops for Task 4. The two
  lists a teammate committed beside them, `train_backgrounds.txt` / `test_backgrounds.txt`
  (29,199 / 7,299), split **80/20 over files** — measured, **all 365 categories appear in both**,
  so the model trained on 80 `airfield` photos and was tested on 20 more. That is the id-level
  split this project rejects everywhere else. `src/data/places365_backgrounds.py` supersedes
  them with a **category-level** split (292 train / 73 held out, derived from a fixed seed
  rather than a file so it cannot drift), and crops a 3:4 window at native resolution before
  downscaling — the old loader resized a 512×683 scene straight to 60×80, which squashes the
  aspect ratio and blurs away the texture that makes a backdrop hard.

## Model state

- **Task 1** — final `ItemTypeCNN` at 120×160, 92 classes after dropping 32 rare ones;
  horizontal-flip augmentation only, CrossEntropyLoss with label smoothing 0.05 and no
  inverse-frequency class weights, AdamW + OneCycleLR for 40 epochs. Test accuracy **90.23%**,
  weighted-F1 **89.68%**, macro-F1 **73.11%**, balanced accuracy **72.84%**. Checkpoint:
  `artifacts/task1_120x160/task1_120x160_onecycle_best.pt`.

  **That checkpoint is the catalogue reference, not the deployed model.** Since 2026-09-11 the
  served and submitted model is `task1_120x160_background_adapted.pt`, the same network
  fine-tuned with photographic backdrops: 88.24% accuracy, 87.14% weighted-F1, 63.88% macro-F1,
  62.59% balanced accuracy, and 57.96% weighted-F1 through held-out Places365 scenes where the
  checkpoint above scores 16.84%. The recipe above is still the recipe - the adapted model is
  that model plus eight epochs - so the architecture and training arguments carry over unchanged.

  The old `task1_cnn.pt` results remain historical 60×80 development and OOD evidence. They are
  not the service or submission model.

  The old ensemble-plus-label-shift submission recipe is historical 60×80 evidence only. It is
  not the current final-model report or serving path; do not apply its prior correction to uploads.
  `outputs/task1_item_type_predictions_prior_corrected.csv` was a stale artefact of a superseded
  run (1,024 rows disagreed with what ships); **deleted on 2026-09-09**. Two files still name it
  in prose - `scripts/build_submission.py`'s usage example and `tests/test_submission.py`'s
  docstring - but neither reads it.

  **Error structure - the ceiling is label convention, not the model.** For the shipped 120x160
  checkpoint on its 5,495 held-out rows, family (`subCategory`) accuracy is **97.20%** against
  **90.23%** at `articleType`, and **71.7%** of the error mass is *symmetric* confusion between
  adjacent classes: `Casual Shoes`/`Sports Shoes` (73 errors), `Tops`/`Tshirts` (42),
  `Flats`/`Heels` (42), `Casual Shoes`/`Formal Shoes` (17). The 60x80 checkpoint reproduces the
  structure on its pooled val+test rows (96.97% family against 88.29%, 73.2% symmetric), so this
  is a property of the catalogue rather than of one network. The model almost always knows what
  the object is; the residual is which label the catalogue chose.
  Regenerate with `python scripts/task1_error_structure.py` ->
  `outputs/evaluation/task1_error_structure{,_pairs}.csv`; notebook 02's CELL 34b reads it. Do not treat this as a
  bug, and **do not relabel** — the held-out `articleType` values are the grading target, so
  "correcting" them moves away from it.

  **Pairwise specialists were measured and declined — and the null is the strongest evidence for
  the ceiling above.** `src/training/train_pair_specialist.py` trains a binary CNN per confusion
  pair and consults it only when the main model's top-2 set is exactly that pair. The specialists
  are individually good — `Sports Shoes` vs `Casual Shoes` 87.71% and `Tshirts` vs `Tops` 93.49%
  validation accuracy on their whole pair populations. On the **gated** rows they are not: 102 rows
  changed on validation, **43 fixed against 55 broken** (net −12, weighted-F1 −0.181); test 124
  changed, 58 fixed, 63 broken (−0.085). That is 44% correct on the rows it moves — worse than a
  coin flip. A model with all of its capacity on one boundary, a balanced prior and no competition
  from 90 other logits still cannot beat the multi-class head where the multi-class head is
  unsure, which is what "the label is not in the pixels" looks like when you test it directly.
  Table: `outputs/evaluation/task1_pair_specialist_ab.csv`. Do not re-try this without a new idea
  about *where the information would come from*.

  **Wider TTA was measured and declined.** `predict_proba`/`predict_split` take an opt-in `views`
  argument (`item_type_classifier.TTA_VIEWS`); `views=None` is the shipped two-view flip average,
  bit-for-bit. Adding scales and shifts *loses*: validation weighted-F1 88.389 → 87.943
  (`flip_scale`, 6 passes) → 87.874 (`flip_shift_scale`, 10 passes), both past the ±0.24 noise
  floor, at 3–5× the inference cost. Catalogue tiles are centred and tightly cropped, so shifting
  one moves the garment off the framing every training image shared.
  Table: `outputs/evaluation/task1_tta_views_ab.csv`.
- **Task 2** — PyTorch season CNN, test accuracy 67.5%, macro-F1 64.3 (baseline 49.6% / 16.6).
- **Task 3** — VGG-style CNN with **early branching** (shared blocks, then a per-attribute conv
  pathway and head), back at **60×80** with four blocks total (`branch_widths (128, 256)`,
  4.25M parameters, 256×5×3 → 3,840 flattened). Block count is **derived from the input height**,
  so the same definition gives five blocks at 120×160. Trained on the shared `splits_120x160.csv`
  (an id-level partition — nothing to do with resolution) with mixed precision, 50-epoch cap,
  patience 8. Four configurations (`base`, `balanced`, `balanced_aug`, `class_weighted`) ×
  **three seeds**. **`base` (seed 42) is retained** — no imbalance correction. Test: gender
  90.33% / 76.37 macro-F1, usage 90.24% / 82.66, exact match 81.59%.

  **The provisional selection is now settled, and it went back to `base`.** Measured training
  variance is **±0.46**, not the ±1.17 that was carried over — the old prior was 2.5× too wide,
  so the threshold is now set by the validation sampling error (±0.67). Three configurations tie
  inside it (`base` 81.68, `class_weighted` 81.42, `balanced` 81.27; `balanced_aug` 80.30 is
  excluded), no tie-break clears the same margin, and the rule falls through to the simplest
  pipeline. **Balanced sampling costs gender macro-F1 again (−1.90 at seed 42)** — the 120×160
  run's +2.12 gain did not reproduce, which is itself evidence that these differences are
  noise-dominated.

  One qualifier remains: **4 of 12 runs hit the 50-epoch cap while still improving** (2×
  `balanced_aug`, 2× `class_weighted`), so their scores are floors. Every `base` and `balanced`
  run stopped on patience, so the truncation works against the challengers, not the winner — but
  the 0.26 margin over `class_weighted` is an upper bound.

  **`balanced_aug` was deployed for `Sports` recall, measured, and reverted — do not re-try it.**
  It delivered what it promised: test `Sports` recall 64.8% → 83.4%. But precision fell 77.7% →
  63.3% in the same step, so `Sports` F1 moved only 70.7 → 72.0, and it cost **gender macro-F1
  −2.69 and exact match −2.70** (76.37 → 73.68, 81.59 → 78.89), with `Casual` recall down to
  0.885 and `Unisex` F1 to 0.518. The sampler equalises `usage` through a trunk `gender` shares,
  which is the mechanism CELL 17's exposure table predicts. A decision-threshold sweep on `base`
  traces the same recall/precision curve with `Sports` F1 flat (70.7 → 72.8 → 70.4), and it does
  not touch `gender` at all — so if recall on `Sports` is ever wanted again, **use the threshold,
  not the sampler**. `DEPLOY_NAME` in CELL 20 is the switch; it is left equal to the rule's answer.

  **Note for the writeup: the test partition has now been read twice for Task 3** — once for the
  retained configuration and once for that override. The evidence that reverses the decision is
  present in validation alone (`balanced_aug` costs 1.18 gender macro-F1 there at seed 42, and the
  rule had already excluded it), so no conclusion depends on the second look, but say so rather
  than let a reader assume one look.

  **`Sports` recall (0.648) is a data ceiling, not a model failure — CELL 24 establishes it.**
  Predicting `usage` from `articleType` alone, an oracle that knows exactly what the garment is,
  gets **54.9% `Sports` recall** against the network's 64.8% (and 75.91 macro-F1 against 82.66).
  Knowing the category recovers less than the image does. `usage` records what an item was
  marketed for: training holds 4,169 `Tshirts` as `Casual` against 679 as `Sports`. Of the 199
  missed `Sports` items, 198 are called `Casual`; 62 are `Tshirts` and 60 are `Sports Shoes`. Do not treat this as a bug to fix.

  **`class_weighted` (per-head inverse-frequency weights in the loss) was evaluated and declined.**
  It is the only correction that improved gender at all (+0.23) and it lifts the weak classes'
  recall — `Unisex` 60.5 → 72.1, `Sports` 68.9 → 73.0, `Girls` 69.1 → 71.8 — but the gain does not
  reach macro-F1: recall bought on a rare class is paid for in precision on it. Sampling is left
  alone, so it avoids the gender/usage trade the sampler forces. Keep it in the notebook as a
  measured negative, alongside thresholds, balanced sampling and `Other` pooling.

  **The resolution comparison is now like-for-like, and a loose end came with it.** Both runs
  used `splits_120x160.csv`, so validation scores are directly comparable. `base` vs `base`:
  81.68 (60×80, 3 seeds) against 81.35 (120×160, 1 seed) — +0.33, inside the threshold, which is
  the cleanest evidence yet that resolution bought nothing. **But `balanced` and `balanced_aug`
  each scored ≈1.5 higher at 120×160** (82.82 / 81.79 against 81.27 / 80.30), from one seed each,
  against a measured SD of ±0.46. Either that is single-seed noise, or the sampler interacts with
  resolution. It is why the shipped 120×160 checkpoint's test usage macro-F1 (84.10) is 1.44 above
  the current one (82.66) — a gap that is configuration, not pixels, on this reading. Three seeds
  of `balanced` at 120×160 (~3.5 h) would settle it; nothing else depends on the answer.

  **Temperature scaling is deployed.** As trained the model reported 98.6% mean confidence on
  gender against 90.3% accuracy (ECE 8.24) and 99.0% against 90.2% on usage (ECE 8.81). One
  temperature per head, fitted on validation (gender 5.83, usage 5.94), brings ECE to 1.23 and
  1.35 with **no change to any prediction** — the notebook asserts it. Stored in the checkpoint as
  `temperature`; `task3_service.py` divides logits by it before the softmax and defaults to 1.0
  when the field is absent, so older checkpoints serve unchanged. Effect on the app: user photos
  went from 20/31 at p ≥ 0.99 and several at exactly 1.0000, to **1/31 and none at 1.0000**.
  It is fitted on catalogue images, so it does not fix out-of-distribution overconfidence.

  **Task 3 collapses out of domain on BOTH heads, and `usage` accuracy hides it.** Measured
  2026-09-11 by `scripts/eval_task3_ood.py` (which reuses Task 1's corruption ladder unchanged,
  so the two tasks are comparable), through `letterbox`, gender accuracy falls 90.11 -> 33.81
  and macro-F1 75.74 -> 17.84. `usage` **accuracy** falls only 90.04 -> 70.87 and looks
  survivable - that is a trap. `Casual` is 77.3% of the split, so a model that has degenerated
  into guessing `Casual` still scores about 70%; usage macro-F1 falls 82.70 -> 25.59. **Never
  quote Task 3 robustness from accuracy.** Table: `outputs/evaluation/task3_ood_results.csv`.

  **Ingestion routing is deployed and was worth ~29 points, free.** `Task3Service` hardcoded
  `mode="letterbox"`, so an upload reached the model with its background intact. It now uses the
  same `looks_like_catalogue` router Task 1 has: catalogue tiles keep `letterbox`, uploads get
  `nobg`. Gender accuracy under mild corruption 36.86 -> 65.85. The catalogue branch is
  `letterbox` rather than `squash` deliberately - clean scores are identical across squash /
  letterbox / crop (90.11), so **the graded submission is bit-for-bit unchanged**, verified on
  40 graded tiles. `build_submission.py` calls this service, so that mattered.

  **That 40-tile verification was not enough, and the gap was found on 2026-09-11.** Checked
  across all 5,829, **four graded tiles routed to the photograph branch**: 52166, 56624, 59593
  and 59606 are 53x80 or 54x80 rather than 60x80, and `looks_like_catalogue` applied its
  aspect-ratio gate *before* its size test, so an off-aspect tile was called a photograph and
  sent to `nobg`, which segments. Two consequences, both bad. The segmentation ladder's top tier
  is `rembg`, a **pretrained** network, so with rembg installed those four tiles put a pretrained
  component in the graded path - which the assignment forbids and the README disclaims. And the
  routing silently disagreed with the committed predictions: `outputs/task3_gender_usage_predictions.csv`
  holds the `letterbox` answers, so regenerating the submission would have flipped 52166 from
  `Unisex` to `Men` and 56624 from `Women` to `Men`, with the whole suite green.

  **The fix makes the size test absolute and runs it first.** A dataset tile is at most 100px on
  its long side; the smallest real upload is 225px. A simple reorder is wrong - `small_ratio *
  model_input` is 320px against the 120x160 checkpoint, which swallows the two 225x225
  photographs in `input_images/` - so the ceiling cannot be a multiple of the model input. After
  the fix all 5,829 graded tiles take the catalogue branch and all 31 photographs take the
  photograph branch. `tests/test_graded_path_never_segments.py` walks both sets and asserts each
  direction, because a rule that called everything a tile would pass the first check and break
  the app.

  **PROMOTED 2026-09-11. The paragraphs below describe how the arm was built and
  measured; the deployment decision they end with has since been reversed, on the
  evidence recorded in "The promotion" further down.**

  **A background-adapted arm exists.**
  `scripts/train_task3_background_adaptation.py` fine-tunes the deployed `base` seed-42
  checkpoint for 8 epochs at p_bg 0.70 on Task 4's 70% Places365 + 30% procedural bank, ~8 min.
  Only `base` is adapted - `balanced`, `balanced_aug` and `class_weighted` lost the selection
  and are not touched. Primary metric is the mean of the two heads' macro-F1.

  **Two people wrote this script independently on 2026-09-11**, at the same path, writing the
  same outputs. The committed one is the teammate's; a second implementation is not in the repo.
  They cross-validate: the two harnesses agree on the baseline to sixteen digits (gender
  macro-F1 0.7522152973801136, usage 0.8185825212508178), which is the strongest available
  evidence that both are correct. The committed script originally had **no argparse at all** -
  epochs, lr, p_bg and the output paths were hardcoded, so the control below could not be run
  and a control run would have overwritten the real results. `--epochs --lr --p-bg --seed --tag`
  were added, every default equal to the committed value, so a bare invocation still reproduces
  the committed run.

  **The headline delta is confounded, and the control is what shows it.** The adapted arm gets 8
  epochs the baseline never got, so `--p-bg 0 --tag ...` runs the same schedule with no
  compositing. Mean macro-F1, both learning rates measured:

  All five rows below come from the committed script, mean of the two heads' macro-F1:

  | | clean | held-out Places365 |
  |---|---|---|
  | baseline (deployed) | 0.7854 | 0.2574 |
  | control, 8 clean epochs, lr 1e-4 | **0.8003** | 0.2598 |
  | background arm, lr 1e-4 (committed run) | 0.7932 | 0.4298 |
  | control, 8 clean epochs, lr 3e-4 | **0.8019** | 0.2585 |
  | background arm, lr 3e-4 | 0.7995 | **0.5524** |

  Eight clean epochs alone move clean **+0.0149 / +0.0165** and out-of-domain **+0.0024 /
  +0.0011** - that is, extra training on white tiles buys essentially *nothing* out of domain at
  either learning rate, which is the cleanest possible confirmation that the backdrops are doing
  the work. It also means the arm's apparent clean *gain* against the baseline is the extra
  training. Against its own matched control:

  | lr | clean vs control | OOD vs control | trade |
  |---|---|---|---|
  | 1e-4 | -0.0071 | +0.1699 | 24:1 |
  | 3e-4 | **-0.0024** | **+0.2939** | **121:1** |

  **3e-4 dominates 1e-4 on both axes** - a third of the in-domain cost and 1.7x the
  out-of-domain gain in the same 8-epoch budget. That is not a trade, it is strictly better, and
  the committed run uses 1e-4; re-run it with `--lr 3e-4`. Per head at 3e-4, out of domain:
  gender macro-F1 0.2457 -> 0.4636, usage 0.2691 -> 0.6412, exact match 0.4076 -> 0.6433.

  One run per cell and seed variance is unmeasured for Task 3. The out-of-domain gap between the
  two learning rates (0.4298 against 0.5524) is far too large to be noise; the clean-side
  numbers all sit within 0.007 of each other and could reorder, so do not rank them on that.

  **Tasks 1 and 2 had the same confound; both controls were run on 2026-09-11** with the
  `--tag` flag added to `scripts/train_t12_background_adaptation.py` (which already had
  `--p-bg`). The direction of the correction is **task-dependent**, so it cannot be predicted -
  it has to be measured:

  | task | metric | reported vs baseline | corrected vs control | |
  |---|---|---|---|---|
  | 1 | weighted-F1 | -2.62 / +40.99, 15.7:1 | **-2.41 / +40.86, 16.9:1** | slightly *better* |
  <!-- Task 1's row is the teammate's COMMITTED run, not the deployed local reproduction. -->
  | 2 | macro-F1 | -3.50 / +8.99, 2.6:1 | **-4.35 / +9.84, 2.3:1** | slightly *worse* |
  | 3 (lr 3e-4) | mean macro-F1 | +1.41 / +29.50 | **-0.24 / +29.39, 121:1** | sign of the clean delta flips |

  Task 1's control *lost* clean performance over its 8 extra epochs (0.8968 -> 0.8947) while
  Tasks 2 and 3's controls gained, which is why the correction runs in opposite directions. An
  earlier note here predicted the cost would be understated in every task; that held for 2 and 3
  and was wrong for 1.

  **What does hold in every task is the out-of-domain half.** The controls moved out-of-domain
  by +0.0012 (task 1), -0.0085 (task 2) and +0.0024 / +0.0011 (task 3) - nothing, in all four
  runs. Extra training on white tiles buys in-domain accuracy and **no** robustness, so the
  entire out-of-domain gain in all three tasks is attributable to the backdrops. That is the
  claim the controls were run to secure, and it survives everywhere.

  Two qualifiers: validation was still climbing at epoch 8 in every run, so all of these are
  **floors**, not converged; and the ingest router above attacks the same failure, so its gain
  and this one do not simply add.

  **The promotion (2026-09-11).** Task 3's adapted arm at lr 3e-4 is now the deployed model:
  `artifacts/task3/task3_cnn_model.pt` holds it, `model_name` is
  `base + background adaptation (lr 3e-4)`, and the previous `base` weights are recoverable from
  git. Tasks 1 and 2 were **not** promoted - see their entries.

  **Why this task and not the others.** A deployment decision compares the candidate against the
  incumbent, which is a different comparison from the one the control exists to settle. Against
  the deployed baseline, within the adaptation harness:

  | task | metric | clean | held-out Places365 |
  |---|---|---|---|
  | 1 | weighted-F1 | 0.8968 -> 0.8714 (**-2.53**) | 0.1684 -> 0.5796 (+41.11) |
  | 2 | macro-F1 | 0.6306 -> 0.5956 (**-3.50**) | 0.3126 -> 0.4025 (+8.99) |
  | 3 | mean macro-F1 | 0.7854 -> **0.7995 (+1.41)** | 0.2574 -> 0.5524 (+29.50) |

  Only Task 3 improves on **both** axes, so only Task 3 needs no trade. As ratios: Task 1
  **16.2:1**, Task 2 **2.6:1**, Task 3 no trade at all.

  **Task 2's row is the teammate's committed run**, which is the one every other Task 2 figure
  in this file also reports. A local reproduction trained on 2026-09-11
  (`--tag _local`, weights at `artifacts/task2_bgaug_local/`) lands at -2.99 / +8.00, so the two
  agree on the decision and differ by about half a point; quote the committed numbers and treat
  the reproduction as the repeat that confirms them. Tasks 1 and 3's rows are single runs. Every clean test metric
  moved the right way: gender accuracy 89.78 -> 90.20, gender macro-F1 75.22 -> 77.23, gender
  balanced accuracy 74.33 -> 77.24, usage accuracy 90.04 -> 90.33, usage macro-F1 81.86 -> 82.67,
  exact match 80.93 -> 81.59.

  **Do not quote the clean gain as evidence for backdrops.** The matched `--p-bg 0` control
  reaches 0.8019 clean, *above* the adapted arm, so the +1.41 is the 8 extra epochs and the
  backdrops cost about 0.24 of it. The out-of-domain half is entirely the backdrops: that
  control moves out-of-domain by +0.0011. Both statements are true at once and the promotion
  rests on the first, the science on the second.

  **Two inherited fields were stale and both were fixed; a future promotion must check them.**
  `train_task3_background_adaptation.py` copies the source checkpoint's metadata forward.

  - **Temperature.** The arm inherited the base model's `{gender: 5.83, usage: 5.94}`, fitted for
    weights that no longer exist. On the adapted weights that scores **ECE 20.30 / 26.73** -
    worse than no temperature at all (5.16 / 5.69), because fine-tuning left the model far less
    overconfident (mean confidence 95.7% against the base's 98.6%). Refit with
    `python scripts/refit_task3_temperature.py --checkpoint <ckpt>`: gender **2.1390**, usage
    **1.9985**, ECE **2.20 / 1.03**, comparable to the base model's 1.23 / 1.35. Temperature is
    monotonic, so **no prediction moved** - the script asserts it - and the submission is
    unaffected by this step alone.
  - **`test_metrics`.** Also the baseline's. Now the adapted model's, with
    `test_metrics_source` recording which harness produced them.

  **The two harnesses do not score the same rows, and the difference is not noise-sized.** On the
  *same baseline weights*, notebook 04 reports gender accuracy **90.33** and the adaptation
  script reports **89.78**. Compare adapted against baseline **within one harness**; never mix
  them. This is why `artifacts/task3/task3_cnn_summary.json` now carries
  `baseline_within_same_harness` beside the headline.

  **The submission changed and was regenerated.** `python scripts/build_submission.py
  --force-task3` rewrote `outputs/predictions/COSC2753_A2_HN_G2.csv`: **494 gender rows (8.47%)
  and 278 usage rows (4.77%)** differ; `articleType` and `season` are byte-identical. The churn
  is large for a ~2-point macro-F1 gain and is the expected shape of it - the baseline
  over-predicted `Women` (4,148 -> 3,944) and the adapted model is more balanced (`Men`
  1,504 -> 1,688, `Unisex` 116 -> 135), which is the +2.9 gender balanced-accuracy gain landing
  on the graded set.

  **Task 3 now reports Task 4's five benchmark families**, built by
  `scripts/eval_task3_benchmarks.py` from the same calls `build_queries` uses
  (`make_eval_backgrounds(600)`, `load_background_bank(2000, split="test")`, then
  `simulate_ingestion(degrade(..., EVAL_DEGRADATIONS))`), so a Task 3 row and a Task 4 row mean
  the same thing. 2,000 held-out rows, one sample shared by every arm and benchmark, mean of the
  two heads' macro-F1. Table: `outputs/evaluation/task3_benchmarks.csv`.

  | arm | clean | hard | photo | wild | wildphoto |
  |---|---|---|---|---|---|
  | baseline (deployed) | **77.92** | 25.90 | 23.61 | 27.15 | 23.81 |
  | control 8ep lr1e-4 | 79.48 | 26.75 | 23.40 | 29.56 | 24.78 |
  | bg arm lr1e-4 | 78.28 | 41.46 | 37.53 | 39.86 | 33.16 |
  | control 8ep lr3e-4 | **79.59** | 27.35 | 23.91 | 28.68 | 25.03 |
  | **bg arm lr3e-4** | 78.77 | **50.58** | **46.01** | **43.38** | **41.41** |

  Three things this says that the single-family numbers could not. **The controls are flat on
  all four out-of-domain columns** at both learning rates, so every point of movement belongs to
  the backdrops. **Task 3 does not repeat arm P's mistake**: its `hard` minus `photo` gap is
  **+4.57** against arm P's **17.91**, which is the 70/30 mix doing its job, and is the arm P
  lesson transferring to a different task and architecture. And **the accuracy trap is visible
  in one row** - the deployed baseline's *usage accuracy* across the five is 89.80 / 71.70 /
  74.30 / 71.35 / 73.10, which looks like a model that barely degrades, while its usage macro-F1
  on the identical frames is 81.20 / 27.40 / 25.14 / 29.67 / 26.06.

  `wildphoto` is the closest proxy to a real upload, but Task 4's real-photo work showed even
  that flatters a model (16.52 measured against 55.78 on `photo`), so read it as an upper bound.

  Weak points are unchanged and are **not** resolution-limited: `Unisex` F1 0.599, `Girls` 0.607,
  `Sports` recall 0.648. **Accessories is still the ceiling** — 85.5% gender accuracy, 38.7% of all
  gender errors (85.2% / 38.5% at 120×160).
- **Task 4** - deployed: `ImprovedEncoder`, recorded in the manifest as
  `Improved+TTA+places365`, 128-dim, 38,612-item served index, **trained at 120x160** with
  Places365 backdrops. Its benchmark table is further down this section; clean P@10 is **76.19**
  and the real-photograph number is **16.52**. The checkpoint is
  `artifacts/task4_120x160/task4_encoder_mixed_seed42.pt`, served as
  `artifacts/task4/task4_improved_encoder.pt` (same weights, promotion copies it).

  The retired 60x80 encoder's figures - clean P@10 80.2, disjoint bank 60.6 - are **not**
  comparable to the above and are not this model: the 120x160 gallery drops 41 rows, which
  reshuffles the product holdout so completely that only 6% of held-out queries are shared. One
  lesson from that era still applies: the old `hard_metrics` 52.8 was measured against the
  encoder's own training backgrounds and is circular, so never grade an encoder on backdrops it
  trained on.
  `?mode=` selects ingestion (`nobg` default); the confidence gate is advisory and is now
  surfaced in the UI rather than discarded.
  Augmentation models the camera and the serve path as well as the backdrop (`degrade`,
  `simulate_ingestion`), which adds a third benchmark, `wild`, beside `clean` and `hard`.
  `colour@10` is always reported beside `colourfam@10`, which merges lexical colour variants
  only (`Navy Blue` → Blue): naming explains 3.04 of the 26-point gap to `P@10`, the other 23
  are real.
  **One model remains in Task 4**: this encoder. The clustering was removed entirely on
  2026-09-09 and notebook 06 was folded into 05 and then deleted, so an earlier version of this
  line naming "the clustering model in notebook 06" is wrong. The four methods notebook 05
  compared against — Classical (HSV + gradient
  histograms), Task3-CNN reused as a feature extractor, a convolutional autoencoder and a plain
  triplet network — were removed. Their measurements survive as a table in notebook 05 §5 and as
  CSVs in `artifacts/task4/superseded/`; their checkpoints are deleted. The Classical
  descriptor's 128-bin HSV block survives as `colour_histogram`, because fusion and re-ranking
  need a colour representation that owes nothing to the encoder.

  **From-scratch training with augmentation on from epoch 1 collapses — measured, do not
  retry it.** The triplet loss sits at exactly the margin (0.3003) from epoch 2, mean pairwise
  embedding distance falls 0.006 → 0.001 and stays, and the run ends at clean P@10 **48.5**
  against the 80.2 the same architecture reaches otherwise. The cause is a **plateau, not a bad
  recipe**: notebook 05's stored training log shows the working run also sits at the margin for
  **8 epochs**, then escapes at epoch 9 (aux type loss 3.78 → 3.18). Ramping the backdrop in
  over epochs 1–4 lands in the middle of that plateau and prevents the escape.
  `train_task4_120x160.py` now trains **12 clean warmup epochs** before any augmentation, which
  clears the plateau with margin; `--warmup 0` reproduces the collapse.

  **There is a second, separate collapse at 120×160, and it happens on clean frames.** With the
  warmup in place, a from-scratch 120×160 run still organised to a spread of 0.011 by epoch 5
  and then fell back monotonically to 0.0016 by epoch 9 — while monitor clean P@10 *degraded*
  62.66 → 42.46, before any backdrop was composited. Four times the pixels reach the same
  `AdaptiveAvgPool2d(1)`, so the pooled feature is a mean over 80 spatial positions rather than
  20 and its scale differs; `learning_rate` 1e-3 (what notebook 05 used, and fine at 60×80) is
  too large for it. **LR is now 5e-4 with gradient clipping at 1.0, for BOTH resolutions** — the
  arm that did not need it is changed too, because tuning the LR per arm would make the arms
  differ in two things and destroy the resolution comparison. Verified at 120×160: spread
  escapes at epoch 5–6 (0.045 → 0.334 → 0.995) and holds ~1.15.

  The collapse guard was fixed twice. It first compared spread against a peak set by the
  *random init* (~0.006), so 15% of it (0.0009) was a floor a fully collapsed run (0.0010) sat
  above and never tripped; an absolute floor of 0.05 after the warmup was added. That relative
  test then had to be gated on the peak having cleared the floor as well, because during the
  plateau every spread is ~0.01 and a ratio between two such numbers is noise — with the lower
  LR the plateau is longer, and it would have killed healthy runs.

  **The deployed model is the 120×160 mixed-background encoder**, seed 42, best epoch 40 of 42,
  102 min, promoted by `scripts/promote_task4_encoder.py`. Its benchmarks:

  | benchmark | P@1 | P@10 | colour@10 | both@10 |
  |---|---|---|---|---|
  | clean | 79.70 | 76.19 | 55.22 | 41.98 |
  | hard (checker/stripes) | 56.15 | 56.32 | 48.52 | 28.84 |
  | photo (held-out Places categories) | 57.55 | 55.78 | 48.68 | 29.40 |
  | wild | 53.20 | 53.07 | 47.27 | 27.33 |
  | wildphoto | 52.80 | 51.62 | 45.36 | 26.20 |

  **The label sheet is filled in (2026-09-09) and the composited benchmark turns out to be an
  optimistic proxy.** `A2_FashionDataset/input_images_labels.csv` now carries 23 scorable rows;
  8 are marked `none` - two stickers, a set of sleep masks with no catalogue class, a six-item
  flat-lay, and four street-style shots where two garments are co-equal. On those 23:

  | encoder | P@1 | P@10 | vs its composited `photo` P@10 |
  |---|---|---|---|
  | arm D, deployed | 17.39 | **16.52** | 55.78 |
  | arm P, procedural | 17.39 | 19.57 | 42.22 |
  | arm C, control | 13.04 | 12.17 | 11.84 |

  **The absolute level does not survive**: 16.52 against the 55.78 the `photo` benchmark
  reports for the same encoder. Compositing a catalogue cutout onto a Places365 scene is easier
  than a real upload, so `photo` and `wildphoto` should be read as upper bounds, not estimates.
  Quote 16.52 whenever a real-world number is wanted.

  **These figures replaced 21.74 / 21.30 / 8.70 on 2026-09-11 when `rembg` was installed, and
  the model did not change - the segmentation did.** Every earlier real-photo number in this
  repo was measured through `cv2.grabCut`, the fallback tier `foreground_mask` uses when `rembg`
  is absent, whose GMM initialisation draws from OpenCV's process-global RNG. `rembg` (u2netp)
  is the ladder's top tier, is deterministic, and is now installed, so **the harness reproduces
  exactly**: two consecutive invocations of `eval_task4_real_photos.py` were diffed and are
  byte-identical, and Task 1's 7-repeat spread collapsed from +/-1.64 to **0.00**. It also
  segments all 31 photographs, where grabCut declined 1 and fell back to a centre crop. This is
  the third real-photo table (26.09 / 23.04, then 21.74, now 16.52) and it is the first that
  does not ride on an RNG; earlier values are superseded, not alternatives.

  **The gap between the augmented arms and the control narrowed sharply, and that is a finding
  rather than a regression.** Arm D was 2.5x arm C under grabCut (21.74 against 8.70) and is
  **1.36x** under `rembg` (16.52 against 12.17), while on composites it stays 4.7x. Arm C is the
  catalogue-only control, so a clean cutout on white is precisely the distribution it was
  trained on: better segmentation hands it back most of what the missing augmentation cost it.
  **Segmentation and background augmentation are substitutes, not complements** - they attack
  the same failure, so their gains do not add. That is the same effect already recorded for
  Task 3's ingest router, now measured for retrieval.

  **Arm P is no longer behind arm D on real photographs - it is ahead on P@10** (19.57 against
  16.52) and tied on P@1, while the composited benchmark separates them by 13.6 points the other
  way (42.22 against 55.78). At n=23 this does not refute the composited ranking; it says 23
  photographs cannot confirm it, and that a 13-point gap on composites must never be quoted as a
  gap on uploads. Nothing here justifies promoting arm P - see the standing rule above.

  Two caveats travel with that row: **n=23**, so the sampling error is roughly +/-10 points,
  which makes the gap to 55.78 safe and every ordering within the table unsafe; and one P@1 is
  worth 4.35 points, so single-photograph moves dominate.

  **The labels were drafted by a vision model and hand-verified on 2026-09-10**, including the
  three rows on the `Casual Shoes` / `Sports Shoes` boundary that the draft had flagged. They
  are ground truth now, not a second opinion. They agree with the shipped Task 1 classifier's
  top-1 on only 3 of 23, so they are not an echo of a model this project also scores.

  `EXCLUDED_ARTICLE_TYPES` in `src/evaluation/real_photo_arms.py` is the single definition of
  the `none` marker, imported by `scripts/make_label_contact_sheet.py`. It exists because
  `real_photo_labels` originally kept every non-empty `articleType`: the 8 excluded rows would
  have been scored against a class no catalogue item has, counted as misses, and quietly cost
  about a quarter of the reported P@10 while claiming `labelled = 31`.

  Two control arms were trained to establish the claims below and then **deleted**, because the
  project keeps one encoder. Their measurements are recorded here and nowhere else.

  **Resolution (60×80 mixed vs 120×160 mixed, same recipe).** P@10: clean 76.89 → 76.19, hard
  55.69 → 56.32, photo 54.12 → 55.78, wild 52.51 → 53.07, wildphoto 49.40 → 51.62. Against a
  ±1.1 sampling floor, `clean`, `hard` and `wild` are ties; only `photo` (+1.66) and `wildphoto`
  (+2.22) clear it. `colour@10` rises on all five (+1.44 to +3.98) — the first evidence that part
  of the 60×80 colour deficit is resolution rather than labelling. Cost: 102 min against 24,
  **4.3x**, one seed each. Halving the LR to 5e-4 for both arms cost 60×80 about 2.5–3 points out
  of domain against its own 1e-3 run; accepted so the arms differ in resolution alone.

  **Places365 backdrops beat procedural ones on photographs by a wide margin, and lose on
  procedural benchmarks.** `photo` **+12.92** and `wildphoto` **+14.14**; `hard` −3.18 and
  `wild` −3.72. Both directions are many times the noise floor.

  **The important consequence: the procedural encoder is not background-invariant, and the old
  benchmark could not tell.** (That encoder is **arm P** of the background grid below, and the
  same rows now appear in `05` Part 6b. One run, two places, not two experiments.) Between two background families it never trained on, it scores
  59.50 (checker/stripes) and 42.86 (real photographs) — a **16.6-point** gap. The mixed encoder
  scores 56.32 and 55.78, a **0.5-point** gap. The claim inherited from the
  old background notebook — "the encoder learned that backgrounds are irrelevant, rather than
  memorising which backgrounds it had seen" — is therefore **too strong**: it had learned that
  *procedural* backgrounds are irrelevant. Checkerboards and
  stripes are still formulae, so the "disjoint" eval bank was never neutral — it is much closer
  in family to solid/gradient/noise/blobs than to a photograph, and it favours procedurally
  trained models. Grade on both families or the invariance claim is not supported.

  **Promotion moves four files as one.** The encoder trains on the 38,571-row
  `task4_gallery_120x160.csv`; an older served index held 38,612. `SearchEngine.load` raises on
  the mismatch by design, so `scripts/promote_task4_encoder.py` rebuilds the index, the
  evaluation index, `gallery_metadata.csv` and `gallery_ids.npy` together and rewrites the
  manifest. Promoting a checkpoint by hand will break the app.

  **Two galleries, and conflating them hides 41 products.** The encoder trains and is measured
  on `task4_gallery_120x160.csv` (38,571 rows — the 41 conflicting-label images are dropped,
  because a row with an unreliable label cannot referee anything). The **served** index covers
  all **38,612**, because a conflicting `gender` label is no reason a customer cannot find the
  product. The first promotion served the training gallery by mistake and
  `test_the_served_index_covers_the_whole_catalogue` caught it.

  **Band regions are now masked to wearable rows BEFORE the top-k, not filtered after it.**
  The upper band of `600_google-pattern-socks.jpg` has all 120 of its nearest neighbours in
  `Personal Care` under this encoder, so the post-filter found nothing to keep and fell back to
  returning perfume bottles for a photograph of socks — the exact failure the filter exists to
  prevent. Masking first guarantees a band ranks among garments however far down the nearest one
  sits. Note the filter keys on `masterCategory`, so a perfume mis-filed under `Accessories`
  still gets through; that is a catalogue labelling problem, not a retrieval one.

  **A resolution mismatch is silent in this architecture, and it cost a wrong number.**
  `ImprovedEncoder` is fully convolutional with an `AdaptiveAvgPool2d(1)`, so it accepts any
  input size without raising. Notebook 06's cluster-stability cell hardcoded
  `search_cache_60x80.npy` and composited at `(80, 60, 3)`; against the 120×160 encoder it ran
  happily and embedded every frame at half the scale it was trained for, reporting **19.9%**
  cluster stability. Driven from `MANIFEST["image_size_pil"]` instead, the same test reports
  **58.7%** — and that is now the *harder* test, because it composites onto held-out Places365
  scenes rather than the procedural patterns the old figure used. Any cell that loads a cache by
  literal size is a bug waiting for the next resolution change.

  **The served metadata fills missing labels with `"Unknown"`.** `clean_train_metadata.csv`
  leaves 14 rows without a `baseColour`, 72 without a `usage` and 7 without a
  `productDisplayName`. Left as NaN they are floats, so any consumer that slices a label for a
  plot title dies with `'float' object is not subscriptable` — and *which* rows surface depends
  on the encoder, so the failure moves when the model changes. `promote_task4_encoder.py` fills
  them.

  **The background grid — what the catalogue's uniformity costs, and what fixes it.** One
  architecture, three training distributions, all scored on identical query frames
  (`build_queries(seed=123)`), so every difference is paired. Same `ImprovedEncoder`, same
  recipe, same seed, same split, same 42-epoch budget; the training backdrop is the only thing
  that differs. P@10, exact search in every cell:

  | benchmark | C catalogue only | P procedural backdrops | D mixed backdrops (deployed) |
  |---|---|---|---|
  | clean | **81.64** | 76.35 | 76.19 |
  | hard | 9.10 | **60.13** | 56.32 |
  | photo | 11.84 | 42.22 | **55.78** |
  | wild | 9.21 | **57.17** | 53.07 |
  | wildphoto | 6.36 | 36.54 | **51.62** |

  **Arm P was retrained on 2026-09-11 and now has a live checkpoint and an interval.** The rows
  above are that run, scored beside C and D by `compare_task4_background_arms.py` in one pass;
  the notebook reads that one file and no longer splices from
  `outputs/task4_resolution_comparison.csv`. The checkpoint is
  `artifacts/task4_120x160/task4_encoder_procedural_seed42.pt`, gitignored like the others, so a
  clone still has only the table. **Do not promote it** - `promote_task4_encoder.py` would serve
  it happily and the only thing on disk that distinguishes it is `background_source: procedural`.

  **That retrain measured Task 4's run-to-run variance for the first time.** Same seed, same
  recipe, same script: the earlier procedural run scored 76.68 / 59.50 / 42.86 / 56.79 / 37.48
  against this one's 76.35 / 60.13 / 42.22 / 57.17 / 36.54 - **max drift 0.94 points, mean
  0.59**. CUDA training is not bit-reproducible because `cudnn.benchmark` picks its convolution
  algorithms at runtime. That is the same order as the ±1.10 query-sampling floor, so the two
  noise sources are comparable and a gap of about a point is not a result however the bootstrap
  interval reads.

  **Arm P is why the invariance claim is now falsifiable.** `hard` and `photo` are both
  background families no encoder trained on. Arm P scores 60.13 and 42.22 — a **17.91-point
  gap** — while arm D scores 56.32 and 55.78, a **0.54-point gap**. Procedural augmentation
  teaches that *formulae* are irrelevant, not that backgrounds are. A checkerboard eval bank
  would have ranked arm P first on `hard` and `wild` (it beats the deployed encoder on both) and
  hidden the failure that matters. Its trade against arm C is 5.74:1 against the deployed
  8.06:1, so the ratio alone does not reveal it either. On `both@10`, paired: arm P costs 6.30
  in domain and buys 19.52 out of it (3.1:1) against arm D's 5.62 for 24.36 (4.3:1).

  **Arm C is the best model in the project on clean catalogue images and near the worst on a
  photograph** — 81.64 clean against the deployed encoder's 76.19, and 11.84 on `photo`. The
  catalogue's uniformity does not merely fail to generalise; it yields a model that beats the
  deployed one on the academic metric and is useless on an upload. Do not quote a clean-only
  P@10 as a headline for this task without the out-of-domain column beside it.

  **The intervention's price is measured, not assumed.** D − C on `both@10`, paired over 2,000
  queries: clean **−5.62** [−6.63, −4.65], hard **+26.03**, photo **+24.17**, wild +23.78,
  wildphoto +23.47 — every one significant. About a 4.3:1 trade, and the first measurement that
  justifies `DEPLOYMENT_WEIGHT = 3.0` in the epoch-selection rule, which was already valuing
  out-of-domain at 3:1 on no evidence.

  **A detail worth keeping**: arm C does not degrade, it collapses. 81.64 clean against 9.10,
  11.84, 9.21 and 6.36 out of domain, against a random-retrieval floor of 5.91. Given a
  background that never varies, it leaned on it all the way.

  **The two classical arms were removed from notebook 05 on 2026-09-10, at the user's explicit
  request** — they found the two-model-family axis confusing, and the three-arm version says
  what the section is for more directly. The grid used to be a 2x2 (k-Means over
  `classical_features` crossed with the same data treatment) plus arm P. Do not re-add them
  without being asked. Their measurements, recorded here because they are now recorded nowhere
  else in prose:

  | benchmark | A classical/catalogue | B classical/augmented |
  |---|---|---|
  | clean | 69.31 | 35.74 |
  | hard | 16.39 | 28.56 |
  | photo | 10.17 | 20.93 |
  | wild | 13.17 | 18.79 |
  | wildphoto | 8.58 | 13.13 |

  What went with them, and what it cost:

  - **The claim "the improvement is the representation, not just the augmentation" no longer has
    a control.** The same intervention had *opposite* value in the two families — classical
    −33.57 clean for +10.76 photo (0.32:1), encoder −5.45 for +43.94 (8.06:1) — which is what
    established that augmentation only pays when a representation can absorb it. Arm P now
    carries a weaker version of the same idea (what the augmentation *contains* matters), which
    is why removing A and B is defensible rather than free.
  - **Arm B's trade curve was monotone, with no sweet spot**: averaging N composited views gives
    clean/photo 69.31/10.17 (N=0), 49.73/14.49 (N=1), 35.74/20.93 (N=3). No setting of N buys
    both. Arm A also reproduced to the digit across two runs, a free determinism check.
  - **On `hard` and `wild`, arm C scored *below* arm A** (9.10 vs 16.39, 9.21 vs 13.17) — a
    network trained only on clean frames is more background-brittle than a hand-built histogram.
  - **Clustering was never the bottleneck.** Exact search and k-Means routing (k=120, 3 probes,
    ~5% of the catalogue) differed by at most 1.15 points, one hair outside the ±1.10 floor,
    while the arms differed by tens.
  - **The project now contains no unsupervised learning at all.** That k-Means was the last of
    it. Tasks 1-3 are supervised CNNs and the Task 4 encoder is a supervised triplet net. The
    user was told this before deciding.
  - `scripts/eval_task4_classical_clustering.py` and its
    `outputs/evaluation/task4_classical_clustering_arms_v{1,3}.csv` **still exist and still run**
    — nothing reads them now. Left in place deliberately: the CSVs are the evidence for the rows
    above, and the repo's convention is that a script generating a committed measurement table
    stays. Recoverable notebook prose is in git at the commit before this change.

  One seed per arm; the intervals are paired query-sampling error, not seed variance, which is
  still unmeasured for retrieval. **Arm C is a control and must never be promoted** —
  `promote_task4_encoder.py` would serve it happily. Its checkpoint records
  `background_augmented: False`, which is the only thing on disk that distinguishes it.

  **`ImprovedEncoderV2` was deleted on 2026-09-10** (GeM pooling, CosFace head, block-2 colour
  branch). It was implemented and unit-tested but never trained, and no trainer could reach it:
  `train_task4_120x160.py` has no `--arch` flag and never calls `build_encoder`. No checkpoint ever
  recorded `architecture: improved_v2`, so nothing selected it. Removed with its `GeM` and
  `CosineHead` helpers, used nowhere else, and its nine tests; the suite went 140/6 to 132/5,
  exactly the nine removed. Recoverable from git if it is ever wanted.

  `ARCHITECTURES` in `src/visual_search/search_engine.py` now has one entry, `"improved"`. Keep the
  table rather than hardcoding the class: it makes an unknown `architecture` name fail loudly with
  the list of valid names instead of loading weights into the wrong network, and its default keeps
  checkpoints written before that field existed loadable.

### Task 3 label policy (do not silently change)

`usage`: 8 raw classes → 4. `Smart Casual` (55) and `Travel` (25, all bags) merge into `Casual`;
`Party` (13) merges into `Formal`; `Home` (1, a cushion cover) and 72 missing-label rows are
dropped — 73 rows total, 0.19%. The pipeline drops a further **41 rows** whose duplicate
images carry conflicting gender/usage labels (`use_for_supervised=False`), leaving 38,498.
Pooling the rare ones into an `Other` class was tried and failed (F1 = 0.000, ~15 points off
usage macro-F1). `CONFIG["rare_strategy"]` in the EDA notebook still
supports `merge` / `drop` / `group`.

`gender`: 5 classes. `Unisex` (F1 0.599) and `Girls` (0.607) are the weak ones, and neither is
resolution-limited or fixable by reweighting — see the `class_weighted` result above. An earlier
note that makeup is systematically predicted `Men` **did not reproduce** — cosmetics score 72%
and fragrance 100% on the held-out set. The real limitation is **Accessories: 38.7% of all gender
errors**, and CELL 25 establishes what kind of limitation it is. It is **not** mainly men's vs
women's ranges looking alike: **39% of all gender errors place or displace `Unisex`**, and
forgiving those alone would take gender accuracy from 90.33% to **94.07%**. The label is often not
in the pixels at all. Seven families of byte-identical photographs carry different genders (one
Ray-Ban aviator is filed as both `Men` and `Unisex`, one Maxima watch as both `Men` and `Women`),
and a gender word in `productDisplayName` matches the label on **98.8% of test rows and 100% of
accessories** — the attribute records how the seller listed an item. That is diagnosis, not a
shortcut: `styles_prediction_template.csv` carries no product name, and a text model would be off
brief. Neither 120×160 nor class weighting moved it, and neither will. If makeup errors still show in the app, that is on
user photos (out-of-distribution), not catalogue data.

## Conventions

- **One definition per architecture.** Services import from `src/models/` (e.g. Task 1's
  `ItemTypeCNN`); they must never redeclare a network. Task 3 is the exception and predates the
  rule: `EarlyBranchCNN` is declared in `task3_service.py`, and `tests/test_task3_service.py`
  covers it there. A duplicated definition previously left a
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
