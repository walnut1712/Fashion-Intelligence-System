# Scripts

Run these commands from the repository root with the project environment active.
The API and saved notebooks use the existing checkpoints; these scripts rebuild
data, run experiments, or refresh outputs when needed.

## Main commands

| Command | Purpose |
|---|---|
| `python scripts/check_project.py --load-models` | Verify dependencies, tracked model/data files and all six API services without rebuilding. |
| `python scripts/build_submission.py` | Build the classification submission from the saved task models. |
| `python scripts/build_task1_cache.py` | Prepare the historical Task 1 60x80 image cache. |
| `python scripts/build_task4_cache.py --resolution 120x160` | Prepare Task 4's image and foreground-mask caches. |
| `python scripts/build_task4_outputs.py` | Export Task 4 retrieval results for the test images. |
| `python scripts/promote_task4_encoder.py --encoder PATH` | Deploy an encoder and rebuild its matching indexes and metadata together. |
| `python scripts/refresh_task1_fixture.py` | Refresh saved Task 1 predictions from the deployed checkpoint. |
| `python scripts/train_task1_task2_background_adaptation.py --task 1 --tag _local` | Fine-tune a Task 1 or Task 2 checkpoint with mixed backgrounds. |
| `python scripts/train_task3_background_adaptation.py --help` | Show Task 3 background-adaptation options. |
| `python scripts/refit_task3_temperature.py --checkpoint PATH --dry-run` | Inspect confidence calibration before updating a Task 3 checkpoint. |
| `python scripts/eval_task1_task2_benchmarks.py --task 1` | Compare Task 1 or Task 2 background arms on five benchmark families. Requires the experiment checkpoints listed in the script. |
| `python scripts/eval_task3_benchmarks.py --help` | Show the equivalent Task 3 benchmark options. |
| `python scripts/compare_task4_background_arms.py` | Compare Task 4's clean and mixed-background encoders. |
| `python scripts/eval_task1_task4_semantic_rerank.py --help` | Show options for comparing Task 1 semantic reranking against Task 4 cosine search. |
| `python scripts/analyze_task1_errors.py --checkpoint PATH` | Measure Task 1 family errors and symmetric confusion pairs. Defaults to the catalogue-only 120x160 reference. |

Training and background benchmarks need the local dataset, prepared caches and
Places365 backgrounds. Some experiment checkpoints and all large caches are
ignored by Git; pulling the repository does not create them. Benchmark tables
already saved under `outputs/evaluation/` can be read without rerunning training.

## Task 1 adaptation and Task 4 reranking experiment on Windows

```powershell
# Show the exact commands without starting training or evaluation.
.\scripts\run_task1_task4_reranking_experiment.ps1 -DryRun

# Fine-tune Task 1 for eight epochs, then evaluate that exact new checkpoint.
.\scripts\run_task1_task4_reranking_experiment.ps1 -Tag _rerank_experiment
```

The launcher finds the repository using its own location, so an absolute launcher
path also works from another directory. It uses `.venv\Scripts\python.exe` by
default; pass `-Python PATH` to select another interpreter. `-Epochs`,
`-TuneQueries`, `-TestQueries` and `-Pool` control the run. Python failures stop
the sequence immediately and restore the caller's working directory.

If Windows blocks `.ps1` execution, run the check in a separate PowerShell process:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_task1_task4_reranking_experiment.ps1 -DryRun
```

This applies only to that process; it does not change the machine's execution policy.

The default tag is `_rerank_experiment`. The new checkpoint is written to
`artifacts/task1_bgaug_rerank_experiment/task1_bgadapt_best.pt` and passed explicitly to
the second stage. Comparison and reranking tables receive the same suffix, for
example `outputs/evaluation/task1_t4_rerank_test_rerank_experiment.csv`. Use a new tag to
keep another run's outputs; repeating a tag replaces that experiment's outputs.
The launcher does not promote a model into the application.

The standalone reranking evaluator defaults to the deployed background-adapted
Task 1 checkpoint. `--task1-adapted PATH` selects an experiment checkpoint,
`--task1-baseline PATH` selects its reference, and `--tag _name` separates its
output tables. Checkpoint paths can be absolute or relative to the repository.

## Names and historical tools

Script names use a verb followed by explicit task numbers. Existing result CSV
names stay stable so notebooks can continue reading the recorded evidence.

| Previous path | Current path |
|---|---|
| `scripts/train_t12_background_adaptation.py` | `scripts/train_task1_task2_background_adaptation.py` |
| `scripts/eval_t12_benchmarks.py` | `scripts/eval_task1_task2_benchmarks.py` |
| `scripts/evaluate_t1_t4_semantic_rerank.py` | `scripts/eval_task1_task4_semantic_rerank.py` |
| `scripts/evaluate_tta_views.py` | `scripts/eval_task1_tta_views.py` |
| `scripts/task1_error_structure.py` | `scripts/analyze_task1_errors.py` |
| `scripts/run_t1_t4_overnight.ps1` | `scripts/run_task1_task4_reranking_experiment.ps1` |
| `src/training/test_task3b_recommendation.py` | `scripts/demo_task3b_recommendation.py` |

`eval_task1_tta_views.py` measures the historical 60x80 Task 1 reference.
`eval_task4_classical_clustering.py` preserves a historical comparison; clustering
is not part of the current Task 4 notebook or application. These are optional
experiments, not setup steps.

`demo_task3b_recommendation.py` prints example occasion recommendations from the
optional SFS data and `artifacts/task2b_sfs/` bundle. It is a manual demonstration;
automated tests live under `tests/`. Real-photo evaluators, contact sheets and
`build_similarity_review.py` are optional inspection tools.
