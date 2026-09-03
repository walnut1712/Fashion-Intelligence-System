#!/usr/bin/env python
"""Adopt a candidate as the Task 1 model, and bring every artefact with it.

Why this is a script and not a `cp`
-----------------------------------
Promotion is six coupled steps, and skipping any one of them leaves the
repository in the state its own history warns about: a checkpoint on disk that
nothing in the tree can rebuild, with metrics tables describing a different
model. `tests/test_prediction.py::test_saved_predictions_are_reproducible`
exists precisely to catch that - it replays the committed
`artifacts/task1/task1_predictions.csv` through the serving path and asserts the
labels and confidences still come back. Swap the checkpoint without regenerating
that CSV and the test fails, which is the intended behaviour, not an obstacle.

So this does all six, in order:

    1. back the current model up, tagged with the run it came from
    2. copy the candidate into artifacts/task1/task1_cnn.pt
    3. regenerate artifacts/task1/task1_predictions.csv (the regression fixture)
    4. regenerate outputs/task1_item_type_predictions.csv (the Task 1 column)
    5. rebuild outputs/predictions/styles_prediction.csv (all four columns)
    6. re-sync task1_summary.json and the checkpoint's own recorded config

and then tells you to run the tests, which is the only step it will not do for
you - a promotion that has not been verified is not finished.

The candidate must already be calibrated (`--calibrate`), because the serving
path reads `logit_adjustment_tau` and `class_counts` from the checkpoint and a
candidate straight out of `train()` carries neither.

Usage
-----
    python scripts/promote_checkpoint.py artifacts/task1/candidate_baseline_augnone.pt
    python scripts/promote_checkpoint.py <candidate> --dry-run
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "task1"
DEPLOYED = ARTIFACT_DIR / "task1_cnn.pt"
FIXTURE = ARTIFACT_DIR / "task1_predictions.csv"
TASK1_OUT = PROJECT_ROOT / "outputs" / "task1_item_type_predictions.csv"
TEST_IMAGES = (PROJECT_ROOT / "A2_FashionDataset" / "FashionDataset" / "test"
               / "images_test")


def _check_calibrated(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    missing = [key for key in ("class_log_prior", "class_counts")
               if checkpoint.get(key) is None]
    if missing:
        raise SystemExit(
            "{} is missing {} - run\n"
            "  python -m src.training.train_item_type --calibrate {}\n"
            "first, or serving will fall back to an unadjusted operating point."
            .format(path.name, " and ".join(missing), path))
    return checkpoint


def _regenerate_predictions(verbose=True):
    """Rewrite the regression fixture and the Task 1 submission column."""
    from src.models.item_type_classifier import (choose_device, load_item_type_model,
                                                 predict_proba)

    device = choose_device()
    model, checkpoint = load_item_type_model(DEPLOYED, device)
    class_names = np.asarray(checkpoint["class_names"])
    paths = sorted((p for p in TEST_IMAGES.iterdir() if p.suffix.lower() == ".jpg"),
                   key=lambda p: int(p.stem))
    if verbose:
        print("scoring {} graded tiles".format(len(paths)))

    probabilities = predict_proba(model, checkpoint, paths, device=device,
                                  tta=bool(checkpoint.get("tta", False)))
    labels = class_names[probabilities.argmax(1)]
    ids = [int(p.stem) for p in paths]

    # The fixture carries two extra columns the submission does not: the image
    # path, and the confidence the test asserts against.
    pd.DataFrame({
        "id": ids, "gender": "", "articleType": labels, "season": "", "usage": "",
        "image_path": ["A2_FashionDataset/FashionDataset/test/images_test/{}.jpg".format(i)
                       for i in ids],
        "articleType_confidence": probabilities.max(1).round(4),
    }).to_csv(FIXTURE, index=False)

    pd.DataFrame({"id": ids, "gender": "", "articleType": labels,
                  "season": "", "usage": ""}).to_csv(TASK1_OUT, index=False)
    if verbose:
        print("wrote {} and {}".format(FIXTURE.name, TASK1_OUT.name))
    return labels


def promote(candidate, dry_run=False, verbose=True):
    candidate = Path(candidate)
    if not candidate.exists():
        raise SystemExit("no such candidate: {}".format(candidate))
    incoming = _check_calibrated(candidate)

    outgoing = torch.load(DEPLOYED, map_location="cpu", weights_only=False) \
        if DEPLOYED.exists() else {}
    if verbose:
        print("replacing {} (run {}) with {} (run {})".format(
            outgoing.get("model_name", "?"), outgoing.get("run_id", "?"),
            incoming.get("model_name", "?"), incoming.get("run_id", "?")))
        for key in ("weighted_f1", "macro_f1", "balanced_acc"):
            before = (outgoing.get("test_metrics") or {}).get(key)
            after = (incoming.get("test_metrics") or {}).get(key)
            if before is not None and after is not None:
                print("  {:<14} {:.2f} -> {:.2f}  ({:+.2f})".format(
                    key, before, after, after - before))
        # These are each checkpoint's *self-recorded* metrics, computed at
        # whichever tau its own training run happened to pick. They are not a
        # like-for-like comparison: calibration may since have moved tau, and it
        # does - candidate_baseline_augnone records balanced_acc 72.10 at its
        # training tau and scores 74.32 at the calibrated 0.40. For the numbers
        # to compare against, read the table that scores every candidate on one
        # split with tau re-selected per candidate on validation.
        print("  (self-recorded, per-run tau; for a like-for-like ranking see"
              " outputs/evaluation/task1_recipe_comparison.csv)")
    if dry_run:
        print("\ndry run - nothing written")
        return

    stamp = outgoing.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ARTIFACT_DIR / "superseded_task1_cnn_{}.pt".format(stamp)
    if DEPLOYED.exists():
        shutil.copy2(DEPLOYED, backup)
        if verbose:
            print("backed up to {}".format(backup.name))
    shutil.copy2(candidate, DEPLOYED)

    _regenerate_predictions(verbose=verbose)

    for command in (
        [sys.executable, "-m", "src.training.train_item_type", "--sync-summary"],
        [sys.executable, str(PROJECT_ROOT / "scripts" / "build_submission.py")],
    ):
        if verbose:
            print("\n$ {}".format(" ".join(Path(c).name if "/" in c or "\\\\" in c else c
                                            for c in command)))
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    print("\npromoted. Now verify, because nothing above proves the swap is sound:")
    print("  python -m pytest tests/ -q")
    print("Expect test_saved_predictions_are_reproducible to pass - it replays the "
          "regenerated fixture through the serving path.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="show the metric change without writing anything")
    args = parser.parse_args(argv)
    promote(args.candidate, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
