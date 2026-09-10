"""Where Task 1's remaining error actually sits: family agreement and symmetric pairs.

Why this exists
---------------
Notebook 02's ``CELL 34b`` computed this from kernel state - ``y_test_pred`` and
``y_test_true``, which only exist after the notebook has trained its models. That
made a five-hour run the price of one table, and it meant the cell described the
notebook's 60x80 model rather than the checkpoint that ships. It had never been
executed, so the notebook showed a blank cell where the argument should be.

This computes the same thing from a checkpoint, writes it to
``outputs/evaluation/``, and lets the notebook read a number instead of
reproducing one.

What it measures
----------------
Two things, and the pair is the point:

``family accuracy`` marginalises the 92 ``articleType`` classes up to their 33
``subCategory`` families and asks whether the coarse answer is right. It runs far
above the fine-grained score, which says the model almost always knows what kind
of object it is looking at.

``symmetric confusion mass`` is the share of errors sitting in pairs confused in
BOTH directions - ``Sports Shoes`` called ``Casual Shoes`` and ``Casual Shoes``
called ``Sports Shoes``. Symmetric is the diagnostic word. A class the model had
simply not learned would be confused in one direction; confusion running both
ways at similar rates is two labels competing to describe one appearance.

Together they say the residual is the catalogue's choice of word, not the
model's grasp of the image, which is why pairwise specialists were measured and
declined rather than pursued.

    python scripts/task1_error_structure.py                 # the shipped 120x160 model
    python scripts/task1_error_structure.py --checkpoint <path>

Writes ``outputs/evaluation/task1_error_structure.csv`` (the headline numbers)
and ``..._pairs.csv`` (every symmetric pair, largest first).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import load_or_create_splits  # noqa: E402
from src.data.taxonomy import subcategory_of  # noqa: E402
from src.training.candidate_120x160 import CandidateDataset, task1_frame_parts  # noqa: E402
from src.training.train_task1_120x160 import (  # noqa: E402
    RESOLUTIONS,
    NormalizeOnly,
    make_model,
)

DEFAULT_CHECKPOINT = (PROJECT_ROOT / "artifacts" / "task1_120x160"
                      / "task1_120x160_onecycle_best.pt")
OUT_MAIN = PROJECT_ROOT / "outputs" / "evaluation" / "task1_error_structure.csv"
OUT_PAIRS = PROJECT_ROOT / "outputs" / "evaluation" / "task1_error_structure_pairs.csv"


def test_predictions(checkpoint_path, resolution, batch_size=256):
    """(true, predicted, class_names) on the held-out split, through the training path.

    Deliberately reuses ``train_task1_120x160``'s own loaders rather than a second
    copy of the preprocessing. The checkpoint's recorded ``test_metrics`` were
    produced by that code, so any private reimplementation here could disagree
    with the number the notebook prints two cells earlier.
    """
    from torch.utils.data import DataLoader

    arm = RESOLUTIONS[resolution]
    train, val, test = load_or_create_splits()
    (train, val, test), classes = task1_frame_parts(train, val, test)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    names = [str(c) for c in checkpoint["class_names"]]
    if len(names) != len(classes):
        raise SystemExit("checkpoint has {} classes, split has {}".format(
            len(names), len(classes)))

    # Normalisation comes from the checkpoint, which records the statistics it was
    # trained under, rather than from a side file that can go missing or drift.
    mean = np.asarray(checkpoint["channel_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["channel_std"], dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(len(names), dropout=0.4).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset = CandidateDataset(test, target="label", source_dir=arm["source"],
                               image_size=arm["size"], transform=NormalizeOnly(mean, std))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    trues, preds = [], []
    # TTA follows the checkpoint, as the service and predict.py both do, so this
    # table describes the same configuration that ships.
    tta = bool(checkpoint.get("tta", False))
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images).float()
            if tta:
                logits = logits + model(torch.flip(images, dims=[3])).float()
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(labels.numpy())

    return np.concatenate(trues), np.concatenate(preds), names


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--resolution", default="120x160", choices=tuple(RESOLUTIONS))
    parser.add_argument("--top-pairs", type=int, default=8)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit("checkpoint not found: {}".format(args.checkpoint))

    y_true, y_pred, names = test_predictions(args.checkpoint, args.resolution)
    class_names = np.asarray(names)
    families = np.asarray(subcategory_of(names))

    errors = y_pred != y_true
    family_correct = families[y_pred] == families[y_true]
    n_errors = int(errors.sum())

    pair_counts = Counter(
        (class_names[t], class_names[p]) for t, p in zip(y_true[errors], y_pred[errors]))
    symmetric = {}
    for (a, b), count in pair_counts.items():
        if (b, a) in pair_counts:
            symmetric[tuple(sorted((a, b)))] = pair_counts[(a, b)] + pair_counts[(b, a)]
    symmetric_mass = sum(symmetric.values())

    summary = pd.DataFrame([{
        "checkpoint": args.checkpoint.name,
        "resolution": args.resolution,
        "rows": int(len(y_true)),
        "articleType_accuracy": round(float((~errors).mean() * 100), 2),
        "family_accuracy": round(float(family_correct.mean() * 100), 2),
        "errors": n_errors,
        "errors_keeping_family": int((errors & family_correct).sum()),
        "errors_keeping_family_pct": round(
            float((errors & family_correct).sum() / n_errors * 100), 1),
        "symmetric_error_mass": symmetric_mass,
        "symmetric_error_mass_pct": round(symmetric_mass / n_errors * 100, 1),
    }])

    pairs = pd.DataFrame(
        [{"pair": "{} <-> {}".format(a, b), "errors": count,
          "pct_of_all_errors": round(count / n_errors * 100, 1)}
         for (a, b), count in sorted(symmetric.items(), key=lambda kv: -kv[1])])

    OUT_MAIN.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_MAIN, index=False)
    pairs.to_csv(OUT_PAIRS, index=False)

    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print()
    print(pairs.head(args.top_pairs).to_string(index=False))
    print()
    print("wrote {}".format(OUT_MAIN.relative_to(PROJECT_ROOT)))
    print("wrote {}".format(OUT_PAIRS.relative_to(PROJECT_ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
