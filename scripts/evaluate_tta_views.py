#!/usr/bin/env python
"""Measure the wider test-time augmentation sets against the shipped flip average.

TTA has always been horizontal-flip only. ``item_type_classifier.TTA_VIEWS`` adds
small scales and shifts on top, each averaged with its mirror. This scores them on
**validation** - the test split is not a knob to tune against - and writes the
table so the result is recorded whether it helps or not.

The 3-seed noise floor on validation weighted-F1 is +/-0.24
(``artifacts/task1/best_config.json``), so anything under ~0.5 here is a tie.

Usage
-----
    python scripts/evaluate_tta_views.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.item_type_classifier import TTA_VIEWS, load_item_type_model  # noqa: E402
from src.training.train_item_type import load_splits, predict_split, score  # noqa: E402

OUT = PROJECT_ROOT / "outputs" / "evaluation" / "task1_tta_views_ab.csv"
NOISE_FLOOR = 0.24


def main():
    device = torch.device("cpu")
    train_df, val_df, test_df, class_names, images = load_splits(verbose=False)

    model, checkpoint = load_item_type_model(
        PROJECT_ROOT / "artifacts" / "task1" / "task1_cnn.pt", device)
    mean = torch.tensor(checkpoint["channel_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(checkpoint["channel_std"]).view(1, 3, 1, 1)

    x_val = torch.from_numpy(np.ascontiguousarray(
        images[val_df["cache_position"].to_numpy()].transpose(0, 3, 1, 2)))
    y_val = val_df["label"].to_numpy()
    del images

    rows = []
    arms = [("flip (shipped)", None, True)] + [(name, name, False) for name in TTA_VIEWS]
    for label, views, tta in arms:
        probabilities = predict_split(model, x_val, mean, std, tta=tta, views=views)
        metrics = score(y_val, probabilities.argmax(1).cpu().numpy())
        passes = 2 if views is None else 2 * len(TTA_VIEWS[views])
        rows.append({
            "arm": label,
            "forward_passes_per_image": passes,
            "accuracy": round(metrics["accuracy"], 3),
            "weighted_f1": round(metrics["weighted_f1"], 3),
            "macro_f1": round(metrics["macro_f1"], 3),
            "balanced_acc": round(metrics["balanced_acc"], 3),
        })
        print("{:<20s} {:>2d} passes  acc {:.3f}  wF1 {:.3f}  mF1 {:.3f}".format(
            label, passes, metrics["accuracy"], metrics["weighted_f1"],
            metrics["macro_f1"]), flush=True)

    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame["arm"] == "flip (shipped)", "weighted_f1"].iloc[0]
    frame["delta_weighted_f1"] = (frame["weighted_f1"] - baseline).round(3)
    frame["verdict"] = np.where(
        frame["delta_weighted_f1"].abs() <= NOISE_FLOOR, "tie (inside noise floor)",
        np.where(frame["delta_weighted_f1"] > 0, "better", "worse"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print("\n" + frame.to_string(index=False))
    print("\nwrote {}".format(OUT.relative_to(PROJECT_ROOT)))


if __name__ == "__main__":
    main()
