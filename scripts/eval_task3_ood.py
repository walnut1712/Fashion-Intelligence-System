"""Task 3 - does the gender/usage model survive a photograph it did not pose for?

Why this exists
---------------
Every Task 3 number in the repository is measured on catalogue tiles: 60x80
product cutouts, one item, centred, on white. The deployed checkpoint scores
90.33 gender accuracy and 90.24 usage accuracy there. Nothing measured what
happens to a photo someone actually uploads - and the app does exactly that,
because ``POST /api/analyze`` runs all four tasks over one upload.

Two independent measurements already say what to expect. Task 1's classifier
falls 87.85 -> 6.58 across the same severity ladder, and Task 4's arm C - an
encoder trained only on catalogue frames - falls P@10 81.64 -> 11.84 on
photographic backdrops. Both models trained on a catalogue where every frame is
a garment on white, so neither had any reason to learn that a background can be
ignored. Task 3 trained on the same catalogue.

This module builds the missing measurement. It reuses the synthesis and
ingestion machinery of ``src.evaluation.ood_benchmark`` unchanged, so Task 1 and
Task 3 are corrupted by the identical code at the identical severities and the
two collapses are directly comparable.

What it does NOT do
-------------------
It does not retrain anything. Task 1 already measured the augmented arm - the
``webphoto`` recipe cost 2.79 points of clean accuracy and bought 22.28 out of
domain, an 8:1 trade - and declined it because the graded set is catalogue
tiles. Task 3's graded set is catalogue tiles too, so the same decision applies
until the grading changes. This script measures the gap and prices the choice;
it does not make it.

Methodology guard
-----------------
``clean`` + ``squash`` passes the catalogue tile through untouched, so it must
reproduce the checkpoint's own recorded ``test_metrics``. If it does not, the
harness is wrong and every other row is meaningless. The check runs first and
raises rather than warns.

The corruption families (checkerboards, stripes, blob fields) are the same
disjoint set Task 1 uses, chosen so no model in this project has trained on
them.

Coupling to Task 1, deliberately
--------------------------------
``SEVERITIES``, ``build_shift_set`` and ``_ingest`` are imported from Task 1's
``src.evaluation.ood_benchmark`` rather than copied. That is the point: the two
tasks are then corrupted by the same code at the same severities, so their
collapses can be read against each other. The cost is that a change to that
module silently moves these numbers, and Task 1 is owned by someone else. So
``_expect_unchanged_ladder`` asserts the ladder this script was written against
and fails loudly rather than reporting figures that are no longer comparable.
If it fires, re-run Task 1's benchmark too and compare them afresh; do not just
widen the assertion.

Usage
-----
    python scripts/eval_task3_ood.py
    python scripts/eval_task3_ood.py --rows 3000
    python scripts/eval_task3_ood.py --modes letterbox nobg

Writes ``outputs/evaluation/task3_ood_results.csv``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# EarlyBranchCNN is declared in the service rather than src/models/. That is the
# documented exception to the one-definition rule, not an oversight, so import
# it from where it lives instead of redeclaring it here.
SERVICES = PROJECT_ROOT / "app" / "backend" / "services"
if str(SERVICES) not in sys.path:
    sys.path.insert(0, str(SERVICES))

from src.data.splits import load_or_create_splits          # noqa: E402
from src.data.user_image import catalogue_alpha             # noqa: E402
from src.evaluation.ood_benchmark import (                  # noqa: E402
    INGEST_MODES,
    SEVERITIES,
    _ingest,
    build_shift_set,
)
from task3_service import EarlyBranchCNN                     # noqa: E402

PROCESSED = PROJECT_ROOT / "A2_FashionDataset" / "processed"
CHECKPOINT = PROJECT_ROOT / "artifacts" / "task3" / "task3_cnn_model.pt"
IMAGE_CACHE = PROCESSED / "image_cache_60x80.npy"
IMAGE_CACHE_IDS = PROCESSED / "image_cache_60x80_ids.npy"
OUT_CSV = PROJECT_ROOT / "outputs" / "evaluation" / "task3_ood_results.csv"

TARGETS = ("gender", "usage")

# The `usage` policy CLAUDE.md records: 8 raw classes collapse to 4. Applying it
# here rather than importing it is a duplication, so the guard below exists to
# catch a drift: get this wrong and `clean` stops reproducing the checkpoint.
USAGE_MERGE = {"Smart Casual": "Casual", "Travel": "Casual", "Party": "Formal"}
USAGE_DROP = {"Home"}


#: The ladder as it stood when this script was written. See the module
#: docstring: Task 1 owns the module these come from.
EXPECTED_SEVERITIES = ("clean", "mild", "moderate", "severe")
EXPECTED_MODES = ("squash", "letterbox", "crop", "nobg")


def _expect_unchanged_ladder():
    """Fail loudly if Task 1's benchmark has moved under us."""
    if tuple(SEVERITIES) != EXPECTED_SEVERITIES:
        raise SystemExit(
            "src.evaluation.ood_benchmark.SEVERITIES has changed: expected %s, "
            "found %s. Task 3's rows are no longer comparable with Task 1's - "
            "re-measure both rather than widening this check."
            % (list(EXPECTED_SEVERITIES), list(SEVERITIES)))
    if tuple(INGEST_MODES) != EXPECTED_MODES:
        raise SystemExit(
            "src.evaluation.ood_benchmark.INGEST_MODES has changed: expected %s, "
            "found %s." % (list(EXPECTED_MODES), list(INGEST_MODES)))


def load_model(device):
    """The deployed checkpoint, rebuilt from the architecture it records."""
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    classes = {t: list(v) for t, v in checkpoint["class_names"].items()}
    size = tuple(checkpoint["image_size_pil"])              # (width, height)
    model = EarlyBranchCNN({t: len(v) for t, v in classes.items()},
                           input_shape=(3, size[1], size[0]),
                           **checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint, classes, size


def held_out_rows():
    """Task 3's test split, under the label policy the training used."""
    _, _, test = load_or_create_splits()
    test = test.copy()
    test["usage"] = test["usage"].replace(USAGE_MERGE)
    test = test[test["usage"].notna() & ~test["usage"].isin(USAGE_DROP)]
    test = test[test["gender"].notna()]

    ids = np.load(IMAGE_CACHE_IDS)
    cache = np.load(IMAGE_CACHE, mmap_mode="r")
    position = {int(value): index for index, value in enumerate(ids)}
    test = test[test["id"].map(lambda value: int(value) in position)]
    tiles = np.asarray(cache[[position[int(value)] for value in test["id"]]])
    return test, tiles


@torch.no_grad()
def predict(model, checkpoint, arrays, device, batch_size=512):
    """Probabilities per target, through the serving normalisation and temperature."""
    mean = np.array(checkpoint["channel_mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.array(checkpoint["channel_std"], dtype=np.float32).reshape(1, 1, 3)
    temperature = checkpoint.get("temperature", {}) or {}

    out = {target: [] for target in TARGETS}
    for start in range(0, len(arrays), batch_size):
        chunk = arrays[start:start + batch_size].astype(np.float32) / 255.0
        chunk = (chunk - mean) / std
        tensor = torch.from_numpy(chunk.transpose(0, 3, 1, 2)).to(device)
        logits = model(tensor)
        for target in TARGETS:
            scale = float(temperature.get(target) or 1.0)
            scale = scale if scale > 0 else 1.0
            out[target].append(
                torch.softmax(logits[target] / scale, dim=1).cpu().numpy())
    return {target: np.concatenate(values) for target, values in out.items()}


def metrics(probabilities, truth, classes):
    """Accuracy, macro-F1 and mean confidence per target, plus exact match."""
    row, predicted = {}, {}
    for target in TARGETS:
        chosen = probabilities[target].argmax(1)
        predicted[target] = chosen
        actual = truth[target]
        row["%s_accuracy" % target] = round(
            float(accuracy_score(actual, chosen)) * 100, 2)
        row["%s_macro_f1" % target] = round(
            float(f1_score(actual, chosen, average="macro", zero_division=0)) * 100, 2)
        # Confidence is reported because the checkpoint carries a temperature
        # fitted on catalogue images. Whether that calibration survives a
        # photograph is exactly what nothing had measured.
        row["%s_confidence" % target] = round(
            float(probabilities[target].max(1).mean()) * 100, 2)
    row["exact_match"] = round(float(np.mean(
        (predicted["gender"] == truth["gender"])
        & (predicted["usage"] == truth["usage"]))) * 100, 2)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1500,
                        help="held-out rows per cell. The full split is used for "
                             "the reproduction guard regardless")
    parser.add_argument("--modes", nargs="+", default=list(INGEST_MODES),
                        choices=list(INGEST_MODES))
    parser.add_argument("--severities", nargs="+", default=list(SEVERITIES),
                        choices=list(SEVERITIES))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    started = time.time()
    _expect_unchanged_ladder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, classes, size = load_model(device)
    test, tiles = held_out_rows()
    truth_full = {target: np.array([classes[target].index(value)
                                    for value in test[target]])
                  for target in TARGETS}
    print("held-out rows: %d | device: %s" % (len(test), device))

    # ---- guard: an untouched tile must reproduce the published metrics -------
    guard = metrics(predict(model, checkpoint, tiles, device), truth_full, classes)
    for target in TARGETS:
        published = checkpoint["test_metrics"][target]["accuracy"]
        if abs(guard["%s_accuracy" % target] - published) > 0.01:
            raise SystemExit(
                "harness is wrong: %s clean accuracy %.2f against the "
                "checkpoint's own %.2f" % (target, guard["%s_accuracy" % target],
                                           published))
    print("guard passed: gender %.2f / usage %.2f reproduce the checkpoint\n"
          % (guard["gender_accuracy"], guard["usage_accuracy"]))

    # ---- the ladder, on one fixed subsample so every cell is paired ----------
    take = np.random.default_rng(args.seed).choice(
        len(test), min(args.rows, len(test)), replace=False)
    sample_tiles = tiles[take]
    truth = {target: values[take] for target, values in truth_full.items()}

    alphas, usable = catalogue_alpha(sample_tiles)
    print("%d/%d sampled rows have a usable matte" % (usable.sum(), len(sample_tiles)))
    sample_tiles, alphas = sample_tiles[usable], alphas[usable]
    truth = {target: values[usable] for target, values in truth.items()}

    rows = []
    for severity in args.severities:
        sources = build_shift_set(sample_tiles, alphas, severity)
        for mode in args.modes:
            arrays = _ingest(sources, mode, size)
            row = {"checkpoint": CHECKPOINT.name, "ingest": mode,
                   "severity": severity, "rows": len(arrays)}
            row.update(metrics(predict(model, checkpoint, arrays, device),
                               truth, classes))
            rows.append(row)
            print("  %-9s %-9s gender %5.2f  usage %5.2f  exact %5.2f  "
                  "confidence %5.2f/%5.2f"
                  % (severity, mode, row["gender_accuracy"], row["usage_accuracy"],
                     row["exact_match"], row["gender_confidence"],
                     row["usage_confidence"]), flush=True)

    table = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)

    print("\ngender accuracy by ingestion mode and severity\n")
    print(table.pivot_table(index="severity", columns="ingest",
                            values="gender_accuracy")
          .reindex([s for s in SEVERITIES if s in args.severities]).to_string())
    print("\nwrote %s in %.1f min" % (OUT_CSV, (time.time() - started) / 60))


if __name__ == "__main__":
    main()
