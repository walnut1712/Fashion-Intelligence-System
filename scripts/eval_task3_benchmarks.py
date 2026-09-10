"""Task 3 across Task 4's five benchmark families, so the two tasks read alike.

Why this exists
---------------
Task 3's two existing out-of-domain measurements use different corruption
families and cannot be put in one table:

- `scripts/eval_task3_ood.py` composites onto PROCEDURAL patterns (checkerboards,
  stripes, blob fields) and walks a severity ladder. That is the `hard` family.
- `scripts/train_task3_background_adaptation.py` composites onto held-out
  Places365 scenes. That is the `photo` family.

Task 4 reports both, plus their degraded variants, as `clean / hard / photo /
wild / wildphoto`. This builds exactly those five for Task 3, using the same
construction `build_queries` in `src/training/train_task4_120x160.py` uses -
`make_eval_backgrounds(600)`, `load_background_bank(2000, split="test")`, then
`simulate_ingestion(degrade(..., EVAL_DEGRADATIONS))` - so a Task 3 row and a
Task 4 row mean the same thing.

Why both families rather than one
---------------------------------
Task 4's arm P is the cautionary result: trained on procedural patterns alone it
scored 60.13 on `hard` and 42.22 on `photo`, a 17.91 point gap between two
families it never saw. Grading on one family cannot tell invariance from a
swapped overfit. Task 3's background arm trains on the 70/30 mix, so it should
show a small gap - and if it does not, that is worth knowing.

Reading the table
-----------------
**Use macro-F1, not accuracy.** `Casual` is 77.3% of the usage split, so a model
that has collapsed into guessing `Casual` still scores about 70% usage accuracy.
Accuracy is reported beside macro-F1 only to make that visible.

Usage
-----
    python scripts/eval_task3_benchmarks.py
    python scripts/eval_task3_benchmarks.py --rows 2000

Writes `outputs/evaluation/task3_benchmarks.csv`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SERVICES = PROJECT_ROOT / "app" / "backend" / "services"
if str(SERVICES) not in sys.path:
    sys.path.insert(0, str(SERVICES))

from src.data.places365_backgrounds import load_background_bank  # noqa: E402
from src.data.splits import load_or_create_splits  # noqa: E402
from src.data.synthetic_backgrounds import (  # noqa: E402
    EVAL_DEGRADATIONS,
    composite,
    degrade,
    make_eval_backgrounds,
    simulate_ingestion,
)
from task3_service import EarlyBranchCNN  # noqa: E402

TARGETS = ("gender", "usage")
TARGET_SIZE_WH = (60, 80)
BENCHMARKS = ("clean", "hard", "photo", "wild", "wildphoto")
USAGE_MERGE = {"Smart Casual": "Casual", "Travel": "Casual", "Party": "Formal"}
USAGE_DROP = {"Home"}

PROCESSED = PROJECT_ROOT / "A2_FashionDataset" / "processed"
OUT_CSV = PROJECT_ROOT / "outputs" / "evaluation" / "task3_benchmarks.csv"

#: label -> checkpoint, control arms included so the table can separate "extra
#: training" from "backdrops" the way the adaptation script's control does.
ARMS = {
    "baseline (deployed)": "artifacts/task3/task3_cnn_model.pt",
    "control 8ep lr1e-4": "artifacts/task3_bgaug_nobg_control_lr1e-4/task3_bgadapt_best.pt",
    "bg arm lr1e-4": "artifacts/task3_bgaug/task3_bgadapt_best.pt",
    "control 8ep lr3e-4": "artifacts/task3_bgaug_nobg_control_lr3e-4/task3_bgadapt_best.pt",
    "bg arm lr3e-4": "artifacts/task3_bgaug_lr3e-4/task3_bgadapt_best.pt",
}


def held_out_rows(classes):
    _, _, test = load_or_create_splits()
    test = test.copy()
    test["usage"] = test["usage"].replace(USAGE_MERGE)
    test = test[test["usage"].notna() & ~test["usage"].isin(USAGE_DROP)]
    test = test[test["gender"].notna()]
    for target in TARGETS:
        unknown = sorted(set(test[target]) - set(classes[target]))
        if unknown:
            raise ValueError("%s labels absent from the checkpoint: %s" % (target, unknown))
    return test


def build_benchmarks(images, masks, positions, seed):
    """The same five families `build_queries` makes, at 120x160."""
    height, width = images.shape[1], images.shape[2]
    evaluation_backgrounds = make_eval_backgrounds(600, size=(width, height))
    photographic = load_background_bank(count=2000, shape=(height, width, 3),
                                        split="test", seed=seed)
    generator = np.random.default_rng(seed)

    clean = np.stack([np.asarray(images[p]) for p in positions])

    def composite_onto(bank):
        return np.stack([
            composite(np.asarray(images[p]), np.asarray(masks[p]),
                      bank[generator.integers(len(bank))], generator)
            for p in positions])

    def weather(frames):
        return np.stack([
            simulate_ingestion(degrade(frame, generator, EVAL_DEGRADATIONS), generator)
            for frame in frames])

    hard = composite_onto(evaluation_backgrounds)
    photo = composite_onto(photographic)
    return {"clean": clean, "hard": hard, "photo": photo,
            "wild": weather(hard), "wildphoto": weather(photo)}


def to_model_input(frames, mean, std):
    """120x160 uint8 frames -> the 60x80 normalised tensor Task 3 expects."""
    resized = np.stack([
        np.asarray(Image.fromarray(frame).resize(TARGET_SIZE_WH, Image.BILINEAR))
        for frame in frames])
    array = (resized.astype(np.float32) / 255.0 - np.asarray(mean, dtype=np.float32)
             ) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(array.transpose(0, 3, 1, 2))


@torch.no_grad()
def score(model, tensor, truth, device, batch_size=512):
    predicted = {t: [] for t in TARGETS}
    for start in range(0, len(tensor), batch_size):
        logits = model(tensor[start:start + batch_size].to(device))
        for target in TARGETS:
            predicted[target].extend(logits[target].argmax(1).cpu().tolist())

    row = {}
    for target in TARGETS:
        chosen = np.asarray(predicted[target])
        row["%s_accuracy" % target] = round(
            float(accuracy_score(truth[target], chosen)) * 100, 2)
        row["%s_macro_f1" % target] = round(float(
            f1_score(truth[target], chosen, average="macro", zero_division=0)) * 100, 2)
    row["exact_match"] = round(float(np.mean(
        (np.asarray(predicted["gender"]) == truth["gender"])
        & (np.asarray(predicted["usage"]) == truth["usage"]))) * 100, 2)
    row["mean_macro_f1"] = round(
        0.5 * (row["gender_macro_f1"] + row["usage_macro_f1"]), 2)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2000,
                        help="held-out rows to score, sampled once and shared by "
                             "every arm and benchmark so the table is paired")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = torch.load(PROJECT_ROOT / ARMS["baseline (deployed)"],
                           map_location=device, weights_only=False)
    classes = {t: list(v) for t, v in reference["class_names"].items()}
    mean, std = reference["channel_mean"], reference["channel_std"]

    gallery = pd.read_csv(PROCESSED / "task4_gallery_120x160.csv")
    id_to_pos = {int(v): i for i, v in enumerate(gallery["id"])}
    images = np.load(PROCESSED / "task4_cache_120x160.npy", mmap_mode="r")
    masks = np.load(PROCESSED / "task4_masks_120x160.npy", mmap_mode="r")

    test = held_out_rows(classes)
    test = test[test["id"].map(lambda v: int(v) in id_to_pos)]
    take = np.random.default_rng(args.seed).choice(
        len(test), min(args.rows, len(test)), replace=False)
    test = test.iloc[np.sort(take)]
    positions = np.asarray([id_to_pos[int(v)] for v in test["id"]])
    truth = {t: np.asarray([classes[t].index(v) for v in test[t]]) for t in TARGETS}
    print("scoring %d held-out rows on %s\n" % (len(test), device))

    print("building the five benchmark families...")
    frames = build_benchmarks(images, masks, positions, args.seed)
    tensors = {name: to_model_input(batch, mean, std) for name, batch in frames.items()}

    rows = []
    for label, relative in ARMS.items():
        path = PROJECT_ROOT / relative
        if not path.exists():
            print("  %-22s missing, skipped (%s)" % (label, relative))
            continue
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = EarlyBranchCNN({t: len(v) for t, v in classes.items()},
                               input_shape=(3, TARGET_SIZE_WH[1], TARGET_SIZE_WH[0]),
                               **reference["architecture"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        for name in BENCHMARKS:
            rows.append({"arm": label, "benchmark": name, "rows": len(truth["gender"]),
                         **score(model, tensors[name], truth, device)})
        got = [r for r in rows if r["arm"] == label]
        print("  %-22s %s" % (label, "  ".join(
            "%s %5.2f" % (r["benchmark"], r["mean_macro_f1"]) for r in got)))

    table = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)

    for metric in ("mean_macro_f1", "gender_macro_f1", "usage_macro_f1"):
        print("\n%s\n" % metric)
        print(table.pivot_table(index="arm", columns="benchmark", values=metric)
              [list(BENCHMARKS)]
              .reindex([a for a in ARMS if a in set(table["arm"])]).to_string())

    # The arm P lesson: two families neither model trained on. A model that had
    # learned "backgrounds are irrelevant" scores alike on hard and photo.
    print("\nhard minus photo, the invariance gap (mean macro-F1)\n")
    wide = table.pivot_table(index="arm", columns="benchmark", values="mean_macro_f1")
    for arm in [a for a in ARMS if a in wide.index]:
        print("  %-22s hard %5.2f   photo %5.2f   gap %+6.2f"
              % (arm, wide.loc[arm, "hard"], wide.loc[arm, "photo"],
                 wide.loc[arm, "hard"] - wide.loc[arm, "photo"]))

    print("\nwrote %s in %.1f min" % (OUT_CSV, (time.time() - started) / 60))


if __name__ == "__main__":
    main()
