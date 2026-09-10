"""Tasks 1 and 2 across Task 4's five benchmark families, so all four read alike.

Why this exists
---------------
`scripts/train_t12_background_adaptation.py` reports two columns, `clean` and
`heldout_places365`. Task 4 reports five - `clean / hard / photo / wild /
wildphoto` - and `scripts/eval_task3_benchmarks.py` gives Task 3 the same five.
This closes the set, using the identical construction `build_queries` in
`src/training/train_task4_120x160.py` uses, so a row here means what a row there
means.

The five, all holding the SAME held-out garments with only the surroundings
changing:

    clean      the catalogue frame as it ships
    hard       composited onto machine-drawn patterns (checkerboards, stripes)
    photo      composited onto held-out Places365 scene photographs
    wild       `hard` put through a simulated camera and upload path
    wildphoto  `photo` put through the same

Why two background families rather than one
-------------------------------------------
Task 4's arm P is the cautionary result: trained on procedural patterns alone it
scored 60.13 on `hard` and 42.22 on `photo`, a 17.91 point gap between two
families it had never seen. It had learned that *formulae* were irrelevant, not
that backgrounds were, and a checkerboard-only benchmark would have ranked it
first. One family cannot separate invariance from a swapped overfit.

Why the control row
-------------------
The background arm receives 8 epochs the baseline never got, so a clean-side
change may be the extra training rather than the backdrops. The control is the
same schedule with no compositing (`--p-bg 0`). Measured across three tasks the
control moves out-of-domain by at most 0.009 either way, so the out-of-domain
column is attributable to the backdrops; the clean column is not, and its
correction runs in *different directions* per task, so it has to be measured.

Note on the background arms
---------------------------
`.gitignore` excludes the adaptation checkpoints, so a clone has the evidence but
not the weights. `ARMS` therefore points at `*_bgaug_local/`, reproduced with
`--tag _local`. Re-create them with, e.g.:

    python scripts/train_t12_background_adaptation.py --task 1 --tag _local

CUDA training is not bit-reproducible (`cudnn.benchmark` picks its algorithms at
runtime; measured drift on Task 4 was up to 0.94 points), so a reproduction is
close to but not identical with the committed run.

Usage
-----
    python scripts/eval_t12_benchmarks.py --task 1
    python scripts/eval_t12_benchmarks.py --task 2 --rows 3000

Writes `outputs/evaluation/task{1,2}_benchmarks.csv`.
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.services.task2_service import SeasonCNN  # noqa: E402
from src.data.places365_backgrounds import load_background_bank  # noqa: E402
from src.data.splits import load_or_create_splits  # noqa: E402
from src.data.synthetic_backgrounds import (  # noqa: E402
    EVAL_DEGRADATIONS,
    composite,
    degrade,
    make_eval_backgrounds,
    simulate_ingestion,
)
from src.models.item_type_classifier import ItemTypeCNN  # noqa: E402
from src.training.candidate_120x160 import task1_frame_parts  # noqa: E402

BENCHMARKS = ("clean", "hard", "photo", "wild", "wildphoto")
SEASONS = ["Fall", "Spring", "Summer", "Winter"]
PROCESSED = PROJECT_ROOT / "A2_FashionDataset" / "processed"

#: label -> checkpoint, per task. The control arms are here so the table can
#: separate "extra training" from "backdrops"; see the module docstring.
ARMS = {
    1: {"baseline (deployed)": "artifacts/task1_120x160/task1_120x160_onecycle_best.pt",
        "control 8ep, no bg": "artifacts/task1_bgaug_nobg_control/task1_bgadapt_best.pt",
        "bg arm": "artifacts/task1_bgaug_local/task1_bgadapt_best.pt"},
    2: {"baseline (deployed)": "artifacts/task2/task2_season_best_pytorch.pth",
        "control 8ep, no bg": "artifacts/task2_bgaug_nobg_control/task2_bgadapt_best.pth",
        "bg arm": "artifacts/task2_bgaug_local/task2_bgadapt_best.pth"},
}
#: The committed run's own baseline score, used as the harness guard below.
EXPECTED_CLEAN = {1: ("weighted_f1", 0.8968), 2: ("macro_f1", 0.6306)}
TARGET_SIZE_WH = {1: (120, 160), 2: (60, 80)}


def load_state(path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj
    return obj, {}


def build_model(task, num_classes, device):
    if task == 1:
        return ItemTypeCNN(num_classes, widths=(16, 32, 64, 128), head_hidden=384,
                           pool_grid=(1, 1), pool_mode="avgmax").to(device)
    return SeasonCNN(num_classes).to(device)


def task_frames(task):
    """Held-out rows and their integer labels, matching the adaptation script."""
    train, val, test = load_or_create_splits()
    if task == 1:
        (_, _, test), classes = task1_frame_parts(train, val, test)
        return test, classes
    test = test[test["season"].notna()].copy()
    unexpected = sorted(set(test["season"].astype(str)) - set(SEASONS))
    if unexpected:
        raise ValueError("Unexpected season labels: %s" % unexpected)
    test["label"] = test["season"].map({n: i for i, n in enumerate(SEASONS)}).astype("int64")
    return test, SEASONS


def build_benchmarks(images, masks, positions, seed):
    """The same five families `build_queries` makes, at the cache's 120x160."""
    height, width = images.shape[1], images.shape[2]
    patterns = make_eval_backgrounds(600, size=(width, height))
    photographs = load_background_bank(count=2000, shape=(height, width, 3),
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

    hard = composite_onto(patterns)
    photo = composite_onto(photographs)
    return {"clean": clean, "hard": hard, "photo": photo,
            "wild": weather(hard), "wildphoto": weather(photo)}


def to_tensor(frames, size_wh, mean, std):
    """Resize after compositing, exactly as the adaptation script does."""
    if frames.shape[2] != size_wh[0] or frames.shape[1] != size_wh[1]:
        frames = np.stack([
            np.asarray(Image.fromarray(f).resize(size_wh, Image.BILINEAR))
            for f in frames])
    array = frames.astype(np.float32) / 255.0
    if mean is not None:
        array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(array.transpose(0, 3, 1, 2))


@torch.no_grad()
def score(model, tensor, truth, device, batch_size=256):
    predicted = []
    for start in range(0, len(tensor), batch_size):
        logits = model(tensor[start:start + batch_size].to(device))
        predicted.extend(logits.argmax(1).cpu().tolist())
    predicted = np.asarray(predicted)
    return {
        "accuracy": round(float(accuracy_score(truth, predicted)) * 100, 2),
        "weighted_f1": round(float(f1_score(truth, predicted, average="weighted",
                                            zero_division=0)) * 100, 2),
        "macro_f1": round(float(f1_score(truth, predicted, average="macro",
                                         zero_division=0)) * 100, 2),
        "balanced_accuracy": round(float(balanced_accuracy_score(truth, predicted)) * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, choices=[1, 2], required=True)
    parser.add_argument("--rows", type=int, default=2000,
                        help="held-out rows, sampled once and shared by every arm "
                             "and benchmark so the table is paired")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size_wh = TARGET_SIZE_WH[args.task]
    out_csv = PROJECT_ROOT / "outputs" / "evaluation" / ("task%d_benchmarks.csv" % args.task)

    test, classes = task_frames(args.task)
    gallery = pd.read_csv(PROCESSED / "task4_gallery_120x160.csv")
    id_to_pos = {int(v): i for i, v in enumerate(gallery["id"])}
    images = np.load(PROCESSED / "task4_cache_120x160.npy", mmap_mode="r")
    masks = np.load(PROCESSED / "task4_masks_120x160.npy", mmap_mode="r")

    test = test[test["id"].map(lambda v: int(v) in id_to_pos)]
    all_positions = np.asarray([id_to_pos[int(v)] for v in test["id"]])
    all_truth = test["label"].to_numpy()
    take = np.random.default_rng(args.seed).choice(
        len(test), min(args.rows, len(test)), replace=False)
    test = test.iloc[np.sort(take)]
    positions = np.asarray([id_to_pos[int(v)] for v in test["id"]])
    truth = test["label"].to_numpy()
    print("task %d | %d held-out rows | %d classes | %s\n"
          % (args.task, len(test), len(classes), device))

    print("building the five benchmark families...")
    frames = build_benchmarks(images, masks, positions, args.seed)

    rows = []
    for label, relative in ARMS[args.task].items():
        path = PROJECT_ROOT / relative
        if not path.exists():
            print("  %-22s MISSING, skipped (%s)" % (label, relative))
            continue
        state, meta = load_state(path, device)
        model = build_model(args.task, len(classes), device)
        model.load_state_dict(state)
        model.eval()
        mean, std = meta.get("channel_mean"), meta.get("channel_std")
        tensors = {n: to_tensor(f, size_wh, mean, std) for n, f in frames.items()}
        for name in BENCHMARKS:
            rows.append({"arm": label, "benchmark": name, "rows": len(truth),
                         **score(model, tensors[name], truth, device)})
        got = {r["benchmark"]: r for r in rows if r["arm"] == label}
        print("  %-22s %s" % (label, "  ".join(
            "%s %5.2f" % (n, got[n][EXPECTED_CLEAN[args.task][0]]) for n in BENCHMARKS)))

        # The clean row must land on the committed run's own baseline, or the
        # label mapping or resize path is wrong and nothing below is worth
        # reading. It runs on EVERY held-out row rather than the sample the
        # table uses: the committed baseline was computed on the full split, and
        # comparing a subsample against it would fail on sampling error alone.
        if label.startswith("baseline"):
            metric, expected = EXPECTED_CLEAN[args.task]
            full = np.stack([np.asarray(images[p]) for p in all_positions])
            observed = score(model, to_tensor(full, size_wh, mean, std),
                             all_truth, device)[metric] / 100.0
            flag = "" if abs(observed - expected) < 0.005 else "   <- CHECK THE HARNESS"
            print("      guard: clean %s on all %d rows %.4f against the committed "
                  "baseline %.4f%s" % (metric, len(all_truth), observed, expected, flag))

    table = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)

    metric = EXPECTED_CLEAN[args.task][0]
    present = [a for a in ARMS[args.task] if a in set(table["arm"])]
    print("\n%s by arm and benchmark\n" % metric)
    wide = table.pivot_table(index="arm", columns="benchmark", values=metric)[list(BENCHMARKS)]
    print(wide.reindex(present).to_string())

    print("\nhard minus photo, the invariance gap\n")
    for arm in present:
        print("  %-22s hard %5.2f   photo %5.2f   gap %+6.2f"
              % (arm, wide.loc[arm, "hard"], wide.loc[arm, "photo"],
                 wide.loc[arm, "hard"] - wide.loc[arm, "photo"]))
    print("  (Task 4's procedural-only arm P: gap +17.91)")
    print("\nwrote %s in %.1f min" % (out_csv, (time.time() - started) / 60))


if __name__ == "__main__":
    main()
