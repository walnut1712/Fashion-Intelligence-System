"""Task 1 - binary specialists for the dominant confusion pairs.

Why this exists
---------------
73.2% of the adopted model's test error mass is *symmetric* confusion between
adjacent classes - A predicted as B about as often as B predicted as A - and two
pairs carry 24% of it between them:

    Sports Shoes <-> Casual Shoes    82 errors (12.9%)
    Tshirts      <-> Tops            71 errors (11.1%)

Symmetry in near-equal numbers is the signature of a boundary the multi-class
head is close to guessing on. A binary model trained on one pair sees a balanced
prior instead of competing with 90 other logits, and spends all of its capacity
on the one distinction.

The gate is deliberately narrow: a specialist is consulted only when the main
model's **top-2 set is exactly its pair**, so every other row is returned
untouched. ``outputs/evaluation/task1_family_rerank_ab.csv`` is the reason to
keep it narrow - re-ranking inside a whole family moved 35 rows and changed
nothing measurable, so a wide gate is already known not to pay.

Nothing here redeclares an architecture or a split. ``ItemTypeCNN`` comes from
``src/models/item_type_classifier.py`` and the partition from
``train_item_type.load_splits`` - the same name-or-hash connected-component group
key the adopted model used - so a specialist is directly comparable with the
numbers in ``artifacts/task1/``.

Usage
-----
    python -m src.training.train_pair_specialist --all
    python -m src.training.train_pair_specialist --pair "Tshirts" "Tops"
    python -m src.training.train_pair_specialist --evaluate
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.item_type_classifier import ItemTypeCNN  # noqa: E402
from src.training.train_item_type import (  # noqa: E402
    ARCHITECTURE,
    ARTIFACT_DIR,
    IMAGE_SIZE_PIL,
    RANDOM_STATE,
    RECIPES,
    choose_device,
    load_splits,
    normalise,
    predict_split,
    score,
    split_tensors,
)

# The two pairs Section 16's confusion table puts at the top. Both are decided by
# merchandising convention as much as by shape, which is why they are worth a
# dedicated boundary rather than more capacity everywhere.
DEFAULT_PAIRS = [
    ("Sports Shoes", "Casual Shoes"),
    ("Tshirts", "Tops"),
]

EVALUATION_OUT = PROJECT_ROOT / "outputs" / "evaluation" / "task1_pair_specialist_ab.csv"

# Adoption gate, fixed before any run so the result cannot be talked into being a
# win afterwards. The 3-seed noise floor on validation weighted-F1 is +/-0.24
# (artifacts/task1/best_config.json), so a single run has to clear ~3 SD.
ADOPTION_MIN_GAIN = 0.7


def specialist_path(pair):
    """One flat artefact per pair, matching the artifacts/task1/ convention."""
    slug = "__".join(name.lower().replace(" ", "_") for name in pair)
    return ARTIFACT_DIR / f"specialist_{slug}.pt"


def channel_stats(x_train, device):
    """Train-split channel statistics, chunked exactly as ``train()`` does.

    Recomputed on the pair's own rows rather than inherited from the 92-class
    run: this model only ever sees two classes, and normalising by the whole
    catalogue's mean would be a preprocessing step it was not trained under.
    """
    total = x_train.shape[0]
    pixel_count = total * x_train.shape[2] * x_train.shape[3]
    channel_sum = torch.zeros(3, dtype=torch.float64, device=device)
    channel_sq = torch.zeros(3, dtype=torch.float64, device=device)
    for start in range(0, total, 2048):
        chunk = x_train[start:start + 2048].double().div_(255.0)
        channel_sum += chunk.sum(dim=(0, 2, 3))
        channel_sq += chunk.pow(2).sum(dim=(0, 2, 3))
        del chunk
    mean_v = channel_sum / pixel_count
    var_v = (channel_sq / pixel_count - mean_v.pow(2)).clamp_min(0)
    return mean_v.float().view(1, 3, 1, 1), var_v.sqrt().float().view(1, 3, 1, 1)


def pair_frames(pair, class_names, train_df, val_df, test_df):
    """The three splits reduced to one pair, relabelled 0/1 in ``pair`` order."""
    lookup = {name: index for index, name in enumerate(class_names)}
    missing = [name for name in pair if name not in lookup]
    if missing:
        raise SystemExit(f"not a class in the adopted model: {missing}")
    codes = [lookup[name] for name in pair]
    remap = {code: position for position, code in enumerate(codes)}

    out = []
    for frame in (train_df, val_df, test_df):
        subset = frame[frame["label"].isin(codes)].copy()
        subset["label"] = subset["label"].map(remap)
        out.append(subset)
    return out


def train_pair(pair, epochs=40, batch_size=128, learning_rate=2e-3,
               seed=RANDOM_STATE, verbose=True):
    """Train one binary specialist. Same recipe as RECIPES['baseline']."""
    recipe = RECIPES["baseline"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = choose_device()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    train_df, val_df, test_df, class_names, images = load_splits(verbose=False)
    p_train, p_val, p_test = pair_frames(pair, class_names, train_df, val_df, test_df)
    if verbose:
        print(f"\n{pair[0]} vs {pair[1]}: "
              f"{len(p_train)} train / {len(p_val)} val / {len(p_test)} test rows")

    x_train, y_train = split_tensors(p_train, images, device)
    x_val, y_val = split_tensors(p_val, images, device)
    x_test, y_test = split_tensors(p_test, images, device)
    del images

    mean, std = channel_stats(x_train, device)

    model = ItemTypeCNN(
        2,
        widths=ARCHITECTURE["widths"],
        dropout=recipe["dropout"],
        head_hidden=ARCHITECTURE["head_hidden"],
        pool_grid=ARCHITECTURE["pool_grid"],
        pool_mode=ARCHITECTURE["pool_mode"],
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=recipe["weight_decay"])
    steps = max(1, int(np.ceil(x_train.shape[0] / batch_size)))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=learning_rate * 3.0, total_steps=epochs * steps)

    best_state, best, history = None, None, []
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(x_train.shape[0], device=device, generator=generator)
        losses = []
        for start in range(0, order.shape[0], batch_size):
            index = order[start:start + batch_size]
            batch = normalise(x_train[index], mean, std)
            if recipe["flip"]:
                flip_mask = torch.rand(batch.shape[0], device=device,
                                       generator=generator) < 0.5
                batch[flip_mask] = torch.flip(batch[flip_mask], dims=[3])

            optimiser.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch), y_train[index],
                                   label_smoothing=recipe["label_smoothing"])
            loss.backward()
            optimiser.step()
            scheduler.step()
            losses.append(float(loss.item()))

        probabilities = predict_split(model, x_val, mean, std)
        metrics = score(y_val.cpu().numpy(), probabilities.argmax(1).cpu().numpy())
        metrics.update(epoch=epoch, loss=float(np.mean(losses)))
        history.append(metrics)

        if best is None or metrics["accuracy"] > best["accuracy"]:
            best = dict(metrics)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            print(f"  epoch {epoch:3d}  loss {metrics['loss']:.4f}  "
                  f"val acc {metrics['accuracy']:.2f}", flush=True)

    model.load_state_dict(best_state)
    test_probabilities = predict_split(model, x_test, mean, std)
    test_metrics = score(y_test.cpu().numpy(),
                         test_probabilities.argmax(1).cpu().numpy())

    if verbose:
        print(f"  best epoch {best['epoch']}  val acc {best['accuracy']:.2f}  "
              f"test acc {test_metrics['accuracy']:.2f}  "
              f"({time.time() - started:.0f}s)")

    checkpoint = {
        "state_dict": model.state_dict(),
        "model_name": "PairSpecialist_" + "__".join(pair),
        "num_classes": 2,
        "class_names": list(pair),
        "channel_mean": mean.flatten().tolist(),
        "channel_std": std.flatten().tolist(),
        "image_size_pil": list(IMAGE_SIZE_PIL),
        "architecture": {
            "widths": list(ARCHITECTURE["widths"]),
            "dropout": recipe["dropout"],
            "head_hidden": ARCHITECTURE["head_hidden"],
            "pool_grid": list(ARCHITECTURE["pool_grid"]),
            "pool_mode": ARCHITECTURE["pool_mode"],
        },
        "config": {"recipe": "baseline", "epochs": epochs, "batch_size": batch_size,
                   "learning_rate": learning_rate, "seed": seed, "pair": list(pair)},
        "val_metrics": best,
        "test_metrics": test_metrics,
        "history": history,
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "tta": False,
    }
    return checkpoint


# ------------------------------------------------------------------ evaluation
def _gated_predictions(base_probabilities, class_names, specialists, x_split,
                       device):
    """Apply every specialist under the top-2 gate, returning new label indices.

    The gate: the main model's top-2 set is exactly the specialist's pair. Rows
    that do not match are returned exactly as the base model left them, so the
    only rows this can change are ones the base model was already unsure about
    in precisely the way the specialist was trained for.
    """
    lookup = {name: index for index, name in enumerate(class_names)}
    labels = base_probabilities.argmax(1).cpu().numpy().copy()
    top2 = base_probabilities.topk(2, dim=1).indices.cpu().numpy()

    changed = {}
    for pair, checkpoint in specialists.items():
        codes = {lookup[pair[0]], lookup[pair[1]]}
        gate = np.array([set(row) == codes for row in top2])
        if not gate.any():
            changed[pair] = 0
            continue

        model = ItemTypeCNN(
            2,
            widths=tuple(checkpoint["architecture"]["widths"]),
            dropout=checkpoint["architecture"]["dropout"],
            head_hidden=checkpoint["architecture"]["head_hidden"],
            pool_grid=tuple(checkpoint["architecture"]["pool_grid"]),
            pool_mode=checkpoint["architecture"]["pool_mode"],
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])

        mean = torch.tensor(checkpoint["channel_mean"], device=device).view(1, 3, 1, 1)
        std = torch.tensor(checkpoint["channel_std"], device=device).view(1, 3, 1, 1)
        selected = np.flatnonzero(gate)
        probabilities = predict_split(model, x_split[selected], mean, std)
        verdict = probabilities.argmax(1).cpu().numpy()

        new_labels = np.array([lookup[pair[index]] for index in verdict])
        changed[pair] = int((new_labels != labels[selected]).sum())
        labels[selected] = new_labels

    return labels, changed


def evaluate(pairs, verbose=True):
    """Score base vs gated on validation, then report the net on test."""
    import pandas as pd

    from src.models.item_type_classifier import load_item_type_model

    device = choose_device()
    train_df, val_df, test_df, class_names, images = load_splits(verbose=False)
    x_val, y_val = split_tensors(val_df, images, device)
    x_test, y_test = split_tensors(test_df, images, device)
    del images

    deployed, checkpoint = load_item_type_model(ARTIFACT_DIR / "task1_cnn.pt", device)
    mean = torch.tensor(checkpoint["channel_mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(checkpoint["channel_std"], device=device).view(1, 3, 1, 1)
    use_tta = bool(checkpoint.get("tta", False))

    specialists = {}
    for pair in pairs:
        path = specialist_path(pair)
        if not path.exists():
            print(f"skipping {pair} - {path.name} not trained yet")
            continue
        specialists[tuple(pair)] = torch.load(path, map_location=device,
                                              weights_only=False)
    if not specialists:
        raise SystemExit("no specialists trained; run --all first")

    rows = []
    for split_name, x_split, y_split in (("validation", x_val, y_val),
                                         ("test", x_test, y_test)):
        base_probabilities = predict_split(deployed, x_split, mean, std, tta=use_tta)
        truth = y_split.cpu().numpy()

        base_labels = base_probabilities.argmax(1).cpu().numpy()
        gated_labels, changed = _gated_predictions(
            base_probabilities, class_names, specialists, x_split, device)

        moved = gated_labels != base_labels
        fixed = int(((gated_labels == truth) & ~(base_labels == truth) & moved).sum())
        broken = int(((base_labels == truth) & ~(gated_labels == truth) & moved).sum())

        base_score = score(truth, base_labels)
        gated_score = score(truth, gated_labels)
        rows.append({
            "split": split_name,
            "n": len(truth),
            "rows_gated": int(moved.sum()),
            "fixed": fixed,
            "broken": broken,
            "net": fixed - broken,
            "base_accuracy": round(base_score["accuracy"], 3),
            "gated_accuracy": round(gated_score["accuracy"], 3),
            "base_weighted_f1": round(base_score["weighted_f1"], 3),
            "gated_weighted_f1": round(gated_score["weighted_f1"], 3),
            "delta_weighted_f1": round(
                gated_score["weighted_f1"] - base_score["weighted_f1"], 3),
            "base_macro_f1": round(base_score["macro_f1"], 3),
            "gated_macro_f1": round(gated_score["macro_f1"], 3),
            "per_pair_changed": json.dumps(
                {" vs ".join(k): v for k, v in changed.items()}),
        })
        if verbose:
            row = rows[-1]
            print(f"\n{split_name}: {row['rows_gated']} rows changed  "
                  f"fixed {fixed}  broken {broken}  net {fixed - broken:+d}")
            print(f"  weighted-F1 {row['base_weighted_f1']:.3f} -> "
                  f"{row['gated_weighted_f1']:.3f} "
                  f"({row['delta_weighted_f1']:+.3f})")

    frame = pd.DataFrame(rows)
    EVALUATION_OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(EVALUATION_OUT, index=False)
    print(f"\nwrote {EVALUATION_OUT.relative_to(PROJECT_ROOT)}")

    validation = frame[frame["split"] == "validation"].iloc[0]
    gain = float(validation["delta_weighted_f1"])
    net = int(validation["net"])
    verdict = gain > ADOPTION_MIN_GAIN and net > 0
    print("\nadoption gate: validation weighted-F1 gain {:+.3f} "
          "(needs > {:.2f}) and net {:+d} (needs > 0)".format(
              gain, ADOPTION_MIN_GAIN, net))
    print("VERDICT:", "ADOPT" if verdict else
          "DECLINE - record as evaluated-and-declined")
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", nargs=2, metavar=("CLASS_A", "CLASS_B"),
                        action="append", default=None,
                        help="a confusion pair to train; repeatable")
    parser.add_argument("--all", action="store_true",
                        help=f"train the default pairs: {DEFAULT_PAIRS}")
    parser.add_argument("--evaluate", action="store_true",
                        help="score the gated ensemble against the base model")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args(argv)

    pairs = [tuple(p) for p in (args.pair or [])] or (DEFAULT_PAIRS if
                                                      (args.all or args.evaluate) else [])
    if not pairs:
        parser.error("pass --pair A B, or --all")

    if not args.evaluate:
        for pair in pairs:
            checkpoint = train_pair(pair, epochs=args.epochs,
                                    batch_size=args.batch_size,
                                    learning_rate=args.learning_rate,
                                    seed=args.seed)
            path = specialist_path(pair)
            torch.save(checkpoint, path)
            print(f"  saved {path.relative_to(PROJECT_ROOT)}")

    evaluate(pairs)


if __name__ == "__main__":
    main()
