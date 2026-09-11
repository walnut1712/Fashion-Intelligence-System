from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset

from src.data.splits import load_or_create_splits
from src.data.places365_backgrounds import load_background_bank
from src.data.synthetic_backgrounds import composite

# Reuse EXACT same cache/background machinery already used for T1/T2.
from scripts.train_task1_task2_background_adaptation import (
    CacheIndex,
    SOURCE_SHAPE,
    build_training_background_bank,
    get_device,
    resolve_project_root,
    seed_all,
)

try:
    from app.backend.services.task3_service import EarlyBranchCNN
except ImportError:
    from app.backend.services.task3_service import EarlyBranchNet as EarlyBranchCNN


SEED = 42
USAGE_MERGE = {
    "Smart Casual": "Casual",
    "Travel": "Casual",
    "Party": "Formal",
}


def find_checkpoint(project: Path) -> Path:
    candidates = [
        project / "artifacts" / "task3" / "task3_cnn_model.pt",
        project / "artifacts" / "task3_cnn" / "task3_cnn_model.pt",
        project / "artifacts" / "task3" / "task3_multitask_cnn.pt",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Cannot find Task 3 production checkpoint.\nChecked:\n" +
        "\n".join(str(x) for x in candidates)
    )


def prepare_parts(class_names):
    train, val, test = load_or_create_splits()

    gender_names = list(class_names["gender"])
    usage_names = list(class_names["usage"])
    gender_map = {x: i for i, x in enumerate(gender_names)}
    usage_map = {x: i for i, x in enumerate(usage_names)}

    output = []
    for frame in (train, val, test):
        f = frame[
            frame["gender"].notna()
            & frame["usage"].notna()
            & ~frame["usage"].eq("Home")
        ].copy()

        f["usage_final"] = f["usage"].replace(USAGE_MERGE)
        f["gender_y"] = f["gender"].map(gender_map)
        f["usage_y"] = f["usage_final"].map(usage_map)

        if f["gender_y"].isna().any():
            bad = sorted(f.loc[f["gender_y"].isna(), "gender"].astype(str).unique())
            raise ValueError(f"Unknown gender labels: {bad}")

        if f["usage_y"].isna().any():
            bad = sorted(f.loc[f["usage_y"].isna(), "usage_final"].astype(str).unique())
            raise ValueError(f"Unknown usage labels: {bad}")

        f["gender_y"] = f["gender_y"].astype("int64")
        f["usage_y"] = f["usage_y"].astype("int64")
        output.append(f)

    return tuple(output)


class Task3Dataset(Dataset):
    def __init__(
        self,
        frame,
        cache,
        image_size,
        mean,
        std,
        backgrounds=None,
        p_bg=0.0,
        seed=42,
        deterministic=False,
    ):
        self.frame = frame.reset_index(drop=True)
        self.positions = cache.positions_for(self.frame)
        self.cache = cache
        self.image_size = tuple(image_size)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.backgrounds = backgrounds
        self.p_bg = float(p_bg)
        self.seed = int(seed)
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.frame)

    def rng_for(self, index):
        if self.deterministic:
            return np.random.default_rng(self.seed + index * 10007)
        return self.rng

    def __getitem__(self, index):
        rng = self.rng_for(index)
        pos = int(self.positions[index])

        image = np.asarray(self.cache.images[pos]).copy()

        if (
            self.backgrounds is not None
            and self.p_bg > 0
            and rng.random() < self.p_bg
        ):
            bg = self.backgrounds[int(rng.integers(len(self.backgrounds)))]
            mask = np.asarray(self.cache.masks[pos])

            image = composite(
                image,
                mask,
                bg,
                rng,
                scale_range=(0.55, 1.00),
            )

        pil = Image.fromarray(image)

        if pil.size != self.image_size:
            pil = pil.resize(self.image_size, Image.Resampling.BILINEAR)

        arr = np.asarray(pil, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()
        tensor = (tensor - self.mean) / self.std

        row = self.frame.iloc[index]

        return (
            tensor,
            int(row["gender_y"]),
            int(row["usage_y"]),
        )


def metric_block(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    gt, gp = [], []
    ut, up = [], []

    for images, gender, usage in loader:
        images = images.to(device, non_blocking=True)

        out = model(images)

        gt.extend(gender.tolist())
        ut.extend(usage.tolist())

        gp.extend(out["gender"].argmax(1).cpu().tolist())
        up.extend(out["usage"].argmax(1).cpu().tolist())

    gender_m = metric_block(gt, gp)
    usage_m = metric_block(ut, up)

    result = {
        "gender": gender_m,
        "usage": usage_m,
        "mean_macro_f1": (
            gender_m["macro_f1"] + usage_m["macro_f1"]
        ) / 2.0,
        "exact_match": float(
            np.mean(
                (np.asarray(gt) == np.asarray(gp))
                & (np.asarray(ut) == np.asarray(up))
            )
        ),
    }

    raw = {
        "gender_true": gt,
        "gender_pred": gp,
        "usage_true": ut,
        "usage_pred": up,
    }

    return result, raw


def flatten_result(model_name, domain, result):
    return {
        "model": model_name,
        "domain": domain,
        "gender_accuracy": result["gender"]["accuracy"],
        "gender_macro_f1": result["gender"]["macro_f1"],
        "gender_weighted_f1": result["gender"]["weighted_f1"],
        "gender_balanced_accuracy": result["gender"]["balanced_accuracy"],
        "usage_accuracy": result["usage"]["accuracy"],
        "usage_macro_f1": result["usage"]["macro_f1"],
        "usage_weighted_f1": result["usage"]["weighted_f1"],
        "usage_balanced_accuracy": result["usage"]["balanced_accuracy"],
        "mean_macro_f1": result["mean_macro_f1"],
        "exact_match": result["exact_match"],
    }


def parse_args():
    """Flags only; every default is the value this script was committed with,
    so a bare `python scripts/train_task3_background_adaptation.py` reproduces
    the committed run exactly.

    They exist for one reason: the adapted arm gets `--epochs` more gradient
    steps than the baseline it is compared against, so a clean-side gain may be
    the extra training rather than the backdrops. `--p-bg 0 --tag _nobg_control`
    is that control - same schedule, no compositing - and `--tag` keeps it from
    overwriting the real run. Measured for Task 3 at lr 3e-4: 8 clean epochs
    alone move clean +1.55 and out-of-domain -0.33, so the clean delta against a
    zero-epoch baseline is not attributable to backgrounds.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--p-bg", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="",
                        help="suffix for the output directory and CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    project = resolve_project_root()
    seed_all(args.seed)

    device = get_device()
    baseline_path = find_checkpoint(project)

    ckpt = torch.load(
        baseline_path,
        map_location=device,
        weights_only=False,
    )

    class_names = {
        k: list(v)
        for k, v in ckpt["class_names"].items()
    }

    num_classes = ckpt.get(
        "num_classes",
        {k: len(v) for k, v in class_names.items()},
    )

    image_size = tuple(
        ckpt.get("image_size_pil", [60, 80])
    )

    arch = ckpt.get("architecture", {})

    model = EarlyBranchCNN(
        num_classes,
        input_shape=(3, image_size[1], image_size[0]),
        shared_widths=tuple(
            arch.get("shared_widths", (32, 64))
        ),
        branch_widths=tuple(
            arch.get("branch_widths", (128, 256))
        ),
        hidden=int(arch.get("hidden", 256)),
        dropout=float(arch.get("dropout", 0.5)),
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])

    baseline_model = copy.deepcopy(model).to(device)

    mean = ckpt["channel_mean"]
    std = ckpt["channel_std"]

    train, val, test = prepare_parts(class_names)

    cache = CacheIndex(project)

    print("=" * 78)
    print("TASK 3 BACKGROUND-ADAPTATION EXPERIMENT")
    print("=" * 78)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU   :", torch.cuda.get_device_name(0))
    print("Baseline:", baseline_path)
    print("Image size:", image_size)
    print(
        f"Train/Val/Test: "
        f"{len(train):,} / {len(val):,} / {len(test):,}"
    )
    print("Targets: gender + usage")
    print("Primary selection metric: mean macro-F1")
    print("Epochs: %d | lr=%g | p_bg=%.2f" % (args.epochs, args.lr, args.p_bg))
    print(
        "Training background mix: "
        "70% Places365 + 30% procedural"
    )
    print(
        "IMPORTANT: no generic strong augmentation; "
        "background is the intervention."
    )

    print("\nBuilding background banks...")

    train_bgs = build_training_background_bank(
        cache, 4000, SEED
    )

    val_bgs = load_background_bank(
        400,
        shape=SOURCE_SHAPE,
        split="train",
        seed=SEED + 101,
    )

    # Current Places365 helper names the held-out category split "test".
    test_bgs = load_background_bank(
        800,
        shape=SOURCE_SHAPE,
        split="test",
        seed=SEED + 202,
    )

    print(
        "Train/val/test background banks:",
        train_bgs.shape,
        val_bgs.shape,
        test_bgs.shape,
    )

    common = dict(
        cache=cache,
        image_size=image_size,
        mean=mean,
        std=std,
    )

    train_ds = Task3Dataset(
        train,
        **common,
        backgrounds=train_bgs,
        p_bg=args.p_bg,
        seed=args.seed,
        deterministic=False,
    )

    clean_val_ds = Task3Dataset(
        val,
        **common,
        backgrounds=None,
        p_bg=0,
        seed=SEED + 1,
        deterministic=True,
    )

    bg_val_ds = Task3Dataset(
        val,
        **common,
        backgrounds=val_bgs,
        p_bg=1.0,
        seed=SEED + 2,
        deterministic=True,
    )

    clean_test_ds = Task3Dataset(
        test,
        **common,
        backgrounds=None,
        p_bg=0,
        seed=SEED + 3,
        deterministic=True,
    )

    bg_test_ds = Task3Dataset(
        test,
        **common,
        backgrounds=test_bgs,
        p_bg=1.0,
        seed=SEED + 4,
        deterministic=True,
    )

    loader_kw = dict(
        batch_size=64,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    train_loader = DataLoader(
        train_ds, shuffle=True, **loader_kw
    )
    clean_val_loader = DataLoader(
        clean_val_ds, shuffle=False, **loader_kw
    )
    bg_val_loader = DataLoader(
        bg_val_ds, shuffle=False, **loader_kw
    )
    clean_test_loader = DataLoader(
        clean_test_ds, shuffle=False, **loader_kw
    )
    bg_test_loader = DataLoader(
        bg_test_ds, shuffle=False, **loader_kw
    )

    base_clean, _ = evaluate(
        baseline_model, clean_test_loader, device
    )
    base_bg, _ = evaluate(
        baseline_model, bg_test_loader, device
    )

    print("\nBASELINE CURRENT-PROTOCOL TEST")
    print(
        " clean:",
        {
            "gender_acc": round(
                base_clean["gender"]["accuracy"], 4
            ),
            "gender_macro_f1": round(
                base_clean["gender"]["macro_f1"], 4
            ),
            "usage_acc": round(
                base_clean["usage"]["accuracy"], 4
            ),
            "usage_macro_f1": round(
                base_clean["usage"]["macro_f1"], 4
            ),
            "mean_macro_f1": round(
                base_clean["mean_macro_f1"], 4
            ),
            "exact_match": round(
                base_clean["exact_match"], 4
            ),
        },
    )

    print(
        " photo:",
        {
            "gender_acc": round(
                base_bg["gender"]["accuracy"], 4
            ),
            "gender_macro_f1": round(
                base_bg["gender"]["macro_f1"], 4
            ),
            "usage_acc": round(
                base_bg["usage"]["accuracy"], 4
            ),
            "usage_macro_f1": round(
                base_bg["usage"]["macro_f1"], 4
            ),
            "mean_macro_f1": round(
                base_bg["mean_macro_f1"], 4
            ),
            "exact_match": round(
                base_bg["exact_match"], 4
            ),
        },
    )

    ce_gender = nn.CrossEntropyLoss()
    ce_usage = nn.CrossEntropyLoss()

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=8,
        eta_min=5e-6,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    best_score = float("-inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []

    started = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        gt, gp, ut, up = [], [], [], []
        losses = []

        for images, gender, usage in train_loader:
            images = images.to(device, non_blocking=True)
            gender = gender.long().to(
                device, non_blocking=True
            )
            usage = usage.long().to(
                device, non_blocking=True
            )

            optimiser.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                out = model(images)

                loss_gender = ce_gender(
                    out["gender"], gender
                )
                loss_usage = ce_usage(
                    out["usage"], usage
                )

                # Equal task weighting, matching the two-head objective.
                loss = loss_gender + loss_usage

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0
            )

            scaler.step(optimiser)
            scaler.update()

            losses.append(float(loss.item()))

            gt.extend(gender.detach().cpu().tolist())
            ut.extend(usage.detach().cpu().tolist())
            gp.extend(
                out["gender"]
                .argmax(1)
                .detach()
                .cpu()
                .tolist()
            )
            up.extend(
                out["usage"]
                .argmax(1)
                .detach()
                .cpu()
                .tolist()
            )

        scheduler.step()

        train_gender = metric_block(gt, gp)
        train_usage = metric_block(ut, up)

        train_mean = (
            train_gender["macro_f1"]
            + train_usage["macro_f1"]
        ) / 2

        clean_val, _ = evaluate(
            model, clean_val_loader, device
        )
        bg_val, _ = evaluate(
            model, bg_val_loader, device
        )

        # Equal importance: preserve catalogue quality + gain OOD robustness.
        selection = (
            0.5 * clean_val["mean_macro_f1"]
            + 0.5 * bg_val["mean_macro_f1"]
        )

        is_best = selection > best_score

        if is_best:
            best_score = selection
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_mean_macro_f1": train_mean,
            "clean_val_gender_macro_f1":
                clean_val["gender"]["macro_f1"],
            "clean_val_usage_macro_f1":
                clean_val["usage"]["macro_f1"],
            "clean_val_mean_macro_f1":
                clean_val["mean_macro_f1"],
            "clean_val_exact_match":
                clean_val["exact_match"],
            "bg_val_gender_macro_f1":
                bg_val["gender"]["macro_f1"],
            "bg_val_usage_macro_f1":
                bg_val["usage"]["macro_f1"],
            "bg_val_mean_macro_f1":
                bg_val["mean_macro_f1"],
            "bg_val_exact_match":
                bg_val["exact_match"],
            "selection_score": selection,
            "lr": optimiser.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train meanMF1={train_mean:.4f} | "
            f"clean-val={clean_val['mean_macro_f1']:.4f} | "
            f"bg-val={bg_val['mean_macro_f1']:.4f} | "
            f"mean={selection:.4f} | "
            f"exact(clean/bg)="
            f"{clean_val['exact_match']:.4f}/"
            f"{bg_val['exact_match']:.4f} | "
            f"{row['seconds']:.1f}s"
            + (" | NEW BEST" if is_best else "")
        )

    model.load_state_dict(best_state)

    adapted_clean, raw_clean = evaluate(
        model, clean_test_loader, device
    )
    adapted_bg, raw_bg = evaluate(
        model, bg_test_loader, device
    )

    # --tag keeps a control run from overwriting the real one; empty by default,
    # so the committed paths are unchanged.
    out_dir = (
        project / "artifacts" / ("task3_bgaug" + args.tag)
    )
    eval_dir = (
        project / "outputs" / "evaluation"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history).to_csv(
        out_dir / "training_history.csv",
        index=False,
    )

    comparison = pd.DataFrame([
        flatten_result(
            "baseline", "clean", base_clean
        ),
        flatten_result(
            "baseline",
            "heldout_places365",
            base_bg,
        ),
        flatten_result(
            "bg_adapted",
            "clean",
            adapted_clean,
        ),
        flatten_result(
            "bg_adapted",
            "heldout_places365",
            adapted_bg,
        ),
    ])

    comparison.to_csv(
        eval_dir / ("task3_bgadapt_comparison" + args.tag + ".csv"),
        index=False,
    )

    for target in ("gender", "usage"):
        names = class_names[target]
        truth = raw_clean[f"{target}_true"]
        pred = raw_clean[f"{target}_pred"]

        pd.DataFrame(
            classification_report(
                truth,
                pred,
                labels=list(range(len(names))),
                target_names=names,
                zero_division=0,
                output_dict=True,
            )
        ).T.to_csv(
            out_dir /
            f"classification_report_clean_{target}.csv"
        )

        pd.DataFrame(
            confusion_matrix(
                truth,
                pred,
                labels=list(range(len(names))),
            ),
            index=names,
            columns=names,
        ).to_csv(
            out_dir /
            f"confusion_matrix_clean_{target}.csv"
        )

        truth = raw_bg[f"{target}_true"]
        pred = raw_bg[f"{target}_pred"]

        pd.DataFrame(
            classification_report(
                truth,
                pred,
                labels=list(range(len(names))),
                target_names=names,
                zero_division=0,
                output_dict=True,
            )
        ).T.to_csv(
            out_dir /
            f"classification_report_heldout_places365_{target}.csv"
        )

        pd.DataFrame(
            confusion_matrix(
                truth,
                pred,
                labels=list(range(len(names))),
            ),
            index=names,
            columns=names,
        ).to_csv(
            out_dir /
            f"confusion_matrix_heldout_places365_{target}.csv"
        )

    new_ckpt = copy.deepcopy(ckpt)
    new_ckpt["state_dict"] = best_state
    new_ckpt["model_name"] = (
        str(ckpt.get("model_name", "Task3"))
        + "_bgadapt"
    )
    new_ckpt["background_adaptation"] = {
        "source_checkpoint": str(baseline_path),
        "training_mix":
            "70% Places365 + 30% procedural",
        "p_bg": 0.70,
        "scale_range": [0.55, 1.00],
        "selection":
            "0.5*clean_val_mean_macro_f1 + "
            "0.5*bg_val_mean_macro_f1",
        "final_ood_test":
            "held-out Places365 scene categories",
        "best_epoch": best_epoch,
        "epochs": 8,
        "seed": SEED,
    }
    new_ckpt["baseline_test"] = {
        "clean": base_clean,
        "heldout_places365": base_bg,
    }
    new_ckpt["adapted_test"] = {
        "clean": adapted_clean,
        "heldout_places365": adapted_bg,
    }

    save_path = (
        out_dir / "task3_bgadapt_best.pt"
    )
    torch.save(new_ckpt, save_path)

    delta_clean = (
        adapted_clean["mean_macro_f1"]
        - base_clean["mean_macro_f1"]
    )
    delta_bg = (
        adapted_bg["mean_macro_f1"]
        - base_bg["mean_macro_f1"]
    )

    summary = {
        "task": 3,
        "primary_metric": "mean_macro_f1",
        "best_epoch": best_epoch,
        "baseline_clean": base_clean,
        "baseline_heldout_places365": base_bg,
        "adapted_clean": adapted_clean,
        "adapted_heldout_places365": adapted_bg,
        "delta_clean_mean_macro_f1":
            delta_clean,
        "delta_ood_mean_macro_f1":
            delta_bg,
        "training_minutes":
            (time.time() - started) / 60,
        "checkpoint": str(save_path),
    }

    (
        out_dir / "summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("TASK 3 BACKGROUND ADAPTATION COMPLETE")
    print("=" * 78)
    print("Best epoch:", best_epoch)

    print("\nCLEAN")
    print(
        f"Gender Macro-F1: "
        f"{base_clean['gender']['macro_f1']:.4f}"
        f" -> "
        f"{adapted_clean['gender']['macro_f1']:.4f}"
    )
    print(
        f"Usage  Macro-F1: "
        f"{base_clean['usage']['macro_f1']:.4f}"
        f" -> "
        f"{adapted_clean['usage']['macro_f1']:.4f}"
    )
    print(
        f"Mean   Macro-F1: "
        f"{base_clean['mean_macro_f1']:.4f}"
        f" -> "
        f"{adapted_clean['mean_macro_f1']:.4f}"
        f" ({delta_clean:+.4f})"
    )
    print(
        f"Exact match     : "
        f"{base_clean['exact_match']:.4f}"
        f" -> "
        f"{adapted_clean['exact_match']:.4f}"
    )

    print("\nHELD-OUT PLACES365 OOD")
    print(
        f"Gender Macro-F1: "
        f"{base_bg['gender']['macro_f1']:.4f}"
        f" -> "
        f"{adapted_bg['gender']['macro_f1']:.4f}"
    )
    print(
        f"Usage  Macro-F1: "
        f"{base_bg['usage']['macro_f1']:.4f}"
        f" -> "
        f"{adapted_bg['usage']['macro_f1']:.4f}"
    )
    print(
        f"Mean   Macro-F1: "
        f"{base_bg['mean_macro_f1']:.4f}"
        f" -> "
        f"{adapted_bg['mean_macro_f1']:.4f}"
        f" ({delta_bg:+.4f})"
    )
    print(
        f"Exact match     : "
        f"{base_bg['exact_match']:.4f}"
        f" -> "
        f"{adapted_bg['exact_match']:.4f}"
    )

    print("\nCheckpoint:", save_path)
    print(
        "Comparison:",
        eval_dir / ("task3_bgadapt_comparison" + args.tag + ".csv"),
    )
    print(
        "\nDO NOT overwrite production Task 3 "
        "until trade-off is reviewed."
    )


if __name__ == "__main__":
    main()
