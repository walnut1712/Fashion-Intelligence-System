from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

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
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.data.config import CANDIDATE_ARTIFACT_DIRS, DATA_DIR, IMAGE_SIZE_PIL
from src.data.splits import load_or_create_splits

CLASS_NAMES = ["Fall", "Spring", "Summer", "Winter"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

OLD_SAME_SPLIT = {
    "accuracy": 0.6863,
    "macro_f1": 0.6567,
    "weighted_f1": 0.6884,
    "balanced_accuracy": 0.7082,
}


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_parts(train, val, test):
    parts = []
    for frame in (train, val, test):
        frame = frame[frame["season"].notna()].copy()
        invalid = sorted(set(frame["season"].astype(str)) - set(CLASS_NAMES))
        if invalid:
            raise ValueError(f"Unexpected season labels: {invalid}")
        frame["label"] = frame["season"].map(CLASS_TO_INDEX).astype("int64")
        parts.append(frame)

    train, val, test = parts
    assert (len(train), len(val), len(test)) == (27533, 5510, 5508)

    category_names = sorted(
        train["masterCategory"].fillna("Unknown").astype(str).unique().tolist()
    )
    category_to_index = {name: i for i, name in enumerate(category_names)}

    for frame in (train, val, test):
        values = frame["masterCategory"].fillna("Unknown").astype(str)
        unseen = sorted(set(values) - set(category_names))
        if unseen:
            raise ValueError(
                "Auxiliary masterCategory appears outside training vocabulary: "
                f"{unseen}"
            )
        frame["category_label"] = values.map(category_to_index).astype("int64")

    return train, val, test, category_names, category_to_index


class MultiTaskDataset(Dataset):
    def __init__(self, frame, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(DATA_DIR / f"{int(row['id'])}.jpg") as image:
            array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).float() / 255.0
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, int(row["label"]), int(row["category_label"])


def make_transforms():
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.05, 0.05),
            scale=(0.92, 1.08),
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.ColorJitter(contrast=0.10),
    ])
    return train_tf, transforms.Compose([])


class SeasonCNNMultiTask(nn.Module):
    def __init__(self, num_season_classes=4, num_category_classes=7):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(128, num_season_classes),
        )

        self.category_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_category_classes),
        )

    def forward(self, x, return_aux=False):
        features = self.features(x)
        season_logits = self.classifier(features)
        if not return_aux:
            return season_logits
        category_logits = self.category_head(features)
        return season_logits, category_logits


def metric_dict(y, p):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
    }


@torch.no_grad()
def evaluate(model, loader, frame, device, season_criterion):
    model.eval()
    loss_sum = 0.0
    n = 0
    y_true, y_pred = [], []
    category_true, category_pred = [], []

    for images, season_labels, category_labels in loader:
        images = images.to(device, non_blocking=True)
        season_labels = season_labels.long().to(device, non_blocking=True)
        category_labels = category_labels.long().to(device, non_blocking=True)

        season_logits, category_logits = model(images, return_aux=True)
        batch_n = len(season_labels)
        loss_sum += float(
            season_criterion(season_logits, season_labels).item()
        ) * batch_n
        n += batch_n

        season_pred = season_logits.argmax(1)
        cat_pred = category_logits.argmax(1)

        y_true.extend(season_labels.cpu().tolist())
        y_pred.extend(season_pred.cpu().tolist())
        category_true.extend(category_labels.cpu().tolist())
        category_pred.extend(cat_pred.cpu().tolist())

    result = metric_dict(y_true, y_pred)
    result["loss"] = float(loss_sum / n)
    result["category_accuracy"] = float(
        accuracy_score(category_true, category_pred)
    )

    acc_mask = (
        frame.reset_index(drop=True)["masterCategory"]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq("accessories")
        .to_numpy()
    )
    if acc_mask.any():
        y_np = np.asarray(y_true)
        p_np = np.asarray(y_pred)
        result["accessory_accuracy"] = float(
            accuracy_score(y_np[acc_mask], p_np[acc_mask])
        )
        result["accessory_macro_f1"] = float(
            f1_score(
                y_np[acc_mask],
                p_np[acc_mask],
                average="macro",
                zero_division=0,
            )
        )

    return result, y_true, y_pred


def season_entropy(values):
    p = (
        pd.Series(values)
        .value_counts(normalize=True)
        .reindex(CLASS_NAMES, fill_value=0)
        .to_numpy(float)
    )
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(4)) if len(p) else 0.0


def build_all_season_policy(
    train, output_dir, min_samples=20, entropy_threshold=0.75
):
    accessories = train[
        train["masterCategory"].astype(str).str.lower().eq("accessories")
    ].copy()

    rows = []
    for article_type, group in accessories.groupby("articleType", dropna=True):
        rows.append({
            "articleType": str(article_type),
            "count": int(len(group)),
            "entropy": season_entropy(group["season"]),
        })

    stats = pd.DataFrame(rows)
    selected = [] if stats.empty else sorted(
        stats.loc[
            (stats["count"] >= min_samples)
            & (stats["entropy"] >= entropy_threshold),
            "articleType",
        ].astype(str).tolist()
    )

    policy = {
        "display_label": "All Season",
        "official_task2_classes": CLASS_NAMES,
        "source_split": "train_only",
        "min_samples": int(min_samples),
        "entropy_threshold": float(entropy_threshold),
        "article_types": selected,
        "note": (
            "Post-processing/UI policy only. "
            "All Season is not a fifth Task-2 model class."
        ),
    }

    policy_path = output_dir / "task2_all_season_policy.json"
    stats_path = output_dir / "task2_accessory_season_entropy.csv"
    policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    stats.to_csv(stats_path, index=False)

    return policy, policy_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.15)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_all(args.seed)

    train, val, test, category_names, category_to_index = prepare_parts(
        *load_or_create_splits()
    )

    output_dir = CANDIDATE_ARTIFACT_DIRS["task2"]
    output_dir.mkdir(parents=True, exist_ok=True)

    policy, policy_path = build_all_season_policy(train, output_dir)

    train_tf, eval_tf = make_transforms()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=use_cuda,
    )
    train_loader = DataLoader(
        MultiTaskDataset(train, train_tf), shuffle=True, **loader_kw
    )
    val_loader = DataLoader(
        MultiTaskDataset(val, eval_tf), shuffle=False, **loader_kw
    )
    test_loader = DataLoader(
        MultiTaskDataset(test, eval_tf), shuffle=False, **loader_kw
    )

    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(4),
        y=train["label"].to_numpy(),
    )
    class_weights = torch.tensor(
        class_weights_np, dtype=torch.float32, device=device
    )

    model = SeasonCNNMultiTask(
        num_season_classes=4,
        num_category_classes=len(category_names),
    ).to(device)

    season_criterion = nn.CrossEntropyLoss(weight=class_weights)
    category_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.scheduler_patience,
        min_lr=1e-6,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    full_path = output_dir / "task2_v2_multitask_120x160_full.pt"
    backend_path = output_dir / "task2_v2_multitask_120x160_backend.pth"
    history_path = output_dir / "training_history_v2_multitask.csv"

    best_macro = -1.0
    best_epoch = None
    bad = 0
    history = []
    started = time.time()

    print("=" * 72)
    print("TASK 2 120x160 — V2 AUXILIARY masterCategory")
    print("=" * 72)
    print(f"Train/Val/Test: {len(train)} / {len(val)} / {len(test)}")
    print(f"Season classes: {CLASS_NAMES}")
    print(f"Auxiliary categories ({len(category_names)}): {category_names}")
    print(f"Device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Season loss: class-weighted CrossEntropy")
    print("Category loss: CrossEntropy")
    print(f"Auxiliary weight: {args.aux_weight}")
    print("Optimizer: Adam | Scheduler: ReduceLROnPlateau(val Macro-F1)")
    print("Checkpoint/early-stop: validation Season Macro-F1")
    print("Accessory oversampling: OFF")
    print(f"All Season articleTypes: {len(policy['article_types'])}")
    print()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        season_loss_sum = 0.0
        category_loss_sum = 0.0
        n = 0
        train_y, train_p = [], []
        cat_true, cat_pred = [], []

        for images, season_labels, category_labels in train_loader:
            images = images.to(device, non_blocking=True)
            season_labels = season_labels.long().to(device, non_blocking=True)
            category_labels = category_labels.long().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_cuda,
            ):
                season_logits, category_logits = model(
                    images, return_aux=True
                )
                season_loss = season_criterion(
                    season_logits, season_labels
                )
                category_loss = category_criterion(
                    category_logits, category_labels
                )
                total_loss = (
                    season_loss + args.aux_weight * category_loss
                )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_n = len(season_labels)
            n += batch_n
            season_loss_sum += float(season_loss.item()) * batch_n
            category_loss_sum += float(category_loss.item()) * batch_n

            train_y.extend(season_labels.detach().cpu().tolist())
            train_p.extend(
                season_logits.argmax(1).detach().cpu().tolist()
            )
            cat_true.extend(category_labels.detach().cpu().tolist())
            cat_pred.extend(
                category_logits.argmax(1).detach().cpu().tolist()
            )

        train_metrics = metric_dict(train_y, train_p)
        train_season_loss = season_loss_sum / n
        train_category_loss = category_loss_sum / n
        train_category_acc = float(
            accuracy_score(cat_true, cat_pred)
        )

        val_metrics, _, _ = evaluate(
            model, val_loader, val, device, season_criterion
        )

        scheduler.step(val_metrics["macro_f1"])
        lr = float(optimizer.param_groups[0]["lr"])
        elapsed = time.time() - t0

        improved = val_metrics["macro_f1"] > best_macro + 1e-4

        if improved:
            best_macro = val_metrics["macro_f1"]
            best_epoch = epoch
            bad = 0

            torch.save({
                "state_dict": model.state_dict(),
                "model_name": "SeasonCNNMultiTask_120x160",
                "num_classes": 4,
                "class_names": CLASS_NAMES,
                "category_names": category_names,
                "category_to_index": category_to_index,
                "image_size_pil": list(IMAGE_SIZE_PIL),
                "architecture": {
                    "season_head": [128, 128, 4],
                    "season_dropout": 0.40,
                    "category_head": [128, 64, len(category_names)],
                    "category_dropout": 0.25,
                },
                "config": {
                    **vars(args),
                    "season_class_weights": class_weights_np.tolist(),
                    "season_loss": "class_weighted_cross_entropy",
                    "category_loss": "cross_entropy",
                    "scheduler": "ReduceLROnPlateau",
                    "scheduler_metric": "val_macro_f1",
                    "selection_metric": "val_macro_f1",
                    "accessory_oversampling": False,
                    "augmentation": "flip_affine_contrast",
                },
                "val_metrics": {**val_metrics, "epoch": epoch},
                "all_season_policy_path": str(policy_path),
            }, full_path)
            status = "NEW BEST"
        else:
            bad += 1
            status = ""

        history.append({
            "epoch": epoch,
            "train_season_loss": train_season_loss,
            "train_category_loss": train_category_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_category_accuracy": train_category_acc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_category_accuracy": val_metrics["category_accuracy"],
            "lr": lr,
            "seconds": elapsed,
        })

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train acc={train_metrics['accuracy']:.4f} "
            f"mF1={train_metrics['macro_f1']:.4f} "
            f"catAcc={train_category_acc:.4f} | "
            f"val acc={val_metrics['accuracy']:.4f} "
            f"mF1={val_metrics['macro_f1']:.4f} "
            f"wF1={val_metrics['weighted_f1']:.4f} "
            f"bal={val_metrics['balanced_accuracy']:.4f} "
            f"catAcc={val_metrics['category_accuracy']:.4f} | "
            f"lr={lr:.6g} | {elapsed:.1f}s"
            + (f" | {status}" if status else ""),
            flush=True,
        )

        if bad >= args.early_stopping_patience:
            print(
                f"Early stopping after {bad} epochs without "
                "validation Macro-F1 improvement."
            )
            break

    pd.DataFrame(history).to_csv(history_path, index=False)

    best = torch.load(full_path, map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    model.eval()

    test_metrics, y_true, y_pred = evaluate(
        model, test_loader, test, device, season_criterion
    )

    pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(4)),
            target_names=CLASS_NAMES,
            zero_division=0,
            output_dict=True,
        )
    ).T.to_csv(output_dir / "classification_report_v2_multitask.csv")

    pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=list(range(4))),
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(output_dir / "confusion_matrix_v2_multitask.csv")

    backend_state_dict = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("category_head.")
    }
    torch.save(backend_state_dict, backend_path)

    best["test_metrics"] = test_metrics
    best["best_epoch"] = best_epoch
    torch.save(best, full_path)

    deltas = {
        key: float(test_metrics[key] - OLD_SAME_SPLIT[key])
        for key in OLD_SAME_SPLIT
    }
    promote = (
        test_metrics["macro_f1"] > OLD_SAME_SPLIT["macro_f1"]
        and test_metrics["accuracy"] >= OLD_SAME_SPLIT["accuracy"] - 0.005
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro,
        "test_metrics": test_metrics,
        "old_60x80_same_split": OLD_SAME_SPLIT,
        "delta_candidate_minus_old": deltas,
        "promotion_rule": (
            "candidate macro_f1 > old macro_f1 AND "
            "candidate accuracy >= old accuracy - 0.005"
        ),
        "promote": bool(promote),
        "all_season_article_types": policy["article_types"],
    }
    (output_dir / "task2_v2_multitask_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("TASK 2 V2 120x160 COMPLETE")
    print("=" * 72)
    print(f"Best epoch: {best_epoch}")
    print(f"Best val Macro-F1: {best_macro:.4f}")
    print(f"Test Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"Test Macro-F1:          {test_metrics['macro_f1']:.4f}")
    print(f"Test Weighted-F1:       {test_metrics['weighted_f1']:.4f}")
    print(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print()
    print("OLD 60x80 ON SAME TEST SPLIT")
    print(f"Accuracy:          {OLD_SAME_SPLIT['accuracy']:.4f}")
    print(f"Macro-F1:          {OLD_SAME_SPLIT['macro_f1']:.4f}")
    print(f"Weighted-F1:       {OLD_SAME_SPLIT['weighted_f1']:.4f}")
    print(f"Balanced Accuracy: {OLD_SAME_SPLIT['balanced_accuracy']:.4f}")
    print()
    print("DELTA V2 - OLD")
    for key, value in deltas.items():
        print(f"{key:18s}: {value:+.4f}")
    print()
    print("DECISION:", "PROMOTE V2" if promote else "KEEP OLD 60x80")
    print(f"Training time: {(time.time() - started) / 60:.2f} minutes")
    print(f"Full model: {full_path}")
    print(f"Backend candidate: {backend_path}")
    print(f"All Season policy: {policy_path}")


if __name__ == "__main__":
    main()
