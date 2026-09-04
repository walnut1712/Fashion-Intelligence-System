from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.data.config import CANDIDATE_ARTIFACT_DIRS, IMAGE_SIZE_PIL
from src.data.splits import load_or_create_splits
from src.models.item_type_classifier import ItemTypeCNN
from src.training.candidate_120x160 import CandidateDataset, compute_train_normalization, task1_frame_parts


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NormalizeOnly:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image):
        return (image - self.mean) / self.std


class BaselineTrainTransform:
    """Match the old Task-1 baseline recipe: horizontal flip only."""
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image):
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=[2])
        return (image - self.mean) / self.std


def make_model(num_classes: int):
    return ItemTypeCNN(
        num_classes,
        widths=(16, 32, 64, 128),
        head_hidden=384,
        pool_grid=(1, 1),
        pool_mode="avgmax",
    )


def calc_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    losses, y_true, y_pred = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        losses.append(float(loss.item()))

        pred = logits.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

    out = calc_metrics(y_true, y_pred)
    out["loss"] = float(np.mean(losses))
    return out, y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--select-on",
        choices=["weighted_f1", "macro_f1", "balanced_acc", "accuracy"],
        default="weighted_f1",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    train, val, test = load_or_create_splits()
    (train, val, test), classes = task1_frame_parts(train, val, test)
    num_classes = len(classes)

    assert num_classes == 92
    assert len(train) == 27458
    assert len(val) == 5497
    assert len(test) == 5495

    artifact_dir = CANDIDATE_ARTIFACT_DIRS["task1"]
    artifact_dir.mkdir(parents=True, exist_ok=True)

    norm_path = artifact_dir / "normalization_120x160.json"
    if norm_path.exists():
        saved = json.loads(norm_path.read_text())
        mean = np.asarray(saved["mean"], dtype=np.float32)
        std = np.asarray(saved["std"], dtype=np.float32)
    else:
        mean, std = compute_train_normalization(train)
        norm_path.write_text(json.dumps({
            "mean": mean.tolist(),
            "std": std.tolist(),
            "image_size_pil": list(IMAGE_SIZE_PIL),
        }, indent=2))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_ds = CandidateDataset(train, target="label", transform=BaselineTrainTransform(mean, std))
    val_ds = CandidateDataset(val, target="label", transform=NormalizeOnly(mean, std))
    test_ds = CandidateDataset(test, target="label", transform=NormalizeOnly(mean, std))

    loader_args = dict(batch_size=args.batch_size, num_workers=0, pin_memory=use_cuda)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    model = make_model(num_classes).to(device)

    # Match the old published baseline: NO inverse-frequency class weights.
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=args.learning_rate * 3.0,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    best_score = float("-inf")
    best_epoch = None
    best_path = artifact_dir / "task1_120x160_onecycle_best.pt"
    history = []

    print("=" * 68)
    print("TASK 1 120x160 — FAIR-COMPARISON ONECYCLE RUN")
    print("=" * 68)
    print(f"Classes: {num_classes}")
    print(f"Train/Val/Test: {len(train)} / {len(val)} / {len(test)}")
    print(f"Device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Recipe: AdamW + OneCycleLR + flip-only + NO class weights")
    print("Label smoothing: 0.05")
    print(f"Selection metric: validation {args.select_on}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size}")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()

        losses, y_true, y_pred = [], [], []

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()

            losses.append(float(loss.item()))
            pred = logits.argmax(dim=1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())

        train_metrics = calc_metrics(y_true, y_pred)
        train_metrics["loss"] = float(np.mean(losses))
        val_metrics, _, _ = evaluate(model, val_loader, device, criterion)

        lr_now = float(optimiser.param_groups[0]["lr"])
        elapsed = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_acc": val_metrics["balanced_acc"],
            "lr": lr_now,
            "seconds": elapsed,
        })

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train acc={train_metrics['accuracy']:.4f} "
            f"wF1={train_metrics['weighted_f1']:.4f} "
            f"mF1={train_metrics['macro_f1']:.4f} | "
            f"val acc={val_metrics['accuracy']:.4f} "
            f"wF1={val_metrics['weighted_f1']:.4f} "
            f"mF1={val_metrics['macro_f1']:.4f} "
            f"bal={val_metrics['balanced_acc']:.4f} | "
            f"lr={lr_now:.6g} | {elapsed:.1f}s",
            flush=True,
        )

        score = val_metrics[args.select_on]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            checkpoint = {
                "state_dict": model.state_dict(),
                "architecture": {
                    "name": "ItemTypeCNN",
                    "widths": [16, 32, 64, 128],
                    "head_hidden": 384,
                    "pool_grid": [1, 1],
                    "pool_mode": "avgmax",
                },
                "class_names": classes,
                "num_classes": num_classes,
                "channel_mean": mean.tolist(),
                "channel_std": std.tolist(),
                "image_size_pil": list(IMAGE_SIZE_PIL),
                "config": {
                    **vars(args),
                    "class_weights": False,
                    "label_smoothing": 0.05,
                    "augmentation": "horizontal_flip_only",
                    "scheduler": "OneCycleLR",
                    "max_lr": args.learning_rate * 3.0,
                    "gradient_clip_norm": 5.0,
                },
                "val_metrics": {**val_metrics, "epoch": epoch},
                "task": "articleType",
            }
            torch.save(checkpoint, best_path)
            print(f"  NEW BEST -> {best_path}", flush=True)

    pd.DataFrame(history).to_csv(
        artifact_dir / "training_history_onecycle.csv",
        index=False,
    )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    best_model = make_model(num_classes).to(device)
    best_model.load_state_dict(checkpoint["state_dict"])

    test_metrics, y_true, y_pred = evaluate(best_model, test_loader, device, criterion)

    pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=classes,
            zero_division=0,
            output_dict=True,
        )
    ).T.to_csv(artifact_dir / "classification_report_onecycle.csv")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        artifact_dir / "confusion_matrix_onecycle.csv"
    )

    checkpoint["test_metrics"] = test_metrics
    checkpoint["best_epoch"] = best_epoch
    torch.save(checkpoint, best_path)

    total_minutes = (time.time() - start_time) / 60.0

    print()
    print("=" * 68)
    print("TASK 1 120x160 ONECYCLE COMPLETE")
    print("=" * 68)
    print(f"Best epoch: {best_epoch}")
    print(f"Selection metric: {args.select_on}")
    print(f"Best val Weighted-F1: {checkpoint['val_metrics']['weighted_f1']:.4f}")
    print(f"Best val Macro-F1: {checkpoint['val_metrics']['macro_f1']:.4f}")
    print(f"Best val Accuracy: {checkpoint['val_metrics']['accuracy']:.4f}")
    print(f"Best val Balanced Accuracy: {checkpoint['val_metrics']['balanced_acc']:.4f}")
    print(f"Test Weighted-F1: {test_metrics['weighted_f1']:.4f}")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Balanced Accuracy: {test_metrics['balanced_acc']:.4f}")
    print(f"Training time: {total_minutes:.2f} minutes")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()