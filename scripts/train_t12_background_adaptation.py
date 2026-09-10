from __future__ import annotations

"""
Fast overnight background-adaptation experiment for COSC2753 Fashion Intelligence System.

Purpose
-------
Test the SAME deployment-oriented background intervention on Task 1 and Task 2
without rebuilding the whole project.

Task 1:
    existing final 120x160 CNN -> background-adaptation fine-tune
Task 2:
    existing final 60x80 season CNN -> background-adaptation fine-tune

Training backgrounds:
    70% Places365 photographic + 30% procedural, using the existing Task 4 utilities.
OOD evaluation:
    held-out Places365 scene categories only.

The original checkpoints are never overwritten.
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path
import sys

# Allow direct execution from scripts/ while importing repo packages.
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
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from app.backend.services.task2_service import SeasonCNN
from src.data.splits import load_or_create_splits
from src.data.places365_backgrounds import load_background_bank, make_mixed_bank
from src.data.synthetic_backgrounds import composite, make_backgrounds
from src.models.item_type_classifier import ItemTypeCNN
from src.training.candidate_120x160 import task1_frame_parts


SEED = 42
SOURCE_H, SOURCE_W = 160, 120
SOURCE_SHAPE = (SOURCE_H, SOURCE_W, 3)
CLASS_NAMES_T2 = ["Fall", "Spring", "Summer", "Winter"]
CLASS_TO_INDEX_T2 = {name: i for i, name in enumerate(CLASS_NAMES_T2)}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_project_root() -> Path:
    here = Path.cwd()
    if (here / "A2_FashionDataset").exists():
        return here
    if (here.parent / "A2_FashionDataset").exists():
        return here.parent
    raise FileNotFoundError(
        "Cannot find A2_FashionDataset. Run this script from the repository root "
        "or from a direct child directory."
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def load_state_dict_flexible(path: Path, device: torch.device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj
    return obj, {}


def make_task1_model(num_classes: int) -> ItemTypeCNN:
    return ItemTypeCNN(
        num_classes,
        widths=(16, 32, 64, 128),
        head_hidden=384,
        pool_grid=(1, 1),
        pool_mode="avgmax",
    )


def prepare_task2_parts(train, val, test):
    parts = []
    for frame in (train, val, test):
        frame = frame[frame["season"].notna()].copy()
        bad = sorted(set(frame["season"].astype(str)) - set(CLASS_NAMES_T2))
        if bad:
            raise ValueError(f"Unexpected season labels: {bad}")
        frame["label"] = frame["season"].map(CLASS_TO_INDEX_T2).astype("int64")
        parts.append(frame)
    return tuple(parts)


class CacheIndex:
    """Maps assignment IDs to the existing Task 4 120x160 image/mask cache."""

    def __init__(self, project: Path):
        processed = project / "A2_FashionDataset" / "processed"
        self.gallery_path = processed / "task4_gallery_120x160.csv"
        self.images_path = processed / "task4_cache_120x160.npy"
        self.masks_path = processed / "task4_masks_120x160.npy"

        missing = [
            p for p in (self.gallery_path, self.images_path, self.masks_path)
            if not p.exists()
        ]
        if missing:
            names = ", ".join(p.name for p in missing)
            raise FileNotFoundError(
                f"Missing Task 4 cache file(s): {names}\n"
                "Build once with:\n"
                "  python scripts/build_task4_cache.py --resolution 120x160"
            )

        gallery = pd.read_csv(self.gallery_path).reset_index(drop=True)
        if "position" not in gallery.columns:
            gallery["position"] = np.arange(len(gallery))

        self.images = np.load(self.images_path, mmap_mode="r")
        self.masks = np.load(self.masks_path, mmap_mode="r")

        if not (len(gallery) == len(self.images) == len(self.masks)):
            raise RuntimeError("Task 4 gallery/image/mask cache lengths disagree.")

        self.id_to_pos = {
            str(row_id): int(pos)
            for row_id, pos in zip(gallery["id"], gallery["position"])
        }

    def positions_for(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [str(x) for x in frame["id"] if str(x) not in self.id_to_pos]
        if missing:
            raise RuntimeError(
                f"{len(missing)} requested IDs are absent from task4_gallery_120x160.csv. "
                f"First few: {missing[:5]}"
            )
        return np.asarray([self.id_to_pos[str(x)] for x in frame["id"]], dtype=np.int64)


def build_training_background_bank(
    cache: CacheIndex,
    n_total: int,
    seed: int,
) -> np.ndarray:
    """Match Task 4's 70% photographic / 30% procedural background mixture."""
    n_photo = int(round(n_total * 0.70))
    n_proc = n_total - n_photo

    rng = np.random.default_rng(seed)
    source_n = min(800, len(cache.images))
    source_pos = np.sort(rng.choice(len(cache.images), source_n, replace=False))
    source_images = np.asarray(cache.images[source_pos])

    procedural = make_backgrounds(
        n_proc,
        shape=SOURCE_SHAPE,
        seed=seed,
        source_images=source_images,
    )
    photographic = load_background_bank(
        n_photo,
        shape=SOURCE_SHAPE,
        split="train",
        seed=seed,
    )
    return make_mixed_bank(procedural, photographic, 0.70, seed=seed)


class ClassificationCacheDataset(Dataset):
    """
    Uses the existing 120x160 catalogue cache and item masks.

    Training:
      with probability p_bg, paste the item onto a randomly sampled background.
    Validation/test:
      deterministic compositing when p_bg=1, making model comparisons paired.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        cache: CacheIndex,
        target_size_wh: tuple[int, int],
        label_col: str = "label",
        backgrounds: np.ndarray | None = None,
        p_bg: float = 0.0,
        scale_range=(0.55, 1.00),
        seed: int = 42,
        deterministic: bool = False,
        normalize_mean=None,
        normalize_std=None,
        task2_train_transform: bool = False,
        horizontal_flip: bool = False,
    ):
        self.frame = frame.reset_index(drop=True)
        self.positions = cache.positions_for(self.frame)
        self.cache = cache
        self.target_size_wh = target_size_wh
        self.label_col = label_col
        self.backgrounds = backgrounds
        self.p_bg = float(p_bg)
        self.scale_range = scale_range
        self.seed = int(seed)
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        self.horizontal_flip = horizontal_flip

        self.mean = None
        self.std = None
        if normalize_mean is not None:
            self.mean = torch.tensor(normalize_mean, dtype=torch.float32).view(3, 1, 1)
            self.std = torch.tensor(normalize_std, dtype=torch.float32).view(3, 1, 1)

        self.task2_tf = None
        if task2_train_transform:
            self.task2_tf = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.05, 0.05),
                    scale=(0.92, 1.08),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ColorJitter(contrast=0.10),
            ])

    def __len__(self):
        return len(self.frame)

    def _rng(self, index: int):
        if self.deterministic:
            return np.random.default_rng(self.seed + index * 10007)
        return self.rng

    def __getitem__(self, index):
        pos = int(self.positions[index])
        image = np.asarray(self.cache.images[pos]).copy()
        rng = self._rng(index)

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
                scale_range=self.scale_range,
            )

        pil = Image.fromarray(image)

        # Resize after compositing. T1 stays at 120x160; T2 becomes 60x80.
        if pil.size != self.target_size_wh:
            pil = pil.resize(self.target_size_wh, Image.BILINEAR)

        if self.task2_tf is not None:
            pil = self.task2_tf(pil)

        array = np.asarray(pil, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1))

        if self.horizontal_flip and rng.random() < 0.5:
            tensor = torch.flip(tensor, dims=[2])

        if self.mean is not None:
            tensor = (tensor - self.mean) / self.std

        label = int(self.frame.iloc[index][self.label_col])
        return tensor, label


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    losses, y_true, y_pred = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)
        logits = model(images)
        losses.append(float(criterion(logits, labels).item()))
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())

    result = metrics(y_true, y_pred)
    result["loss"] = float(np.mean(losses))
    return result, y_true, y_pred


def train_adaptation(
    task: int,
    project: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    p_bg: float,
    train_bank_size: int,
    val_bank_size: int,
    test_bank_size: int,
    seed: int,
):
    seed_all(seed)
    device = get_device()
    cache = CacheIndex(project)

    train, val, test = load_or_create_splits()

    if task == 1:
        (train, val, test), class_names = task1_frame_parts(train, val, test)
        num_classes = len(class_names)
        baseline_path = (
            project / "artifacts" / "task1_120x160" /
            "task1_120x160_onecycle_best.pt"
        )
        baseline_state, baseline_meta = load_state_dict_flexible(baseline_path, device)
        model = make_task1_model(num_classes).to(device)
        model.load_state_dict(baseline_state)

        mean = baseline_meta.get("channel_mean")
        std = baseline_meta.get("channel_std")
        if mean is None or std is None:
            raise RuntimeError("Task 1 checkpoint does not contain channel mean/std.")

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        primary = "weighted_f1"
        target_size = (120, 160)
        task2_tf = False
        flip = True
        out_dir = project / "artifacts" / "task1_bgaug"
        output_name = "task1_bgadapt_best.pt"
        optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    elif task == 2:
        train, val, test = prepare_task2_parts(train, val, test)
        class_names = CLASS_NAMES_T2
        num_classes = 4
        baseline_path = project / "artifacts" / "task2" / "task2_season_best_pytorch.pth"
        baseline_state, baseline_meta = load_state_dict_flexible(baseline_path, device)
        model = SeasonCNN(num_classes).to(device)
        model.load_state_dict(baseline_state)

        weights = compute_class_weight(
            "balanced",
            classes=np.arange(num_classes),
            y=train["label"].to_numpy(),
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device)
        )
        mean = std = None
        primary = "macro_f1"
        target_size = (60, 80)
        task2_tf = True
        flip = False
        out_dir = project / "artifacts" / "task2_bgaug"
        output_name = "task2_bgadapt_best.pth"
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    else:
        raise ValueError("--task must be 1 or 2")

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = project / "outputs" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"TASK {task} BACKGROUND-ADAPTATION EXPERIMENT")
    print("=" * 76)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU   :", torch.cuda.get_device_name(0))
    print("Baseline:", baseline_path)
    print(f"Train/Val/Test: {len(train):,} / {len(val):,} / {len(test):,}")
    print(f"Primary metric: {primary}")
    print(f"Fine-tune epochs: {epochs} | lr={lr:g} | p_bg={p_bg:.2f}")
    print("Training background mix: 70% Places365 + 30% procedural")
    print("Final OOD test: held-out Places365 scene categories only")

    print("\nBuilding background banks...")
    train_bgs = build_training_background_bank(cache, train_bank_size, seed)
    val_bgs = load_background_bank(
        val_bank_size, shape=SOURCE_SHAPE, split="train", seed=seed + 101
    )
    test_bgs = load_background_bank(
        test_bank_size, shape=SOURCE_SHAPE, split="test", seed=seed + 202
    )
    print("Train/val/test background banks:",
          train_bgs.shape, val_bgs.shape, test_bgs.shape)

    common = dict(
        cache=cache,
        target_size_wh=target_size,
        normalize_mean=mean,
        normalize_std=std,
    )

    train_ds = ClassificationCacheDataset(
        train, **common,
        backgrounds=train_bgs,
        p_bg=p_bg,
        scale_range=(0.55, 1.00),
        seed=seed,
        deterministic=False,
        task2_train_transform=task2_tf,
        horizontal_flip=flip,
    )
    clean_val_ds = ClassificationCacheDataset(
        val, **common,
        backgrounds=None,
        p_bg=0,
        seed=seed + 1,
        deterministic=True,
    )
    bg_val_ds = ClassificationCacheDataset(
        val, **common,
        backgrounds=val_bgs,
        p_bg=1.0,
        scale_range=(0.55, 1.00),
        seed=seed + 2,
        deterministic=True,
    )
    clean_test_ds = ClassificationCacheDataset(
        test, **common,
        backgrounds=None,
        p_bg=0,
        seed=seed + 3,
        deterministic=True,
    )
    bg_test_ds = ClassificationCacheDataset(
        test, **common,
        backgrounds=test_bgs,
        p_bg=1.0,
        scale_range=(0.55, 1.00),
        seed=seed + 4,
        deterministic=True,
    )

    pin = device.type == "cuda"
    loader_kw = dict(batch_size=batch_size, num_workers=0, pin_memory=pin)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    clean_val_loader = DataLoader(clean_val_ds, shuffle=False, **loader_kw)
    bg_val_loader = DataLoader(bg_val_ds, shuffle=False, **loader_kw)
    clean_test_loader = DataLoader(clean_test_ds, shuffle=False, **loader_kw)
    bg_test_loader = DataLoader(bg_test_ds, shuffle=False, **loader_kw)

    # Evaluate the unchanged final checkpoint under the exact same current protocol.
    baseline_model = copy.deepcopy(model).to(device)
    base_clean_test, _, _ = evaluate(
        baseline_model, clean_test_loader, device, criterion
    )
    base_bg_test, _, _ = evaluate(
        baseline_model, bg_test_loader, device, criterion
    )

    print("\nBASELINE CURRENT-PROTOCOL TEST")
    print(" clean:", {k: round(v, 4) for k, v in base_clean_test.items() if k != "loss"})
    print(" photo:", {k: round(v, 4) for k, v in base_bg_test.items() if k != "loss"})

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, epochs), eta_min=lr * 0.05
    )

    best_score = float("-inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    started = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_losses, tr_y, tr_p = [], [], []

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimiser)
            scaler.update()

            tr_losses.append(float(loss.item()))
            tr_y.extend(labels.detach().cpu().tolist())
            tr_p.extend(logits.argmax(1).detach().cpu().tolist())

        scheduler.step()

        train_m = metrics(tr_y, tr_p)
        train_m["loss"] = float(np.mean(tr_losses))
        clean_val, _, _ = evaluate(model, clean_val_loader, device, criterion)
        bg_val, _, _ = evaluate(model, bg_val_loader, device, criterion)

        # Equal-weight validation objective: keep catalogue skill AND gain robustness.
        selection = 0.5 * clean_val[primary] + 0.5 * bg_val[primary]

        row = {
            "epoch": epoch,
            "selection_score": selection,
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"clean_val_{k}": v for k, v in clean_val.items()},
            **{f"bg_val_{k}": v for k, v in bg_val.items()},
            "lr": optimiser.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train {primary}={train_m[primary]:.4f} | "
            f"clean-val={clean_val[primary]:.4f} | "
            f"bg-val={bg_val[primary]:.4f} | "
            f"mean={selection:.4f} | "
            f"{row['seconds']:.1f}s"
            + (" | NEW BEST" if selection > best_score else "")
        )

        if selection > best_score:
            best_score = selection
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    adapted_clean_test, clean_y, clean_p = evaluate(
        model, clean_test_loader, device, criterion
    )
    adapted_bg_test, bg_y, bg_p = evaluate(
        model, bg_test_loader, device, criterion
    )

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    rows = []
    for model_name, domain, result in [
        ("baseline", "clean", base_clean_test),
        ("baseline", "heldout_places365", base_bg_test),
        ("bg_adapted", "clean", adapted_clean_test),
        ("bg_adapted", "heldout_places365", adapted_bg_test),
    ]:
        rows.append({"model": model_name, "domain": domain, **result})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        eval_dir / f"task{task}_bgadapt_comparison.csv",
        index=False,
    )

    pd.DataFrame(
        classification_report(
            clean_y,
            clean_p,
            labels=list(range(num_classes)),
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        )
    ).T.to_csv(out_dir / "classification_report_clean.csv")

    pd.DataFrame(
        confusion_matrix(clean_y, clean_p, labels=list(range(num_classes))),
        index=class_names,
        columns=class_names,
    ).to_csv(out_dir / "confusion_matrix_clean.csv")

    pd.DataFrame(
        classification_report(
            bg_y,
            bg_p,
            labels=list(range(num_classes)),
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        )
    ).T.to_csv(out_dir / "classification_report_heldout_places365.csv")

    pd.DataFrame(
        confusion_matrix(bg_y, bg_p, labels=list(range(num_classes))),
        index=class_names,
        columns=class_names,
    ).to_csv(out_dir / "confusion_matrix_heldout_places365.csv")

    checkpoint = {
        "state_dict": best_state,
        "task": "articleType" if task == 1 else "season",
        "class_names": class_names,
        "num_classes": num_classes,
        "background_adaptation": {
            "source_checkpoint": str(baseline_path),
            "training_mix": "70% Places365 + 30% procedural",
            "p_bg": p_bg,
            "scale_range": [0.55, 1.00],
            "selection": f"0.5*clean_val_{primary} + 0.5*bg_val_{primary}",
            "final_ood_test": "held-out Places365 scene categories",
            "epochs": epochs,
            "best_epoch": best_epoch,
            "seed": seed,
        },
        "baseline_test": {
            "clean": base_clean_test,
            "heldout_places365": base_bg_test,
        },
        "adapted_test": {
            "clean": adapted_clean_test,
            "heldout_places365": adapted_bg_test,
        },
    }

    if task == 1:
        checkpoint["architecture"] = {
            "name": "ItemTypeCNN",
            "widths": [16, 32, 64, 128],
            "head_hidden": 384,
            "pool_grid": [1, 1],
            "pool_mode": "avgmax",
        }
        checkpoint["channel_mean"] = mean
        checkpoint["channel_std"] = std
        checkpoint["image_size_pil"] = [120, 160]
    else:
        checkpoint["model_name"] = "SeasonCNN_bgadapt_60x80"
        checkpoint["image_size_pil"] = [60, 80]

    save_path = out_dir / output_name
    torch.save(checkpoint, save_path)

    delta_clean = adapted_clean_test[primary] - base_clean_test[primary]
    delta_bg = adapted_bg_test[primary] - base_bg_test[primary]

    summary = {
        "task": task,
        "primary_metric": primary,
        "best_epoch": best_epoch,
        "baseline_clean": base_clean_test,
        "baseline_heldout_places365": base_bg_test,
        "adapted_clean": adapted_clean_test,
        "adapted_heldout_places365": adapted_bg_test,
        "delta_clean": delta_clean,
        "delta_heldout_places365": delta_bg,
        "training_minutes": (time.time() - started) / 60.0,
        "checkpoint": str(save_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print(f"TASK {task} BACKGROUND ADAPTATION COMPLETE")
    print("=" * 76)
    print(f"Best epoch: {best_epoch}")
    print(f"Baseline clean {primary}: {base_clean_test[primary]:.4f}")
    print(f"Adapted  clean {primary}: {adapted_clean_test[primary]:.4f}")
    print(f"Delta clean            : {delta_clean:+.4f}")
    print(f"Baseline OOD   {primary}: {base_bg_test[primary]:.4f}")
    print(f"Adapted  OOD   {primary}: {adapted_bg_test[primary]:.4f}")
    print(f"Delta OOD              : {delta_bg:+.4f}")
    print(f"Checkpoint: {save_path}")
    print(f"Comparison: {eval_dir / f'task{task}_bgadapt_comparison.csv'}")
    print("\nDo NOT overwrite the production checkpoint until the clean/OOD trade-off is reviewed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=[1, 2], required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--p-bg", type=float, default=0.70)
    parser.add_argument("--train-bank-size", type=int, default=4000)
    parser.add_argument("--val-bank-size", type=int, default=400)
    parser.add_argument("--test-bank-size", type=int, default=800)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Default: 2e-4 for Task 1, 1e-4 for Task 2.",
    )
    args = parser.parse_args()

    project = resolve_project_root()
    lr = args.lr
    if lr is None:
        lr = 2e-4 if args.task == 1 else 1e-4

    train_adaptation(
        task=args.task,
        project=project,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=lr,
        p_bg=args.p_bg,
        train_bank_size=args.train_bank_size,
        val_bank_size=args.val_bank_size,
        test_bank_size=args.test_bank_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


