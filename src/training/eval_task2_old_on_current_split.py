from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from app.backend.services.task2_service import SeasonCNN
from src.data.splits import load_or_create_splits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OLD_IMAGE_DIR = PROJECT_ROOT / "A2_FashionDataset" / "FashionDataset" / "train" / "images_train"
OLD_MODEL_PATH = PROJECT_ROOT / "artifacts" / "task2" / "task2_season_best_pytorch.pth"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "task2_120x160"

CLASS_NAMES = ["Fall", "Spring", "Summer", "Winter"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

EVAL_TF = transforms.Compose([
    transforms.Resize((80, 60), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])


class OldTask2Dataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = OLD_IMAGE_DIR / f"{int(row['id'])}.jpg"
        if not path.is_file():
            raise FileNotFoundError(f"Missing original 60x80 source image: {path}")
        with Image.open(path) as image:
            tensor = EVAL_TF(image.convert("RGB"))
        return tensor, int(row["label"])


def metrics(y, p):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
    }


def main():
    if not OLD_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Old Task-2 model not found: {OLD_MODEL_PATH}")
    if not OLD_IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Original training images not found: {OLD_IMAGE_DIR}")

    _, _, test = load_or_create_splits()
    test = test[test["season"].notna()].copy()
    test["label"] = test["season"].map(CLASS_TO_INDEX).astype("int64")
    assert len(test) == 5508, len(test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        OldTask2Dataset(test),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    payload = torch.load(OLD_MODEL_PATH, map_location=device, weights_only=False)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload

    model = SeasonCNN(4).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            pred = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(labels.tolist())

    result = metrics(y_true, y_pred)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "old60x80_on_current_split_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
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
    ).T.to_csv(OUTPUT_DIR / "old60x80_on_current_split_classification_report.csv")

    pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=list(range(4))),
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(OUTPUT_DIR / "old60x80_on_current_split_confusion_matrix.csv")

    print("=" * 68)
    print("OLD TASK 2 60x80 — CURRENT LEAKAGE-SAFE TEST SPLIT")
    print("=" * 68)
    print(f"Test rows: {len(test)}")
    print(f"Accuracy:          {result['accuracy']:.4f}")
    print(f"Macro-F1:          {result['macro_f1']:.4f}")
    print(f"Weighted-F1:       {result['weighted_f1']:.4f}")
    print(f"Balanced Accuracy: {result['balanced_accuracy']:.4f}")
    print()
    print("Current 120x160 result:")
    print("Accuracy:          0.6233")
    print("Macro-F1:          0.5937")
    print("Weighted-F1:       0.6182")
    print("Balanced Accuracy: 0.6182")


if __name__ == "__main__":
    main()
