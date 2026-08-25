from io import BytesIO
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SeasonCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 128), nn.ReLU(),
            nn.Dropout(0.40), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class Task2Service:
    def __init__(self):
        self.device = choose_device()
        self.project_root = Path(__file__).resolve().parents[3]
        self.artifact_dir = self.project_root / "artifacts" / "task2"
        self.model_path = self.artifact_dir / "task2_season_best_pytorch.pth"
        self.mapping_path = self.artifact_dir / "task2_season_class_mapping.json"
        if not self.model_path.exists():
            raise FileNotFoundError("Task 2 model not found: {}".format(self.model_path))
        if not self.mapping_path.exists():
            raise FileNotFoundError("Task 2 mapping not found: {}".format(self.mapping_path))
        with open(self.mapping_path, "r", encoding="utf-8") as file:
            self.class_to_index = json.load(file)
        self.index_to_class = {int(index): name for name, index in self.class_to_index.items()}
        self.num_classes = len(self.class_to_index)
        self.model = SeasonCNN(self.num_classes)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image_bytes):
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image = image.resize((60, 80), Image.BILINEAR)
        except (UnidentifiedImageError, OSError):
            raise ValueError("Cannot decode uploaded image")
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image_bytes):
        probabilities = F.softmax(self.model(self.preprocess(image_bytes)), dim=1)[0]
        top_probs, top_indices = torch.topk(probabilities, k=min(3, self.num_classes))
        results = [
            {"label": self.index_to_class[index], "confidence": round(float(probability), 4)}
            for probability, index in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist())
        ]
        return {"label": results[0]["label"], "confidence": results[0]["confidence"], "top3": results}
