from io import BytesIO
from pathlib import Path

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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class MultiTaskCNN(nn.Module):
    def __init__(self, num_classes, widths=(32, 64, 128, 256), dropout=0.4, head_hidden=128):
        super().__init__()
        channels = 3
        blocks = []
        for width in widths:
            blocks.append(ConvBlock(channels, width))
            channels = width
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

        def make_head(n_out):
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(channels, head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, n_out),
            )

        self.heads = nn.ModuleDict({
            target: make_head(n_classes)
            for target, n_classes in num_classes.items()
        })

    def forward(self, x):
        features = self.pool(self.backbone(x)).flatten(1)
        return {target: head(features) for target, head in self.heads.items()}


class Task3Service:
    def __init__(self):
        self.device = choose_device()
        self.project_root = Path(__file__).resolve().parents[3]
        self.model_path = self.project_root / "artifacts" / "task3" / "task3_multitask_cnn.pt"
        if not self.model_path.exists():
            raise FileNotFoundError("Task 3 model not found: {}".format(self.model_path))

        self.checkpoint = self._load_checkpoint()
        architecture = self.checkpoint["architecture"]
        self.class_names = self.checkpoint["class_names"]
        self.model = MultiTaskCNN(
            num_classes=self.checkpoint["num_classes"],
            widths=tuple(architecture["widths"]),
            dropout=float(architecture["dropout"]),
            head_hidden=int(architecture["head_hidden"]),
        )
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.num_classes = {
            target: len(classes) for target, classes in self.class_names.items()
        }
        self.mean = np.array(self.checkpoint["channel_mean"], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(self.checkpoint["channel_std"], dtype=np.float32).reshape(1, 1, 3)
        self.image_size = tuple(self.checkpoint["image_size_pil"])

    def _load_checkpoint(self):
        try:
            return torch.load(self.model_path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(self.model_path, map_location=self.device)

    def preprocess(self, image_bytes):
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image = image.resize(self.image_size, Image.BILINEAR)
        except (UnidentifiedImageError, OSError):
            raise ValueError("Cannot decode uploaded image")
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = ((array - self.mean) / self.std).transpose(2, 0, 1)
        return torch.from_numpy(array).float().unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image_bytes):
        logits = self.model(self.preprocess(image_bytes))
        output = {}
        for target, values in logits.items():
            probabilities = F.softmax(values.float(), dim=1)[0]
            top_probs, top_indices = torch.topk(
                probabilities, k=min(3, len(self.class_names[target]))
            )
            results = [
                {
                    "label": self.class_names[target][index],
                    "confidence": round(float(probability), 4),
                }
                for probability, index in zip(
                    top_probs.cpu().tolist(), top_indices.cpu().tolist()
                )
            ]
            output[target] = {
                "label": results[0]["label"],
                "confidence": results[0]["confidence"],
                "top3": results,
            }
        return output
