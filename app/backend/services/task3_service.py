"""Task 3 - gender and occasion prediction.

Serves the model trained in ``notebooks/04_task3_cnn_architectures.ipynb``: a
VGG-style convolutional network with early task branching, where the first two
blocks are shared and each attribute then has its own convolutional pathway and
classifier.

Artefact: ``artifacts/task3/task3_cnn_model.pt``

The checkpoint records its own architecture, so the network is rebuilt from what
was saved rather than from assumptions held here. It also carries a per-class
probability multiplier for each attribute, applied before the argmax so the served
output matches the notebook's evaluation. Threshold adjustment was evaluated in the
notebook and not adopted, so those multipliers are currently all ones and prediction
is a plain argmax; the multiply is kept so that a checkpoint which does use them
serves correctly without a code change.

The checkpoint also carries a temperature per attribute, fitted on the validation
split, which divides the logits before the softmax. Trained to convergence with
cross-entropy the network saturates its softmax and reports near-certainty on
almost everything, including what it gets wrong, and this response is what the
caller displays as a confidence. Dividing every logit by the same positive number
cannot reorder them, so the label served is unchanged and only the probability
beside it moves. A checkpoint without the field is served at temperature 1, exactly
as before.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import UnidentifiedImageError

from src.data.user_image import load_user_image


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------- network ----
ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
POOLINGS = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d}


def conv_block(in_channels, width, activation="relu", pooling="max"):
    """Two 3x3 convolutions, each with batch normalisation and a non-linearity,
    then a 2x2 downsample. Must match the training definition exactly."""
    act = ACTIVATIONS[activation]
    pool = POOLINGS[pooling]
    return [
        nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
        nn.BatchNorm2d(width), act(),
        nn.Conv2d(width, width, 3, padding=1, bias=False),
        nn.BatchNorm2d(width), act(),
        pool(2),
    ]


def make_classifier(in_features, n_classes, hidden=256, dropout=0.5):
    return nn.Sequential(
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, n_classes),
    )


class EarlyBranchCNN(nn.Module):
    """Shared low-level blocks, then one convolutional pathway per attribute."""

    def __init__(self, num_classes, input_shape=(3, 80, 60), shared_widths=(32, 64),
                 branch_widths=(128, 256), hidden=256, dropout=0.5,
                 activation="relu", pooling="max", head="flatten"):
        super().__init__()
        layers, channels = [], input_shape[0]
        for width in shared_widths:
            layers += conv_block(channels, width, activation, pooling)
            channels = width
        self.shared = nn.Sequential(*layers)

        self.branches = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        for target, n_classes in num_classes.items():
            branch_layers, branch_channels = [], channels
            for width in branch_widths:
                branch_layers += conv_block(branch_channels, width, activation, pooling)
                branch_channels = width
            branch = nn.Sequential(*branch_layers)
            self.branches[target] = branch

            if head == "flatten":
                probe = nn.Sequential(self.shared, branch).eval()
                with torch.no_grad():
                    flat = int(np.prod(probe(torch.zeros(1, *input_shape)).shape[1:]))
                self.heads[target] = make_classifier(flat, n_classes, hidden, dropout)
            else:
                self.heads[target] = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                    nn.Linear(branch_channels, hidden), nn.ReLU(inplace=True),
                    nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        shared = self.shared(x)
        return {target: self.heads[target](self.branches[target](shared))
                for target in self.branches}


# ------------------------------------------------------------- service ----
class Task3Service:
    TARGETS = ("gender", "usage")

    def __init__(self, model_path=None):
        self.device = choose_device()
        self.project_root = Path(__file__).resolve().parents[3]
        self.model_path = Path(model_path) if model_path else (
            self.project_root / "artifacts" / "task3" / "task3_cnn_model.pt")
        if not self.model_path.exists():
            raise FileNotFoundError(
                "Task 3 model not found: {}. Run notebooks/04_task3_cnn_architectures.ipynb"
                .format(self.model_path))

        checkpoint = self._load_checkpoint()
        self.model_name = checkpoint.get("model_name", "early-branch CNN")

        state_dict = checkpoint["state_dict"]
        if not any(key.startswith("branches.") for key in state_dict):
            raise ValueError(
                "Checkpoint '{}' is not an early-branching model.".format(self.model_name))

        self.class_names = {t: list(v) for t, v in checkpoint["class_names"].items()}
        self.num_classes = {t: len(v) for t, v in self.class_names.items()}
        self.image_size = tuple(checkpoint["image_size_pil"])          # (width, height)

        architecture = checkpoint.get("architecture", {})
        self.architecture = {
            "shared_widths": tuple(architecture.get("shared_widths", (32, 64))),
            "branch_widths": tuple(architecture.get("branch_widths", (128, 256))),
            "hidden": int(architecture.get("hidden", 256)),
            "dropout": float(architecture.get("dropout", 0.5)),
            "activation": architecture.get("activation", "relu"),
            "pooling": architecture.get("pooling", "max"),
            "head": architecture.get("head", "flatten"),
        }

        self.model = EarlyBranchCNN(
            self.num_classes,
            input_shape=(3, self.image_size[1], self.image_size[0]),
            **self.architecture)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        # Probability multipliers from threshold adjustment; ones when not applied.
        saved_weights = checkpoint.get("class_weights", {})
        self.class_weights = {
            target: np.asarray(saved_weights.get(target,
                                                 np.ones(self.num_classes[target])),
                               dtype=np.float32)
            for target in self.TARGETS
        }
        self.uses_thresholds = any(not np.allclose(w, 1.0)
                                   for w in self.class_weights.values())

        # Calibration temperature, one per attribute. Absent, or non-positive from a
        # malformed checkpoint, means serve the probabilities as the network produced them.
        saved_temperature = checkpoint.get("temperature", {}) or {}
        self.temperature = {
            target: (float(saved_temperature[target])
                     if target in saved_temperature and float(saved_temperature[target]) > 0
                     else 1.0)
            for target in self.TARGETS
        }
        self.is_calibrated = any(t != 1.0 for t in self.temperature.values())

        self.mean = np.array(checkpoint["channel_mean"], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(checkpoint["channel_std"], dtype=np.float32).reshape(1, 1, 3)
        self.test_metrics = checkpoint.get("test_metrics", {})

        print("Task 3 loaded:", self.model_name,
              "| threshold adjustment:", "on" if self.uses_thresholds else "off",
              "| temperature:", ", ".join(f"{t} {self.temperature[t]:.2f}"
                                          for t in self.TARGETS)
              if self.is_calibrated else "off")

    def _load_checkpoint(self):
        try:
            return torch.load(self.model_path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(self.model_path, map_location=self.device)

    def preprocess(self, image_bytes):
        """Match the training pipeline: RGB on white, resized, normalised."""
        try:
            image = load_user_image(image_bytes, size=self.image_size, mode="letterbox")
        except (UnidentifiedImageError, OSError):
            raise ValueError("Cannot decode uploaded image")

        array = np.asarray(image, dtype=np.float32) / 255.0
        array = ((array - self.mean) / self.std).transpose(2, 0, 1)
        return torch.from_numpy(array).float().unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image_bytes, top_k=4):
        """Ranked labels with probabilities, for each attribute."""
        logits = self.model(self.preprocess(image_bytes))
        output = {}
        for target in self.TARGETS:
            names = self.class_names[target]
            probabilities = F.softmax(logits[target][0].float() / self.temperature[target],
                                      dim=0).cpu().numpy()

            # Ranking uses the adjusted scores; the reported figure stays the model's
            # own calibrated probability, which is what a confidence display needs.
            order = np.argsort(-(probabilities * self.class_weights[target]))
            output[target] = [
                {"label": names[index], "p": round(float(probabilities[index]), 4)}
                for index in order[:min(top_k, len(names))]
            ]
        return output
