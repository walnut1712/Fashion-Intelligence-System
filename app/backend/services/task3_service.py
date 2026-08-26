"""Task 3 - gender and occasion prediction.

Serves the model trained in ``notebooks/04b_task3_cnn_architectures.ipynb``:
an EarlyBranch multi-task CNN with a shared low-level trunk and a separate
convolutional pathway per attribute.

Artefact: ``artifacts/task3_cnn/task3_cnn_model.pt``
"""

from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------- model ----
def vgg_block(in_channels, width):
    """Two 3x3 convolutions then max-pool - the VGG pattern used in training."""
    return [
        nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
        nn.BatchNorm2d(width), nn.ReLU(inplace=True),
        nn.Conv2d(width, width, 3, padding=1, bias=False),
        nn.BatchNorm2d(width), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
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


class EarlyBranchNet(nn.Module):
    """Shared low-level blocks, then a convolutional pathway per task.

    Must match the notebook definition exactly or the state_dict will not load.
    """

    def __init__(self, num_classes, input_shape=(3, 80, 60), shared_widths=(32, 64),
                 branch_widths=(128, 256), hidden=256, dropout=0.5):
        super().__init__()

        layers, channels = [], input_shape[0]
        for width in shared_widths:
            layers += vgg_block(channels, width)
            channels = width
        self.shared = nn.Sequential(*layers)

        self.branches = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        for target, n_classes in num_classes.items():
            branch_layers, branch_channels = [], channels
            for width in branch_widths:
                branch_layers += vgg_block(branch_channels, width)
                branch_channels = width
            branch = nn.Sequential(*branch_layers)
            self.branches[target] = branch

            probe = nn.Sequential(self.shared, branch).eval()
            with torch.no_grad():
                flat = int(np.prod(probe(torch.zeros(1, *input_shape)).shape[1:]))
            self.heads[target] = make_classifier(flat, n_classes, hidden, dropout)

    def forward(self, x):
        shared = self.shared(x)
        return {target: self.heads[target](self.branches[target](shared))
                for target in self.branches}


# ------------------------------------------------------------- service ----
class Task3Service:
    TARGETS = ("gender", "usage")

    def __init__(self):
        self.device = choose_device()
        self.project_root = Path(__file__).resolve().parents[3]
        self.model_path = self.project_root / "artifacts" / "task3_cnn" / "task3_cnn_model.pt"
        if not self.model_path.exists():
            raise FileNotFoundError(
                "Task 3 model not found: {}. Run notebooks/04b_task3_cnn_architectures.ipynb"
                .format(self.model_path)
            )

        checkpoint = self._load_checkpoint()
        self.model_name = checkpoint.get("model_name", "EarlyBranch")

        state_dict = checkpoint["state_dict"]
        if not any(key.startswith("branches.") for key in state_dict):
            raise ValueError(
                "Checkpoint '{}' is not an EarlyBranch model. Set FINAL_NAME = "
                "'EarlyBranch' in CELL 21 of the notebook and re-run cells 21-24."
                .format(self.model_name)
            )

        self.class_names = {t: list(v) for t, v in checkpoint["class_names"].items()}
        self.num_classes = {t: len(v) for t, v in self.class_names.items()}
        self.image_size = tuple(checkpoint["image_size_pil"])          # (W, H)

        # Architecture is recorded by newer checkpoints; older ones used these defaults.
        architecture = checkpoint.get("architecture", {})
        self.model = EarlyBranchNet(
            self.num_classes,
            input_shape=(3, self.image_size[1], self.image_size[0]),
            shared_widths=tuple(architecture.get("shared_widths", (32, 64))),
            branch_widths=tuple(architecture.get("branch_widths", (128, 256))),
            hidden=int(architecture.get("hidden", 256)),
            dropout=float(architecture.get("dropout", 0.5)),
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        self.mean = np.array(checkpoint["channel_mean"], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(checkpoint["channel_std"], dtype=np.float32).reshape(1, 1, 3)
        self.test_metrics = checkpoint.get("test_metrics", {})

    def _load_checkpoint(self):
        try:
            return torch.load(self.model_path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(self.model_path, map_location=self.device)

    def preprocess(self, image_bytes):
        """Match the training pipeline: RGB on white, resized to 60x80, normalised."""
        try:
            image = Image.open(BytesIO(image_bytes))
            image = ImageOps.exif_transpose(image)
            if image.mode == "P":
                image = image.convert("RGBA" if "transparency" in image.info else "RGB")
            if image.mode in ("RGBA", "LA"):
                canvas = Image.new("RGB", image.size, (255, 255, 255))
                canvas.paste(image, mask=image.split()[-1])
                image = canvas
            else:
                image = image.convert("RGB")
            image = image.resize(self.image_size, Image.BILINEAR)
        except (UnidentifiedImageError, OSError):
            raise ValueError("Cannot decode uploaded image")

        array = np.asarray(image, dtype=np.float32) / 255.0
        array = ((array - self.mean) / self.std).transpose(2, 0, 1)
        return torch.from_numpy(array).float().unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image_bytes, top_k=4):
        """Return, per attribute, the ranked labels with softmax probabilities."""
        logits = self.model(self.preprocess(image_bytes))
        output = {}
        for target in self.TARGETS:
            names = self.class_names[target]
            probabilities = F.softmax(logits[target][0].float(), dim=0)
            k = min(top_k, len(names))
            top_probs, top_indices = torch.topk(probabilities, k=k)
            ranked = [
                {"label": names[index], "p": round(float(probability), 4)}
                for probability, index in zip(top_probs.cpu().tolist(),
                                              top_indices.cpu().tolist())
            ]
            output[target] = ranked
        return output
