"""Task 1 - fashion item type (articleType) classification.

Single source of truth for the Task 1 architecture and its inference
contract. Imported by both

  * ``notebooks/02_task1_item_type.ipynb``    (training and evaluation)
  * ``app/backend/services/task1_service.py`` (serving)

Both used to carry their own private copy of ``ItemTypeCNN``. That is how the
checkpoint written by run ``20260824_210822`` - which uses a spatial avg+max
pooling head - ended up unloadable by the only ``ItemTypeCNN`` left in the
tree, which declares a global-average-pool head. Keep this the only
definition; ``tests/test_models.py`` fails if a checkpoint stops loading.

Checkpoint contract
-------------------
``torch.save`` dict written by the notebook, read by everything else::

    state_dict      model weights
    model_name      e.g. "CNN_tuned"
    num_classes     int
    class_names     list[str], index i is class i (LabelEncoder order)
    channel_mean    list[float], 3 values, computed on the TRAIN split only
    channel_std     list[float], 3 values
    image_size_pil  [width, height] for PIL.Image.resize
    architecture    kwargs for build_from_checkpoint (below)
    config          the notebook CONFIG dict for the run
    test_metrics    dict of held-out metrics
    run_id          timestamp of the run that produced the file
"""

from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

__all__ = [
    "ConvBlock",
    "ItemTypeCNN",
    "build_from_checkpoint",
    "choose_device",
    "load_item_type_model",
    "load_image_array",
    "preprocess_arrays",
    "preprocess_image",
    "predict_proba",
]

# Pooling head layouts. "bn_spatial" is the current one; "legacy_gap" only
# exists so checkpoints written before run 20260824_210822 still load.
HEAD_BN_SPATIAL = "bn_spatial"
HEAD_LEGACY_GAP = "legacy_gap"


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ConvBlock(nn.Module):
    """Conv-BN-ReLU x2 followed by max-pooling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class ItemTypeCNN(nn.Module):
    """Small CNN trained from scratch on 60x80 catalogue photos.

    ``pool_grid`` and ``pool_mode`` control how the final feature map is
    reduced before the classifier head:

    * ``pool_grid=(1, 1)``, ``pool_mode="avg"`` - plain global average pool.
    * ``pool_grid=(2, 1)``, ``pool_mode="avgmax"`` - keep a 2x1 spatial grid
      and concatenate average- and max-pooled features. Catalogue photos are
      vertically structured, so keeping one vertical division is worth about
      +3 validation weighted-F1 over a plain global pool.

    ``head_style`` selects the head module layout, and therefore the
    ``state_dict`` key indices. Do not reorder these ``Sequential`` members -
    saved checkpoints address them positionally (``head.0``, ``head.2``, ...).
    """

    def __init__(
        self,
        num_classes,
        widths=(16, 32, 64, 128),
        dropout=0.4,
        head_hidden=128,
        pool_grid=(1, 1),
        pool_mode="avg",
        head_style=HEAD_BN_SPATIAL,
    ):
        super().__init__()
        if pool_mode not in ("avg", "max", "avgmax"):
            raise ValueError("pool_mode must be avg, max or avgmax, got {!r}".format(pool_mode))
        if head_style not in (HEAD_BN_SPATIAL, HEAD_LEGACY_GAP):
            raise ValueError("unknown head_style {!r}".format(head_style))

        self.pool_grid = tuple(pool_grid)
        self.pool_mode = pool_mode
        self.head_style = head_style

        channels = 3
        blocks = []
        for width in widths:
            blocks.append(ConvBlock(channels, width))
            channels = width
        self.backbone = nn.Sequential(*blocks)

        self.avg_pool = nn.AdaptiveAvgPool2d(self.pool_grid)
        self.max_pool = nn.AdaptiveMaxPool2d(self.pool_grid)

        cells = self.pool_grid[0] * self.pool_grid[1]
        features = channels * cells * (2 if pool_mode == "avgmax" else 1)
        self.pooled_features = features

        if head_style == HEAD_LEGACY_GAP:
            # head.1 / head.4 carry the weights
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(features, head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, num_classes),
            )
        else:
            # head.0 (BatchNorm1d) / head.2 / head.5 carry the weights
            self.head = nn.Sequential(
                nn.BatchNorm1d(features),
                nn.Dropout(dropout),
                nn.Linear(features, head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, num_classes),
            )

    def forward(self, x):
        feature_map = self.backbone(x)
        if self.pool_mode == "avg":
            pooled = self.avg_pool(feature_map).flatten(1)
        elif self.pool_mode == "max":
            pooled = self.max_pool(feature_map).flatten(1)
        else:
            # Average first, then max. The concatenation order is baked into
            # the head BatchNorm1d running statistics, so it must not change.
            pooled = torch.cat(
                [self.avg_pool(feature_map).flatten(1), self.max_pool(feature_map).flatten(1)],
                dim=1,
            )
        return self.head(pooled)


def build_from_checkpoint(checkpoint):
    """Rebuild the exact architecture a checkpoint was trained with.

    Checkpoints written before run ``20260824_210822`` have no ``pool_grid``
    key; those are global-average-pool models with the older head layout.
    """
    architecture = checkpoint["architecture"]
    pool_grid = architecture.get("pool_grid")
    legacy = pool_grid is None

    return ItemTypeCNN(
        int(checkpoint["num_classes"]),
        widths=tuple(architecture["widths"]),
        dropout=float(architecture["dropout"]),
        head_hidden=int(architecture["head_hidden"]),
        pool_grid=(1, 1) if legacy else tuple(pool_grid),
        pool_mode="avg" if legacy else architecture.get("pool_mode", "avg"),
        head_style=HEAD_LEGACY_GAP if legacy else HEAD_BN_SPATIAL,
    )


def load_item_type_model(path, device=None):
    """Return ``(model, checkpoint)`` ready for inference."""
    device = device or choose_device()
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only argument
        checkpoint = torch.load(path, map_location=device)

    model = build_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def load_image_array(source, size):
    """Read one image as a uint8 HWC array. The single decode path.

    ``source`` may be a path, raw bytes, or a PIL image. ``size`` is
    ``(width, height)``, matching ``checkpoint["image_size_pil"]``.
    """
    try:
        if isinstance(source, Image.Image):
            image = source
        elif isinstance(source, (bytes, bytearray)):
            image = Image.open(BytesIO(bytes(source)))
        else:
            image = Image.open(Path(source))
        with image:
            image = image.convert("RGB").resize(tuple(size), Image.BILINEAR)
            return np.asarray(image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Cannot decode image: {}".format(error))


def preprocess_arrays(arrays, checkpoint):
    """uint8 NHWC -> normalised float32 NCHW tensor."""
    mean = np.asarray(checkpoint["channel_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["channel_std"], dtype=np.float32)
    batch = np.asarray(arrays, dtype=np.float32) / 255.0
    batch = (batch - mean) / std
    return torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2))).float()


def preprocess_image(source, checkpoint, device=None):
    """One image -> 1xCxHxW tensor on ``device``."""
    array = load_image_array(source, checkpoint["image_size_pil"])
    tensor = preprocess_arrays(array[None, ...], checkpoint)
    return tensor.to(device) if device is not None else tensor


@torch.no_grad()
def predict_proba(model, checkpoint, sources, batch_size=256, device=None, tta=False):
    """Softmax probabilities for an iterable of images.

    ``tta=True`` averages the probabilities of each image and its horizontal
    mirror. Catalogue photos are near-symmetric and training already uses
    random horizontal flips, so this is a free consistency gain at inference.
    """
    device = device or next(model.parameters()).device
    size = checkpoint["image_size_pil"]
    sources = list(sources)
    was_training = model.training
    model.eval()

    chunks = []
    for start in range(0, len(sources), batch_size):
        arrays = np.stack(
            [load_image_array(s, size) for s in sources[start:start + batch_size]]
        )
        tensor = preprocess_arrays(arrays, checkpoint).to(device)
        probabilities = F.softmax(model(tensor).float(), dim=1)
        if tta:
            flipped = F.softmax(model(torch.flip(tensor, dims=[3])).float(), dim=1)
            probabilities = (probabilities + flipped) / 2
        chunks.append(probabilities.cpu().numpy())

    if was_training:
        model.train()
    if not chunks:
        return np.zeros((0, int(checkpoint["num_classes"])), dtype=np.float32)
    return np.concatenate(chunks)
