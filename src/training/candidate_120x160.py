"""Real-model 120x160 candidate wiring and smoke validation.

This module prepares the four production candidate pipelines without running
full training. Use ``python -m src.training.candidate_120x160 --smoke`` for the
one-batch checks. Training entry points remain the historical notebooks/scripts;
this candidate runner is the shared 120x160 data/model contract they can call.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from app.backend.services.task2_service import SeasonCNN
from app.backend.services.task3_service import EarlyBranchCNN
from src.data.config import (
    CANDIDATE_ARTIFACT_DIRS,
    DATA_DIR,
    EXPECTED_TENSOR_SHAPE,
    IMAGE_SIZE_PIL,
    IMAGE_SIZE_TORCH,
)
from src.data.splits import load_or_create_splits, load_supervised_metadata
from src.models.item_type_classifier import ItemTypeCNN
from src.visual_search.search_engine import ImprovedEncoder

BATCH_SIZE = 64
REAL_TASK4_EMBEDDING_DIM = 128


class CandidateDataset(Dataset):
    def __init__(self, frame, target=None, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(DATA_DIR / f"{int(row['id'])}.jpg") as image:
            array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).float() / 255.0
        if self.transform:
            tensor = self.transform(tensor)
        if self.target is None:
            return tensor
        return tensor, row[self.target]


def compute_train_normalization(train):
    total = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    for position, image_id in enumerate(train["id"], start=1):
        with Image.open(DATA_DIR / f"{int(image_id)}.jpg") as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        flattened = pixels.reshape(-1, 3).astype(np.float64)
        channel_sum += flattened.sum(axis=0)
        channel_square_sum += np.square(flattened).sum(axis=0)
        total += len(flattened)
        if position % 5000 == 0:
            print(f"normalization: {position}/{len(train)} train images", flush=True)
    mean = channel_sum / total
    std = np.sqrt(np.maximum(channel_square_sum / total - np.square(mean), 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


def task1_frame_parts(train, val, test):
    counts = load_supervised_metadata().articleType.value_counts()
    retained = set(counts[counts >= 10].index)
    parts = [frame[frame.articleType.isin(retained)].copy() for frame in (train, val, test)]
    classes = sorted(retained)
    encoding = {name: index for index, name in enumerate(classes)}
    for frame in parts:
        frame["label"] = frame.articleType.map(encoding)
    return parts, classes


def task3_frame_parts(train, val, test):
    usage_merge = {"Smart Casual": "Casual", "Travel": "Casual", "Party": "Formal"}
    parts = []
    for frame in (train, val, test):
        frame = frame[frame.usage != "Home"].copy()
        frame["usage"] = frame["usage"].replace(usage_merge)
        parts.append(frame)
    return parts


def two_view_tensor(image):
    """Create two independently augmented views without changing image geometry."""
    left = image.clone()
    right = image.clone()
    if torch.rand(()) < 0.5:
        left = torch.flip(left, dims=[2])
    if torch.rand(()) < 0.5:
        right = torch.flip(right, dims=[2])
    return left, right


def instance_contrastive_loss(
    left, right, article_types=None, base_colours=None, temperature=0.2,
    hard_negative_weight=1.5,
):
    """Product-aware two-view InfoNCE with optional hard-negative weighting.

    The positive for each view is the other augmented view of the same item.
    Other products remain negatives even when their article type matches. Those
    visually close negatives receive a larger logit penalty when metadata is
    supplied; metadata never changes the positive definition.
    """
    left = F.normalize(left, dim=1)
    right = F.normalize(right, dim=1)
    embeddings = torch.cat([left, right], dim=0)
    logits = embeddings @ embeddings.T / temperature
    size = left.shape[0]
    positive = torch.cat([torch.arange(size, 2 * size), torch.arange(0, size)])
    diagonal = torch.eye(2 * size, dtype=torch.bool, device=embeddings.device)
    if article_types is not None or base_colours is not None:
        hard = torch.zeros((size, size), dtype=torch.bool, device=embeddings.device)
        if article_types is not None:
            values = torch.as_tensor(article_types, device=embeddings.device)
            hard |= values[:, None] == values[None, :]
        if base_colours is not None:
            values = torch.as_tensor(base_colours, device=embeddings.device)
            hard |= values[:, None] == values[None, :]
        hard.fill_diagonal_(False)
        hard = torch.cat([torch.cat([hard, hard], dim=1),
                          torch.cat([hard, hard], dim=1)], dim=0)
        logits = logits - hard.float() * float(hard_negative_weight)
    logits = logits.masked_fill(diagonal, float("-inf"))
    return F.cross_entropy(logits, positive.to(logits.device))


def candidate_checkpoint(model, mean, std, extra=None):
    payload = {
        "state_dict": model.state_dict(),
        "image_size_pil": list(IMAGE_SIZE_PIL),
        "image_size_torch": list(IMAGE_SIZE_TORCH),
        "expected_tensor_shape": list(EXPECTED_TENSOR_SHAPE),
        "channel_mean": mean.tolist(),
        "channel_std": std.tolist(),
    }
    if extra:
        payload.update(extra)
    return payload


def smoke_task1(batch, mean, std, device):
    model = ItemTypeCNN(92, widths=(16, 32, 64, 128), head_hidden=384,
                        pool_grid=(1, 1), pool_mode="avgmax").to(device)
    logits = model(batch.to(device))
    loss = F.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=device))
    optimiser = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimiser.zero_grad(); loss.backward(); optimiser.step()
    checkpoint = candidate_checkpoint(model, mean, std, {"num_classes": 92})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "task1.pt"
        torch.save(checkpoint, path)
        restored = ItemTypeCNN(92, widths=(16, 32, 64, 128), head_hidden=384,
                               pool_grid=(1, 1), pool_mode="avgmax").to(device)
        restored.load_state_dict(torch.load(path, map_location=device,
                                            weights_only=False)["state_dict"])
        with torch.no_grad():
            assert restored(batch.to(device)).shape == (len(batch), 92)
    return "PASS"


def smoke_task2(batch, mean, std, device):
    model = SeasonCNN(4).to(device)
    logits = model(batch.to(device))
    assert logits.shape == (len(batch), 4)
    loss = F.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=device))
    loss.backward()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "task2.pt"
        torch.save(model.state_dict(), path)
        restored = SeasonCNN(4).to(device)
        restored.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        assert restored(batch.to(device)).shape == (len(batch), 4)
    return "PASS"


def smoke_task3(batch, mean, std, device):
    model = EarlyBranchCNN({"gender": 5, "usage": 4}, input_shape=EXPECTED_TENSOR_SHAPE,
                           shared_widths=(32, 64), branch_widths=(128, 256),
                           hidden=256, dropout=0.0).to(device)
    outputs = model(batch.to(device))
    assert outputs["gender"].shape == (len(batch), 5)
    assert outputs["usage"].shape == (len(batch), 4)
    loss = F.cross_entropy(outputs["gender"], torch.zeros(len(batch), dtype=torch.long, device=device))
    loss += F.cross_entropy(outputs["usage"], torch.zeros(len(batch), dtype=torch.long, device=device))
    loss.backward()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "task3.pt"
        torch.save(model.state_dict(), path)
        restored = EarlyBranchCNN({"gender": 5, "usage": 4}, input_shape=EXPECTED_TENSOR_SHAPE,
                                   shared_widths=(32, 64), branch_widths=(128, 256),
                                   hidden=256, dropout=0.0).to(device)
        restored.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        assert set(restored(batch.to(device))) == {"gender", "usage"}
    return "PASS"


def smoke_task4(batch, mean, std, device):
    model = ImprovedEncoder(embedding_dim=REAL_TASK4_EMBEDDING_DIM,
                            widths=(32, 64, 128, 256)).to(device)
    left, right = zip(*(two_view_tensor(image) for image in batch))
    left, right = torch.stack(left).to(device), torch.stack(right).to(device)
    left_embedding, right_embedding = model.embed(left), model.embed(right)
    loss = instance_contrastive_loss(
        left_embedding, right_embedding,
        article_types=torch.arange(len(batch)) % 2,
        base_colours=torch.arange(len(batch)) % 3,
    )
    assert torch.isfinite(loss)
    optimiser = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimiser.zero_grad(); loss.backward(); optimiser.step()
    assert left_embedding.shape == (len(batch), REAL_TASK4_EMBEDDING_DIM)
    assert torch.allclose(left_embedding.norm(dim=1), torch.ones(len(batch), device=device), atol=1e-5)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "task4.pt"
        torch.save(candidate_checkpoint(model, mean, std, {
            "embedding_dim": REAL_TASK4_EMBEDDING_DIM,
        }), path)
        restored = ImprovedEncoder(embedding_dim=REAL_TASK4_EMBEDDING_DIM,
                                   widths=(32, 64, 128, 256)).to(device)
        restored.load_state_dict(torch.load(path, map_location=device,
                                            weights_only=False)["state_dict"])
        assert restored.embed(left).shape == (len(batch), REAL_TASK4_EMBEDDING_DIM)
    return "PASS"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run one-batch checks only")
    args = parser.parse_args()
    if not args.smoke:
        raise SystemExit("This candidate runner only supports --smoke; full training is intentionally disabled.")

    train, val, test = load_or_create_splits()
    parts, classes = task1_frame_parts(train, val, test)
    task3_parts = task3_frame_parts(train, val, test)
    normalization_path = CANDIDATE_ARTIFACT_DIRS["task1"] / "normalization_120x160.json"
    if normalization_path.exists():
        saved = json.loads(normalization_path.read_text())
        mean = np.asarray(saved["mean"], dtype=np.float32)
        std = np.asarray(saved["std"], dtype=np.float32)
        print("normalization: loaded cached train-only statistics", flush=True)
    else:
        print("normalization: computing from train split only", flush=True)
        mean, std = compute_train_normalization(train)
    normalization = {"image_size_pil": list(IMAGE_SIZE_PIL), "mean": mean.tolist(), "std": std.tolist()}
    for directory in CANDIDATE_ARTIFACT_DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "normalization_120x160.json").write_text(json.dumps(normalization, indent=2))

    loader = DataLoader(CandidateDataset(train.head(4)), batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {
        "task1": smoke_task1(batch, mean, std, device),
        "task2": smoke_task2(batch, mean, std, device),
        "task3": smoke_task3(batch, mean, std, device),
        "task4": smoke_task4(batch, mean, std, device),
    }
    print("real entry data: train/val/test", len(train), len(val), len(test))
    print("task1 retained classes:", len(classes), "split rows:", [len(frame) for frame in parts])
    for target in ("season", "gender"):
        print(f"{target} distributions:", {
            name: frame[target].value_counts().to_dict()
            for name, frame in (("train", train), ("val", val), ("test", test))
        })
    print("usage distributions after Task 3 policy:", {
        name: frame["usage"].value_counts().to_dict()
        for name, frame in (("train", task3_parts[0]), ("val", task3_parts[1]),
                            ("test", task3_parts[2]))
    })
    print("train-only mean:", mean.round(6).tolist())
    print("train-only std:", std.round(6).tolist())
    print("task4 production embedding_dim:", REAL_TASK4_EMBEDDING_DIM)
    print("batch shape:", list(batch.shape))
    print("smoke:", results)


if __name__ == "__main__":
    main()
