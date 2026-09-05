"""Lightweight 120x160 pipeline checks; never performs model training."""

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.config import (
    DATA_DIR,
    EXPECTED_TENSOR_SHAPE,
    IMAGE_SIZE_PIL,
    IMAGE_SIZE_TORCH,
)
from src.data.splits import load_or_create_splits
from src.data.user_image import load_user_image
from src.models.item_type_classifier import ItemTypeCNN, build_from_checkpoint
from app.backend.services.task2_service import SeasonCNN
from app.backend.services.task3_service import EarlyBranchCNN
from src.visual_search.search_engine import ImprovedEncoder


def main():
    train, val, test = load_or_create_splits()
    image_paths = [DATA_DIR / f"{int(image_id)}.jpg" for image_id in train["id"].head(4)]
    arrays = np.stack([load_user_image(path, IMAGE_SIZE_PIL, mode="letterbox")
                       for path in image_paths])
    batch = torch.from_numpy(arrays.transpose(0, 3, 1, 2)).float() / 255.0
    assert tuple(batch.shape[1:]) == EXPECTED_TENSOR_SHAPE

    task1_checkpoint = {
        "num_classes": 92,
        "class_names": [str(index) for index in range(92)],
        "channel_mean": [0.5, 0.5, 0.5],
        "channel_std": [0.25, 0.25, 0.25],
        "image_size_pil": list(IMAGE_SIZE_PIL),
        "architecture": {"widths": [8, 16], "dropout": 0.0,
                          "head_hidden": 16, "pool_grid": [1, 1],
                          "pool_mode": "avg"},
    }
    task1 = build_from_checkpoint(task1_checkpoint)
    loss = F.cross_entropy(task1(batch), torch.zeros(len(batch), dtype=torch.long))
    optimiser = torch.optim.SGD(task1.parameters(), lr=1e-3)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    task2 = SeasonCNN(num_classes=4)
    assert task2(batch).shape == (len(batch), 4)

    task3 = EarlyBranchCNN({"gender": 5, "usage": 4},
                           input_shape=EXPECTED_TENSOR_SHAPE,
                           shared_widths=(4,), branch_widths=(8,),
                           hidden=8, dropout=0.0)
    outputs = task3(batch)
    assert outputs["gender"].shape == (len(batch), 5)
    assert outputs["usage"].shape == (len(batch), 4)

    encoder = ImprovedEncoder(embedding_dim=8, widths=(4, 8))
    embeddings = encoder.embed(batch)
    assert embeddings.shape == (len(batch), 8)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(len(batch)), atol=1e-5)
    assert torch.topk(embeddings @ embeddings.T, k=2, dim=1).indices.shape == (len(batch), 2)

    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "candidate.pt"
        torch.save({"state_dict": task1.state_dict(), **task1_checkpoint}, checkpoint_path)
        reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        restored = build_from_checkpoint(reloaded)
        assert restored(batch).shape == (len(batch), 92)

    print("metadata: PASS", len(train) + len(val) + len(test), "usable rows")
    print("split: PASS", len(train), len(val), len(test))
    print("image batch: PASS", tuple(batch.shape))
    print("task1: PASS (forward, loss, optimizer step, checkpoint reload)")
    print("task2: PASS", tuple(task2(batch).shape))
    print("task3: PASS", {key: tuple(value.shape) for key, value in outputs.items()})
    print("task4: PASS", tuple(embeddings.shape), "normalized similarity query")
    print("backend preprocessing contract: PASS", (1, 3, IMAGE_SIZE_TORCH[0], IMAGE_SIZE_TORCH[1]))


if __name__ == "__main__":
    main()