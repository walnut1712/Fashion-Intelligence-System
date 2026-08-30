"""Task 1 architecture regression tests.

These exist because the shipped checkpoint once became unloadable: the
notebook was rolled back to a version whose ``ItemTypeCNN`` used a
global-average-pool head, while ``artifacts/task1/task1_cnn.pt`` had been
trained with a spatial avg+max head. Nothing caught it until the API 503'd.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.item_type_classifier import (  # noqa: E402
    HEAD_BN_SPATIAL,
    HEAD_LEGACY_GAP,
    ItemTypeCNN,
    build_from_checkpoint,
    load_item_type_model,
)

CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "task1" / "task1_cnn.pt"
LABEL_CLASSES_PATH = PROJECT_ROOT / "artifacts" / "task1" / "label_classes.json"


@pytest.fixture(scope="module")
def loaded():
    if not CHECKPOINT_PATH.exists():
        pytest.skip("{} not present".format(CHECKPOINT_PATH))
    return load_item_type_model(CHECKPOINT_PATH, torch.device("cpu"))


def test_shipped_checkpoint_loads_strictly(loaded):
    """A strict load is the whole point - a silently partial load is the bug."""
    model, checkpoint = loaded
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=True), None
    assert missing.missing_keys == []
    assert missing.unexpected_keys == []


def test_checkpoint_metadata_is_self_consistent(loaded):
    _, checkpoint = loaded
    assert int(checkpoint["num_classes"]) == len(checkpoint["class_names"])
    assert len(checkpoint["channel_mean"]) == 3
    assert len(checkpoint["channel_std"]) == 3
    assert len(checkpoint["image_size_pil"]) == 2
    assert all(std > 0 for std in checkpoint["channel_std"])


def test_class_names_match_label_classes_artifact(loaded):
    """The app maps an argmax index to a name; the two orderings must agree."""
    if not LABEL_CLASSES_PATH.exists():
        pytest.skip("label_classes.json not present")
    _, checkpoint = loaded
    with open(LABEL_CLASSES_PATH, encoding="utf-8") as handle:
        assert json.load(handle) == list(checkpoint["class_names"])


def test_output_width_matches_class_count(loaded):
    model, checkpoint = loaded
    width, height = checkpoint["image_size_pil"]
    with torch.no_grad():
        logits = model(torch.zeros(2, 3, height, width))
    assert logits.shape == (2, len(checkpoint["class_names"]))


def test_avgmax_head_doubles_the_feature_width():
    avg = ItemTypeCNN(10, widths=(8, 16), pool_grid=(2, 1), pool_mode="avg")
    avgmax = ItemTypeCNN(10, widths=(8, 16), pool_grid=(2, 1), pool_mode="avgmax")
    assert avg.pooled_features == 16 * 2
    assert avgmax.pooled_features == 16 * 2 * 2


def test_legacy_checkpoints_without_pool_grid_still_build():
    """Pre-20260824 checkpoints have no pool_grid and a head with no BatchNorm."""
    legacy = {"num_classes": 44,
              "architecture": {"widths": [16, 32, 64, 128], "dropout": 0.4, "head_hidden": 128}}
    model = build_from_checkpoint(legacy)
    assert model.head_style == HEAD_LEGACY_GAP
    keys = set(model.state_dict())
    assert "head.1.weight" in keys and "head.4.weight" in keys
    assert not any(key.startswith("head.0.") for key in keys)


def test_spatial_checkpoints_use_the_batchnorm_head():
    spatial = {"num_classes": 92,
               "architecture": {"widths": [16, 32, 64, 128], "dropout": 0.2,
                                "head_hidden": 384, "pool_grid": [2, 1], "pool_mode": "avgmax"}}
    model = build_from_checkpoint(spatial)
    assert model.head_style == HEAD_BN_SPATIAL
    keys = set(model.state_dict())
    assert {"head.0.weight", "head.2.weight", "head.5.weight"} <= keys
    assert model.state_dict()["head.2.weight"].shape == (384, 512)
