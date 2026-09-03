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


def test_shipped_checkpoint_matches_what_the_docs_claim():
    """The deployed model pools 1x1, not 2x1.

    The test above exercises the *builder* with a 2x1 grid, which is a supported
    configuration - but for a long time the class docstring, the notebook CONFIG
    and this file together implied 2x1 was what ships. It is not, and the
    checkpoint is the only authority on that question.
    """
    import torch

    path = PROJECT_ROOT / "artifacts" / "task1" / "task1_cnn.pt"
    if not path.exists():
        pytest.skip("deployed checkpoint not present")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    architecture = checkpoint["architecture"]
    assert architecture["pool_grid"] == [1, 1]
    assert architecture["head_hidden"] == 384
    model = build_from_checkpoint(checkpoint)
    assert model.state_dict()["head.2.weight"].shape == (384, 256)


def test_recorded_config_does_not_contradict_the_architecture():
    """``config`` used to be the notebook's template rather than the run's settings.

    It claimed ``head_hidden: 256``, ``dropout: 0.3``, ``pool_grid: [2, 1]`` and
    ``use_class_weights: True`` for a model named ``CNN_weights_none_full`` with
    384 hidden units and a 1x1 pool. ``Task1Service.model_card`` reads this dict,
    so the drift was user-visible. ``--sync-summary`` repairs it.
    """
    import torch

    path = PROJECT_ROOT / "artifacts" / "task1" / "task1_cnn.pt"
    if not path.exists():
        pytest.skip("deployed checkpoint not present")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    config = checkpoint.get("config") or {}
    architecture = checkpoint["architecture"]
    for key in ("widths", "head_hidden", "dropout", "pool_grid", "pool_mode"):
        if key in config:
            assert config[key] == architecture[key], (
                "config[{0!r}]={1!r} contradicts architecture[{0!r}]={2!r}; run "
                "`python -m src.training.train_item_type --sync-summary`".format(
                    key, config[key], architecture[key]))
    assert not config.get("use_class_weights"), (
        "config claims class weights for a run named {!r}".format(
            checkpoint.get("model_name")))


def test_temperature_is_applied_by_predict_proba():
    """Batch inference and the API must sit on the same operating point.

    ``Task1Service.predict`` divided logits by ``checkpoint["temperature"]`` and
    ``predict_proba`` did not, so the two paths disagreed on confidence for any
    checkpoint carrying one - while the module docstring claimed they agree by
    construction. Temperature cannot move the argmax, only the confidence.
    """
    import numpy as np
    import torch
    from PIL import Image

    from src.models.item_type_classifier import predict_proba

    checkpoint = {
        "num_classes": 4, "class_names": ["a", "b", "c", "d"],
        "channel_mean": [0.5, 0.5, 0.5], "channel_std": [0.25, 0.25, 0.25],
        "image_size_pil": [60, 80],
        "architecture": {"widths": [8, 16], "dropout": 0.0, "head_hidden": 16,
                         "pool_grid": [1, 1], "pool_mode": "avg"},
    }
    model = build_from_checkpoint(checkpoint).eval()
    sources = [Image.new("RGB", (60, 80), (120, 90, 60)),
               Image.new("RGB", (60, 80), (30, 200, 10))]

    with torch.no_grad():
        cold = predict_proba(model, checkpoint, sources)
        warm = predict_proba(model, {**checkpoint, "temperature": 3.0}, sources)

    assert (cold.argmax(1) == warm.argmax(1)).all(), "temperature moved the argmax"
    assert warm.max(1).mean() < cold.max(1).mean(), "temperature did not soften confidence"


def test_adjust_false_returns_raw_posteriors():
    """Label-shift estimators need posteriors the tau adjustment has not touched."""
    import numpy as np
    from PIL import Image

    from src.models.item_type_classifier import apply_logit_adjustment, predict_proba
    import torch

    checkpoint = {
        "num_classes": 4, "class_names": ["a", "b", "c", "d"],
        "channel_mean": [0.5, 0.5, 0.5], "channel_std": [0.25, 0.25, 0.25],
        "image_size_pil": [60, 80],
        "architecture": {"widths": [8, 16], "dropout": 0.0, "head_hidden": 16,
                         "pool_grid": [1, 1], "pool_mode": "avg"},
        "class_log_prior": np.log([0.7, 0.2, 0.07, 0.03]).tolist(),
        "logit_adjustment_tau": 0.5,
    }
    model = build_from_checkpoint(checkpoint).eval()
    sources = [Image.new("RGB", (60, 80), (120, 90, 60))]

    raw = predict_proba(model, checkpoint, sources, adjust=False)
    adjusted = predict_proba(model, checkpoint, sources, adjust=True)
    replayed = apply_logit_adjustment(torch.from_numpy(raw), checkpoint).numpy()

    assert not np.allclose(raw, adjusted), "adjust=False returned the adjusted matrix"
    assert np.allclose(replayed, adjusted, atol=1e-6), (
        "adjust=True is not reproducible from the raw posteriors")
