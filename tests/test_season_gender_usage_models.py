"""Task 2 and Task 3 architecture and serving regression tests.

Neither task had any test. That is the same exposure `tests/test_models.py` was
written for after a duplicated Task 1 network definition left a trained
checkpoint the API could not load - and it is worse here, because both
architectures are declared inside `app/backend/services/`, not in `src/models/`
(`src/models/season_classifier.py`, `gender_classifier.py` and
`usage_classifier.py` are still empty). Editing a service therefore silently
edits the architecture a checkpoint has to fit.

The two checkpoints carry very different risk:

* Task 3 records its own architecture, so the service rebuilds from the file and
  a notebook change cannot orphan it.
* Task 2 is a bare ``OrderedDict`` - no architecture, no class names, no
  preprocessing constants. `SeasonCNN` is reconstructed from a hardcoded default
  and every one of those assumptions is unchecked at load time. These tests are
  the only thing standing between a widened layer and a wrong prediction.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TASK2_CHECKPOINT = PROJECT_ROOT / "artifacts" / "task2" / "task2_season_best_pytorch.pth"
TASK2_MAPPING = PROJECT_ROOT / "artifacts" / "task2" / "task2_season_class_mapping.json"
TASK3_CHECKPOINT = PROJECT_ROOT / "artifacts" / "task3" / "task3_cnn_model.pt"


# =============================================================== Task 2 ====
@pytest.fixture(scope="module")
def task2_state():
    if not TASK2_CHECKPOINT.exists():
        pytest.skip("{} not present".format(TASK2_CHECKPOINT))
    return torch.load(TASK2_CHECKPOINT, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def task2_mapping():
    if not TASK2_MAPPING.exists():
        pytest.skip("{} not present".format(TASK2_MAPPING))
    with open(TASK2_MAPPING, encoding="utf-8") as handle:
        return json.load(handle)


def test_task2_checkpoint_loads_strictly(task2_state, task2_mapping):
    """A strict load is the point - a silently partial load is the bug."""
    from app.backend.services.task2_service import SeasonCNN

    model = SeasonCNN(len(task2_mapping))
    model.load_state_dict(task2_state)          # strict by default
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, 3, 80, 60))
    assert logits.shape == (2, len(task2_mapping))


def test_task2_class_mapping_matches_the_trained_head(task2_state, task2_mapping):
    """The mapping file and the checkpoint must agree on how many seasons exist.

    They are separate files and nothing else compares them, so a mapping edited
    without retraining would relabel every prediction rather than raise.
    """
    final_weight = [v for k, v in task2_state.items() if k.endswith("weight")][-1]
    assert final_weight.shape[0] == len(task2_mapping), (
        "checkpoint predicts {} classes, mapping names {}".format(
            final_weight.shape[0], len(task2_mapping))
    )
    assert sorted(task2_mapping.values()) == list(range(len(task2_mapping))), (
        "class indices must be contiguous from zero"
    )


def test_task2_preprocessing_matches_the_training_transform():
    """Training used Resize((80, 60)) + ToTensor() and NO normalisation.

    Tasks 1, 3 and 4 all normalise by channel statistics, so the natural
    assumption when reading the service is that Task 2 does too. It must not:
    adding normalisation here would shift every input away from what the network
    was fitted on. This pins the actual contract.
    """
    from PIL import Image

    from app.backend.services.task2_service import Task2Service

    if not (TASK2_CHECKPOINT.exists() and TASK2_MAPPING.exists()):
        pytest.skip("Task 2 artefacts not present")

    service = Task2Service()
    buffer = __import__("io").BytesIO()
    Image.fromarray(np.full((120, 90, 3), 200, dtype=np.uint8)).save(buffer, format="PNG")
    tensor = service.preprocess(buffer.getvalue())

    assert tensor.shape == (1, 3, 80, 60), "expected 80 tall by 60 wide"
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0, (
        "values must stay in [0, 1] - ToTensor scaling only, no normalisation"
    )
    # a flat 200/255 image must come back as ~0.784 everywhere, not re-centred
    assert float(tensor.mean()) == pytest.approx(200 / 255, abs=1e-3)


def test_task2_predicts_a_known_season_label(task2_mapping):
    from PIL import Image

    from app.backend.services.task2_service import Task2Service

    if not (TASK2_CHECKPOINT.exists() and TASK2_MAPPING.exists()):
        pytest.skip("Task 2 artefacts not present")

    service = Task2Service()
    buffer = __import__("io").BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(
        0, 256, (80, 60, 3), dtype=np.uint8)).save(buffer, format="PNG")
    result = service.predict(buffer.getvalue())

    labels = {entry["label"] for entry in result["top3"]} if "top3" in result else set()
    known = set(task2_mapping)
    assert result["label"] in known
    assert labels <= known
    assert 0.0 <= result["confidence"] <= 1.0


# =============================================================== Task 3 ====
@pytest.fixture(scope="module")
def task3_checkpoint():
    if not TASK3_CHECKPOINT.exists():
        pytest.skip("{} not present".format(TASK3_CHECKPOINT))
    return torch.load(TASK3_CHECKPOINT, map_location="cpu", weights_only=False)


def test_task3_checkpoint_records_its_own_architecture(task3_checkpoint):
    """The convention that makes Task 3 safe to edit in the notebook.

    Widths live in the file, so the service rebuilds whatever was trained rather
    than whatever the source currently declares.
    """
    architecture = task3_checkpoint.get("architecture")
    assert isinstance(architecture, dict) and architecture, (
        "checkpoint must record its architecture"
    )
    for key in ("shared_widths", "branch_widths", "hidden"):
        assert key in architecture, "architecture is missing {}".format(key)


def test_task3_rebuilds_from_the_checkpoint_and_loads_strictly(task3_checkpoint):
    from app.backend.services.task3_service import EarlyBranchCNN

    class_names = {t: list(v) for t, v in task3_checkpoint["class_names"].items()}
    num_classes = {t: len(v) for t, v in class_names.items()}
    width, height = tuple(task3_checkpoint["image_size_pil"])
    architecture = dict(task3_checkpoint["architecture"])
    architecture["shared_widths"] = tuple(architecture["shared_widths"])
    architecture["branch_widths"] = tuple(architecture["branch_widths"])

    model = EarlyBranchCNN(num_classes, input_shape=(3, height, width), **architecture)
    model.load_state_dict(task3_checkpoint["state_dict"])       # strict
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, height, width))
    assert set(out) == {"gender", "usage"}
    assert out["gender"].shape == (2, num_classes["gender"])
    assert out["usage"].shape == (2, num_classes["usage"])


def test_task3_is_early_branching_not_a_shared_trunk(task3_checkpoint):
    """The design decision the notebook selected. A checkpoint without per-attribute
    branches is a different architecture wearing the same filename."""
    keys = task3_checkpoint["state_dict"].keys()
    assert any(k.startswith("branches.gender.") for k in keys)
    assert any(k.startswith("branches.usage.") for k in keys)
    assert any(k.startswith("shared.") for k in keys)


def test_task3_ships_without_threshold_adjustment(task3_checkpoint):
    """Threshold adjustment was evaluated and removed.

    The service multiplies probabilities by `class_weights` before argmax, so a
    checkpoint whose weights are not all 1.0 would silently re-introduce a
    technique the notebook rejected.
    """
    assert task3_checkpoint.get("thresholds_applied") is False
    for target, weights in task3_checkpoint["class_weights"].items():
        assert all(float(w) == 1.0 for w in weights), (
            "{} carries non-unit class weights: {}".format(target, weights)
        )


def test_task3_label_policy_is_five_genders_and_four_usages(task3_checkpoint):
    """Pinned because the usage merge is a documented, deliberate choice.

    Eight raw usage classes collapse to four; pooling the rare ones into `Other`
    was tried and scored F1 0.000 on it. A checkpoint with a different count
    means that policy changed.
    """
    class_names = task3_checkpoint["class_names"]
    assert len(class_names["gender"]) == 5
    assert len(class_names["usage"]) == 4
    assert set(class_names["gender"]) == {"Boys", "Girls", "Men", "Unisex", "Women"}


def test_task3_preprocessing_constants_are_present(task3_checkpoint):
    """Unlike Task 2, these travel with the weights - keep it that way."""
    assert len(task3_checkpoint["channel_mean"]) == 3
    assert len(task3_checkpoint["channel_std"]) == 3
    assert all(s > 0 for s in task3_checkpoint["channel_std"])
    assert tuple(task3_checkpoint["image_size_pil"]) == (60, 80)


def test_task3_reported_metrics_travel_with_the_checkpoint(task3_checkpoint):
    """The API's model card reads these; a checkpoint without them advertises
    numbers from whatever ran last."""
    metrics = task3_checkpoint.get("test_metrics", {})
    assert {"gender", "usage"} <= set(metrics)
    for target in ("gender", "usage"):
        assert 0.0 <= metrics[target]["accuracy"] <= 100.0
        assert 0.0 <= metrics[target]["macro_f1"] <= 100.0
