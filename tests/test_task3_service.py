"""Task 3 checkpoint and serving regression tests.

These mirror the Task 1 tests, and exist for the same reason: the checkpoint is
self-describing, so nothing outside it pins the architecture down, and a shape
that no longer matches shows up as a 503 rather than as a wrong answer.

They also guard the calibration contract added when the model started reporting
temperature-scaled probabilities. Dividing every logit by one positive number
cannot reorder them, so the served *label* must be identical with and without the
temperature. That invariant is easy to break by dividing in the wrong place, and
the failure is silent: the labels stay plausible and only the confidence is wrong.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.services.task3_service import (  # noqa: E402
    EarlyBranchCNN,
    Task3Service,
)

CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "task3" / "task3_cnn_model.pt"
SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "task3" / "task3_cnn_summary.json"
TARGETS = ("gender", "usage")


@pytest.fixture(scope="module")
def service():
    if not CHECKPOINT_PATH.exists():
        pytest.skip("{} not present".format(CHECKPOINT_PATH))
    return Task3Service(CHECKPOINT_PATH)


@pytest.fixture(scope="module")
def checkpoint():
    if not CHECKPOINT_PATH.exists():
        pytest.skip("{} not present".format(CHECKPOINT_PATH))
    return torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def sample_image():
    """A small PNG, encoded in memory - the service takes bytes, not an array."""
    from io import BytesIO

    from PIL import Image

    rng = np.random.default_rng(0)
    array = rng.integers(180, 255, size=(160, 120, 3), dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def test_shipped_checkpoint_loads_strictly(service, checkpoint):
    """A partial load is the bug: it leaves random weights behind a working API."""
    result = service.model.load_state_dict(checkpoint["state_dict"], strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_checkpoint_metadata_is_self_consistent(checkpoint):
    for target in TARGETS:
        assert checkpoint["num_classes"][target] == len(checkpoint["class_names"][target])
    assert len(checkpoint["channel_mean"]) == 3
    assert len(checkpoint["channel_std"]) == 3
    assert all(std > 0 for std in checkpoint["channel_std"])
    assert len(checkpoint["image_size_pil"]) == 2


def test_service_rebuilds_the_architecture_the_checkpoint_records(service, checkpoint):
    """The service must not hold its own opinion about widths or input size."""
    recorded = checkpoint["architecture"]
    assert tuple(service.architecture["shared_widths"]) == tuple(recorded["shared_widths"])
    assert tuple(service.architecture["branch_widths"]) == tuple(recorded["branch_widths"])
    assert service.image_size == tuple(checkpoint["image_size_pil"])


def test_each_head_outputs_one_score_per_class(service):
    width, height = service.image_size
    with torch.no_grad():
        outputs = service.model(torch.zeros(2, 3, height, width, device=service.device))
    for target in TARGETS:
        assert outputs[target].shape == (2, len(service.class_names[target]))


def test_prediction_is_ranked_and_labelled(service, sample_image):
    output = service.predict(sample_image, top_k=3)
    for target in TARGETS:
        ranked = output[target]
        probabilities = [row["p"] for row in ranked]
        assert 0 < len(ranked) <= 3
        assert probabilities == sorted(probabilities, reverse=True)
        assert all(row["label"] in service.class_names[target] for row in ranked)
        assert all(0.0 <= row["p"] <= 1.0 for row in ranked)


def test_temperature_is_present_and_positive(service):
    """Absent or zero would divide the logits by nothing, or by zero."""
    for target in TARGETS:
        assert service.temperature[target] > 0


def test_temperature_changes_confidence_but_never_the_label(service, sample_image):
    """The whole claim behind temperature scaling, asserted where it is served."""
    tensor = service.preprocess(sample_image)
    with torch.no_grad():
        logits = service.model(tensor)

    served = service.predict(sample_image, top_k=1)
    for target in TARGETS:
        raw = F.softmax(logits[target][0].float(), dim=0).cpu().numpy()
        scaled = F.softmax(logits[target][0].float() / service.temperature[target],
                           dim=0).cpu().numpy()
        # The label is invariant to the scaling, and must match what was served.
        assert int(raw.argmax()) == int(scaled.argmax())
        assert service.class_names[target][int(raw.argmax())] == served[target][0]["label"]
        # The probability is not invariant, and the served one must be the scaled one.
        # Comparing against `scaled` rather than merely checking that the two differ is
        # what makes this fail if the service ever stops dividing.
        assert served[target][0]["p"] == pytest.approx(float(scaled.max()), abs=1e-4)
        if service.temperature[target] != 1.0:
            assert float(scaled.max()) < float(raw.max()), "T > 1 must soften the peak"
            assert served[target][0]["p"] != pytest.approx(float(raw.max()), abs=1e-3)


def test_a_checkpoint_without_a_temperature_serves_unscaled(tmp_path, checkpoint):
    """Older checkpoints predate the field; they must still load and serve at T=1."""
    legacy = {key: value for key, value in checkpoint.items() if key != "temperature"}
    path = tmp_path / "legacy_task3.pt"
    torch.save(legacy, path)
    legacy_service = Task3Service(path)
    assert legacy_service.temperature == {target: 1.0 for target in TARGETS}
    assert legacy_service.is_calibrated is False


def test_class_weights_are_ones_so_serving_is_a_plain_argmax(service):
    """Threshold adjustment was evaluated and not adopted; ones record that."""
    for target in TARGETS:
        assert np.allclose(service.class_weights[target], 1.0)
    assert service.uses_thresholds is False


def test_summary_artifact_agrees_with_the_checkpoint(checkpoint):
    """Two files record the same run; a mismatch means one was written by another."""
    if not SUMMARY_PATH.exists():
        pytest.skip("summary json not present")
    with open(SUMMARY_PATH, encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["final_model"] == checkpoint["model_name"]
    for target in TARGETS:
        assert summary["temperature"][target] == pytest.approx(
            checkpoint["temperature"][target])
        assert summary["test_{}_macro_f1".format(target)] == pytest.approx(
            checkpoint["test_metrics"][target]["macro_f1"], abs=0.01)


def test_branch_blocks_follow_the_input_height():
    """60x80 takes two branch blocks and 120x160 three, both leaving a 5x3 map.

    The notebook derives the count; this pins the shape the served classifier was
    sized for, so a resolution change cannot silently reshape the head.
    """
    for (height, width), widths in [((80, 60), (128, 256)), ((160, 120), (128, 256, 512))]:
        model = EarlyBranchCNN({"gender": 5, "usage": 4}, input_shape=(3, height, width),
                               branch_widths=widths)
        with torch.no_grad():
            features = model.branches["gender"](model.shared(torch.zeros(1, 3, height, width)))
        assert features.shape[2:] == (5, 3)
