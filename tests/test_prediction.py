"""End-to-end Task 1 inference tests: checkpoint -> service -> label.

The strongest check here is ``test_saved_predictions_are_reproducible``: it
replays rows of the committed ``task1_predictions.csv`` through the serving
path. If the architecture, the class ordering, the normalisation constants or
the resize contract ever drift again, that test fails loudly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.services.task1_service import Task1Service  # noqa: E402
from src.models.item_type_classifier import (  # noqa: E402
    load_item_type_model,
    predict_proba,
    preprocess_image,
)

CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "task1" / "task1_cnn.pt"
PREDICTIONS_CSV = PROJECT_ROOT / "artifacts" / "task1" / "task1_predictions.csv"
TEST_IMAGE_DIR = PROJECT_ROOT / "A2_FashionDataset" / "FashionDataset" / "test" / "images_test"


@pytest.fixture(scope="module")
def service():
    if not CHECKPOINT_PATH.exists():
        pytest.skip("{} not present".format(CHECKPOINT_PATH))
    return Task1Service()


@pytest.fixture(scope="module")
def sample_image_bytes():
    if not TEST_IMAGE_DIR.exists():
        pytest.skip("dataset images not present (gitignored)")
    images = sorted(TEST_IMAGE_DIR.glob("*.jpg"))
    if not images:
        pytest.skip("no test images found")
    return images[0].read_bytes()


def test_service_loads(service):
    assert service.num_classes == len(service.class_names)
    assert service.num_classes > 1


def test_predict_returns_a_known_label(service, sample_image_bytes):
    prediction = service.predict(sample_image_bytes)
    assert prediction["label"] in service.class_names
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert prediction["top3"][0]["label"] == prediction["label"]


def test_top3_is_ordered_and_distinct(service, sample_image_bytes):
    top3 = service.predict(sample_image_bytes)["top3"]
    assert len(top3) == 3
    confidences = [entry["confidence"] for entry in top3]
    assert confidences == sorted(confidences, reverse=True)
    assert len({entry["label"] for entry in top3}) == 3


def test_rejects_data_that_is_not_an_image(service):
    with pytest.raises(ValueError):
        service.predict(b"this is not a JPEG")


def test_service_and_module_agree_on_logits(service, sample_image_bytes):
    """The notebook and the API must run identical maths on identical pixels."""
    model, checkpoint = load_item_type_model(CHECKPOINT_PATH, torch.device("cpu"))
    with torch.no_grad():
        reference = model(preprocess_image(sample_image_bytes, checkpoint, torch.device("cpu")))
        served = service.model(service.preprocess(sample_image_bytes))
    assert torch.allclose(reference, served.cpu(), atol=1e-5)


def test_saved_predictions_are_reproducible(service):
    """Replay committed predictions through the serving path."""
    if not PREDICTIONS_CSV.exists() or not TEST_IMAGE_DIR.exists():
        pytest.skip("predictions CSV or dataset images not present")
    pandas = pytest.importorskip("pandas")

    saved = pandas.read_csv(PREDICTIONS_CSV).head(64)
    paths = [TEST_IMAGE_DIR / "{}.jpg".format(image_id) for image_id in saved["id"]]
    if not all(path.exists() for path in paths):
        pytest.skip("sampled images missing")

    probabilities = predict_proba(service.model, service.checkpoint, paths, batch_size=64)
    predicted = [service.class_names[index] for index in probabilities.argmax(1)]

    assert predicted == list(saved["articleType"])
    assert np.allclose(probabilities.max(1), saved["articleType_confidence"], atol=1e-4)
