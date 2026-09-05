"""Shared paths and image-shape configuration for candidate experiments."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "A2_FashionDataset" / "processed"

IMAGE_WIDTH = 120
IMAGE_HEIGHT = 160
IMAGE_SIZE_PIL = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_SIZE_TORCH = (IMAGE_HEIGHT, IMAGE_WIDTH)
EXPECTED_TENSOR_SHAPE = (3, IMAGE_HEIGHT, IMAGE_WIDTH)

DATA_DIR = PROCESSED_DIR / "images_train_120x160"
SUPERVISED_METADATA = PROCESSED_DIR / "train_metadata_120x160_supervised.csv"
SPLIT_IDS_PATH = PROCESSED_DIR / "splits_120x160.csv"

CANDIDATE_ARTIFACT_DIRS = {
    "task1": PROJECT_ROOT / "artifacts" / "task1_120x160",
    "task2": PROJECT_ROOT / "artifacts" / "task2_120x160",
    "task3": PROJECT_ROOT / "artifacts" / "task3_120x160",
    "task4": PROJECT_ROOT / "artifacts" / "task4_120x160",
}


def require_candidate_inputs():
    """Validate the immutable 120x160 inputs before an experiment starts."""
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"120x160 image directory not found: {DATA_DIR}")
    if not SUPERVISED_METADATA.is_file():
        raise FileNotFoundError(
            f"Required supervised metadata not found: {SUPERVISED_METADATA}"
        )
