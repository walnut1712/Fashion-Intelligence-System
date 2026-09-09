"""Both Task 4 encoder arms on the hand-labelled real photographs.

Notebook 05 Part 8b shows this table; this script writes it to
``outputs/evaluation/task4_real_photo_arms.csv`` so notebook 07 can read a
number rather than recompute one. Same code path in both - ``score_arms`` from
``src.evaluation.real_photo_arms`` - so the two cannot disagree.

Why it matters. Every out-of-domain figure Task 4 publishes comes from
*compositing*: a held-out catalogue cutout pasted onto a held-out Places365
scene. That is ground truth by construction and it scales to 2,000 queries, but
it is a proxy for a real upload rather than the thing itself. These 23 rows are
the only measurement in the project that scores real photographs against labels,
and they are what says whether the proxy is honest.

It is not, quite: the deployed encoder scores 23.04 here against 55.78 on the
composited ``photo`` benchmark. Compositing preserves the ordering between the
arms and overstates the level, so ``photo`` and ``wildphoto`` are upper bounds.

Three limits travel with the number and belong beside it wherever it is quoted:
n=23 (so roughly +/-10 points of sampling error), the labels were produced by a
vision model rather than a person and are unverified, and 1 of the 31 defeats
ingestion and falls back to a centre crop.

    python scripts/eval_task4_real_photos.py        # ~3 min, builds both indexes
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.evaluation import real_photo_arms as rpa  # noqa: E402

PROCESSED_DIR = PROJECT_DIR / "A2_FashionDataset" / "processed"
GALLERY_CSV = PROCESSED_DIR / "task4_gallery_120x160.csv"
CACHE_X = PROCESSED_DIR / "task4_cache_120x160.npy"
TRAINED_DIR = PROJECT_DIR / "artifacts" / "task4_120x160"
TEST_IMAGE_DIR = PROJECT_DIR / "A2_FashionDataset" / "FashionDataset" / "test" / "images_test"
OUT_CSV = PROJECT_DIR / "outputs" / "evaluation" / "task4_real_photo_arms.csv"
OUT_PER_PHOTO = PROJECT_DIR / "outputs" / "evaluation" / "task4_real_photo_per_image.csv"


def main():
    gallery = pd.read_csv(GALLERY_CSV)
    images = np.load(CACHE_X, mmap_mode="r")

    engines = {}
    for arm_name, arm_file in rpa.ARMS.items():
        checkpoint = TRAINED_DIR / arm_file
        if not checkpoint.exists():
            print("missing {} - train it with:".format(arm_file))
            print("    python -m src.training.train_task4_120x160 "
                  "--backgrounds {} --seed 42".format(
                      "none" if "none" in arm_file else "mixed"))
            return 1
        engines[arm_name] = rpa.build_engine(checkpoint, gallery, images)
        print("{:<20s} index {}  background_augmented={}".format(
            arm_name, engines[arm_name].index.shape,
            engines[arm_name].manifest["background_augmented"]))

    paths = rpa.real_photo_paths(PROJECT_DIR)
    labels = rpa.real_photo_labels(PROJECT_DIR)
    print()
    print(rpa.describe_label_state(labels, paths))
    if labels is None:
        print("nothing to score; run scripts/make_label_contact_sheet.py first")
        return 1

    reference = sorted(TEST_IMAGE_DIR.iterdir())[:60] if TEST_IMAGE_DIR.exists() else None
    summary, per_photo = rpa.score_arms(
        engines, paths, labels=labels, reference=reference)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    per_photo.to_csv(OUT_PER_PHOTO, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print()
    print(summary.to_string(index=False))
    print()
    print("wrote {}".format(OUT_CSV.relative_to(PROJECT_DIR)))
    print("wrote {}".format(OUT_PER_PHOTO.relative_to(PROJECT_DIR)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
