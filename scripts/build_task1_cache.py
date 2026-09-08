"""Rebuild Task 1's uint8 image cache.

``src/training/train_item_type.py`` and the five split tests in
``tests/test_splits.py`` read ``processed/image_cache_task1_60x80{,_ids}.npy``.
Notebook ``02`` builds it as a side effect of training, so a checkout that has
never run that notebook - or one where the file was cleaned up - fails those
tests with ``FileNotFoundError`` rather than anything that explains itself.
This script builds the same file without a training run.

Task 1 keeps its own cache rather than reusing Task 3's ``image_cache_60x80``
because the two hold different row subsets: Task 3's is 38,539 rows, Task 1's
is the 38,491 that survive the rare-class floor (``MIN_CLASS_SIZE``). The two
are not interchangeable and neither is a superset of the other.

Row selection and decode path both mirror notebook ``02`` CELL 11 exactly:

* rows are ``clean_train_metadata.csv`` minus the classes below the floor, in
  CSV order, which is what ``load_splits`` reconstructs;
* tiles are decoded by ``item_type_classifier.load_image_array`` at
  ``(60, 80)``, the project's single decode path - ``convert("RGB")`` then
  ``resize(..., BILINEAR)``.

Paths come from ``id``, not from the metadata's ``image_path`` column, which
holds absolute paths baked in on the machine that wrote it.

Order does not actually matter to correctness: ``load_splits`` maps ids to rows
through the ``_ids`` file rather than assuming a position. It is matched anyway
so a rebuilt cache is byte-identical to the notebook's.

    python scripts/build_task1_cache.py
    python scripts/build_task1_cache.py --check    # verify, build nothing
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.item_type_classifier import load_image_array  # noqa: E402
from src.training.train_item_type import (  # noqa: E402
    CLEAN_METADATA,
    IMAGE_CACHE,
    IMAGE_CACHE_IDS,
    IMAGE_SIZE_PIL,
    MIN_CLASS_SIZE,
    TARGET,
)

TRAIN_IMAGES = (PROJECT_ROOT / "A2_FashionDataset" / "FashionDataset" / "train"
                / "images_train")
IMAGE_SHAPE = (IMAGE_SIZE_PIL[1], IMAGE_SIZE_PIL[0], 3)   # (H, W, 3)


def task1_rows(min_class_size=MIN_CLASS_SIZE):
    """The rows Task 1 models, in the order ``load_splits`` reconstructs."""
    data = pd.read_csv(CLEAN_METADATA)
    counts = data[TARGET].value_counts()
    rare = counts[counts < min_class_size].index.tolist()
    kept = data[~data[TARGET].isin(rare)].copy()
    return kept, sorted(rare)


def verify(sample=25, seed=0):
    """Re-decode a sample and compare it against what the cache holds."""
    if not (IMAGE_CACHE.exists() and IMAGE_CACHE_IDS.exists()):
        return False, "cache files are absent"

    kept, _ = task1_rows()
    ids = np.load(IMAGE_CACHE_IDS)
    images = np.load(IMAGE_CACHE, mmap_mode="r")

    if len(ids) != len(images):
        return False, "cache has %d rows but %d ids" % (len(images), len(ids))
    if len(ids) != len(kept):
        return False, ("cache holds %d rows, the row selection wants %d"
                       % (len(ids), len(kept)))
    if not np.array_equal(np.asarray(ids), kept["id"].to_numpy()):
        return False, "cache ids do not match the row selection"
    if tuple(images.shape[1:]) != IMAGE_SHAPE:
        return False, "cache frames are %s, expected %s" % (images.shape[1:], IMAGE_SHAPE)

    rng = np.random.default_rng(seed)
    for position in rng.choice(len(ids), min(sample, len(ids)), replace=False):
        expected = load_image_array(TRAIN_IMAGES / ("%d.jpg" % int(ids[position])),
                                    IMAGE_SIZE_PIL)
        if not np.array_equal(np.asarray(images[position]), expected):
            return False, "row %d does not match a fresh decode" % position
    return True, "%d rows, %d spot-checked and byte-identical" % (len(ids), min(sample, len(ids)))


def build(force=False):
    kept, rare = task1_rows()
    ids = kept["id"].to_numpy()
    print("clean metadata %d rows | %d classes below the floor of %d dropped "
          "| caching %d" % (len(pd.read_csv(CLEAN_METADATA)), len(rare),
                            MIN_CLASS_SIZE, len(kept)))

    if IMAGE_CACHE.exists() and not force:
        ok, detail = verify()
        if ok:
            print("cache already valid:", detail)
            return 0
        print("rebuilding, existing cache is unusable:", detail)

    missing = [int(i) for i in ids
               if not (TRAIN_IMAGES / ("%d.jpg" % int(i))).exists()]
    if missing:
        print("%d source images are absent, e.g. %s" % (len(missing), missing[:5]))
        print("this cache cannot be built without A2_FashionDataset/.../images_train")
        return 1

    IMAGE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    # Written straight to a memmap: the finished array is ~554 MB and there is
    # no reason to hold a second copy in RAM while building it.
    images = np.lib.format.open_memmap(
        IMAGE_CACHE, mode="w+", dtype=np.uint8, shape=(len(ids),) + IMAGE_SHAPE)

    started = time.time()
    for position, image_id in enumerate(ids):
        images[position] = load_image_array(
            TRAIN_IMAGES / ("%d.jpg" % int(image_id)), IMAGE_SIZE_PIL)
        if (position + 1) % 5000 == 0:
            print("  %d/%d (%.0fs)" % (position + 1, len(ids), time.time() - started))
    images.flush()
    del images
    np.save(IMAGE_CACHE_IDS, ids)
    print("built in %.0fs -> %s" % (time.time() - started, IMAGE_CACHE.name))

    ok, detail = verify()
    print("verification:", detail)
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="verify the existing cache and exit")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the existing cache verifies")
    args = parser.parse_args()

    if args.check:
        ok, detail = verify()
        print(("cache is valid: " if ok else "cache is NOT usable: ") + detail)
        return 0 if ok else 1
    return build(force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
