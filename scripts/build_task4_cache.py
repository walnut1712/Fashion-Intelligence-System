#!/usr/bin/env python
"""Build the image and item-mask caches Task 4 trains against, at either resolution.

Why this exists
---------------
Notebook 06 indexes ``IMAGES[position]`` and ``ITEM_MASKS[position]`` thousands
of times an epoch, so both live in memmapped caches rather than being decoded
per sample. Those caches were built inside the notebook and only ever at 60x80,
which is why moving Task 4 to the 120x160 catalogue needed a script rather than
an edit.

It also fixes the gallery. Task 4's gallery was every row of
``clean_train_metadata.csv``; the 120x160 pipeline marks 41 rows
``use_for_supervised = False`` for conflicting task labels, and those are
dropped here. That matters more than 41 rows suggests: ``RetrievalProtocol``
shuffles *products* to pick its holdout, so removing any row reshuffles the
whole split. Measured, the reduced gallery shares only 6% of its held-out
queries with the original - which is why the published 80.2 cannot be compared
against a 120x160 run, and why ``--resolution 60x80`` exists here. Building both
caches from one gallery is what makes the comparison a comparison.

Outputs, per resolution:

    A2_FashionDataset/processed/task4_cache_<res>.npy        (N, H, W, 3) uint8
    A2_FashionDataset/processed/task4_cache_<res>_ids.npy    (N,) int
    A2_FashionDataset/processed/task4_masks_<res>.npy        (N, H, W) bool
    A2_FashionDataset/processed/task4_gallery_<res>.csv      the gallery rows

The ``task4_`` prefix is deliberate. ``search_cache_60x80.npy`` and
``item_masks_60x80.npy`` are notebook 06's, hold all 38,612 rows and are indexed
by position against ``clean_train_metadata.csv``; writing a 38,571-row array
over either would silently misalign every executed cell in that notebook.

Usage
-----
    python scripts/build_task4_cache.py --resolution 120x160
    python scripts/build_task4_cache.py --resolution 60x80     # the re-baseline
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthetic_backgrounds import _border_colour_mask  # noqa: E402

PROCESSED = PROJECT_ROOT / "A2_FashionDataset" / "processed"
SUPERVISED = PROCESSED / "train_metadata_120x160_supervised.csv"

#: name -> (width, height, source directory). The 60x80 entry is not legacy
#: baggage: it is how the resolution comparison is made on one gallery.
RESOLUTIONS = {
    "60x80": (60, 80,
              PROJECT_ROOT / "A2_FashionDataset" / "FashionDataset" / "train" / "images_train"),
    "120x160": (120, 160, PROCESSED / "images_train_120x160"),
}

METADATA_COLUMNS = ["id", "articleType", "subCategory", "masterCategory",
                    "baseColour", "gender", "usage", "productDisplayName"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", choices=sorted(RESOLUTIONS), default="120x160")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N rows, for a smoke test")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the caches already exist")
    return parser.parse_args()


def build_gallery(limit=None):
    """The rows Task 4 may train on, in a stable order.

    Sorted by id so that the two resolutions produce identical row ordering and
    an embedding at one can be compared position-for-position with the other.
    """
    if not SUPERVISED.is_file():
        sys.exit("Missing {}. Pull the 120x160 pipeline first.".format(SUPERVISED))
    supervised = pd.read_csv(SUPERVISED)

    usable = supervised["use_for_supervised"].map(
        lambda v: v is True or str(v).strip().lower() == "true")
    gallery = supervised[usable].copy()
    dropped = int((~usable).sum())

    missing = [c for c in METADATA_COLUMNS if c not in gallery.columns]
    if missing:
        sys.exit("Supervised metadata is missing {}".format(missing))

    gallery = gallery[METADATA_COLUMNS].sort_values("id").reset_index(drop=True)
    print("Gallery: {:,} rows ({} excluded for conflicting task labels)".format(
        len(gallery), dropped))
    return gallery.head(limit) if limit else gallery


def main():
    args = parse_args()
    width, height, image_dir = RESOLUTIONS[args.resolution]
    if not image_dir.is_dir():
        sys.exit("Image directory not found: {}".format(image_dir))

    gallery = build_gallery(args.limit)
    ids = gallery["id"].to_numpy()

    image_path = PROCESSED / "task4_cache_{}.npy".format(args.resolution)
    ids_path = PROCESSED / "task4_cache_{}_ids.npy".format(args.resolution)
    mask_path = PROCESSED / "task4_masks_{}.npy".format(args.resolution)
    gallery_path = PROCESSED / "task4_gallery_{}.csv".format(args.resolution)

    if image_path.exists() and not args.force:
        sys.exit("{} exists. Pass --force to rebuild.".format(image_path.name))

    print("Writing {} x {} for {:,} items".format(width, height, len(ids)))
    images = np.lib.format.open_memmap(
        image_path, mode="w+", dtype=np.uint8, shape=(len(ids), height, width, 3))
    masks = np.lib.format.open_memmap(
        mask_path, mode="w+", dtype=bool, shape=(len(ids), height, width))

    started = time.perf_counter()
    resized = 0
    for position, item in enumerate(ids):
        with Image.open(image_dir / "{}.jpg".format(item)) as opened:
            frame = opened.convert("RGB")
            if frame.size != (width, height):
                # A handful of catalogue files are not the nominal size. Resize
                # rather than skip: a gallery with holes cannot be indexed by
                # position, which is how everything downstream addresses it.
                frame = frame.resize((width, height), Image.BILINEAR)
                resized += 1
            array = np.asarray(frame, dtype=np.uint8)

        images[position] = array
        masks[position] = _border_colour_mask(array)

        if (position + 1) % 5000 == 0:
            print("  {:>6,}/{:,}".format(position + 1, len(ids)), flush=True)

    images.flush()
    masks.flush()
    np.save(ids_path, ids)
    gallery.to_csv(gallery_path, index=False)

    occupancy = masks.mean(axis=(1, 2))
    print("\nDone in {:.1f} min ({} files resized to the nominal size)".format(
        (time.perf_counter() - started) / 60, resized))
    print("  images {} -> {}".format(images.shape, image_path.name))
    print("  masks  {} -> {}".format(masks.shape, mask_path.name))
    print("  gallery -> {}".format(gallery_path.name))
    print("  mask occupancy: mean {:.3f} p10 {:.3f} p90 {:.3f}".format(
        occupancy.mean(), np.percentile(occupancy, 10), np.percentile(occupancy, 90)))
    suspicious = int(((occupancy < 0.02) | (occupancy > 0.95)).sum())
    print("  implausible masks: {} ({:.2%})".format(suspicious, suspicious / len(ids)))


if __name__ == "__main__":
    main()
