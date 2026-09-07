"""Photographic backgrounds for Task 4, drawn from Places365.

Why this exists beside ``synthetic_backgrounds``
------------------------------------------------
The procedural bank (solid, gradient, noise, blobs, blurred crop) took the hard
benchmark from 12.2 to 60.6 P@10, so it is not a failure. But none of its five
families is a photograph: they have no objects, no depth of field, no clutter
competing with the garment for the encoder's attention. A real upload's backdrop
does. Places365 supplies 36,500 scene photographs, which is exactly that missing
family.

This module produces a ``(N, H, W, 3)`` uint8 array with the same contract as
``make_backgrounds``, so ``WildDataset`` consumes it without modification and the
composite -> degrade -> ingestion chain is unchanged. The background *source* is
the only variable.

The split is by scene category, not by file
-------------------------------------------
An earlier split (``train_backgrounds.txt`` / ``test_backgrounds.txt``, 29,199 /
7,299) drew 80/20 over files. Measured: **all 365 categories appear in both
lists**. So the model trained on 80 ``airfield`` photographs and was tested on 20
more ``airfield`` photographs, which is the id-level split this project rejected
everywhere else. Holding out whole categories makes "a background it has never
seen" mean an unseen *kind of place*, which is what an upload actually is.

The split is derived from the sorted category list and a fixed seed rather than
read from a file, so it cannot drift between machines or go stale when the
corpus changes. ``write_split_manifest`` records it for the write-up.

Crops, not resizes
------------------
Places photographs are around 512x683. Resizing one straight to 60x80 - which
the earlier script did - squashes the aspect ratio and blurs away every texture,
producing a backdrop far easier than the thing it stands for. A random window at
the target aspect ratio, taken at native resolution and then downscaled, keeps
the local texture that makes a background hard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["PLACES_ROOT", "category_split", "list_category_images",
           "load_background_bank", "make_mixed_bank", "write_split_manifest"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLACES_ROOT = PROJECT_ROOT / "A2_FashionDataset" / "external_data" / "places365" / "images"

#: Share of scene CATEGORIES held out for evaluation. 20% of 365 is 73 scene
#: kinds the training loop never composites onto.
DEFAULT_TEST_FRACTION = 0.20

#: Seed for the category split. Fixed, so train and eval banks built in separate
#: processes agree without passing anything between them.
SPLIT_SEED = 20260906

#: Fraction of the source photograph a crop covers, as a linear scale. Below
#: 0.35 the window is small enough to land on a flat wall, which reproduces the
#: procedural "solid" family rather than adding to it.
DEFAULT_CROP_SCALE = (0.35, 1.00)


def _categories():
    if not PLACES_ROOT.is_dir():
        raise FileNotFoundError(
            "Places365 images not found at {}. Expected "
            "A2_FashionDataset/external_data/places365/images/<category>/*.jpg"
            .format(PLACES_ROOT))
    return sorted(p.name for p in PLACES_ROOT.iterdir() if p.is_dir())


def category_split(test_fraction=DEFAULT_TEST_FRACTION, seed=SPLIT_SEED):
    """Partition the scene categories into train and test sets.

    Deterministic given the corpus and the seed, so no file needs to be read or
    kept in sync. Returns ``(train_categories, test_categories)``, both sorted.
    """
    categories = _categories()
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(categories))
    n_test = int(round(len(categories) * test_fraction))
    test = sorted(categories[i] for i in order[:n_test])
    train = sorted(categories[i] for i in order[n_test:])
    assert not set(train) & set(test), "category split overlaps"
    return train, test


def list_category_images(categories):
    """Every jpg under the given categories, sorted for reproducibility."""
    paths = []
    for category in categories:
        paths.extend(sorted((PLACES_ROOT / category).glob("*.jpg")))
    return paths


def _crop_to_shape(image, shape, generator, crop_scale=DEFAULT_CROP_SCALE):
    """A random window at the target aspect ratio, then one downscale."""
    height, width = shape[0], shape[1]
    target_aspect = width / height
    source_width, source_height = image.size

    # The largest window of the target aspect that fits inside the source.
    if source_width / source_height > target_aspect:
        max_height = source_height
        max_width = max_height * target_aspect
    else:
        max_width = source_width
        max_height = max_width / target_aspect

    scale = generator.uniform(*crop_scale)
    crop_width = max(width, int(max_width * scale))
    crop_height = max(height, int(max_height * scale))
    crop_width = min(crop_width, source_width)
    crop_height = min(crop_height, source_height)

    left = int(generator.integers(0, max(1, source_width - crop_width + 1)))
    top = int(generator.integers(0, max(1, source_height - crop_height + 1)))

    window = image.crop((left, top, left + crop_width, top + crop_height))
    return np.asarray(window.resize((width, height), Image.BILINEAR), dtype=np.uint8)


def load_background_bank(count=8000, shape=(80, 60, 3), split="train",
                         seed=42, test_fraction=DEFAULT_TEST_FRACTION,
                         crop_scale=DEFAULT_CROP_SCALE, verbose=False):
    """Render ``count`` photographic backgrounds at ``shape``.

    ``split`` is ``"train"`` or ``"test"``; they draw from disjoint sets of scene
    categories. Photographs are sampled without replacement while any remain,
    then with replacement using a fresh crop, so ``count`` above the corpus size
    still yields distinct frames.
    """
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test', got {!r}".format(split))

    train_categories, test_categories = category_split(test_fraction)
    categories = train_categories if split == "train" else test_categories
    paths = list_category_images(categories)
    if not paths:
        raise FileNotFoundError("No Places365 images for split {!r}".format(split))

    generator = np.random.default_rng(seed)
    order = generator.permutation(len(paths))
    if count > len(paths):
        extra = generator.integers(0, len(paths), size=count - len(paths))
        order = np.concatenate([order, extra])
    else:
        order = order[:count]

    bank = np.zeros((count,) + tuple(shape), dtype=np.uint8)
    for index, source in enumerate(order):
        with Image.open(paths[source]) as opened:
            frame = opened.convert("RGB")
            bank[index] = _crop_to_shape(frame, shape, generator, crop_scale)
        if verbose and (index + 1) % 2000 == 0:
            print("  {:>6,}/{:,} backgrounds".format(index + 1, count), flush=True)

    return bank


def make_mixed_bank(procedural, photographic, photographic_share=0.70, seed=42):
    """Interleave a procedural bank with a photographic one.

    Both kinds are kept because they model different halves of the serve path.
    ``outputs/task4_ingestion_fallback.csv`` measures that 38% of uploads arrive
    with the backdrop intact - which is what a Places365 photograph stands for -
    while the other 62% arrive segmented onto a flat field, which is what the
    procedural solid and gradient families stand for. Training on only one
    teaches invariance to only one.
    """
    if not 0.0 <= photographic_share <= 1.0:
        raise ValueError("photographic_share must lie in [0, 1]")

    total = len(procedural) + len(photographic)
    n_photo = int(round(total * photographic_share))
    n_proc = total - n_photo

    generator = np.random.default_rng(seed)
    photo_idx = generator.integers(0, len(photographic), size=n_photo)
    proc_idx = generator.integers(0, len(procedural), size=n_proc)

    bank = np.concatenate([photographic[photo_idx], procedural[proc_idx]], axis=0)
    return bank[generator.permutation(len(bank))]


def write_split_manifest(path=None, test_fraction=DEFAULT_TEST_FRACTION):
    """Record the category split, so the write-up can quote it."""
    import json

    train, test = category_split(test_fraction)
    manifest = {
        "corpus": str(PLACES_ROOT.relative_to(PROJECT_ROOT)),
        "split_by": "scene category",
        "split_seed": SPLIT_SEED,
        "test_fraction": test_fraction,
        "n_train_categories": len(train),
        "n_test_categories": len(test),
        "n_train_images": len(list_category_images(train)),
        "n_test_images": len(list_category_images(test)),
        "train_categories": train,
        "test_categories": test,
        "supersedes": (
            "A2_FashionDataset/external_data/places365/{train,test}_backgrounds.txt, "
            "which split "
            "80/20 over files and left all 365 categories on both sides."),
    }
    path = path or (PROJECT_ROOT / "outputs" / "places365_category_split.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
