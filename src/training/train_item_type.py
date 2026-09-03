"""Task 1 - train the item-type CNN outside the notebook.

Why this exists
---------------
``notebooks/02_task1_item_type.ipynb`` ranks its candidate configurations on
``ablation_max_epochs = 3``. That budget is fine for separating architectures,
but it systematically penalises *regularisers*: augmentation slows convergence
before it improves generalisation, so a three-epoch comparison scores it while
it is still behind. ``CNN_aug_heavy`` was rejected that way (78.22 vs 80.90
validation weighted-F1).

The published run ``20260830_215803`` reaches 97.80% train accuracy against
87.64% test - a 10.2 point generalisation gap, and 97.15 vs 73.09 on macro-F1.
That model is memorising, most severely on the rare classes, so the lever is
more regularisation evaluated at a budget long enough to show it, not more
capacity.

This module reproduces the notebook's split exactly - same cleaning, same
rare-class drop, same name-or-hash connected-component grouping, same
``StratifiedGroupKFold`` seeds - so anything trained here is directly
comparable with the published numbers. It writes the same checkpoint contract
that ``src/models/item_type_classifier.py`` reads, so a model trained here
loads in the notebook and in the FastAPI service without changes.

Usage
-----
    python -m src.training.train_item_type --recipe cutout --epochs 40
    python -m src.training.train_item_type --recipe mixup --epochs 40 --tag mix
    python -m src.training.train_item_type --evaluate artifacts/task1/task1_cnn.pt

Training runs on tensor batches held in RAM rather than a DataLoader with PIL
transforms: the images are 60x80, so per-sample Python overhead dominated the
step time. Everything is batched on the GPU-shaped path even on CPU.

Known limitation - epoch selection for the "webphoto" recipe
------------------------------------------------------------
Early stopping and best-epoch selection score validation on **clean catalogue
tiles**, which is the right criterion for every recipe here except "webphoto".
That recipe exists to survive photographs, so the epoch that is best on clean
tiles is not necessarily the epoch that is best at the job. In practice the
OneCycle schedule anneals the learning rate to near zero, so the best clean epoch
lands at or near the end of the run and the criterion rarely binds - check
``checkpoint["val_metrics"]["epoch"]`` against the epoch count before trusting
that. Selecting on a held-out *shifted* validation set would be the correct fix
and needs the set built once up front, not re-synthesised every epoch.
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.user_image import catalogue_alpha
from src.models.item_type_classifier import ItemTypeCNN, choose_device

# ----------------------------------------------------------------- constants
# These mirror notebooks/02_task1_item_type.ipynb CELL 3. Changing any of them
# breaks comparability with the published run.
RANDOM_STATE = 42
MIN_CLASS_SIZE = 10
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15
GROUP_COLUMN = "productDisplayName"
TARGET = "articleType"
IMAGE_SIZE_PIL = (60, 80)

PROCESSED = PROJECT_ROOT / "A2_FashionDataset" / "processed"
CLEAN_METADATA = PROCESSED / "clean_train_metadata.csv"
IMAGE_HASHES = PROCESSED / "image_hashes.csv"
IMAGE_CACHE = PROCESSED / "image_cache_task1_60x80.npy"
IMAGE_CACHE_IDS = PROCESSED / "image_cache_task1_60x80_ids.npy"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "task1"


# ------------------------------------------------------------------- recipes
# Each recipe is a full training configuration. "baseline" reproduces the
# published run's settings so the harness itself can be validated before any
# comparison is trusted; the others add regularisation on top of it.
RECIPES = {
    "baseline": {
        "dropout": 0.2, "weight_decay": 1e-4, "label_smoothing": 0.05,
        "flip": True, "affine": 0.0, "erase": 0.0, "mixup": 0.0, "cutmix": 0.0,
    },
    # The domain recipe. Every other recipe here fights the ~10 point
    # train/test generalisation gap on catalogue tiles; this one fights the ~62
    # point gap between a catalogue tile and a photograph, which is the larger
    # number by far. Background randomisation is the load-bearing part - see
    # random_background() - with wide geometry and photometric jitter alongside,
    # because a real upload is off-angle and un-colour-managed as well as
    # un-white. Regularisation is kept at the "cutout" level so the two effects
    # stay separable when the results are compared.
    "webphoto": {
        "dropout": 0.3, "weight_decay": 3e-4, "label_smoothing": 0.1,
        "flip": True, "affine": 2.5, "erase": 0.5, "mixup": 0.0, "cutmix": 0.0,
        "background": 0.6, "photometric": 0.5, "aspect": 0.5,
    },
    # More of the same, for the photograph path only. Worth trying because the
    # serving router decouples the two objectives: catalogue tiles are answered
    # by task1_cnn.pt, so this checkpoint is never asked about them and its clean
    # accuracy is no longer a cost worth protecting. Optimise it purely for
    # photographs. Regularisers are held at the "webphoto" level so the only
    # variable is the amount of domain randomisation.
    "webphoto_strong": {
        "dropout": 0.35, "weight_decay": 3e-4, "label_smoothing": 0.1,
        "flip": True, "affine": 3.5, "erase": 0.5, "mixup": 0.0, "cutmix": 0.0,
        "background": 0.9, "photometric": 0.75, "aspect": 0.7,
    },
    # Geometric jitter plus cutout. Cutout is the cheapest way to stop a small
    # CNN memorising individual garments: it removes a patch, so the model
    # cannot rely on one distinctive region (a logo, a collar) per training row.
    "cutout": {
        "dropout": 0.3, "weight_decay": 3e-4, "label_smoothing": 0.1,
        "flip": True, "affine": 0.5, "erase": 0.5, "mixup": 0.0, "cutmix": 0.0,
    },
    # Adds mixup/cutmix. Both are label-space regularisers, which is what a
    # 97.15 train / 73.09 test macro-F1 gap calls for: they stop the head
    # driving rare-class logits to saturation on the handful of rows it has.
    "mixup": {
        "dropout": 0.3, "weight_decay": 3e-4, "label_smoothing": 0.1,
        "flip": True, "affine": 0.5, "erase": 0.25, "mixup": 0.2, "cutmix": 0.5,
    },
    # Strongest setting, for the tail. Only worth running if "mixup" wins.
    "strong": {
        "dropout": 0.4, "weight_decay": 5e-4, "label_smoothing": 0.1,
        "flip": True, "affine": 0.75, "erase": 0.5, "mixup": 0.4, "cutmix": 0.5,
    },
    # ---------------------------------------------------------------- prior shift
    # The three recipes below exist because the graded submission is not the
    # population every number above was measured on. Its class mix sits ~45%
    # total-variation away from training, and the classes carrying that mass are
    # the starved ones: Lipstick has 15 training rows and 5.04% of the
    # submission, Foundation and Primer has 12 and 1.85%. 17.4% of the
    # deliverable rides on classes with <=5 test rows.
    #
    # "use_class_weights: false" in best_config.json is not being overturned.
    # Inverse-frequency weighting genuinely failed (66.88 against 82.20), and it
    # failed for a good reason: at 92 classes with an 11-row class it multiplies
    # that class's gradient by ~250 and destabilises training. Balanced softmax
    # reaches the same re-prioritisation by shifting logits rather than scaling
    # gradients, which is stable, and it is scored here under the deployment
    # prior rather than the training one.
    "balanced_softmax": {
        "dropout": 0.3, "weight_decay": 3e-4, "label_smoothing": 0.05,
        "flip": True, "affine": 0.5, "erase": 0.25, "mixup": 0.0, "cutmix": 0.0,
        "loss": "balanced_softmax",
    },
    # Resampling instead of reweighting. beta=0.5 and the x8 cap are both
    # deliberate: full balancing would show each Lipstick tile ~27 times an
    # epoch, which is memorisation rather than learning. affine=1.0 is safe here
    # because the starved classes are small rigid objects on white.
    "tail_sqrt": {
        "dropout": 0.3, "weight_decay": 3e-4, "label_smoothing": 0.1,
        "flip": True, "affine": 1.0, "erase": 0.25, "mixup": 0.0, "cutmix": 0.0,
        "sampler": "sqrt", "sampler_cap": 8.0,
    },
    # Both levers at once, over the label-space regularisers that have never
    # actually been run.
    "tail_mixup": {
        "dropout": 0.3, "weight_decay": 3e-4, "label_smoothing": 0.1,
        "flip": True, "affine": 0.5, "erase": 0.25, "mixup": 0.2, "cutmix": 0.5,
        "loss": "balanced_softmax",
    },
}

ARCHITECTURE = {
    "widths": (16, 32, 64, 128), "head_hidden": 384,
    "pool_grid": (1, 1), "pool_mode": "avgmax",
}


# ---------------------------------------------------------------------- data
def build_group_key(frame, name_column=GROUP_COLUMN, hash_column="image_md5"):
    """Connected components of the rows-share-a-name-or-an-image-hash graph.

    Grouping on the product name alone leaks: the dataset holds groups of
    byte-identical JPEGs whose rows carry different names, so the notebook
    unions on the image hash too. Reproduced here verbatim.
    """
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression
            parent[node], node = root, parent[node]
        return root

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for row_id, name, digest in zip(frame["id"], frame[name_column], frame[hash_column]):
        find(("row", row_id))
        if isinstance(name, str) and name.strip():
            union(("row", row_id), ("name", name))
        union(("row", row_id), ("md5", digest))

    return [str(find(("row", row_id))) for row_id in frame["id"]]


def _grouped_split(df, strat, grp, fraction, seed):
    n_splits = max(2, round(1 / fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep_idx, hold_idx = next(splitter.split(df, strat, grp))
    return df.iloc[keep_idx], df.iloc[hold_idx]


MERGE_MAP = ARTIFACT_DIR / "dropped_class_merge_map.json"
IMAGE_CACHE_EXTRA = PROCESSED / "image_cache_task1_60x80_extra.npy"
IMAGE_CACHE_EXTRA_IDS = PROCESSED / "image_cache_task1_60x80_extra_ids.npy"
TRAIN_IMAGES_DIR = (PROJECT_ROOT / "A2_FashionDataset" / "FashionDataset" / "train"
                    / "images_train")


def _extend_cache(missing_ids, verbose=True):
    """Decode tiles the notebook's cache never covered, and memoise them.

    ``image_cache_task1_60x80.npy`` was built *after* the rare-class floor, so it
    holds 38,491 of the 38,612 rows. Merging a dropped class back in therefore
    asks for tiles that were never cached. There are at most 121 of them, so this
    decodes exactly the missing ones and keeps them in a small sidecar rather
    than rebuilding the 554 MB cache.

    The decode path is deliberately the same one the cache itself used -
    ``load_image_array`` at ``IMAGE_SIZE_PIL`` - so an extended row is
    byte-identical to what a full rebuild would have produced.
    """
    from src.models.item_type_classifier import load_image_array

    missing_ids = np.asarray(sorted(int(i) for i in missing_ids))
    if IMAGE_CACHE_EXTRA.exists() and IMAGE_CACHE_EXTRA_IDS.exists():
        stored_ids = np.load(IMAGE_CACHE_EXTRA_IDS)
        if set(missing_ids).issubset(set(int(i) for i in stored_ids)):
            return np.load(IMAGE_CACHE_EXTRA), stored_ids

    if verbose:
        print(f"decoding {len(missing_ids)} tiles absent from the cache")
    tiles = np.stack([
        load_image_array(TRAIN_IMAGES_DIR / f"{image_id}.jpg", IMAGE_SIZE_PIL)
        for image_id in missing_ids
    ]).astype(np.uint8)
    np.save(IMAGE_CACHE_EXTRA, tiles)
    np.save(IMAGE_CACHE_EXTRA_IDS, missing_ids)
    return tiles, missing_ids


def load_merge_map(path=MERGE_MAP):
    """``{dropped class: kept class}`` from the committed, justified mapping."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: entry["into"] for name, entry in payload["merges"].items()}


def load_splits(verbose=True, min_class_size=MIN_CLASS_SIZE, merge_dropped=False):
    """Return ``(train_df, val_df, test_df, class_names, images)``.

    ``images`` is the uint8 NHWC cache built by the notebook. The split is
    asserted leakage-free on groups, ids and image hashes, exactly as notebook
    cell 10 does, because a silent split change would invalidate every
    comparison this module is for.

    ``merge_dropped`` relabels the rows of a dropped class to their nearest kept
    class instead of deleting them, using the justified mapping in
    ``artifacts/task1/dropped_class_merge_map.json``, and adds them **to train
    only**. Lipstick goes 11 -> 19 training rows and Foundation and Primer
    9 -> 16, which is the largest increase in tail training data available for
    this task, and it lands on two classes carrying 6.9% of the graded set.

    Train-only is not a detail, it is the whole experimental design. Merging
    before the split - which is how this was first written - changes the class
    counts, which changes the strata, which makes ``StratifiedGroupKFold``
    deal the groups differently. The result was a split sharing 3,675 rows
    between the merged run's train set and the unmerged run's test set, and a
    checkpoint that scored a fictional 91.33 weighted-F1. Assigning every merged
    row to train instead keeps the partition of the original 38,491 rows
    byte-identical, so a merged run and an unmerged one are scored on exactly
    the same held-out data and the difference is attributable to the training
    data alone.
    """
    data = pd.read_csv(CLEAN_METADATA)

    mapping = load_merge_map() if merge_dropped else {}
    extra = data[data[TARGET].isin(mapping)].copy() if mapping else None

    counts = data[TARGET].value_counts()
    rare = counts[counts < min_class_size].index.tolist()
    model_df = data[~data[TARGET].isin(rare)].copy()

    hashes = pd.read_csv(IMAGE_HASHES).set_index("id")["md5"]
    model_df["image_md5"] = model_df["id"].map(hashes)
    assert model_df["image_md5"].notna().all(), "image_hashes.csv misses rows"
    model_df["split_group"] = build_group_key(model_df)

    # LabelEncoder order is sorted-unique, which is what the checkpoint stores.
    class_names = sorted(model_df[TARGET].unique())
    model_df["label"] = model_df[TARGET].map({n: i for i, n in enumerate(class_names)})

    trainval_df, test_df = _grouped_split(
        model_df, model_df["label"], model_df["split_group"], TEST_FRACTION, RANDOM_STATE
    )
    val_adj = VAL_FRACTION / (1 - TEST_FRACTION)
    train_df, val_df = _grouped_split(
        trainval_df, trainval_df["label"], trainval_df["split_group"], val_adj, RANDOM_STATE
    )

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        frames = {"train": train_df, "val": val_df, "test": test_df}
        for column in ("split_group", "image_md5", "id"):
            overlap = set(frames[a][column]) & set(frames[b][column])
            assert not overlap, f"leakage between {a} and {b} on {column}"

    merged_rows = 0
    if extra is not None and len(extra):
        # Relabel, then append to train only. Rows whose target class did not
        # survive the floor are still dropped - a merge cannot resurrect a class
        # that does not exist.
        extra[TARGET] = extra[TARGET].map(mapping)
        extra = extra[extra[TARGET].isin(set(class_names))].copy()
        extra["image_md5"] = extra["id"].map(hashes)
        extra["split_group"] = extra["id"].astype(str) + "_merged"
        extra["label"] = extra[TARGET].map({n: i for i, n in enumerate(class_names)})
        # Never let a merged row land in train when its own group is already in
        # val or test: that would be leakage of the ordinary kind.
        blocked = set(val_df["image_md5"]) | set(test_df["image_md5"])
        extra = extra[~extra["image_md5"].isin(blocked)]
        merged_rows = len(extra)
        train_df = pd.concat([train_df, extra], ignore_index=True)

    cache_ids = np.load(IMAGE_CACHE_IDS)
    position = {int(i): p for p, i in enumerate(cache_ids)}

    # Memory-mapped, not read into RAM. It is only ever fancy-indexed into the
    # three splits, and holding all 554 MB resident on a 16 GB machine alongside
    # the split tensors was enough to push a run into paging - which cost far
    # more than the mmap does, dropping it from 20 cores to under one.
    images = np.load(IMAGE_CACHE, mmap_mode="r")

    wanted = pd.concat([train_df["id"], val_df["id"], test_df["id"]])
    missing = sorted(set(wanted) - set(position))
    if missing:
        extra, extra_ids = _extend_cache(missing, verbose=verbose)
        # Concatenating drops the mmap, so only do it when a merge actually
        # pulled uncached rows in - the default path stays memory-mapped.
        base_rows = images.shape[0]
        images = np.concatenate([np.asarray(images), extra], axis=0)
        position.update({int(i): base_rows + offset for offset, i in enumerate(extra_ids)})

    for frame in (train_df, val_df, test_df):
        mapped = frame["id"].map(position)
        assert mapped.notna().all(), "image cache does not cover every row"
        frame["cache_position"] = mapped.astype(int)

    if verbose:
        print(f"classes {len(class_names)} | train {len(train_df)} "
              f"val {len(val_df)} test {len(test_df)}")
        if merged_rows:
            print(f"merged {merged_rows} rows from {len(mapping)} dropped classes "
                  f"into their nearest kept sibling, train only "
                  f"(val/test partition unchanged)")
        if min_class_size != MIN_CLASS_SIZE:
            print(f"min_class_size={min_class_size} (default {MIN_CLASS_SIZE})")
    return train_df, val_df, test_df, class_names, images


def starved_classes(frame, max_test_rows=5, target=TARGET):
    """Classes whose held-out support is too small to measure anything with."""
    counts = frame[target].value_counts()
    return sorted(counts[counts <= max_test_rows].index)


def load_splits_cv_starved(fold, folds=3, classes=None, verbose=True, **kwargs):
    """The adopted split, with the starved classes re-partitioned for fold ``fold``.

    Why this exists
    ---------------
    ``classification_report.txt`` records F1 = 1.00 for Lipstick, and Lipstick is
    5.04% of the graded submission. That 1.00 comes from **two** test rows. The
    same is true of Kajal and Eyeliner, Nail Polish, Foundation and Primer and
    Lip Liner: together, classes with <=5 test rows carry 17.4% of the
    deliverable on estimates that are coin flips.

    Importance-weighting cannot fix this - it makes it worse, because it
    multiplies those 1.00 recalls up to 21% of the metric. The only fix is more
    held-out rows for those classes, and the only way to get them is to hold out
    a different slice and retrain.

    So: leave the adopted split alone for every other class, and for the starved
    ones rotate all of their rows through ``folds`` held-out partitions. Pooling
    the out-of-fold predictions gives Lipstick 15 evaluation rows instead of 2.

    Why this is not leakage, and why the bias is negligible
    ------------------------------------------------------
    Every starved class is one row per ``split_group`` and one row per
    ``image_md5`` - verified, not assumed, and the standard leakage assertions
    still run and still pass. So no group spans the fold boundary.

    At ``folds=3`` a fold's model sees ~10 of Lipstick's 15 rows where the
    deployed model saw 11, making the out-of-fold estimate very slightly
    *pessimistic*. At ``folds=5`` it sees 12, making it very slightly optimistic.
    Either way the bias is one training row, against a current estimate drawn
    from two evaluation rows.
    """
    train_df, val_df, test_df, class_names, images = load_splits(verbose=False, **kwargs)
    if classes is None:
        classes = starved_classes(test_df)
    classes = [c for c in classes if c in set(class_names)]

    pooled = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Rotate whole groups, never rows. Most starved classes are one row per
    # product, but not all of them - Cufflinks and Bracelet carry multi-row
    # product groups, and splitting one across the fold boundary would put the
    # same product in train and test. A group that also contains a non-starved
    # class is left exactly where the adopted split put it, so rotating the tail
    # cannot disturb the head.
    class_set = set(classes)
    group_classes = pooled.groupby("split_group")[TARGET].agg(lambda v: set(v))
    rotatable = sorted(g for g, names in group_classes.items() if names <= class_set)
    rotatable_set = set(rotatable)

    pinned = pooled[pooled[TARGET].isin(class_set) & ~pooled["split_group"].isin(rotatable_set)]
    moving = pooled[pooled["split_group"].isin(rotatable_set)]

    # Deal groups of each class round-robin, so every fold gets a share of every
    # class rather than a fold accidentally taking all of one.
    assignment = {}
    for name in classes:
        groups = sorted(moving[moving[TARGET] == name]["split_group"].unique())
        for offset, group in enumerate(groups):
            assignment[group] = offset % folds

    held = moving["split_group"].map(assignment) == (fold % folds)
    stays_in_train = moving[~held]
    goes_to_test = moving[held]

    untouched_test = test_df[~test_df["split_group"].isin(rotatable_set)]
    untouched_train = train_df[~train_df["split_group"].isin(rotatable_set)]
    untouched_val = val_df[~val_df["split_group"].isin(rotatable_set)]

    new_test = pd.concat([untouched_test, goes_to_test], ignore_index=True)
    new_train = pd.concat([untouched_train, stays_in_train], ignore_index=True)
    new_val = untouched_val.reset_index(drop=True)
    del pinned

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        frames = {"train": new_train, "val": new_val, "test": new_test}
        for column in ("split_group", "image_md5", "id"):
            overlap = set(frames[a][column]) & set(frames[b][column])
            assert not overlap, f"fold {fold}: leakage between {a} and {b} on {column}"

    for frame in (new_train, new_val, new_test):
        mapped = frame["id"].map(
            {int(i): p for p, i in enumerate(np.load(IMAGE_CACHE_IDS))})
        if mapped.notna().all():
            frame["cache_position"] = mapped.astype(int)

    if verbose:
        rotated = int(goes_to_test[TARGET].isin(class_set).sum())
        print(f"fold {fold}/{folds}: {len(rotatable)} rotatable groups across "
              f"{len(classes)} starved classes, {rotated} rows held out | "
              f"train {len(new_train)} val {len(new_val)} test {len(new_test)}")
    return new_train, new_val, new_test, class_names, images


def split_tensors(frame, images, device):
    """uint8 NHWC cache rows -> (float NCHW on device, int64 labels)."""
    arrays = images[frame["cache_position"].to_numpy()]
    x = torch.from_numpy(np.ascontiguousarray(arrays.transpose(0, 3, 1, 2)))
    y = torch.from_numpy(frame["label"].to_numpy().astype(np.int64))
    return x.to(device), y.to(device)


def split_alpha(frame, images, device):
    """Subject mattes for a split, aligned row-for-row with ``split_tensors``.

    Catalogue tiles are cutouts on white, so the matte is a brightness threshold
    rather than a segmentation - see ``catalogue_alpha``. Returned as ``(N,1,H,W)``
    so it broadcasts against the image tensor, plus a per-row flag marking the
    mattes that are trustworthy enough to composite with.
    """
    arrays = images[frame["cache_position"].to_numpy()]
    alpha, usable = catalogue_alpha(arrays)
    return (torch.from_numpy(alpha[:, None]).to(device),
            torch.from_numpy(usable).to(device))


# -------------------------------------------------------------- augmentation
def normalise(x_uint8, mean, std):
    """uint8 NCHW -> normalised float, matching preprocess_arrays()."""
    x = x_uint8.float().div_(255.0)
    return (x - mean) / std


def random_affine(x, strength, generator):
    """Batched translate/scale/rotate via a sampled affine grid.

    Catalogue photos are centred and tightly framed, so the jitter stays small:
    +/-6 degrees, +/-8% translation, +/-8% scale at strength 1.0. Edges are
    replicated rather than zero-filled, because these images sit on a white
    background and a black border would be a feature the model could learn.
    """
    n = x.shape[0]
    device = x.device
    degrees = 6.0 * strength * math.pi / 180.0
    translate = 0.08 * strength
    scale_range = 0.08 * strength

    def rand(lo, hi):
        return torch.empty(n, device=device).uniform_(lo, hi, generator=generator)

    angle = rand(-degrees, degrees)
    scale = 1.0 + rand(-scale_range, scale_range)
    tx, ty = rand(-translate, translate), rand(-translate, translate)

    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.zeros(n, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty

    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                         align_corners=False)


def random_erase(x, probability, generator, max_area=0.25):
    """Cutout: blank a rectangle per selected sample.

    The erased patch is set to 0 in normalised space, i.e. the dataset mean
    pixel, so it reads as "no information" rather than as a black box.
    """
    n, _, h, w = x.shape
    device = x.device
    selected = torch.rand(n, device=device, generator=generator) < probability
    if not selected.any():
        return x
    x = x.clone()
    for index in selected.nonzero(as_tuple=True)[0].tolist():
        area = float(torch.empty(1).uniform_(0.02, max_area, generator=generator))
        ratio = float(torch.empty(1).uniform_(0.4, 2.5, generator=generator))
        eh = min(h, max(1, round(math.sqrt(area * h * w * ratio))))
        ew = min(w, max(1, round(math.sqrt(area * h * w / ratio))))
        top = int(torch.randint(0, h - eh + 1, (1,), generator=generator))
        left = int(torch.randint(0, w - ew + 1, (1,), generator=generator))
        x[index, :, top:top + eh, left:left + ew] = 0.0
    return x


def random_background(x_uint8, alpha, usable, probability, generator):
    """Composite the garment onto a random background. The important one.

    The shipped model learned "garment on white", not "garment": on held-out rows
    composited onto a textured background its accuracy falls from 87.92 to 25.80
    and its macro-F1 from 71.98 to 5.19. Inference-time segmentation
    (``src/data/user_image.py``) recovers a usable matte on roughly half of real
    photographs, so something has to carry the other half, and only training can.

    Four families - solid, linear gradient, band-limited noise, and a blurred
    other image from the same batch - all generated from the batch itself, so no
    external image data is used and the "train your own algorithms" constraint
    holds. They are deliberately **disjoint** from the checkerboards, stripes and
    blob fields ``src/evaluation/ood_benchmark.py`` scores on: sharing families
    would measure the model recognising its own augmentation.

    ``usable`` gates out rows whose matte is untrustworthy - a product
    photographed edge to edge, or one that thresholds to almost nothing - because
    compositing those produces a garbage image with a confident label.
    """
    n, channels, height, width = x_uint8.shape
    device = x_uint8.device
    selected = (torch.rand(n, device=device, generator=generator) < probability) & usable
    if not selected.any():
        return x_uint8

    index = selected.nonzero(as_tuple=True)[0]
    count = index.shape[0]
    x = x_uint8.float()

    def rand(*shape):
        return torch.rand(*shape, device=device, generator=generator)

    solid = rand(count, channels, 1, 1) * 255.0
    solid = solid.expand(-1, -1, height, width)

    # Gradient between two colours along a random axis.
    ramp_y = torch.linspace(0, 1, height, device=device).view(1, 1, height, 1)
    ramp_x = torch.linspace(0, 1, width, device=device).view(1, 1, 1, width)
    weight = rand(count, 1, 1, 1)
    ramp = weight * ramp_y + (1 - weight) * ramp_x
    start, end = rand(count, channels, 1, 1) * 255.0, rand(count, channels, 1, 1) * 255.0
    gradient = start + (end - start) * ramp

    # Band-limited noise: random at 1/8 scale, then smoothly upsampled.
    coarse = rand(count, channels, max(2, height // 8), max(2, width // 8)) * 255.0
    noise = F.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False)

    # Another image from the batch, blurred hard enough to be texture rather than
    # a second garment the model could try to classify.
    donor = x[torch.randint(0, n, (count,), device=device, generator=generator)]
    blurred = F.interpolate(F.avg_pool2d(donor, 4), size=(height, width),
                            mode="bilinear", align_corners=False)

    family = torch.randint(0, 4, (count, 1, 1, 1), device=device, generator=generator)
    background = torch.where(family == 0, solid,
                  torch.where(family == 1, gradient,
                   torch.where(family == 2, noise, blurred)))

    # Feather the matte: a hard 0/1 cut leaves a one-pixel step the network can
    # learn as "this was composited".
    matte = alpha[index].float()
    matte = F.avg_pool2d(matte, 3, stride=1, padding=1).clamp(0, 1)

    out = x.clone()
    out[index] = matte * x[index] + (1.0 - matte) * background
    return out.clamp_(0, 255).to(x_uint8.dtype)


def random_photometric(x_uint8, strength, generator):
    """Brightness, contrast, saturation and gamma jitter, in 0-255 space.

    A phone photo is not colour-managed the way a catalogue shot is. Applied
    before normalisation so the statistics the model sees shift the way a real
    upload's do.
    """
    n = x_uint8.shape[0]
    device = x_uint8.device
    x = x_uint8.float()

    def jitter(scale):
        return 1.0 + (torch.rand(n, 1, 1, 1, device=device, generator=generator) - 0.5) \
            * 2.0 * scale * strength

    x = x * jitter(0.30)
    grey = x.mean(dim=1, keepdim=True)
    x = grey + (x - grey) * jitter(0.35)                       # saturation
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = mean + (x - mean) * jitter(0.30)                       # contrast
    gamma = jitter(0.25).clamp(0.5, 2.0)
    x = 255.0 * (x.clamp(0, 255) / 255.0).pow(gamma)
    return x.clamp_(0, 255).to(x_uint8.dtype)


def random_aspect(x, probability, generator, low=0.7, high=1.7):
    """Squash to a random aspect ratio and back, as the serving resize does.

    A non-3:4 upload reaching ``load_image_array`` is resized straight to 60x80,
    so the garment arrives stretched. Worth about one accuracy point on its own -
    small next to the background, but free. Applied per batch rather than per
    sample: one interpolate call instead of N, for the same effect over an epoch.
    """
    if float(torch.rand(1, generator=generator)) >= probability:
        return x
    height, width = x.shape[2], x.shape[3]
    ratio = float(torch.empty(1).uniform_(low, high, generator=generator))
    squashed = F.interpolate(x, size=(max(8, int(height / ratio)), max(6, int(width * ratio))),
                             mode="bilinear", align_corners=False)
    return F.interpolate(squashed, size=(height, width), mode="bilinear", align_corners=False)


def mix_batch(x, y, num_classes, recipe, generator, smoothing):
    """Apply mixup or cutmix, returning soft targets.

    Returns ``(x, soft_targets)``. Label smoothing is folded into the soft
    target here so the loss is a single cross-entropy against a distribution,
    whether or not a mix was applied.
    """
    n = x.shape[0]
    device = x.device
    target = torch.full((n, num_classes), smoothing / num_classes, device=device)
    target.scatter_add_(1, y[:, None],
                        torch.full((n, 1), 1.0 - smoothing, device=device))

    use_mixup = recipe["mixup"] > 0
    use_cutmix = recipe["cutmix"] > 0
    if not (use_mixup or use_cutmix):
        return x, target
    # One of the two per batch, so their probabilities stay interpretable.
    if use_mixup and use_cutmix:
        pick_cutmix = float(torch.rand(1, generator=generator)) < 0.5
    else:
        pick_cutmix = use_cutmix
    alpha = recipe["cutmix"] if pick_cutmix else recipe["mixup"]
    if float(torch.rand(1, generator=generator)) >= 0.5:
        return x, target  # applied to half the batches

    lam = float(np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,),
                generator=generator))).beta(alpha, alpha))
    perm = torch.randperm(n, device=device, generator=generator)

    if pick_cutmix:
        _, _, h, w = x.shape
        cut_h, cut_w = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        if cut_h > 0 and cut_w > 0:
            cy = int(torch.randint(0, h, (1,), generator=generator))
            cx = int(torch.randint(0, w, (1,), generator=generator))
            y1, y2 = max(0, cy - cut_h // 2), min(h, cy + cut_h // 2)
            x1, x2 = max(0, cx - cut_w // 2), min(w, cx + cut_w // 2)
            x = x.clone()
            x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
            lam = 1 - ((y2 - y1) * (x2 - x1) / (h * w))
    else:
        x = lam * x + (1 - lam) * x[perm]

    return x, lam * target + (1 - lam) * target[perm]


# ------------------------------------------------------------------ training
@torch.no_grad()
def predict_split(model, x_uint8, mean, std, batch_size=512, tta=False):
    model.eval()
    outputs = []
    for start in range(0, x_uint8.shape[0], batch_size):
        chunk = normalise(x_uint8[start:start + batch_size], mean, std)
        probabilities = F.softmax(model(chunk).float(), dim=1)
        if tta:
            mirrored = F.softmax(model(torch.flip(chunk, dims=[3])).float(), dim=1)
            probabilities = (probabilities + mirrored) / 2
        outputs.append(probabilities)
    return torch.cat(outputs)


def score(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        "balanced_acc": balanced_accuracy_score(y_true, y_pred) * 100,
    }


def topk_accuracy(probabilities, y_true, k):
    top = probabilities.topk(k, dim=1).indices.cpu().numpy()
    return float((top == y_true[:, None]).any(axis=1).mean() * 100)


def train(recipe_name="cutout", epochs=40, batch_size=128, learning_rate=2e-3,
          patience=10, seed=RANDOM_STATE, tag=None, verbose=True, widths=None,
          select_on="weighted_f1", min_class_size=MIN_CLASS_SIZE, merge_dropped=False,
          cv_fold=None, cv_folds=3):
    # Recipes written before the domain stages existed carry none of their keys,
    # and must keep behaving exactly as they did when their numbers were recorded.
    recipe = {"background": 0.0, "photometric": 0.0, "aspect": 0.0,
              "loss": "ce", "sampler": None, "sampler_cap": 8.0,
              **RECIPES[recipe_name]}
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = choose_device()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    if cv_fold is None:
        train_df, val_df, test_df, class_names, images = load_splits(
            verbose=verbose, min_class_size=min_class_size, merge_dropped=merge_dropped)
    else:
        train_df, val_df, test_df, class_names, images = load_splits_cv_starved(
            cv_fold, folds=cv_folds, verbose=verbose,
            min_class_size=min_class_size, merge_dropped=merge_dropped)
    num_classes = len(class_names)

    x_train, y_train = split_tensors(train_df, images, device)
    x_val, y_val = split_tensors(val_df, images, device)
    x_test, y_test = split_tensors(test_df, images, device)

    # Mattes only for the train split, and only when a recipe composites - they
    # cost another 132 MB of RAM beside the 554 MB image cache.
    if recipe["background"] > 0:
        alpha_train, usable_train = split_alpha(train_df, images, device)
        if verbose:
            print(f"mattes usable on {int(usable_train.sum())}/{usable_train.numel()} "
                  f"train rows", flush=True)
    else:
        alpha_train = usable_train = None

    # Everything needed is now a tensor; drop the mapped cache so the pages can
    # be reclaimed rather than competing with training for physical memory.
    del images

    # Channel statistics come from the TRAIN split only - computing them over
    # the whole set would leak validation and test pixels into preprocessing.
    #
    # Accumulated in chunks rather than by casting the split to float in one go:
    # 27,493 x 3 x 80 x 60 float32 is a 1.6 GB temporary, which on a 16 GB machine
    # is enough to push the run into paging and drop it to a third of a core.
    total = x_train.shape[0]
    pixel_count = total * x_train.shape[2] * x_train.shape[3]
    channel_sum = torch.zeros(3, dtype=torch.float64, device=device)
    channel_sq = torch.zeros(3, dtype=torch.float64, device=device)
    for start in range(0, total, 2048):
        chunk = x_train[start:start + 2048].double().div_(255.0)
        channel_sum += chunk.sum(dim=(0, 2, 3))
        channel_sq += chunk.pow(2).sum(dim=(0, 2, 3))
        del chunk
    mean_v = channel_sum / pixel_count
    # population variance, matching tensor.std(unbiased=False) closely enough at
    # this sample size that the stored channel_std stays comparable across runs
    var_v = (channel_sq / pixel_count - mean_v.pow(2)).clamp_min(0)
    mean = mean_v.float().view(1, 3, 1, 1)
    std = var_v.sqrt().float().view(1, 3, 1, 1)

    # Width is a parameter because the domain recipes change which way the model
    # errs. The published run overfits (+10.16 train/test gap); "webphoto" at the
    # same width *under*fits (+2.03), which says the capacity that was ample for
    # catalogue tiles is not ample for a randomised-background distribution.
    widths = tuple(widths) if widths else ARCHITECTURE["widths"]
    model = ItemTypeCNN(
        num_classes, widths=widths, dropout=recipe["dropout"],
        head_hidden=ARCHITECTURE["head_hidden"], pool_grid=ARCHITECTURE["pool_grid"],
        pool_mode=ARCHITECTURE["pool_mode"],
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=recipe["weight_decay"])
    steps = max(1, math.ceil(x_train.shape[0] / batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=learning_rate * 3.0, total_steps=epochs * steps,
    )

    y_val_np = y_val.cpu().numpy()
    # Balanced softmax (Ren et al. 2020) adds log(train prior) to the logits
    # during training, so the head stops being rewarded for simply predicting the
    # common classes. It is the train-time twin of the post-hoc tau adjustment in
    # src/models/item_type_classifier.py, and unlike inverse-frequency class
    # weights it re-prioritises without scaling any gradient by ~250x.
    class_counts = np.bincount(y_train.cpu().numpy(), minlength=num_classes)
    log_train_prior = torch.tensor(
        np.log(np.clip(class_counts, 1, None) / class_counts.sum()),
        dtype=torch.float32, device=device).view(1, -1)

    # Tail oversampling. beta=0.5 rather than 1.0, capped, because full balancing
    # would draw each of Lipstick's 11 training tiles ~27 times an epoch - that
    # memorises the tile instead of learning the class. Steps per epoch are
    # unchanged, so this costs no extra runtime.
    sample_weight = None
    if recipe["sampler"]:
        beta = 0.5 if recipe["sampler"] == "sqrt" else 1.0
        per_class = np.power(np.clip(class_counts, 1, None), -beta)
        per_class = np.minimum(per_class,
                               recipe["sampler_cap"] / np.clip(class_counts, 1, None))
        weights = per_class[y_train.cpu().numpy()]
        sample_weight = torch.tensor(weights / weights.sum(), dtype=torch.float32,
                                     device=device)
        if verbose:
            print(f"  sampler={recipe['sampler']} (beta={beta}, cap x{recipe['sampler_cap']:.0f})")

    if select_on not in ("weighted_f1", "macro_f1", "balanced_acc", "accuracy"):
        raise ValueError(f"select_on must be a key of score(), got {select_on!r}")
    best = {select_on: -1.0}
    best_state, epochs_without_gain, history = None, 0, []

    for epoch in range(1, epochs + 1):
        model.train()
        started = time.time()
        if sample_weight is not None:
            order = torch.multinomial(sample_weight, x_train.shape[0],
                                      replacement=True, generator=generator)
        else:
            order = torch.randperm(x_train.shape[0], device=device, generator=generator)
        total_loss = 0.0

        for start in range(0, order.shape[0], batch_size):
            index = order[start:start + batch_size]
            if index.shape[0] < 2:
                continue  # BatchNorm1d in the head needs at least two rows
            # Domain stages run in pixel space, before normalisation: compositing
            # a garment onto a background and jittering its colour are operations
            # on an image, not on a standardised tensor.
            raw = x_train[index]
            if recipe["background"] > 0:
                raw = random_background(raw, alpha_train[index], usable_train[index],
                                        recipe["background"], generator)
            if recipe["photometric"] > 0:
                raw = random_photometric(raw, recipe["photometric"], generator)

            batch = normalise(raw, mean, std)
            if recipe["aspect"] > 0:
                batch = random_aspect(batch, recipe["aspect"], generator)
            if recipe["flip"]:
                flip_mask = torch.rand(batch.shape[0], device=device,
                                       generator=generator) < 0.5
                batch[flip_mask] = torch.flip(batch[flip_mask], dims=[3])
            if recipe["affine"] > 0:
                batch = random_affine(batch, recipe["affine"], generator)
            if recipe["erase"] > 0:
                batch = random_erase(batch, recipe["erase"], generator)
            batch, target = mix_batch(batch, y_train[index], num_classes, recipe,
                                      generator, recipe["label_smoothing"])

            optimiser.zero_grad(set_to_none=True)
            logits = model(batch)
            if recipe["loss"] == "balanced_softmax":
                logits = logits + log_train_prior
            loss = torch.sum(-target * F.log_softmax(logits, dim=1), dim=1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            scheduler.step()
            total_loss += float(loss.detach()) * index.shape[0]

        probabilities = predict_split(model, x_val, mean, std)
        metrics = score(y_val_np, probabilities.argmax(1).cpu().numpy())
        history.append({"epoch": epoch, "loss": total_loss / order.shape[0], **metrics})

        if verbose:
            print(f"  epoch {epoch:>3}/{epochs}  loss {history[-1]['loss']:.4f}  "
                  f"val acc {metrics['accuracy']:.2f}  wF1 {metrics['weighted_f1']:.2f}  "
                  f"mF1 {metrics['macro_f1']:.2f}  ({time.time() - started:.0f}s)",
                  flush=True)

        if metrics[select_on] > best[select_on]:
            best = {**metrics, "epoch": epoch}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} "
                          f"(best epoch {best['epoch']})")
                break

    model.load_state_dict(best_state)

    # TTA is a decision, not an assumption: take it only if it helps on
    # VALIDATION, then record the choice so serving matches the reported run.
    val_plain = predict_split(model, x_val, mean, std)
    val_tta = predict_split(model, x_val, mean, std, tta=True)
    plain_f1 = score(y_val_np, val_plain.argmax(1).cpu().numpy())["weighted_f1"]
    tta_f1 = score(y_val_np, val_tta.argmax(1).cpu().numpy())["weighted_f1"]
    use_tta = tta_f1 > plain_f1

    y_test_np = y_test.cpu().numpy()
    test_probabilities = predict_split(model, x_test, mean, std, tta=use_tta)
    test_metrics = score(y_test_np, test_probabilities.argmax(1).cpu().numpy())
    test_metrics["top3_acc"] = topk_accuracy(test_probabilities, y_test_np, 3)
    test_metrics["top5_acc"] = topk_accuracy(test_probabilities, y_test_np, 5)

    train_probabilities = predict_split(model, x_train, mean, std, tta=use_tta)
    train_metrics = score(y_train.cpu().numpy(),
                          train_probabilities.argmax(1).cpu().numpy())

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_name": f"CNN_{recipe_name}" + (f"_{tag}" if tag else ""),
        "num_classes": num_classes,
        "class_names": class_names,
        "channel_mean": mean.flatten().tolist(),
        "channel_std": std.flatten().tolist(),
        "image_size_pil": list(IMAGE_SIZE_PIL),
        "architecture": {
            "widths": list(widths),
            "dropout": recipe["dropout"],
            "head_hidden": ARCHITECTURE["head_hidden"],
            "pool_grid": list(ARCHITECTURE["pool_grid"]),
            "pool_mode": ARCHITECTURE["pool_mode"],
        },
        "config": {"recipe": recipe_name, "epochs": epochs, "batch_size": batch_size,
                   "learning_rate": learning_rate, "seed": seed, **recipe},
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "val_metrics": best,
        "history": history,
        "run_id": run_id,
        "tta": use_tta,
    }
    return checkpoint, (y_test_np, test_probabilities.cpu().numpy(), class_names)


# ---------------------------------------------------------------- evaluation
TAU_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0)


def calibrate_checkpoint(path, tau_grid=TAU_GRID, verbose=True):
    """Fit and store the post-hoc logit adjustment for an already-trained model.

    ``train()`` does not write ``class_log_prior`` / ``logit_adjustment_tau``, but
    the deployed ``task1_cnn.pt`` carries them (tau=0.2). A candidate without them
    would be compared against a calibrated incumbent, and would be deployed on a
    different operating point from the one its metrics describe. This adds them
    in place, so it can be run on checkpoints that already exist.

    Tau is chosen on VALIDATION weighted-F1 - the criterion recorded in the
    deployed checkpoint's ``logit_adjustment_selected_on`` and the one
    ``evaluate_checkpoints`` sweeps on. Choosing it on test would inflate the
    number it is being used to justify.
    """
    from src.models.item_type_classifier import load_item_type_model

    path = Path(path)
    device = choose_device()
    train_df, val_df, _, class_names, images = load_splits(verbose=False)
    model, checkpoint = load_item_type_model(path, device)
    if list(checkpoint["class_names"]) != class_names:
        raise ValueError(f"{path.name}: class order differs from the split")

    x_val, y_val = split_tensors(val_df, images, device)
    y_val = y_val.cpu().numpy()
    mean = torch.tensor(checkpoint["channel_mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(checkpoint["channel_std"], device=device).view(1, 3, 1, 1)

    counts = np.bincount(train_df["label"].to_numpy(), minlength=len(class_names))
    log_prior = np.log(np.maximum(counts, 1) / counts.sum())
    prior = torch.tensor(log_prior, dtype=torch.float32, device=device)

    log_val = torch.log(predict_split(model, x_val, mean, std,
                                      tta=bool(checkpoint.get("tta", False))).clamp_min(1e-12))
    best_tau, best = 0.0, None
    for tau in tau_grid:
        metrics = score(y_val, (log_val - tau * prior).argmax(1).cpu().numpy())
        if best is None or metrics["weighted_f1"] > best["weighted_f1"]:
            best_tau, best = tau, metrics

    checkpoint["class_log_prior"] = log_prior.tolist()
    # The raw counts as well as the normalised prior. Label-shift correction needs
    # to know how much evidence each class's prior rests on, not just its size:
    # Lipstick at 11 rows and Sarees at 110 can carry the same prior mass and
    # deserve very different amounts of trust when the prior is re-estimated.
    checkpoint["class_counts"] = counts.astype(int).tolist()
    checkpoint["logit_adjustment_tau"] = float(best_tau)
    checkpoint["logit_adjustment_selected_on"] = "validation weighted-F1 over tau in [0, 1]"
    torch.save(checkpoint, path)
    if verbose:
        print(f"{path.name}: tau={best_tau:.2f} "
              f"(val wF1 {best['weighted_f1']:.2f}, mF1 {best['macro_f1']:.2f}) written back")
    return best_tau


def pool_starved_cv(checkpoints, folds=3, out=None, verbose=True):
    """Pool out-of-fold predictions for the starved classes into one honest table.

    Each checkpoint must have been trained with ``--cv-fold K`` for a distinct K.
    Its held-out rows for the starved classes are scored, and the results are
    concatenated - so Lipstick's recall comes from 15 rows instead of 2, and
    Cufflinks' from 44 instead of 5.

    The Wilson interval is reported rather than a bare proportion because even
    15 rows is not many, and a point estimate invites exactly the over-reading
    this table exists to stop.
    """
    from src.models.item_type_classifier import load_item_type_model

    device = choose_device()
    records = []
    for fold, path in enumerate(checkpoints):
        path = Path(path)
        train_df, val_df, test_df, class_names, images = load_splits_cv_starved(
            fold, folds=folds, verbose=verbose)
        model, checkpoint = load_item_type_model(path, device)
        mean = torch.tensor(checkpoint["channel_mean"], dtype=torch.float32,
                            device=device).view(1, 3, 1, 1)
        std = torch.tensor(checkpoint["channel_std"], dtype=torch.float32,
                           device=device).view(1, 3, 1, 1)
        x_test, y_test = split_tensors(test_df, images, device)
        probabilities = predict_split(model, x_test, mean, std,
                                      tta=bool(checkpoint.get("tta", False)))
        predicted = probabilities.argmax(1).cpu().numpy()
        truth = y_test.cpu().numpy()
        starved = set(starved_classes(load_splits(verbose=False)[2]))
        for index, name in enumerate(class_names):
            if name not in starved:
                continue
            mask = truth == index
            if not mask.any():
                continue
            records.append({"articleType": name, "fold": fold,
                            "rows": int(mask.sum()),
                            "hits": int((predicted[mask] == index).sum())})

    frame = pd.DataFrame(records)
    if frame.empty:
        raise SystemExit("no starved-class rows scored - were the folds trained?")
    pooled = frame.groupby("articleType")[["rows", "hits"]].sum().reset_index()
    pooled["oof_recall"] = (100 * pooled["hits"] / pooled["rows"]).round(2)

    # Wilson score interval - behaves at small n and near 0 or 1, where the
    # normal approximation does not.
    z = 1.96
    n = pooled["rows"].to_numpy(dtype=float)
    p_hat = pooled["hits"].to_numpy(dtype=float) / n
    denominator = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denominator
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))) / denominator
    pooled["oof_recall_lo"] = (100 * np.clip(centre - margin, 0, 1)).round(2)
    pooled["oof_recall_hi"] = (100 * np.clip(centre + margin, 0, 1)).round(2)

    reference = load_splits(verbose=False)[2]
    test_counts = reference["articleType"].value_counts()
    pooled["test_rows_in_adopted_split"] = pooled["articleType"].map(test_counts).fillna(0).astype(int)

    try:
        from src.evaluation.prior_shift import graded_probabilities

        ids, raw, adjusted, class_names = graded_probabilities(verbose=False)
        graded = pd.Series(np.asarray(class_names)[adjusted.argmax(1)]).value_counts()
        pooled["graded_rows"] = pooled["articleType"].map(graded).fillna(0).astype(int)
        pooled["graded_rows_at_risk"] = (
            pooled["graded_rows"] * (1 - pooled["oof_recall"] / 100)).round(1)
    except Exception as error:
        if verbose:
            print(f"  !! graded counts unavailable ({error})")

    pooled = pooled.sort_values("graded_rows", ascending=False) if "graded_rows" in pooled \
        else pooled.sort_values("rows", ascending=False)
    out = Path(out) if out else PROJECT_ROOT / "outputs" / "evaluation" / "task1_starved_class_cv.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(out, index=False)
    if verbose:
        print(pooled.head(15).to_string(index=False))
        if "graded_rows_at_risk" in pooled:
            print(f"\nestimated graded rows wrong in these classes: "
                  f"{pooled['graded_rows_at_risk'].sum():.0f}")
        print(f"wrote {out}")
    return pooled


def sync_summary(path=ARTIFACT_DIR / "task1_cnn.pt", summary_path=None, verbose=True):
    """Make the checkpoint agree with itself, and the summary agree with the checkpoint.

    Three things were out of sync, and all three are the kind a report would
    quote without noticing:

    ``checkpoint["config"]``
        is the notebook's CONFIG *template*, not the settings the run actually
        used. It records ``use_class_weights: True`` with ``class_weight_scheme:
        "sqrt"`` for a model literally named ``CNN_weights_none_full``, plus
        ``head_hidden: 256``, ``dropout: 0.3``, ``pool_grid: [2, 1]`` and
        ``learning_rate: 0.001`` - every one contradicted by
        ``checkpoint["architecture"]`` in the same file and by
        ``best_config.json``. ``Task1Service.model_card`` reads this dict.
    ``checkpoint["val_metrics"]``
        was empty, so nothing recorded which epoch was selected.
    ``task1_summary.json``
        still headlines the plain-argmax framing (87.64 / 87.13 / 73.09) while
        the checkpoint records the deployed TTA + tau=0.2 operating point
        (87.80 / 87.44 / 72.51). The deployed point is the honest headline,
        because it is what ``predict.py`` actually runs.

    The stale template is kept under ``config_template_superseded`` rather than
    deleted - it is the provenance of every published number to date.
    """
    from src.models.item_type_classifier import load_item_type_model

    path = Path(path)
    device = choose_device()
    train_df, val_df, test_df, class_names, images = load_splits(verbose=False)
    model, checkpoint = load_item_type_model(path, device)

    architecture = dict(checkpoint.get("architecture") or {})
    config = dict(checkpoint.get("config") or {})
    contradictions = {}
    for key in ("widths", "head_hidden", "dropout", "pool_grid", "pool_mode"):
        if key in config and key in architecture and config[key] != architecture[key]:
            contradictions[key] = {"config_said": config[key], "truth": architecture[key]}
            config[key] = architecture[key]
    if config.get("use_class_weights"):
        contradictions["use_class_weights"] = {"config_said": True, "truth": False}
        config["use_class_weights"] = False
        config["class_weight_scheme"] = None

    if contradictions:
        original = dict(checkpoint.get("config") or {})
        checkpoint.setdefault("config_template_superseded", original)
        checkpoint["config"] = config
        checkpoint["config_source"] = (
            "reconciled against checkpoint['architecture'] and best_config.json; "
            "the original notebook CONFIG template is kept under "
            "config_template_superseded")

    mean = torch.tensor(checkpoint["channel_mean"], dtype=torch.float32,
                        device=device).view(1, 3, 1, 1)
    std = torch.tensor(checkpoint["channel_std"], dtype=torch.float32,
                       device=device).view(1, 3, 1, 1)
    tta = bool(checkpoint.get("tta", False))
    x_val, y_val = split_tensors(val_df, images, device)
    val_probabilities = predict_split(model, x_val, mean, std, tta=tta)
    val_metrics = score(y_val.cpu().numpy(), val_probabilities.argmax(1).cpu().numpy())
    existing_val = checkpoint.get("val_metrics") or {}
    checkpoint["val_metrics"] = {**existing_val, **val_metrics}
    torch.save(checkpoint, path)

    summary_path = Path(summary_path) if summary_path else ARTIFACT_DIR / "task1_summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    test_metrics = checkpoint.get("test_metrics") or {}
    summary.update({
        "headline_operating_point": "horizontal-flip TTA + logit adjustment (tau={})".format(
            checkpoint.get("logit_adjustment_tau")),
        "headline_note": ("what predict.py actually runs; the plain-argmax numbers below "
                          "are retained for comparability with the notebook"),
        "test_accuracy_deployed": test_metrics.get("accuracy"),
        "test_weighted_f1_deployed": test_metrics.get("weighted_f1"),
        "test_macro_f1_deployed": test_metrics.get("macro_f1"),
        "test_balanced_acc_deployed": test_metrics.get("balanced_acc"),
        "test_top3_acc_deployed": test_metrics.get("top3_acc"),
        "val_metrics": val_metrics,
        "logit_adjustment_tau": checkpoint.get("logit_adjustment_tau"),
        "config_contradictions_repaired": contradictions or None,
    })
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if verbose:
        print(f"{path.name}: val wF1 {val_metrics['weighted_f1']:.2f} "
              f"mF1 {val_metrics['macro_f1']:.2f} recorded")
        if contradictions:
            print("repaired config contradictions:")
            for key, value in contradictions.items():
                print(f"  {key:<20} config said {value['config_said']!r}, "
                      f"actually {value['truth']!r}")
        else:
            print("config already consistent")
        print(f"wrote {summary_path}")
    return checkpoint, summary


def evaluate_checkpoints(paths, tau_grid=TAU_GRID, deployment_prior=True, out=None):
    """Score checkpoints on the shared split, tau selected on VALIDATION.

    Every candidate gets the same treatment: sweep the post-hoc logit
    adjustment on validation, adopt the tau with the best validation
    weighted-F1 (the criterion the notebook selects on), then report test once
    at that tau. Choosing tau on test would inflate every row here.

    With ``deployment_prior=True`` each row also carries ``dep_accuracy`` and
    ``dep_weighted_f1``: the same predictions re-weighted onto the class prior
    the graded submission actually has, estimated from the unlabelled tiles by
    ``src/evaluation/prior_shift.py``. **Both rankings are printed**, because
    they can disagree - and when they do, the deployment ranking is the one that
    describes the deliverable. A recipe that trades train-prior weighted-F1 for
    tail performance is supposed to look worse on the first ranking; that is the
    point of running it.
    """
    from src.models.item_type_classifier import load_item_type_model

    device = choose_device()
    train_df, val_df, test_df, class_names, images = load_splits(verbose=False)
    x_val, y_val = split_tensors(val_df, images, device)
    x_test, y_test = split_tensors(test_df, images, device)
    y_val, y_test = y_val.cpu().numpy(), y_test.cpu().numpy()

    counts = np.bincount(train_df["label"].to_numpy(), minlength=len(class_names))
    log_prior = torch.tensor(np.log(np.maximum(counts, 1) / counts.sum()),
                             dtype=torch.float32, device=device)

    train_prior, graded_prior, weights = None, None, None
    if deployment_prior:
        try:
            from src.evaluation.prior_shift import (estimate_prior_em,
                                                    graded_probabilities,
                                                    importance_weights, support_shrink)

            train_prior = np.maximum(counts, 1) / counts.sum()
            _, raw, _, _ = graded_probabilities(paths[0], verbose=False)
            graded_prior, _, _, _ = estimate_prior_em(
                raw, train_prior, shrink=support_shrink(counts))
            weights = importance_weights(y_test, graded_prior, train_prior)
        except Exception as error:
            print(f"  !! deployment prior unavailable ({error}); train-prior only")
            deployment_prior = False

    rows = []
    for path in paths:
        path = Path(path)
        model, checkpoint = load_item_type_model(path, device)
        if list(checkpoint["class_names"]) != class_names:
            print(f"  !! {path.name}: class order differs, skipped")
            continue
        mean = torch.tensor(checkpoint["channel_mean"], device=device).view(1, 3, 1, 1)
        std = torch.tensor(checkpoint["channel_std"], device=device).view(1, 3, 1, 1)
        tta = bool(checkpoint.get("tta", False))

        val_p = predict_split(model, x_val, mean, std, tta=tta)
        test_p = predict_split(model, x_test, mean, std, tta=tta)
        log_val, log_test = torch.log(val_p.clamp_min(1e-12)), torch.log(test_p.clamp_min(1e-12))

        best_tau, best_val = 0.0, None
        for tau in tau_grid:
            metrics = score(y_val, (log_val - tau * log_prior).argmax(1).cpu().numpy())
            if best_val is None or metrics["weighted_f1"] > best_val["weighted_f1"]:
                best_tau, best_val = tau, metrics

        adjusted = (log_test - best_tau * log_prior)
        predicted = adjusted.argmax(1).cpu().numpy()
        test_metrics = score(y_test, predicted)
        probabilities = F.softmax(adjusted, dim=1)
        row = {
            "name": checkpoint.get("model_name", path.stem),
            "file": path.name,
            "tta": tta,
            "tau": best_tau,
            "val_weighted_f1": best_val["weighted_f1"],
            "val_macro_f1": best_val["macro_f1"],
            "train_acc": (checkpoint.get("train_metrics") or {}).get("accuracy"),
            "top3": topk_accuracy(probabilities, y_test, 3),
            **test_metrics,
        }
        if deployment_prior:
            labels = np.unique(y_test)
            row["dep_accuracy"] = accuracy_score(y_test, predicted,
                                                 sample_weight=weights) * 100
            row["dep_weighted_f1"] = f1_score(y_test, predicted, average="weighted",
                                              labels=labels, sample_weight=weights,
                                              zero_division=0) * 100
        rows.append(row)

    def show(ordered, criterion, header):
        print(f"\n{header}")
        print(f"{'candidate':<26}{'tau':>5}{'VALwF1':>9}{'TESTacc':>9}{'wF1':>8}"
              f"{'mF1':>8}{'bal':>8}{'top3':>8}{'gap':>8}"
              + (f"{'DEPacc':>9}{'DEPwF1':>9}" if deployment_prior else ""))
        for r in ordered:
            gap = "" if r["train_acc"] is None else f"{r['train_acc'] - r['accuracy']:+.2f}"
            line = (f"{r['name'][:25]:<26}{r['tau']:>5.2f}{r['val_weighted_f1']:>9.2f}"
                    f"{r['accuracy']:>9.2f}{r['weighted_f1']:>8.2f}{r['macro_f1']:>8.2f}"
                    f"{r['balanced_acc']:>8.2f}{r['top3']:>8.2f}{gap:>8}")
            if deployment_prior:
                line += f"{r['dep_accuracy']:>9.2f}{r['dep_weighted_f1']:>9.2f}"
            print(line)
        if ordered:
            print(f"  -> best on {criterion}: {ordered[0]['name']}")

    by_train = sorted(rows, key=lambda r: r["val_weighted_f1"], reverse=True)
    show(by_train, "validation weighted-F1 (train prior)",
         "=== ranked the way every published number was ranked ===")
    if deployment_prior:
        by_dep = sorted(rows, key=lambda r: r["dep_weighted_f1"], reverse=True)
        show(by_dep, "deployment-prior weighted-F1",
             "=== ranked by the population the submission actually has ===")
        if by_train[0]["name"] != by_dep[0]["name"]:
            print(f"\nThe two rankings disagree: {by_train[0]['name']} leads on the "
                  f"train prior, {by_dep[0]['name']} on the deployment prior. The "
                  f"deliverable is the second population.")

    if out:
        frame = pd.DataFrame(rows)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        frame.round(3).to_csv(out, index=False)
        print(f"wrote {out}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", default="cutout", choices=sorted(RECIPES))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--widths", type=int, nargs=4, default=None,
                        metavar=("W1", "W2", "W3", "W4"),
                        help="conv block widths (default 16 32 64 128)")
    parser.add_argument("--out", default=None,
                        help="checkpoint path (default artifacts/task1/candidate_<recipe>.pt)")
    parser.add_argument("--report", action="store_true",
                        help="print a per-class classification report")
    parser.add_argument("--calibrate", nargs="+", metavar="CHECKPOINT",
                        help="fit the post-hoc logit adjustment on validation and "
                             "write class_log_prior / logit_adjustment_tau into "
                             "these checkpoints in place")
    parser.add_argument("--evaluate", nargs="+", metavar="CHECKPOINT",
                        help="score existing checkpoints on the shared split "
                             "instead of training, tau selected on validation")
    parser.add_argument("--cv-fold", type=int, default=None, metavar="K",
                        help="rotate the starved classes into held-out fold K instead "
                             "of using the adopted split, so their recall can be "
                             "estimated from more than two rows")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--merge-dropped", action="store_true",
                        help="relabel rows of dropped classes to their nearest kept "
                             "class instead of deleting them, per "
                             "artifacts/task1/dropped_class_merge_map.json")
    parser.add_argument("--min-class-size", type=int, default=MIN_CLASS_SIZE,
                        help="rare-class floor (default %(default)s)")
    parser.add_argument("--evaluate-out", default=None, metavar="CSV",
                        help="write the --evaluate comparison to this CSV")
    parser.add_argument("--pool-cv", nargs="+", metavar="CHECKPOINT",
                        help="pool out-of-fold starved-class predictions from the "
                             "checkpoints trained with --cv-fold 0..K-1, in order")
    parser.add_argument("--sync-summary", nargs="?", const=str(ARTIFACT_DIR / "task1_cnn.pt"),
                        metavar="CHECKPOINT",
                        help="reconcile a checkpoint's recorded config with the "
                             "architecture it actually has, fill its val_metrics, and "
                             "regenerate task1_summary.json from it")
    parser.add_argument("--select-on", default="weighted_f1",
                        choices=("weighted_f1", "macro_f1", "balanced_acc", "accuracy"),
                        help="early-stopping and best-epoch criterion. The default "
                             "reproduces every published run. Prefer macro_f1 for the "
                             "graded submission: it is prior-invariant, so it targets "
                             "the deployment population without importing the "
                             "uncertainty of an estimated prior into the training loop")
    args = parser.parse_args()

    if args.pool_cv:
        pool_starved_cv(args.pool_cv, folds=args.cv_folds)
        return

    if args.sync_summary:
        sync_summary(args.sync_summary)
        return

    if args.calibrate:
        for path in args.calibrate:
            calibrate_checkpoint(path)
        return

    if args.evaluate:
        evaluate_checkpoints(args.evaluate, out=args.evaluate_out)
        return

    print(f"=== recipe {args.recipe} | {args.epochs} epochs | seed {args.seed} ===")
    print(json.dumps(RECIPES[args.recipe], indent=2))
    started = time.time()
    checkpoint, (y_true, probabilities, class_names) = train(
        recipe_name=args.recipe, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, patience=args.patience,
        seed=args.seed, tag=args.tag, widths=args.widths,
        select_on=args.select_on, min_class_size=args.min_class_size,
        merge_dropped=args.merge_dropped, cv_fold=args.cv_fold, cv_folds=args.cv_folds,
    )

    out = Path(args.out) if args.out else (
        ARTIFACT_DIR / f"candidate_{args.recipe}{'_' + args.tag if args.tag else ''}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)

    train_metrics, test_metrics = checkpoint["train_metrics"], checkpoint["test_metrics"]
    print(f"\n--- {checkpoint['model_name']} ({time.time() - started:.0f}s) ---")
    print(f"best val epoch     : {checkpoint['val_metrics']['epoch']}")
    print(f"TTA adopted        : {checkpoint['tta']}")
    print(f"train accuracy     : {train_metrics['accuracy']:.2f}  "
          f"macro-F1 {train_metrics['macro_f1']:.2f}")
    print(f"TEST accuracy      : {test_metrics['accuracy']:.2f}")
    print(f"TEST weighted-F1   : {test_metrics['weighted_f1']:.2f}")
    print(f"TEST macro-F1      : {test_metrics['macro_f1']:.2f}")
    print(f"TEST balanced acc  : {test_metrics['balanced_acc']:.2f}")
    print(f"TEST top-3 / top-5 : {test_metrics['top3_acc']:.2f} / {test_metrics['top5_acc']:.2f}")
    print(f"generalisation gap : {train_metrics['accuracy'] - test_metrics['accuracy']:+.2f}")
    print("\npublished run 20260830_215803: acc 87.64  wF1 87.13  mF1 73.09  "
          "bal 71.52  (train acc 97.80, gap +10.16)")
    print("saved", out)

    if args.report:
        print()
        print(classification_report(y_true, probabilities.argmax(1),
                                    target_names=class_names, zero_division=0))


if __name__ == "__main__":
    main()
