"""Synthetic backgrounds and item compositing for Task 4.

Catalogue photographs are flat lays on white; real uploads are not. Training the
encoder to ignore a backdrop, and measuring whether it did, both need the same
operation: cut the item out and paste it onto something else at a random scale
and position.

That operation was written twice - once in the background-augmentation notebook
with five procedural families, and again in the clustering notebook with three,
under prose claiming it was done "exactly as that notebook did". The cluster
stability figure was therefore measured against an easier distribution than the
one the encoder trains on. One definition, imported by both.

Two banks, and they must not be confused:

``make_backgrounds``
    the five families used for TRAINING.

``make_eval_backgrounds``
    checkerboards and stripes, from ``src/evaluation/ood_benchmark.py``. Used
    only for measurement, so a model cannot be graded on its own augmentation.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

__all__ = ["make_backgrounds", "make_eval_backgrounds", "composite",
           "TRAINING_FAMILIES", "EVAL_FAMILIES"]

TRAINING_FAMILIES = ("solid", "gradient", "noise", "blobs", "blurred-crop")
EVAL_FAMILIES = ("checker", "stripes")

DEFAULT_SHAPE = (80, 60, 3)          # (H, W, C)
DEFAULT_SCALE_RANGE = (0.35, 0.95)


def make_backgrounds(count=600, shape=DEFAULT_SHAPE, seed=42, source_images=None):
    """The training bank: solid, gradient, noise, soft blobs, blurred crop.

    ``source_images`` supplies the blurred-catalogue-crop family. Passing None
    drops that family and cycles the other four, which keeps the function usable
    where the image cache is not loaded.
    """
    generator = np.random.default_rng(seed)
    height, width, _ = shape
    backgrounds = np.zeros((count,) + shape, dtype=np.uint8)

    yy, xx = np.mgrid[0:height, 0:width]
    yy = yy / height
    xx = xx / width
    n_kinds = 5 if source_images is not None else 4

    for index in range(count):
        kind = index % n_kinds

        if kind == 0:                                    # solid colour
            backgrounds[index] = generator.integers(20, 245, 3)

        elif kind == 1:                                  # linear gradient
            c0 = generator.integers(0, 256, 3).astype(np.float32)
            c1 = generator.integers(0, 256, 3).astype(np.float32)
            ramp = (xx if generator.random() < 0.5 else yy)[..., None]
            backgrounds[index] = (c0 * (1 - ramp) + c1 * ramp).astype(np.uint8)

        elif kind == 2:                                  # noise texture
            base = generator.integers(30, 230, 3).astype(np.float32)
            noise = generator.normal(0, generator.uniform(8, 45), (height, width, 3))
            backgrounds[index] = np.clip(base + noise, 0, 255).astype(np.uint8)

        elif kind == 3:                                  # soft out-of-focus blobs
            canvas = np.zeros((height, width, 3), np.float32)
            canvas[:] = generator.integers(30, 220, 3)
            for _ in range(generator.integers(2, 6)):
                cy, cx = generator.uniform(0, 1, 2)
                radius = generator.uniform(0.15, 0.6)
                weight = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius ** 2)))
                canvas += weight[..., None] * generator.integers(-90, 90, 3)
            backgrounds[index] = np.clip(canvas, 0, 255).astype(np.uint8)

        else:                                            # blurred catalogue crop
            source = np.asarray(
                source_images[generator.integers(len(source_images))]).astype(np.float32)
            for _ in range(3):                           # cheap box blur
                source = (np.roll(source, 1, 0) + np.roll(source, -1, 0)
                          + np.roll(source, 1, 1) + np.roll(source, -1, 1) + source) / 5.0
            tint = generator.uniform(0.6, 1.3, 3)
            backgrounds[index] = np.clip(source * tint, 0, 255).astype(np.uint8)

    return backgrounds


def make_eval_backgrounds(count=600, size=(60, 80), seed=20260904):
    """Evaluation-only bank, from families the training pipeline never generates.

    Reuses the corruption families in ``src/evaluation/ood_benchmark.py``, which
    already established this discipline for Task 1: a model must not be graded on
    the same augmentation it was trained against.
    """
    from src.evaluation.ood_benchmark import _background

    generator = np.random.default_rng(seed)
    return np.stack([
        _background(EVAL_FAMILIES[i % len(EVAL_FAMILIES)], size, generator)
        for i in range(count)
    ])


def composite(image, mask, background, generator, scale_range=DEFAULT_SCALE_RANGE):
    """Paste the masked item onto ``background`` at a random scale and position.

    The scale range is what makes this more than a background swap: catalogue
    items always fill roughly the same fraction of the frame, and a user's photo
    does not, so the encoder has to stop relying on that framing.
    """
    height, width = image.shape[:2]

    ys, xs = np.where(mask)
    if len(ys) < 10:                       # nothing to cut out - keep as is
        return image.copy()
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    item = image[y0:y1 + 1, x0:x1 + 1]
    item_mask = mask[y0:y1 + 1, x0:x1 + 1]

    target_height = max(8, int(height * generator.uniform(*scale_range)))
    ratio = item.shape[1] / item.shape[0]
    target_width = max(6, min(width, int(target_height * ratio)))

    item_img = Image.fromarray(item).resize((target_width, target_height), Image.BILINEAR)
    mask_img = Image.fromarray((item_mask * 255).astype(np.uint8)).resize(
        (target_width, target_height), Image.NEAREST)

    canvas = Image.fromarray(background.copy())
    offset_x = int(generator.integers(0, max(1, width - target_width + 1)))
    offset_y = int(generator.integers(0, max(1, height - target_height + 1)))
    canvas.paste(item_img, (offset_x, offset_y), mask_img)
    return np.asarray(canvas, dtype=np.uint8)
