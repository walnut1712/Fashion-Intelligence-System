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

Swapping the backdrop is only half of what separates a catalogue photograph from
an uploaded one. The composited item is still perfectly upright, perfectly sharp,
lit exactly as the studio lit it, unoccluded and losslessly stored - none of which
survives a phone camera. ``degrade`` supplies the missing half, and splits into
training and evaluation families on the same principle as the backgrounds.

``simulate_ingestion`` closes a third gap. A live upload does not reach the
encoder as a composite: it goes through ``load_user_image(mode="nobg")``, which
segments the subject onto white - or gives up and centre-crops, keeping the
original backdrop, which measurement puts at 38% of images. Training on raw
composites means training on a distribution the service never sends.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

__all__ = ["make_backgrounds", "make_eval_backgrounds", "composite",
           "degrade", "simulate_ingestion",
           "TRAINING_FAMILIES", "EVAL_FAMILIES",
           "TRAINING_DEGRADATIONS", "EVAL_DEGRADATIONS"]

TRAINING_FAMILIES = ("solid", "gradient", "noise", "blobs", "blurred-crop")
EVAL_FAMILIES = ("checker", "stripes")

#: Corruptions a camera applies that compositing does not. Disjoint from
#: ``EVAL_DEGRADATIONS`` for the same reason the background banks are disjoint.
TRAINING_DEGRADATIONS = ("rotate", "blur", "whitebalance", "occlude", "jpeg")
EVAL_DEGRADATIONS = ("perspective", "shadow")

#: Applied in this order whatever order the caller lists them in, so a run does
#: not depend on tuple ordering: geometry, then light, then things in the way,
#: then the lens, then the file format - which is the order a photograph
#: actually acquires them.
_DEGRADATION_ORDER = ("rotate", "perspective", "whitebalance", "shadow",
                      "occlude", "blur", "jpeg")

#: The catalogue resolution this module was written against. Every function
#: takes its size from its input or its argument; this is only the fallback, so
#: that a 120x160 run needs no edits here - it passes a shape and everything
#: downstream follows.
DEFAULT_SHAPE = (80, 60, 3)          # (H, W, C)
DEFAULT_SCALE_RANGE = (0.35, 0.95)

#: Morphology and area thresholds below were tuned on 80-pixel-tall frames. A
#: 3x3 opening removes proportionally less of a 160-tall frame, so the structure
#: is scaled by the linear factor rather than left to shrink in real terms.
REFERENCE_HEIGHT = 80

#: Share of uploads whose segmentation succeeds, measured on ``images_test`` in
#: ``outputs/task4_ingestion_fallback.csv``: 3,636 segmented against 2,193 that
#: fell back to a centre crop. A round number here would be an invention.
DEFAULT_SEGMENT_PROBABILITY = 3636 / (3636 + 2193)


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


def make_eval_backgrounds(count=600, size=None, seed=20260904):
    """Evaluation-only bank, from families the training pipeline never generates.

    Reuses the corruption families in ``src/evaluation/ood_benchmark.py``, which
    already established this discipline for Task 1: a model must not be graded on
    the same augmentation it was trained against.
    """
    from src.evaluation.ood_benchmark import _background

    if size is None:                       # (W, H), matching PIL's convention
        size = (DEFAULT_SHAPE[1], DEFAULT_SHAPE[0])
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


# ------------------------------------------------- photographic degradation ----
def _border_median(image):
    """Background colour estimated from the frame edge, as uint8 RGB.

    Used to fill the corners a rotation opens up. Filling with black instead
    would hand the encoder a perfect rotation detector - a constant, saturated
    wedge no real photograph contains - and it would learn that rather than the
    invariance the augmentation is meant to teach.
    """
    edges = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]], axis=0)
    return tuple(int(v) for v in np.median(edges, axis=0))


def _perspective_coefficients(source_corners, target_corners):
    """Solve the eight coefficients ``Image.PERSPECTIVE`` wants.

    PIL samples the input at ``((ax + by + c) / (gx + hy + 1), ...)`` for each
    output pixel ``(x, y)``, so the unknowns are recovered from four output
    corners and the input points they should read from - an exactly determined
    8x8 system, no least squares needed.
    """
    rows, values = [], []
    for (x, y), (u, v) in zip(target_corners, source_corners):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        values.extend((u, v))
    return np.linalg.solve(np.asarray(rows, dtype=np.float64),
                          np.asarray(values, dtype=np.float64))


def degrade(image, generator, families=TRAINING_DEGRADATIONS, strength=1.0,
            probability=0.5):
    """Apply the corruptions a camera adds and compositing does not.

    ``composite`` produces an item that is upright, sharp, studio-lit, unoccluded
    and losslessly stored. A photograph of the same garment is none of those, and
    an encoder trained only on composites can key on the difference. Each named
    family fires independently with ``probability``; ``strength`` scales every
    magnitude, so 0.0 is the identity and the parameter can be ramped exactly as
    the background probability already is.

    Families are applied in ``_DEGRADATION_ORDER``, not in the order given.

    Returns a new ``uint8`` array of the same shape; ``image`` is not modified.
    """
    if strength <= 0 or not families:
        return np.ascontiguousarray(image, dtype=np.uint8)

    selected = [f for f in _DEGRADATION_ORDER if f in set(families)]
    unknown = set(families) - set(_DEGRADATION_ORDER)
    if unknown:
        raise ValueError("Unknown degradation families: {}. Known: {}".format(
            sorted(unknown), list(_DEGRADATION_ORDER)))

    picture = Image.fromarray(np.asarray(image, dtype=np.uint8))
    width, height = picture.size

    for family in selected:
        # Draw for every family whether or not it fires, so that adding a family
        # to the list does not reshuffle the draws of the ones after it.
        fires = generator.random() < probability

        if family == "rotate":
            angle = strength * generator.uniform(-12.0, 12.0)
            if fires:
                picture = picture.rotate(
                    angle, resample=Image.BILINEAR,
                    fillcolor=_border_median(np.asarray(picture)))

        elif family == "perspective":
            jitter = strength * 0.08 * np.array([width, height], dtype=np.float64)
            offsets = generator.uniform(-1.0, 1.0, (4, 2)) * jitter
            if fires:
                corners = np.array([[0, 0], [width, 0], [width, height], [0, height]],
                                   dtype=np.float64)
                picture = picture.transform(
                    (width, height), Image.PERSPECTIVE,
                    _perspective_coefficients(corners + offsets, corners),
                    resample=Image.BILINEAR,
                    fillcolor=_border_median(np.asarray(picture)))

        elif family == "whitebalance":
            gain = 1.0 + strength * generator.uniform(-0.10, 0.10, 3)
            if fires:
                array = np.asarray(picture, dtype=np.float32) * gain
                picture = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

        elif family == "shadow":
            darkness = 1.0 - strength * generator.uniform(0.15, 0.45)
            angle = generator.uniform(0, 2 * np.pi)
            edge = generator.uniform(0.25, 0.75)
            if fires:
                yy, xx = np.mgrid[0:height, 0:width]
                projection = (np.cos(angle) * xx / width
                              + np.sin(angle) * yy / height)
                low, high = projection.min(), projection.max()
                projection = (projection - low) / max(high - low, 1e-6)
                # smoothstep, so the shadow has a soft edge rather than a line
                ramp = np.clip((projection - edge) / 0.35, 0, 1)
                ramp = ramp * ramp * (3 - 2 * ramp)
                factor = (darkness + (1.0 - darkness) * ramp)[..., None]
                array = np.asarray(picture, dtype=np.float32) * factor
                picture = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

        elif family == "occlude":
            fraction = strength * generator.uniform(0.05, 0.20)
            aspect = generator.uniform(0.4, 2.5)
            centre = generator.uniform(0.15, 0.85, 2)
            colour = generator.integers(0, 256, 3)
            if fires:
                area = fraction * width * height
                box_w = int(np.clip(round(np.sqrt(area * aspect)), 1, width))
                box_h = int(np.clip(round(np.sqrt(area / aspect)), 1, height))
                x0 = int(np.clip(centre[0] * width - box_w / 2, 0, width - box_w))
                y0 = int(np.clip(centre[1] * height - box_h / 2, 0, height - box_h))
                array = np.asarray(picture).copy()
                array[y0:y0 + box_h, x0:x0 + box_w] = colour
                picture = Image.fromarray(array)

        elif family == "blur":
            radius = strength * generator.uniform(0.0, 0.8)
            if fires and radius > 0.01:
                picture = picture.filter(ImageFilter.GaussianBlur(radius))

        elif family == "jpeg":
            quality = int(round(100 - strength * generator.uniform(10.0, 65.0)))
            if fires:
                buffer = io.BytesIO()
                picture.save(buffer, format="JPEG", quality=max(1, min(100, quality)))
                buffer.seek(0)
                picture = Image.open(buffer).convert("RGB")

    return np.asarray(picture, dtype=np.uint8)


# --------------------------------------------------- serve-path simulation ----
def _scaled_structure(height):
    """A 3x3 structure, and how many times to iterate it for this frame.

    Keeping the kernel and scaling the iteration count holds the operation at a
    fixed size *relative to the frame*: a single 3x3 opening removes
    proportionally less of a 160-tall image than of an 80-tall one, so the masks
    would drift apart between the two resolutions.
    """
    return np.ones((3, 3), dtype=bool), max(1, int(round(height / REFERENCE_HEIGHT)))


def _border_colour_mask(image, tolerance=20):
    """Foreground where the pixel differs from the frame-edge background colour.

    The same estimator notebook 05 uses to cut catalogue items off white, and the
    numpy tier of ``src/data/user_image.py``'s segmentation ladder. Modelling the
    backdrop from the border rather than assuming white is what lets it work on a
    composite, where the backdrop is whatever the background bank supplied.
    """
    from scipy import ndimage

    background = np.median(
        np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]], axis=0),
        axis=0)
    mask = np.abs(image.astype(np.float32) - background).max(axis=2) > tolerance

    structure, factor = _scaled_structure(image.shape[0])
    mask = ndimage.binary_opening(mask, structure=structure, iterations=1 * factor)
    mask = ndimage.binary_closing(mask, structure=structure, iterations=2 * factor)

    labels, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return mask
    areas = np.bincount(labels.ravel())
    areas[0] = 0                                   # background label
    keep = np.flatnonzero(areas > 0.15 * areas.max())
    return np.isin(labels, keep)


def simulate_ingestion(image, generator, p_segment=DEFAULT_SEGMENT_PROBABILITY,
                       margin=0.06, tolerance=20, return_info=False):
    """Put a composite through the shape of the pipeline that serves real uploads.

    ``load_user_image(mode="nobg")`` does not hand the encoder the photograph. It
    segments the subject onto white and crops tight, and when no plausible mask
    comes back it declines and centre-crops instead, leaving the backdrop in
    place. Training only on raw composites therefore trains on a distribution the
    service never sends, and never shows the encoder what a *failed* segmentation
    looks like - a halo of leftover backdrop, or a garment with a piece missing.

    ``p_segment`` defaults to the rate measured on ``images_test``, not a round
    number. Declining is modelled by returning the composite untouched.

    This is deliberately the cheap tier only. The served ladder is rembg ->
    GrabCut -> border colour -> flood fill, and its docstring records 510 ms per
    image; at 60x80 that is unaffordable per sample per epoch, and the border
    colour model alone is microseconds. Training therefore sees a slightly worse
    segmentation than production delivers, which errs in the safe direction.
    """
    array = np.asarray(image, dtype=np.uint8)
    info = {"segmented": False, "reason": "declined"}

    if generator.random() >= p_segment:
        return (array.copy(), info) if return_info else array.copy()

    height, width = array.shape[:2]
    mask = _border_colour_mask(array, tolerance=tolerance)
    covered = mask.mean()

    # Production declines on an implausible mask rather than deleting the
    # product; ``load_user_image`` uses an ink fraction of 0.05 for the same
    # call. A mask covering nearly everything has segmented nothing.
    if covered < 0.05 or covered > 0.95:
        info["reason"] = "implausible mask ({:.2f} covered)".format(covered)
        return (array.copy(), info) if return_info else array.copy()

    ys, xs = np.where(mask)
    pad_y = int(round(margin * height))
    pad_x = int(round(margin * width))
    y0 = max(0, ys.min() - pad_y)
    y1 = min(height, ys.max() + 1 + pad_y)
    x0 = max(0, xs.min() - pad_x)
    x1 = min(width, xs.max() + 1 + pad_x)

    subject = np.full_like(array, 255)
    subject[mask] = array[mask]
    tile = Image.fromarray(subject[y0:y1, x0:x1])

    # Letterbox onto white at the original size, so the frame the encoder sees
    # keeps its aspect ratio and the item is not stretched.
    tile.thumbnail((width, height), Image.BILINEAR)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(tile, ((width - tile.width) // 2, (height - tile.height) // 2))

    info.update(segmented=True, reason="border colour", covered=float(covered))
    result = np.asarray(canvas, dtype=np.uint8)
    return (result, info) if return_info else result
