"""Hand-built image descriptors for Task 4 - the non-neural comparison point.

The assignment asks for a thorough investigation across different *types* of
algorithm, and the retrieval study needs something to measure the learned
encoder against that shares none of its machinery. This module is that
something: an HSV colour histogram concatenated with a block-wise
gradient-orientation histogram, computed with numpy only.

It is not a strawman. Under the deployment protocol it scores P@10 68.78 against
the learned encoder's 81.23, and on the ``baseColour`` control - an attribute no
method was trained on - it beats the triplet encoder outright (23.56 vs 18.63),
because a colour histogram cannot help but represent colour while a metric
learned on ``articleType`` actively discards it.

Extracted from ``notebooks/05_task4_triplet_encoder.ipynb`` so the notebook, the
fusion experiments and any future service share one definition.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rgb_to_hsv_array", "colour_histogram", "gradient_histogram",
           "classical_features", "CLASSICAL_COLOUR_DIMS"]

#: Width of the colour block at the default binning, so callers can slice the
#: descriptor into its colour and shape halves without recomputing it.
CLASSICAL_COLOUR_DIMS = 8 * 4 * 4


def rgb_to_hsv_array(rgb):
    """Vectorised RGB->HSV for a batch of uint8 images, shaped (N, H, W, 3)."""
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    mask = delta > 1e-6
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    idx = mask & (maxc == r)
    hue[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6
    idx = mask & (maxc == g)
    hue[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2
    idx = mask & (maxc == b)
    hue[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4

    hue = hue / 6.0
    saturation = np.where(maxc > 1e-6, delta / np.clip(maxc, 1e-6, None), 0.0)
    return hue, saturation, maxc


def colour_histogram(batch, hue_bins=8, sat_bins=4, val_bins=4):
    """Joint HSV histogram per image, L1-normalised.

    HSV rather than RGB because hue survives the brightness variation between
    catalogue shots, which is exactly the invariance ``baseColour`` labels
    assume.
    """
    batch = np.asarray(batch)
    n = len(batch)
    hue, saturation, value = rgb_to_hsv_array(batch)

    h_idx = np.clip((hue * hue_bins).astype(int), 0, hue_bins - 1)
    s_idx = np.clip((saturation * sat_bins).astype(int), 0, sat_bins - 1)
    v_idx = np.clip((value * val_bins).astype(int), 0, val_bins - 1)
    flat = (h_idx * sat_bins * val_bins + s_idx * val_bins + v_idx).reshape(n, -1)

    bins = hue_bins * sat_bins * val_bins
    colour = np.zeros((n, bins), dtype=np.float32)
    for i in range(n):
        colour[i] = np.bincount(flat[i], minlength=bins)
    colour /= np.clip(colour.sum(axis=1, keepdims=True), 1e-8, None)
    return colour


def gradient_histogram(batch, grid=(4, 3), orient_bins=9):
    """Magnitude-weighted gradient orientations over a spatial grid.

    A simplified HOG: the grid keeps coarse layout information that a global
    histogram would discard, which is what separates a shirt from a pair of
    trousers once colour is stripped out.
    """
    batch = np.asarray(batch)
    n = len(batch)
    grey = batch.astype(np.float32).mean(axis=-1) / 255.0
    gy, gx = np.gradient(grey, axis=(1, 2))
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)          # 0..1
    o_idx = np.clip((orientation * orient_bins).astype(int), 0, orient_bins - 1)

    height, width = grey.shape[1], grey.shape[2]
    rows, cols = grid
    row_edges = np.linspace(0, height, rows + 1).astype(int)
    col_edges = np.linspace(0, width, cols + 1).astype(int)

    shape = np.zeros((n, rows * cols * orient_bins), dtype=np.float32)
    cell = 0
    for r in range(rows):
        for c in range(cols):
            block_o = o_idx[:, row_edges[r]:row_edges[r + 1],
                            col_edges[c]:col_edges[c + 1]].reshape(n, -1)
            block_m = magnitude[:, row_edges[r]:row_edges[r + 1],
                                col_edges[c]:col_edges[c + 1]].reshape(n, -1)
            for i in range(n):
                shape[i, cell * orient_bins:(cell + 1) * orient_bins] = np.bincount(
                    block_o[i], weights=block_m[i], minlength=orient_bins)
            cell += 1
    shape /= np.clip(np.linalg.norm(shape, axis=1, keepdims=True), 1e-8, None)
    return shape


def classical_features(batch, hue_bins=8, sat_bins=4, val_bins=4,
                       grid=(4, 3), orient_bins=9):
    """Colour histogram + gradient histogram, concatenated.

    At the defaults this is 236 dimensions: 128 colour, then 108 gradient. The
    colour block always comes first, so ``features[:, :CLASSICAL_COLOUR_DIMS]``
    is the colour descriptor the fusion and re-ranking experiments use.
    """
    colour = colour_histogram(batch, hue_bins, sat_bins, val_bins)
    shape = gradient_histogram(batch, grid, orient_bins)
    return np.hstack([colour, shape]).astype(np.float32)
