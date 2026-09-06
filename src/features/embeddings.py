"""Combining a learned embedding with a hand-built colour descriptor.

Two ways to put colour back into a retrieval system whose metric was learned on
``articleType`` and therefore discards it:

``fuse``
    concatenate the two descriptors, weighted. Simple, but it doubles the stored
    index and pays the colour cost on every one of the 32,837 comparisons.

``RerankIndex``
    shortlist on the learned embedding alone, then re-order only the top 100 by
    colour agreement. The stored index stays 128-dimensional and the colour
    table is consulted for 100 items per query rather than all of them.

Both were evaluated in ``notebooks/05_task4_triplet_encoder.ipynb`` and both landed
within the noise floor of the plain encoder (both@10 46.08 and 46.09 against
45.92), which is why ``uses_reranking`` is false in the shipped manifest. They
are kept because "we tried it and it did not clear the bar" is a result, and
because that comparison used an *unpaired* floor that was far too conservative -
re-testing them with ``paired_bootstrap`` from ``src/evaluation/metrics.py`` may
yet separate them.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import CatalogueIndex

__all__ = ["l2", "fuse", "RerankIndex", "DEFAULT_COLOUR_WEIGHT", "DEFAULT_ALPHA"]

#: Winners of the sweeps in notebook 05, recorded in ``task4_summary.json``.
DEFAULT_COLOUR_WEIGHT = 0.35
DEFAULT_ALPHA = 0.7


def l2(matrix):
    """Row-wise L2 normalisation - cosine similarity is only a matmul on these."""
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8, None)


def fuse(learned, colour, weight=DEFAULT_COLOUR_WEIGHT):
    """Weighted concatenation of a learned embedding and a colour descriptor.

    Each side is normalised first, so ``weight`` alone controls the balance
    rather than the descriptors' differing scales.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    return np.hstack([l2(learned) * (1.0 - weight), l2(colour) * weight]).astype(np.float32)


class RerankIndex:
    """Shortlist by embedding similarity, then re-order that shortlist by colour.

    Query vectors arrive as ``[learned | colour]`` concatenations so this drops
    into anything that expects an index object.

    ``alpha`` weights the learned similarity against colour agreement; at 1.0 it
    is the plain index. Notebook 05 swept it and the argmax moved between 1.0,
    0.6 and 0.8 across three runs of identical code, which is the clearest
    single piece of evidence that the effect is inside the noise.
    """

    def __init__(self, learned, colour, alpha=DEFAULT_ALPHA, protocol=None,
                 name="rerank", pool=100, catalogue=None):
        self.base = CatalogueIndex(learned, protocol, name=name, catalogue=catalogue)
        self.colour = l2(colour)
        self.split = np.asarray(learned).shape[1]
        self.alpha = float(alpha)
        self.pool = int(pool)
        self.name = name

    def __len__(self):
        return len(self.base)

    @property
    def dim(self):
        return self.base.dim

    @property
    def vectors(self):
        return self.base.vectors

    def search(self, query_vectors, k=10, exclude=None, batch_size=512):
        queries = np.atleast_2d(np.asarray(query_vectors, dtype=np.float32))
        learned = queries[:, :self.split]
        query_colour = l2(queries[:, self.split:])

        scores, positions = self.base.search(learned, k=max(self.pool, k),
                                             batch_size=batch_size)

        out_scores = np.zeros((len(queries), k), dtype=np.float32)
        out_positions = np.zeros((len(queries), k), dtype=positions.dtype)
        for row in range(len(queries)):
            shortlist = positions[row]
            colour_agreement = self.colour[shortlist] @ query_colour[row]
            combined = self.alpha * scores[row] + (1.0 - self.alpha) * colour_agreement
            order = np.argsort(-combined)[:k]
            out_scores[row] = combined[order]
            out_positions[row] = shortlist[order]
        return out_scores, out_positions
