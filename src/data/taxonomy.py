"""The dataset's label hierarchy, and the coarse answer it makes possible.

Why this exists
---------------
``articleType`` has 92 classes and the model gets 54.86% of them right on a
shift-synthesised photograph. Those 92 roll up into 33 ``subCategory`` families,
and the family is right **66.95%** of the time - twelve points better, for free,
because most of the model's mistakes are within-family (Casual Shoes for Sports
Shoes, Tops for Tshirts) and collapsing the family absorbs them.

Measured on 587 held-out rows through the serving path:

    articleType top-1            54.86
    subCategory via top-1 class  66.44
    subCategory marginalised     66.95      <- what this module computes
    subCategory marginalised top-2  73.25

Marginalising - summing the probability of every class in a family - beats
reading the family off the argmax, because it counts a family that is second and
third as well as first.

Note what this is *not* for. Re-ranking the fine alternatives to span different
families was tried and rejected: it drops top-3 from 66.95 to 60.14 at mild and
60.99 to 53.66 at moderate, because the correct answer is very often in the same
family as the top guess, and forcing diversity evicts it. The hierarchy earns its
place as a second, coarser answer - not as a filter on the first one.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["family_matrix", "subcategory_of"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_METADATA = PROJECT_ROOT / "A2_FashionDataset" / "processed" / "clean_train_metadata.csv"


@lru_cache(maxsize=1)
def _article_to_subcategory():
    """``{articleType: subCategory}`` read once from the cleaned metadata.

    The mapping is not a function in the source data, so the aggregation has to
    be the *modal* subCategory rather than whichever row happened to come last.
    ``Kajal and Eyeliner`` is filed under ``Eyes`` on 5 rows and ``Makeup`` on 8;
    ``Perfume and Body Mist`` under ``Fragrance`` on 551 and ``Perfumes`` on 5.
    A ``dict(zip(...))`` over the frame resolves those by row order, which means
    the family a class rolls up into depends on how the CSV happens to be sorted.
    Taking the mode makes it depend on the data instead, and makes this function
    invariant to shuffling the metadata.
    """
    frame = pd.read_csv(CLEAN_METADATA, usecols=["articleType", "subCategory"])
    modal = frame.groupby("articleType")["subCategory"].agg(
        lambda values: values.mode().iat[0])
    return modal.to_dict()


def subcategory_of(class_names):
    """The family for each class, falling back to the class's own name.

    The fallback matters: a class absent from the metadata would otherwise be
    dropped from the hierarchy silently, and it is better for it to be its own
    one-member family than to be mis-grouped.
    """
    mapping = _article_to_subcategory()
    return [str(mapping.get(name, name)) for name in class_names]


@lru_cache(maxsize=4)
def _cached_matrix(class_names):
    families = sorted(set(subcategory_of(class_names)))
    index = {name: i for i, name in enumerate(families)}
    matrix = np.zeros((len(class_names), len(families)), dtype=np.float32)
    for row, family in enumerate(subcategory_of(class_names)):
        matrix[row, index[family]] = 1.0
    return matrix, families


def family_matrix(class_names):
    """``(matrix, family_names)`` for marginalising class probabilities.

    ``probabilities @ matrix`` gives one probability per family. Cached on the
    class-name tuple, so serving pays the CSV read once per process.
    """
    return _cached_matrix(tuple(class_names))
