"""Task 1 splits - the invariants that make two runs comparable.

Every recipe comparison in this project assumes the held-out data is the same
across runs. That assumption broke once, silently and expensively: merging the
dropped classes before the split changed the class counts, which changed the
strata, which made StratifiedGroupKFold deal the groups differently. 3,675 rows
of one run's training set landed in another run's test set and the resulting
checkpoint scored a fictional 91.33 weighted-F1 - high enough to be believed.

These tests are slow-ish because they build the real splits. They are worth it:
nothing else in the suite would have caught that.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLEAN_METADATA = (PROJECT_ROOT / "A2_FashionDataset" / "processed"
                  / "clean_train_metadata.csv")
pytestmark = pytest.mark.skipif(not CLEAN_METADATA.exists(),
                                reason="dataset not present")


@pytest.fixture(scope="module")
def splits():
    from src.training.train_item_type import load_splits

    plain = load_splits(verbose=False)
    merged = load_splits(verbose=False, merge_dropped=True)
    return plain, merged


def test_merging_dropped_classes_leaves_the_held_out_data_alone(splits):
    """The merge is a *training* intervention. It must not touch evaluation."""
    (_, val_a, test_a, names_a, _), (_, val_b, test_b, names_b, _) = splits

    assert names_a == names_b, "merging must not change the class list"
    assert set(val_a["id"]) == set(val_b["id"]), "merge moved validation rows"
    assert set(test_a["id"]) == set(test_b["id"]), "merge moved test rows"


def test_merging_only_adds_training_rows(splits):
    (train_a, _, _, _, _), (train_b, _, _, _, _) = splits

    assert set(train_a["id"]) <= set(train_b["id"]), "merge removed training rows"
    assert len(train_b) > len(train_a), "merge added nothing"


def test_merged_training_rows_do_not_leak_into_evaluation(splits):
    (_, val_a, test_a, _, _), (train_b, _, _, _, _) = splits

    for column in ("id", "image_md5"):
        assert not (set(train_b[column]) & set(test_a[column])), (
            f"merged train set shares {column} with the test set")
        assert not (set(train_b[column]) & set(val_a[column])), (
            f"merged train set shares {column} with the validation set")


def test_merge_lands_where_the_graded_set_needs_it(splits):
    """The merge exists for the beauty tail; check it actually arrives there."""
    (train_a, _, _, _, _), (train_b, _, _, _, _) = splits
    before = train_a["articleType"].value_counts()
    after = train_b["articleType"].value_counts()

    for name in ("Lipstick", "Foundation and Primer"):
        assert after[name] > before[name] * 1.5, (
            f"{name} gained too little: {before[name]} -> {after[name]}")


def test_starved_class_folds_are_leakage_free_and_add_evaluation_rows():
    """The cross-validation that replaces two-row recall estimates."""
    from src.training.train_item_type import (load_splits, load_splits_cv_starved,
                                              starved_classes)

    _, _, test_df, _, _ = load_splits(verbose=False)
    names = starved_classes(test_df)
    baseline = test_df["articleType"].value_counts()

    pooled = {}
    for fold in range(3):
        train_df, val_df, fold_test, _, _ = load_splits_cv_starved(
            fold, folds=3, verbose=False)
        for column in ("id", "image_md5", "split_group"):
            assert not (set(train_df[column]) & set(fold_test[column])), (
                f"fold {fold} leaks {column} between train and test")
            assert not (set(val_df[column]) & set(fold_test[column])), (
                f"fold {fold} leaks {column} between val and test")
        counts = fold_test["articleType"].value_counts()
        for name in names:
            pooled[name] = pooled.get(name, 0) + int(counts.get(name, 0))

    for name in ("Lipstick", "Foundation and Primer"):
        assert pooled[name] >= 4 * baseline.get(name, 0), (
            f"{name} gained too few evaluation rows: "
            f"{baseline.get(name, 0)} -> {pooled[name]}")
