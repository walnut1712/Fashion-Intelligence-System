"""Leakage-safe, reusable splits for the 120x160 candidate experiments."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.config import IMAGE_SIZE_PIL, SPLIT_IDS_PATH, SUPERVISED_METADATA


EXPECTED_SUPERVISED_ROWS = 38571
EXPECTED_EXCLUDED_ROWS = 41
RANDOM_STATE = 42
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15


def _split_once(frame, stratify_column, fraction, seed):
    folds = max(2, round(1 / fraction))
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    keep, held = next(
        splitter.split(frame, frame[stratify_column], frame["split_group"])
    )
    return frame.iloc[keep].copy(), frame.iloc[held].copy()


def load_supervised_metadata(path=SUPERVISED_METADATA):
    """Load only rows explicitly approved for supervised development."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required supervised metadata not found: {path}")
    frame = pd.read_csv(path)
    required = {"id", "split_group", "use_for_supervised"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {sorted(missing)}")
    usable_mask = frame["use_for_supervised"].map(
        lambda value: value is True or str(value).strip().lower() == "true"
    )
    usable = frame[usable_mask].copy()
    assert len(usable) == EXPECTED_SUPERVISED_ROWS, len(usable)
    assert len(frame) - len(usable) == EXPECTED_EXCLUDED_ROWS
    return usable


def load_or_create_splits(
    path=SUPERVISED_METADATA, split_path=SPLIT_IDS_PATH, stratify_column="articleType"
):
    """Return train/validation/test frames and persist one shared assignment."""
    frame = load_supervised_metadata(path)
    split_path = Path(split_path)
    if split_path.exists():
        assignments = pd.read_csv(split_path, usecols=["id", "split"])
        if set(assignments["id"]) != set(frame["id"]):
            raise ValueError("Persisted 120x160 split IDs do not match metadata")
        frame = frame.merge(assignments, on="id", how="left", validate="one_to_one")
    else:
        trainval, test = _split_once(
            frame, stratify_column, TEST_FRACTION, RANDOM_STATE
        )
        val_fraction = VAL_FRACTION / (1 - TEST_FRACTION)
        train, val = _split_once(trainval, stratify_column, val_fraction, RANDOM_STATE)
        frame.loc[train.index, "split"] = "train"
        frame.loc[val.index, "split"] = "val"
        frame.loc[test.index, "split"] = "test"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        frame[["id", "split"]].to_csv(split_path, index=False)

    assert frame["split"].notna().all()
    parts = {name: frame[frame["split"] == name].copy() for name in ("train", "val", "test")}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (set(parts[left]["split_group"]) & set(parts[right]["split_group"]))
        assert not (set(parts[left]["id"]) & set(parts[right]["id"]))
    return parts["train"], parts["val"], parts["test"]
