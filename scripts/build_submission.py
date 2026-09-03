#!/usr/bin/env python
"""Assemble the four-column submission from the four task models.

Why this exists
---------------
``outputs/task1_item_type_predictions.csv`` matches the template's shape but
carries only ``articleType``; ``gender``, ``season`` and ``usage`` ship blank.
That is not a missing capability - every model exists and every one of them
covers all 5,829 graded ids:

    articleType   artifacts/task1/task1_cnn.pt        via predict.py
    season        outputs/task2_season_predictions.csv (already written)
    gender+usage  artifacts/task3/task3_cnn_model.pt   via Task3Service

so three quarters of the deliverable was being left on the floor for want of a
join. This script does the join, and asserts the things that would otherwise
fail silently: row count, id order against the template, and no empty cells.

Note the import direction. ``EarlyBranchCNN`` is defined only inside
``app/backend/services/task3_service.py``, because ``src/models/gender_classifier.py``
and ``src/models/usage_classifier.py`` are still empty placeholders. Task 1 had
the same problem once and commit 37a2a7e8 fixed it by making ``src/models`` the
single definition; Tasks 2 and 3 have not had that treatment yet, so a top-level
script has to reach into the backend package for the class. That is backwards,
and worth fixing separately - it is the same latent bug that once left
``task1_cnn.pt`` unloadable by anything in the repository.

Usage
-----
    python scripts/build_submission.py
    python scripts/build_submission.py --task1 outputs/task1_prior_corrected.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEMPLATE = (PROJECT_ROOT / "A2_FashionDataset" / "FashionDataset" / "test"
            / "styles_prediction_template.csv")
TEST_IMAGES = PROJECT_ROOT / "A2_FashionDataset" / "FashionDataset" / "test" / "images_test"
TASK1 = PROJECT_ROOT / "outputs" / "task1_item_type_predictions.csv"
TASK2 = PROJECT_ROOT / "outputs" / "task2_season_predictions.csv"
TASK3 = PROJECT_ROOT / "outputs" / "task3_gender_usage_predictions.csv"
TARGETS = ("gender", "articleType", "season", "usage")


def score_task3(out=TASK3, images_dir=TEST_IMAGES, force=False, verbose=True):
    """Run the Task 3 model over the graded tiles, writing id,gender,usage.

    Cached, because it is the only part of the merge that costs real time. The
    service takes single-image bytes, so this is a plain loop - a few minutes on
    5,829 60x80 tiles, not hours.
    """
    out = Path(out)
    if out.exists() and not force:
        if verbose:
            print("cached: {}".format(out.name))
        return pd.read_csv(out)

    from app.backend.services.task3_service import Task3Service

    service = Task3Service()
    paths = sorted((p for p in Path(images_dir).iterdir() if p.suffix.lower() == ".jpg"),
                   key=lambda p: int(p.stem))
    rows = []
    for index, path in enumerate(paths, 1):
        prediction = service.predict(path.read_bytes())
        rows.append({
            "id": int(path.stem),
            "gender": prediction["gender"]["label"],
            "gender_confidence": prediction["gender"]["confidence"],
            "usage": prediction["usage"]["label"],
            "usage_confidence": prediction["usage"]["confidence"],
        })
        if verbose and index % 1000 == 0:
            print("  {}/{}".format(index, len(paths)), flush=True)

    frame = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    if verbose:
        print("wrote {} ({} rows)".format(out, len(frame)))
    return frame


def build(task1=TASK1, task2=TASK2, task3=TASK3, out=None, force_task3=False,
          verbose=True):
    template = pd.read_csv(TEMPLATE)
    ids = template["id"].astype(int)

    item = pd.read_csv(task1)[["id", "articleType"]]
    season = pd.read_csv(task2)[["id", "season"]]
    gender_usage = score_task3(task3, force=force_task3, verbose=verbose)[
        ["id", "gender", "usage"]]

    frame = (template[["id"]].astype({"id": int})
             .merge(item.astype({"id": int}), on="id", how="left")
             .merge(season.astype({"id": int}), on="id", how="left")
             .merge(gender_usage.astype({"id": int}), on="id", how="left"))
    frame = frame[["id", "gender", "articleType", "season", "usage"]]

    # The three things that would otherwise be wrong without anyone noticing.
    assert len(frame) == len(template), (
        "row count {} != template {}".format(len(frame), len(template)))
    assert frame["id"].tolist() == ids.tolist(), "id order diverged from the template"
    for column in TARGETS:
        blank = frame[column].isna() | (frame[column].astype(str).str.strip() == "")
        assert not blank.any(), "{} blank on {} rows".format(column, int(blank.sum()))

    out = Path(out) if out else PROJECT_ROOT / "outputs" / "predictions" / "styles_prediction.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    if verbose:
        print("\nwrote {} ({} rows, all four targets filled)".format(out, len(frame)))
        for column in TARGETS:
            counts = frame[column].value_counts()
            print("  {:<12} {:>3} distinct, top: {}".format(
                column, len(counts), ", ".join(
                    "{} {}".format(name, count) for name, count in counts.head(3).items())))
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task1", default=TASK1, type=Path)
    parser.add_argument("--task2", default=TASK2, type=Path)
    parser.add_argument("--task3", default=TASK3, type=Path)
    parser.add_argument("--out", default=None, type=Path)
    parser.add_argument("--force-task3", action="store_true",
                        help="re-run the Task 3 model instead of reusing its CSV")
    args = parser.parse_args(argv)
    build(args.task1, args.task2, args.task3, args.out, args.force_task3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
