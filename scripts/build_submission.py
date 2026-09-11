#!/usr/bin/env python
"""Assemble the four-column submission from the four task models.

Why this exists
---------------
``outputs/task1_item_type_predictions.csv`` matches the template's shape but
carries only ``articleType``; ``gender``, ``season`` and ``usage`` ship blank.
That is not a missing capability - every model exists and every one of them
covers all 5,829 graded ids:

    articleType   artifacts/task1_120x160/task1_120x160_background_adapted.pt via predict.py
    season        artifacts/task2/task2_season_best_pytorch.pth via Task2Service
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
# The PyTorch season CNN is the shipped Task 2 model, so its predictions are the
# ones that belong in the submission. The older `task2_season_predictions.csv`
# is a different model and disagrees from the first graded row onward (52003:
# Fall against Spring); pointing here was a wiring bug that shipped a model the
# app does not serve. Kept as a separate name rather than overwriting that file,
# so the two remain distinguishable.
TASK2 = PROJECT_ROOT / "outputs" / "task2_season_predictions_pytorch.csv"
TASK3 = PROJECT_ROOT / "outputs" / "task3_gender_usage_predictions.csv"
TARGETS = ("gender", "articleType", "season", "usage")


def score_task2(out=TASK2, images_dir=TEST_IMAGES, force=False, verbose=True):
    """Run the shipped season model over the graded tiles, writing id,season.

    Regenerated from ``Task2Service`` rather than trusted from disk, for the same
    reason Task 3 is: a cached CSV records whichever model happened to write it,
    and this one has been wrong twice already. ``catalogue_label`` is the field
    that means the dataset's own ``season`` column; the service also reports a
    ``suitable_label``, which answers a different question and is not the target.
    """
    out = Path(out)
    if out.exists() and not force:
        if verbose:
            print("cached: {}".format(out.name))
        return pd.read_csv(out)

    from app.backend.services.task2_service import Task2Service

    service = Task2Service()
    paths = sorted((p for p in Path(images_dir).iterdir() if p.suffix.lower() == ".jpg"),
                   key=lambda p: int(p.stem))
    rows = []
    for index, path in enumerate(paths, 1):
        prediction = service.predict(path.read_bytes())
        rows.append({
            "id": int(path.stem),
            "season": prediction.get("catalogue_label") or prediction["label"],
            "season_confidence": prediction.get("catalogue_confidence")
            or prediction["confidence"],
        })
        if verbose and index % 1000 == 0:
            print("  {}/{}".format(index, len(paths)), flush=True)

    frame = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    if verbose:
        print("wrote {} ({} rows)".format(out, len(frame)))
    return frame


def score_task3(out=TASK3, images_dir=TEST_IMAGES, force=False, verbose=True):
    """Run the Task 3 model over the graded tiles, writing id,gender,usage.

    Cached, because it is the only part of the merge that costs real time. The
    service takes single-image bytes, so this is a plain loop - a few minutes on
    5,829 60x80 tiles, not hours.

    ``Task3Service.predict`` returns each head as a list of ``{label, p}`` ranked
    best first, not as a single ``{label, confidence}``. This function read the
    latter shape and had therefore been unable to run at all; the cache hid that,
    because a present CSV is returned without ever calling the service, so the
    submission kept a stale gender and usage column while appearing to work.
    ``_top`` is what keeps the two shapes from silently diverging again.
    """
    def _top(head):
        """(label, probability) for one head, whichever shape the service returns."""
        if isinstance(head, dict):
            return head["label"], head.get("confidence", head.get("p"))
        return head[0]["label"], head[0].get("p", head[0].get("confidence"))

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
        gender_label, gender_p = _top(prediction["gender"])
        usage_label, usage_p = _top(prediction["usage"])
        rows.append({
            "id": int(path.stem),
            "gender": gender_label,
            "gender_confidence": gender_p,
            "usage": usage_label,
            "usage_confidence": usage_p,
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
          force_task2=False, verbose=True):
    template = pd.read_csv(TEMPLATE)
    ids = template["id"].astype(int)

    item = pd.read_csv(task1)[["id", "articleType"]]
    season = score_task2(task2, force=force_task2, verbose=verbose)[["id", "season"]]
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

    out = Path(out) if out else PROJECT_ROOT / "outputs" / "predictions" / "COSC2753_A2_HN_G2.csv"
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
    parser.add_argument("--force-task2", action="store_true",
                        help="re-run the Task 2 model instead of reusing its CSV")
    args = parser.parse_args(argv)
    build(args.task1, args.task2, args.task3, args.out, args.force_task3,
          args.force_task2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
