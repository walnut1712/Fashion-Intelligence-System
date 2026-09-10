"""Rebuild the Task 1 serving regression fixture.

Why this exists
---------------
``tests/test_prediction.py::test_saved_predictions_are_reproducible`` replays the
committed ``artifacts/task1/task1_predictions.csv`` through the serving path and
asserts the labels and confidences still come back. That is a real guard: swap
the checkpoint without rebuilding the fixture and the test fails, which is the
intended behaviour rather than an obstacle.

**Which checkpoint it pairs with.** That test loads ``legacy_service``, which is
``artifacts/task1/task1_cnn.pt`` - the 60x80 model - and its docstring says so:
it replays the historical 60x80 predictions through their matching path. So this
fixture is a frozen record of that pairing, not a description of what ships.
Rebuilding it against the deployed 120x160 checkpoint changes 1,644 of 5,829
labels and makes the test fail. ``--check`` exists to confirm the pairing still
holds, which is the only thing that can legitimately go wrong with it.

Until now the only code that could write it lived inside
``scripts/promote_checkpoint.py``. That script adopted a candidate into
``task1_cnn.pt``, a path nothing serves since Task 1 moved to 120x160, and it
rebuilt the submission with the superseded ensemble-plus-label-shift recipe, so
running it silently reverted a deliberate decision. It was deleted; this is the
one piece of it worth keeping.

Note what this does **not** do. It does not touch
``outputs/task1_item_type_predictions.csv`` or the submission. Those are built by

    python predict.py --images A2_FashionDataset/FashionDataset/test/images_test \
                      --out outputs/task1_item_type_predictions.csv --submission
    python scripts/build_submission.py

which is deliberately a separate decision from the fixture: the fixture must sit
on the serving path (one checkpoint, whatever TTA the checkpoint declares), while
the submission recipe is a choice the team makes explicitly.

    python scripts/refresh_task1_fixture.py                     # rebuild the shipped pair
    python scripts/refresh_task1_fixture.py --check             # report drift, write nothing
    python scripts/refresh_task1_fixture.py --pair legacy --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# A fixture is only meaningful next to the checkpoint that produced it, so the
# two are named together here. Rebuilding one against the other's checkpoint
# makes its test fail rather than pass, which is a mistake that has already been
# made once: `shipped` and `legacy` disagree on 1,644 of 5,829 labels, because
# they are different models, not because either is stale.
PAIRS = {
    # The deployed model. Guards against a retrain silently desyncing the
    # submission: every other check in the suite compares one file to another,
    # so only replaying THIS pair can notice that the checkpoint moved.
    "shipped": (
        PROJECT_ROOT / "artifacts" / "task1_120x160" / "task1_120x160_onecycle_best.pt",
        PROJECT_ROOT / "artifacts" / "task1_120x160" / "task1_predictions_120x160.csv",
    ),
    # Frozen historical evidence. `test_saved_predictions_are_reproducible`
    # replays it through `legacy_service`, which loads the 60x80 checkpoint.
    "legacy": (
        PROJECT_ROOT / "artifacts" / "task1" / "task1_cnn.pt",
        PROJECT_ROOT / "artifacts" / "task1" / "task1_predictions.csv",
    ),
}
TEST_IMAGES = (PROJECT_ROOT / "A2_FashionDataset" / "FashionDataset" / "test"
               / "images_test")


def build_frame(checkpoint_path, verbose=True):
    """Score the graded tiles through the serving path and return the fixture frame."""
    from src.models.item_type_classifier import (  # noqa: E402
        choose_device,
        load_item_type_model,
        predict_proba,
    )

    device = choose_device()
    model, checkpoint = load_item_type_model(checkpoint_path, device)
    class_names = np.asarray(checkpoint["class_names"])
    paths = sorted((p for p in TEST_IMAGES.iterdir() if p.suffix.lower() == ".jpg"),
                   key=lambda p: int(p.stem))
    if verbose:
        print("checkpoint {}".format(Path(checkpoint_path).name))
        print("scoring {} graded tiles for the serving fixture".format(len(paths)))

    # TTA follows the checkpoint rather than a flag, because the fixture has to
    # describe the serving path and the service reads the same field.
    probabilities = predict_proba(model, checkpoint, paths, device=device,
                                  tta=bool(checkpoint.get("tta", False)))
    ids = [int(p.stem) for p in paths]

    # Two columns the submission does not carry: the image path, and the
    # confidence the test asserts against.
    return pd.DataFrame({
        "id": ids,
        "gender": "",
        "articleType": class_names[probabilities.argmax(1)],
        "season": "",
        "usage": "",
        "image_path": ["A2_FashionDataset/FashionDataset/test/images_test/{}.jpg".format(i)
                       for i in ids],
        "articleType_confidence": probabilities.max(1).round(4),
    })


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", choices=tuple(PAIRS), default="shipped",
                        help="which checkpoint/fixture pair to act on")
    parser.add_argument("--check", action="store_true",
                        help="report whether the committed fixture is stale, write nothing")
    args = parser.parse_args(argv)
    checkpoint, fixture = PAIRS[args.pair]

    if not checkpoint.exists():
        print("checkpoint not found: {}".format(checkpoint))
        return 1
    if not TEST_IMAGES.exists():
        print("graded tiles not found: {}".format(TEST_IMAGES))
        return 1

    frame = build_frame(checkpoint)

    if args.check:
        if not fixture.exists():
            print("fixture absent: {}".format(fixture))
            return 1
        current = pd.read_csv(fixture)
        if len(current) != len(frame):
            print("STALE: {} rows on disk, {} scored".format(len(current), len(frame)))
            return 1
        changed = int((current["articleType"].astype(str).values
                       != frame["articleType"].astype(str).values).sum())
        # Reported as a count rather than a verdict. A handful of rows can differ
        # from tie-breaking at the top of the softmax without the pairing being
        # wrong; a few hundred means the fixture and the checkpoint have parted.
        print("{} of {} labels differ from {}".format(
            changed, len(frame), checkpoint.name))
        return 1 if changed else 0

    fixture.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(fixture, index=False)
    print("wrote {} ({} rows)".format(fixture.relative_to(PROJECT_ROOT), len(frame)))
    print("now run: .venv/Scripts/python.exe -m pytest tests/test_prediction.py -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
