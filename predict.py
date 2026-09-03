#!/usr/bin/env python
"""Task 1 - predict articleType for a folder of fashion images.

Runs the adopted item-type model over every image in a directory and writes a
CSV. Uses exactly the code path the notebook and the FastAPI service use
(``src/models/item_type_classifier``), so all three agree by construction.

Examples
--------
Reproduce the submission file for the unlabelled test images::

    python predict.py --images A2_FashionDataset/FashionDataset/test/images_test \\
                      --out outputs/task1_item_type_predictions.csv --submission

Score an arbitrary folder, keeping the top-3 alternatives::

    python predict.py --images my_photos --out my_predictions.csv --top-k 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.item_type_classifier import (  # noqa: E402
    choose_device,
    load_item_type_model,
    predict_proba,
)

DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "task1" / "task1_cnn.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, type=Path,
                        help="directory of images to classify")
    parser.add_argument("--out", required=True, type=Path, help="CSV to write")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="checkpoint (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=1,
                        help="also write the top-K alternatives with their confidences")
    parser.add_argument("--submission", action="store_true",
                        help="write the assignment submission columns "
                             "(id,gender,articleType,season,usage) instead")
    parser.add_argument("--tta", dest="tta", action="store_true", default=None,
                        help="force horizontal-flip test-time augmentation on")
    parser.add_argument("--ingest", default="squash",
                        choices=["squash", "letterbox", "crop", "nobg"],
                        help="how a photo becomes a 60x80 tile. 'squash' resizes "
                             "and lets the aspect ratio distort, which is what the "
                             "published metrics used; the others coerce an upload "
                             "towards catalogue framing first")
    parser.add_argument("--no-tta", dest="tta", action="store_false",
                        help="force it off (default: whatever the checkpoint recorded)")
    parser.add_argument("--prior-correct", action="store_true",
                        help="re-weight the posteriors onto the class prior estimated "
                             "from this folder itself (Saerens-Latinne-Decaestecker EM). "
                             "Off by default so the published path is unchanged. The "
                             "graded set sits ~43%% total-variation from the training "
                             "prior; on a simulation of that shift the correction is "
                             "worth +2.1 weighted-F1, and it is measured harmless when "
                             "there is no shift. See src/evaluation/prior_shift.py")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="strength of the prior correction, 0 disables it")
    parser.add_argument("--review-queue", type=Path, default=None,
                        help="also write the rows worth a human look: low confidence, "
                             "or the coarse family disagreeing with the fine label, or "
                             "the label changed by prior correction")
    return parser.parse_args(argv)


def write_review_queue(path, ids, probabilities, class_names, baseline_labels=None):
    """The rows of a prediction run that are worth a human look.

    The submission itself must carry a fine label in every cell, so abstention is
    not available there. This is the artefact that says where those labels are
    weak, without touching them.

    Three flags, each measured rather than assumed:

    ``low_confidence``
        Below 0.50. The catalogue path is well calibrated (measured ECE 0.018),
        so a 0.4 here really does mean roughly a 40% chance of being right.
    ``family_disagrees``
        The subCategory family with the most probability mass is not the family
        of the top-1 class. Marginalising over a family is right 66.95% of the
        time on shifted inputs against 54.86% for the fine label, so when the two
        disagree the fine label is the less reliable of the pair.
    ``changed_by_prior``
        Prior correction moved this label. Makes the swap set inspectable instead
        of asking anyone to take it on trust.
    """
    order = np.argsort(-probabilities, axis=1)[:, :3]
    confidence = probabilities.max(1)
    rows = {
        "id": ids,
        "articleType": class_names[order[:, 0]],
        "confidence": confidence.round(4),
    }
    for rank in (1, 2):
        rows["alt{}".format(rank)] = class_names[order[:, rank]]
        rows["alt{}_confidence".format(rank)] = probabilities[
            np.arange(len(ids)), order[:, rank]].round(4)

    flags = [[] for _ in ids]
    for index, value in enumerate(confidence):
        if value < 0.50:
            flags[index].append("low_confidence")

    try:
        from src.data.taxonomy import family_matrix

        matrix, families = family_matrix(tuple(class_names.tolist()))
        family_probabilities = probabilities @ matrix
        family_top = family_probabilities.argmax(1)
        fine_family = matrix[order[:, 0]].argmax(1)
        rows["family"] = [families[i] for i in family_top]
        rows["family_confidence"] = family_probabilities.max(1).round(4)
        for index in range(len(ids)):
            if family_top[index] != fine_family[index]:
                flags[index].append("family_disagrees")
    except Exception as error:  # the queue is a diagnostic, never a hard dependency
        print("Review   : family roll-up unavailable ({})".format(error))

    if baseline_labels is not None:
        for index, (new_label, old_label) in enumerate(
                zip(class_names[order[:, 0]], baseline_labels)):
            if new_label != old_label:
                flags[index].append("changed_by_prior")
        rows["label_before_prior_correction"] = baseline_labels

    rows["flags"] = ["|".join(f) for f in flags]
    frame = pd.DataFrame(rows)
    flagged = frame[frame["flags"] != ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(path, index=False)
    print("Review   : {} ({} of {} rows flagged, {:.1f}%)".format(
        path, len(flagged), len(frame), 100 * len(flagged) / max(len(frame), 1)))
    return flagged


def main(argv=None):
    args = parse_args(argv)

    if not args.model.exists():
        raise SystemExit("Checkpoint not found: {}".format(args.model))
    if not args.images.is_dir():
        raise SystemExit("Not a directory: {}".format(args.images))

    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit("No images found in {}".format(args.images))

    device = choose_device()
    model, checkpoint = load_item_type_model(args.model, device)
    tta = checkpoint.get("tta", False) if args.tta is None else args.tta
    class_names = np.asarray(checkpoint["class_names"])

    print("Model    : {} (run {}, {} classes)".format(
        checkpoint.get("model_name", "?"), checkpoint.get("run_id", "?"), len(class_names)))
    print("Device   : {} | TTA: {} | ingest: {}".format(device, tta, args.ingest))
    print("Images   : {}".format(len(paths)))

    # Prior correction needs the *raw* posteriors. The checkpoint's tau adjustment
    # is itself a blind push away from the training prior, so running a measured
    # correction on top of it corrects twice - see src/evaluation/prior_shift.py.
    probabilities = predict_proba(model, checkpoint, paths,
                                  batch_size=args.batch_size, device=device, tta=tta,
                                  ingest=args.ingest, adjust=not args.prior_correct)

    baseline_labels = None
    if args.prior_correct:
        from src.evaluation.prior_shift import (apply_prior_correction,
                                                estimate_prior_em, support_shrink)

        if len(paths) < 500:
            raise SystemExit(
                "--prior-correct needs at least 500 images to estimate a prior from; "
                "got {}. On a small folder the estimate has no statistical basis and "
                "would mis-correct rather than correct.".format(len(paths)))

        train_counts = np.asarray(checkpoint.get("class_counts") or [], dtype=float)
        if train_counts.size != len(class_names):
            raise SystemExit(
                "checkpoint records no class_counts, so how much evidence each "
                "class's prior rests on is unknown - and that is exactly what damps "
                "the correction for the starved classes. Run "
                "`python -m src.training.train_item_type --calibrate {}` first."
                .format(args.model))

        train_prior = train_counts / train_counts.sum()
        shrink = support_shrink(train_counts)
        baseline_labels = class_names[probabilities.argmax(1)]
        target_prior, _, iters, converged = estimate_prior_em(
            probabilities, train_prior, shrink=shrink)
        probabilities = apply_prior_correction(probabilities, train_prior, target_prior,
                                               alpha=args.alpha, tau_already_removed=True)
        shift = 50 * np.abs(target_prior - train_prior).sum()
        print("Prior    : EM converged={} in {} iters, shift from train prior "
              "TV={:.1f}%, alpha={}".format(converged, iters, shift, args.alpha))

    # Image ids are the filename stems, matching the dataset convention.
    ids = [path.stem for path in paths]
    top_indices = np.argsort(-probabilities, axis=1)[:, :max(1, args.top_k)]
    labels = class_names[top_indices[:, 0]]
    if baseline_labels is not None:
        changed = int((labels != baseline_labels).sum())
        print("Prior    : {} of {} labels changed ({:.1f}%)".format(
            changed, len(labels), 100 * changed / len(labels)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.submission:
        # The template's column order; the other three targets are left blank
        # for the Task 2 and Task 3 models to fill.
        frame = pd.DataFrame({"id": ids, "gender": "", "articleType": labels,
                              "season": "", "usage": ""})
    else:
        frame = pd.DataFrame({"id": ids, "articleType": labels,
                              "confidence": probabilities.max(1).round(4)})
        for rank in range(1, max(1, args.top_k)):
            column = top_indices[:, rank]
            frame["alt{}".format(rank)] = class_names[column]
            frame["alt{}_confidence".format(rank)] = (
                probabilities[np.arange(len(column)), column].round(4))

    frame.to_csv(args.out, index=False)
    print("Wrote    : {} ({} rows)".format(args.out, len(frame)))

    if args.review_queue:
        write_review_queue(args.review_queue, ids, probabilities, class_names,
                           baseline_labels)

    counts = pd.Series(labels).value_counts()
    print("\nTop predicted classes:")
    print(counts.head(10).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
