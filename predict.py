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
    parser.add_argument("--no-tta", dest="tta", action="store_false",
                        help="force it off (default: whatever the checkpoint recorded)")
    return parser.parse_args(argv)


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
    print("Device   : {} | TTA: {}".format(device, tta))
    print("Images   : {}".format(len(paths)))

    probabilities = predict_proba(model, checkpoint, paths,
                                  batch_size=args.batch_size, device=device, tta=tta)

    # Image ids are the filename stems, matching the dataset convention.
    ids = [path.stem for path in paths]
    top_indices = np.argsort(-probabilities, axis=1)[:, :max(1, args.top_k)]
    labels = class_names[top_indices[:, 0]]

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

    counts = pd.Series(labels).value_counts()
    print("\nTop predicted classes:")
    print(counts.head(10).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
