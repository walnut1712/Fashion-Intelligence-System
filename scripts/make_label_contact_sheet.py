"""Contact sheet for hand-labelling the 31 real-world photographs.

`A2_FashionDataset/input_images_labels.csv` ships as an empty template, so Task 4
has no P@10 on real photographs. Filling it in is a human job; this script makes
it a checking job rather than a typing job.

It renders every photograph at a size you can actually identify a garment in, and
prints beside each one the Task 1 model's five most likely `articleType` values.

Those five are a **vocabulary hint, not an answer.** The catalogue spells things
its own way (`Tshirts`, not `T-Shirts`; `Flip Flops`, not `Flipflops`) and a value
that is not spelled exactly as the catalogue spells it will silently fail to match
at evaluation time. The template is deliberately left blank: pre-filling it with
the model's own prediction would turn any subsequent P@10 into a measurement of
agreement with Task 1 rather than of correctness.

    python scripts/make_label_contact_sheet.py           # render the sheet
    python scripts/make_label_contact_sheet.py --check   # validate a filled-in template

`--check` reports blank rows, unknown `articleType` values (with the nearest legal
spelling), and files present in one of the folder or the CSV but not the other. Write
`none` for a photograph that is not one catalogue garment - a sticker, a flat-lay of
six items, a worn outfit - and it is excluded from scoring rather than counted as
unfinished.
"""

import argparse
import difflib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.evaluation.real_photo_arms import EXCLUDED_ARTICLE_TYPES  # noqa: E402
from src.models.item_type_classifier import (  # noqa: E402
    load_image_array,
    load_item_type_model,
    predict_proba,
)

IMAGE_DIR = PROJECT_DIR / "A2_FashionDataset" / "input_images"
LABEL_CSV = PROJECT_DIR / "A2_FashionDataset" / "input_images_labels.csv"
CHECKPOINT = PROJECT_DIR / "artifacts" / "task1" / "task1_cnn.pt"
SHEET_PNG = PROJECT_DIR / "outputs" / "label_contact_sheet.png"
VOCAB_TXT = PROJECT_DIR / "outputs" / "label_vocabulary.txt"

COLUMNS = 4
CELL_W = 3.6
CELL_H = 4.9
PREVIEW_MAX = 420


def load_template():
    """Return the label template, or build one from the folder if it is missing."""
    if LABEL_CSV.exists():
        return pd.read_csv(LABEL_CSV)
    files = sorted(p.name for p in IMAGE_DIR.iterdir() if p.is_file())
    return pd.DataFrame({
        "file": files,
        "articleType": "",
        "baseColour": "",
        "n_garments": "",
        "notes": "",
    })


def top5_predictions(files):
    """Return {filename: [(class_name, probability), ...]} under the served recipe."""
    model, checkpoint = load_item_type_model(CHECKPOINT)
    class_names = list(checkpoint["class_names"])
    sources = [IMAGE_DIR / name for name in files]

    # Same switches the API serves with: the checkpoint's own TTA flag and the
    # logit adjustment it was selected with. `nobg` is the ingestion the app
    # defaults to for user uploads, which is what these photographs are.
    probabilities = predict_proba(
        model,
        checkpoint,
        sources,
        tta=bool(checkpoint.get("tta", False)),
        ingest="nobg",
        adjust=True,
    )

    out = {}
    for name, row in zip(files, np.asarray(probabilities)):
        order = np.argsort(row)[::-1][:5]
        out[name] = [(class_names[i], float(row[i])) for i in order]
    return out, class_names


def preview_array(path):
    """Decode one photograph for display, longest side capped at PREVIEW_MAX."""
    from PIL import Image

    with Image.open(path) as handle:
        image = handle.convert("RGB")
        scale = PREVIEW_MAX / max(image.size)
        if scale < 1:
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size, Image.LANCZOS)
        return np.asarray(image)


def render(files, predictions):
    rows = -(-len(files) // COLUMNS)
    figure, axes = plt.subplots(
        rows,
        COLUMNS,
        figsize=(COLUMNS * CELL_W, rows * CELL_H),
    )
    axes = np.atleast_1d(axes).ravel()

    for axis in axes:
        axis.axis("off")

    for index, name in enumerate(files):
        axis = axes[index]
        try:
            axis.imshow(preview_array(IMAGE_DIR / name))
        except Exception as error:  # a format PIL cannot open should not lose the sheet
            axis.text(0.5, 0.5, "could not decode\n{}".format(error), ha="center",
                      va="center", fontsize=7, transform=axis.transAxes)

        short = name if len(name) <= 34 else name[:31] + "..."
        guesses = predictions.get(name, [])
        caption = "\n".join(
            "{:<22.22s} {:>5.1f}%".format(label, probability * 100)
            for label, probability in guesses
        )
        axis.set_title("{}\n{}".format(index + 1, short), fontsize=7, loc="left")
        axis.text(
            0.0,
            -0.03,
            caption,
            transform=axis.transAxes,
            fontsize=6.5,
            family="monospace",
            va="top",
            ha="left",
            color="#333333",
        )

    figure.suptitle(
        "Real-world photographs to label - the five values under each image are "
        "Task 1's guesses,\nshown for catalogue spelling only. Judge the photograph, "
        "not the list.",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    SHEET_PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(SHEET_PNG, dpi=150, bbox_inches="tight")
    plt.close(figure)


def check(class_names):
    """Validate a filled-in template against the folder and the legal class list."""
    if not LABEL_CSV.exists():
        print("no template at {}".format(LABEL_CSV))
        return 1

    table = pd.read_csv(LABEL_CSV)
    on_disk = {p.name for p in IMAGE_DIR.iterdir() if p.is_file()}
    listed = set(table["file"])
    legal = set(class_names)

    problems = 0

    for name in sorted(on_disk - listed):
        print("in the folder but not in the CSV : {}".format(name))
        problems += 1
    for name in sorted(listed - on_disk):
        print("in the CSV but not in the folder : {}".format(name))
        problems += 1

    values = table["articleType"].fillna("").astype(str).str.strip()
    blank = table.loc[values == "", "file"]
    for name in blank:
        print("blank articleType                : {}".format(name))
        problems += 1

    # Not every photograph in this folder is one catalogue garment - the set was
    # collected to include non-clothing, multi-garment flat-lays and worn outfits.
    # `none` marks those deliberately, so they are excluded from P@10 rather than
    # counted as unfinished work or forced into a label that is not true.
    excluded = 0
    for name, value in zip(table["file"], values):
        if value.lower() in EXCLUDED_ARTICLE_TYPES:
            excluded += 1
            continue
        if value and value not in legal:
            near = difflib.get_close_matches(value, class_names, n=1, cutoff=0.6)
            hint = "  did you mean {!r}?".format(near[0]) if near else ""
            print("not a catalogue articleType      : {!r} on {}{}".format(value, name, hint))
            problems += 1

    scored = int((values != "").sum()) - excluded
    print()
    print("{} of {} rows carry a catalogue articleType and will be scored".format(
        scored, len(table)))
    print("{} marked {} and excluded, {} problem(s)".format(
        excluded, "/".join(sorted(EXCLUDED_ARTICLE_TYPES)), problems))
    if problems == 0:
        print("template is ready - notebook 05 Part 8b will emit real P@1/P@10")
    return 0 if problems == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="validate a filled-in template instead of rendering the sheet")
    arguments = parser.parse_args()

    template = load_template()
    files = list(template["file"])

    predictions, class_names = top5_predictions(files)

    VOCAB_TXT.parent.mkdir(parents=True, exist_ok=True)
    VOCAB_TXT.write_text("\n".join(class_names) + "\n", encoding="utf-8")

    if arguments.check:
        return check(class_names)

    render(files, predictions)
    print("contact sheet : {}".format(SHEET_PNG.relative_to(PROJECT_DIR)))
    print("vocabulary    : {} ({} legal articleType values)".format(
        VOCAB_TXT.relative_to(PROJECT_DIR), len(class_names)))
    print("template      : {}".format(LABEL_CSV.relative_to(PROJECT_DIR)))
    print()
    print("Fill in the articleType column, then run this script with --check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
