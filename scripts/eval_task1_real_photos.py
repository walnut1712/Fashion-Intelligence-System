"""Task 1's shipped classifier against the hand-labelled real photographs.

Why this exists
---------------
Every Task 1 number in the repository is measured on catalogue tiles: a centred
garment, filling roughly half the frame, on white. The headline is 90.23% test
accuracy and 96.97% at the coarser ``subCategory`` level. Neither says anything
about the input the deployed app actually receives, because the app takes an
upload.

Task 4 already measured that gap for retrieval - its composited ``photo``
benchmark reports 55.78 P@10 while the same encoder scores 23.04 on real
photographs. Task 1 had no equivalent, so the classifier half of the system was
being reported entirely on the distribution it was trained on. This scores the
**shipped 120x160 checkpoint** on the same 23 hand-labelled photographs Task 4
uses, through the same ingestion path the API serves.

Both ingestion modes are reported because they answer different questions.
``squash`` is what a catalogue tile gets; ``nobg`` segments the subject and
re-crops to catalogue framing, which is what ``auto`` routes a photograph to.

Repeats, and why they are not decoration
----------------------------------------
``nobg`` is **order-dependent on this machine**. ``foreground_mask`` prefers
``rembg`` and falls back to ``cv2.grabCut``; ``rembg`` is not installed here, so
every photograph is segmented by grabCut, whose GMM initialisation draws from
OpenCV's RNG.

That RNG is seeded once per process, so this script reproduces exactly from run
to run - two separate invocations were checked and agree to the digit. What it
does *not* do is give one photograph a stable answer: consecutive segmentations
of the same image disagree, and one image's foreground fraction moved 0.238 to
0.444 across three calls. So a photograph's score depends on how many
segmentations preceded it in the process, which is not a property a benchmark
should have, and not one the served API reproduces - it segments a single upload
per request.

A single pass therefore reports one draw rather than the score. This runs
``--repeats`` passes and reports the spread beside the mean, so the instability
is visible instead of being averaged into a number that looks solid. ``squash``
touches none of that path and is exact, which is why it is run once.

    python scripts/eval_task1_real_photos.py              # ~1 min, 7 repeats

Writes ``outputs/evaluation/task1_real_photo_accuracy.csv`` (per image, one row
per photograph per repeat) and ``..._summary.csv`` (the aggregate).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.backend.services.task1_service import Task1Service  # noqa: E402
from src.data.taxonomy import _article_to_subcategory  # noqa: E402
from src.evaluation.real_photo_arms import (  # noqa: E402
    EXCLUDED_ARTICLE_TYPES,
    real_photo_labels,
)

INPUT_DIR = PROJECT_DIR / "A2_FashionDataset" / "input_images"
LABELS_CSV = PROJECT_DIR / "A2_FashionDataset" / "input_images_labels.csv"
OUT_PER_IMAGE = PROJECT_DIR / "outputs" / "evaluation" / "task1_real_photo_accuracy.csv"
OUT_SUMMARY = PROJECT_DIR / "outputs" / "evaluation" / "task1_real_photo_summary.csv"

MODES = ("nobg", "squash")


def family_lookup():
    """``articleType`` to ``subCategory``, from the same definition the service uses.

    Deliberately not a local ``drop_duplicates`` over the metadata. The mapping
    is not a function in the source data - ``Kajal and Eyeliner`` is filed under
    ``Eyes`` on 5 rows and ``Makeup`` on 8 - so first-row-wins depends on how the
    CSV happens to be sorted, while ``src/data/taxonomy.py`` takes the mode. The
    service's ``family_matrix`` is built from that module, so scoring the coarse
    answer against anything else would compare two different taxonomies and
    silently under-report agreement. Four ``articleType`` values differ between
    the two rules today (``Kajal and Eyeliner``, ``Travel Accessory``,
    ``Wallets``, ``Wristbands``); none is currently a label here, which is
    exactly why the disagreement would have gone unnoticed.
    """
    return _article_to_subcategory()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=7,
                        help="passes over the photographs; nobg is stochastic here")
    args = parser.parse_args()

    labels = real_photo_labels(PROJECT_DIR)
    if labels is None:
        print("no scorable labels in A2_FashionDataset/input_images_labels.csv")
        print("run scripts/make_label_contact_sheet.py and fill the template in")
        return 1

    families = family_lookup()
    service = Task1Service()
    print("checkpoint {}  input {}".format(
        service.model_path.name, service.image_size))
    # Counted, not hardcoded: the folder gains photographs from time to time,
    # and a stale literal here would misreport how many were set aside.
    total = len(pd.read_csv(LABELS_CSV))
    print("{} scorable photographs of {} ({} set aside by the {} marker)".format(
        len(labels), total, total - len(labels),
        "/".join(sorted(EXCLUDED_ARTICLE_TYPES))))

    rows = []
    for repeat in range(args.repeats):
        for _, entry in labels.iterrows():
            truth = str(entry["articleType"]).strip()
            path = INPUT_DIR / str(entry["file"])
            payload = path.read_bytes()
            for mode in MODES:
                # squash is deterministic, so one pass of it is the whole answer.
                if mode == "squash" and repeat:
                    continue
                out = service.predict(payload, ingest=mode)
                rows.append({
                    "repeat": repeat,
                    "file": entry["file"],
                    "ingest": mode,
                    "truth": truth,
                    "truth_family": families.get(truth, "?"),
                    "predicted": out["label"],
                    "confidence": out["confidence"],
                    "predicted_family": out["family"]["label"],
                    "top3": " | ".join(t["label"] for t in out["top3"]),
                    "correct": out["label"].lower() == truth.lower(),
                    "family_correct": (out["family"]["label"].lower()
                                       == str(families.get(truth, "?")).lower()),
                    "in_top3": truth.lower() in [
                        t["label"].lower() for t in out["top3"]],
                })
        print("repeat {}/{} done".format(repeat + 1, args.repeats))

    per_image = pd.DataFrame(rows)

    summary = []
    for mode in MODES:
        subset = per_image[per_image["ingest"] == mode]
        by_run = subset.groupby("repeat")[["correct", "in_top3", "family_correct"]].mean() * 100
        summary.append({
            "ingest": mode,
            "n_photos": int(subset["file"].nunique()),
            "repeats": int(subset["repeat"].nunique()),
            "top1": round(by_run["correct"].mean(), 2),
            "top1_sd": round(float(np.std(by_run["correct"], ddof=1)) if len(by_run) > 1 else 0.0, 2),
            "top1_min": round(by_run["correct"].min(), 2),
            "top1_max": round(by_run["correct"].max(), 2),
            "top3": round(by_run["in_top3"].mean(), 2),
            "family": round(by_run["family_correct"].mean(), 2),
            "mean_confidence": round(subset["confidence"].mean(), 4),
        })
    summary = pd.DataFrame(summary)

    OUT_PER_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(OUT_PER_IMAGE, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    pd.set_option("display.width", 200)
    print()
    print(summary.to_string(index=False))
    print()
    print("catalogue reference for the same checkpoint: 90.23 top-1, 96.97 family")
    print()
    print("wrote {}".format(OUT_PER_IMAGE.relative_to(PROJECT_DIR)))
    print("wrote {}".format(OUT_SUMMARY.relative_to(PROJECT_DIR)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
