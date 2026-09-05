from __future__ import annotations

import json
from pathlib import Path

import joblib
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TASK1_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "task1_120x160"
    / "task1_120x160_onecycle_best.pt"
)

TASK2B_MODEL = (
    PROJECT_ROOT
    / "artifacts"
    / "task2b_sfs"
    / "task2b_sfs_logreg.joblib"
)

OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "task2b_sfs"
    / "task1_to_sfs_mapping.json"
)


# ------------------------------------------------------------
# Taxonomy alignment only.
#
# IMPORTANT:
# This mapping does NOT assign seasons.
# It only translates the assignment articleType taxonomy into
# garment categories recognised by the external SFS dataset.
#
# "exact"      = essentially the same garment category
# "approximate"= broader/narrower taxonomy match
# "unsupported"= no defensible SFS garment equivalent
# ------------------------------------------------------------

MAPPING = {
    "Accessory Gift Set": (None, "unsupported"),
    "Backpacks": ("backpack", "exact"),
    "Bangle": ("bracelet", "approximate"),
    "Basketballs": (None, "unsupported"),
    "Belts": ("belt", "exact"),
    "Booties": ("boots", "approximate"),
    "Boxers": (None, "unsupported"),
    "Bra": ("bra", "exact"),
    "Bracelet": ("bracelet", "exact"),
    "Briefs": (None, "unsupported"),
    "Camisoles": ("camisole", "exact"),
    "Capris": ("pants", "approximate"),
    "Caps": ("hat", "approximate"),
    "Casual Shoes": ("shoes", "approximate"),
    "Churidar": ("pants", "approximate"),
    "Clutches": ("clutch", "exact"),
    "Cufflinks": (None, "unsupported"),
    "Deodorant": (None, "unsupported"),
    "Dresses": ("dress", "exact"),
    "Duffel Bag": ("bag", "approximate"),
    "Dupatta": ("scarf", "approximate"),
    "Earrings": ("earrings", "exact"),
    "Flats": ("flats", "exact"),
    "Flip Flops": ("sandals", "approximate"),
    "Formal Shoes": ("shoes", "approximate"),
    "Foundation and Primer": (None, "unsupported"),
    "Fragrance Gift Set": (None, "unsupported"),
    "Free Gifts": (None, "unsupported"),
    "Gloves": ("gloves", "exact"),
    "Handbags": ("bag", "approximate"),
    "Heels": ("heels", "exact"),
    "Innerwear Vests": (None, "unsupported"),
    "Jackets": ("jacket", "exact"),
    "Jeans": ("jeans", "exact"),
    "Jeggings": ("leggings", "approximate"),
    "Jewellery Set": (None, "unsupported"),
    "Jumpsuit": ("jumpsuit", "exact"),
    "Kajal and Eyeliner": (None, "unsupported"),
    "Kurta Sets": ("tunic", "approximate"),
    "Kurtas": ("tunic", "approximate"),
    "Kurtis": ("tunic", "approximate"),
    "Laptop Bag": ("bag", "approximate"),
    "Leggings": ("leggings", "exact"),
    "Lip Liner": (None, "unsupported"),
    "Lipstick": (None, "unsupported"),
    "Lounge Pants": ("pants", "approximate"),
    "Lounge Shorts": ("shorts", "approximate"),
    "Messenger Bag": ("bag", "approximate"),
    "Mobile Pouch": (None, "unsupported"),
    "Mufflers": ("scarf", "approximate"),
    "Nail Polish": (None, "unsupported"),
    "Necklace and Chains": ("necklace", "approximate"),
    "Night suits": (None, "unsupported"),
    "Nightdress": (None, "unsupported"),
    "Patiala": ("pants", "approximate"),
    "Pendant": ("pendant", "exact"),
    "Perfume and Body Mist": (None, "unsupported"),
    "Ring": ("ring", "exact"),
    "Rompers": ("romper", "exact"),
    "Rucksacks": ("backpack", "exact"),
    "Salwar": ("pants", "approximate"),
    "Sandals": ("sandals", "exact"),
    "Sarees": (None, "unsupported"),
    "Scarves": ("scarf", "exact"),
    "Shirts": ("shirt", "exact"),
    "Shoe Accessories": (None, "unsupported"),
    "Shorts": ("shorts", "exact"),
    "Skirts": ("skirt", "exact"),
    "Socks": ("socks", "exact"),
    "Sports Sandals": ("sandals", "approximate"),
    "Sports Shoes": ("sneakers", "approximate"),
    "Stockings": ("tights", "approximate"),
    "Stoles": ("scarf", "approximate"),
    "Sunglasses": ("sunglasses", "exact"),
    "Suspenders": ("suspenders", "exact"),
    "Sweaters": ("sweater", "exact"),
    "Sweatshirts": ("sweatshirt", "exact"),
    "Swimwear": ("swimwear", "exact"),
    "Ties": ("tie", "exact"),
    "Tops": ("top", "exact"),
    "Track Pants": ("pants", "approximate"),
    "Tracksuits": (None, "unsupported"),
    "Travel Accessory": (None, "unsupported"),
    "Trousers": ("pants", "approximate"),
    "Trunk": (None, "unsupported"),
    "Tshirts": ("shirt", "approximate"),
    "Tunics": ("tunic", "exact"),
    "Waist Pouch": ("bag", "approximate"),
    "Waistcoat": ("vest", "exact"),
    "Wallets": ("wallet", "exact"),
    "Watches": ("watch", "exact"),
    "Water Bottle": (None, "unsupported"),
}


def main():

    if not TASK1_CHECKPOINT.exists():
        raise FileNotFoundError(TASK1_CHECKPOINT)

    if not TASK2B_MODEL.exists():
        raise FileNotFoundError(TASK2B_MODEL)

    checkpoint = torch.load(
        TASK1_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    task1_classes = checkpoint.get("class_names")

    if task1_classes is None:
        raise KeyError(
            "Task1 checkpoint does not contain class_names"
        )

    bundle = joblib.load(TASK2B_MODEL)

    sfs_categories = set(bundle["categories"])

    task1_classes = list(task1_classes)

    print("Task1 classes:", len(task1_classes))
    print("SFS categories:", len(sfs_categories))

    missing_mapping = [
        c for c in task1_classes
        if c not in MAPPING
    ]

    extra_mapping = [
        c for c in MAPPING
        if c not in task1_classes
    ]

    if missing_mapping:
        print("\nERROR - Task1 classes missing mapping:")
        for c in missing_mapping:
            print(" -", c)

    if extra_mapping:
        print("\nWARNING - mapping entries not in Task1:")
        for c in extra_mapping:
            print(" -", c)

    invalid_targets = []

    records = {}

    counts = {
        "exact": 0,
        "approximate": 0,
        "unsupported": 0,
    }

    for article_type in task1_classes:

        target, status = MAPPING[article_type]

        if (
            target is not None
            and target not in sfs_categories
        ):
            invalid_targets.append(
                (article_type, target)
            )

        counts[status] += 1

        records[article_type] = {
            "sfs_category": target,
            "status": status,
        }

    if invalid_targets:
        print("\nERROR - targets absent from trained SFS model:")

        for source, target in invalid_targets:
            print(
                f" - {source} -> {target}"
            )

        raise RuntimeError(
            "Invalid Task1 -> SFS mapping"
        )

    print("\nCoverage")
    print("--------")

    for key, value in counts.items():
        print(
            f"{key:12s}: "
            f"{value:2d} / {len(task1_classes)} "
            f"({value / len(task1_classes):.1%})"
        )

    supported = (
        counts["exact"]
        + counts["approximate"]
    )

    print(
        f"\nTotal supported: "
        f"{supported}/{len(task1_classes)} "
        f"({supported / len(task1_classes):.1%})"
    )

    print("\nUnsupported article types")
    print("-------------------------")

    for article_type in task1_classes:
        if MAPPING[article_type][1] == "unsupported":
            print(" -", article_type)

    payload = {
        "description": (
            "Taxonomy alignment from Task1 articleType classes "
            "to SFS garment categories. This mapping contains no "
            "season labels or season heuristics."
        ),
        "task1_class_count": len(task1_classes),
        "sfs_category_count": len(sfs_categories),
        "coverage": counts,
        "supported_count": supported,
        "supported_fraction": (
            supported
            / len(task1_classes)
        ),
        "mapping": records,
    }

    OUTPUT.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()