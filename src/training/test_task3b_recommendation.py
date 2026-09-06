from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SFS_CSV = (
    ROOT
    / "external_data"
    / "sfs"
    / "SFS_metadata.csv"
)

TASK2B_BUNDLE = (
    ROOT
    / "artifacts"
    / "task2b_sfs"
    / "task2b_sfs_logreg.joblib"
)

TASK1_TO_SFS = (
    ROOT
    / "artifacts"
    / "task2b_sfs"
    / "task1_to_sfs_mapping.json"
)


SEASONS = {
    "spring",
    "summer",
    "fall",
    "winter",
}


# Generic SFS labels do not add useful information to the UI.
HIDDEN_OCCASIONS = {
    "Everyday",
    "Other",
}


# Fine-grained occasions allowed for each primary Task3 usage.
# This is deterministic recommendation post-processing;
# it does NOT alter Task3 predictions or metrics.
USAGE_OCCASION_WEIGHTS = {

    "Casual": {
        "Work": 0.85,
        "Brunch": 1.00,
        "Casual Party": 0.90,
        "Dinner Date": 0.75,
        "Going Out With Friends": 1.00,
        "Lunch Date": 0.95,
        "Shopping": 1.00,
        "School": 0.95,
        "Girls Night Out": 0.80,
        "Beach": 0.90,
        "Picnic": 1.00,
        "Birthday": 0.85,
        "Music Concert": 0.75,
        "Traveling": 1.00,
        "Movie night": 1.00,
        "BBQ Party": 0.95,
        "Amusement Park": 1.00,
        "Vacation": 1.00,
        "First Date": 0.80,
        "Shopping Date": 0.95,
        "Farmer/s Market": 1.00,
        "Museum Outing": 0.90,
        "Boys Night Out": 0.80,
        "Walking The Dog": 1.00,
        "The Fair": 0.95,
    },

    "Formal": {
        "Work": 0.90,
        "Dinner Date": 0.95,
        "Cocktail": 1.00,
        "Art Opening": 0.95,
        "Girls Night Out": 0.75,
        "Photo Shoot": 0.85,
        "Dinner Party": 1.00,
        "Fashion Show": 0.95,
        "Clubbing": 0.70,
        "Birthday": 0.75,
        "Music Concert": 0.70,
        "Holiday Party": 1.00,
        "Company Event": 1.00,
        "Anniversary": 1.00,
        "First Date": 0.80,
        "Wedding": 1.00,
        "Romantic Dinner": 1.00,
        "Blind Date": 0.80,
        "Formal": 1.00,
        "Work Happy Hour": 0.85,
        "Conference": 1.00,
        "Valentine/s Day": 0.95,
        "Theatre/Opera/Symphony": 1.00,
        "Interview": 1.00,
        "Wine Tasting": 0.90,
        "Baby Shower": 0.85,
        "Prom": 1.00,
        "Bridal Shower": 0.95,
        "Graduation": 0.95,
        "Bachelorette Party": 0.85,
    },

    "Sports": {
        "Beach": 0.90,
        "Picnic": 0.65,
        "Traveling": 0.70,
        "Amusement Park": 0.75,
        "Vacation": 0.80,
        "Hiking": 1.00,
        "Walking The Dog": 0.75,
        "Pool Party": 0.90,
        "Game Day": 1.00,
        "Gym": 1.00,
        "Yoga": 1.00,
    },

    "Ethnic": {
        "Casual Party": 0.65,
        "Dinner Date": 0.65,
        "Art Opening": 0.65,
        "Photo Shoot": 0.85,
        "Dinner Party": 0.85,
        "Fashion Show": 0.85,
        "Birthday": 0.80,
        "Holiday Party": 0.90,
        "Company Event": 0.65,
        "Anniversary": 0.90,
        "Wedding": 1.00,
        "Romantic Dinner": 0.75,
        "Formal": 0.85,
        "Theatre/Opera/Symphony": 0.75,
        "Baby Shower": 0.80,
        "Bridal Shower": 0.95,
        "Graduation": 0.80,
    },
}


def extract_occasion(styles):
    parts = [
        x.strip()
        for x in str(styles).split(",")
        if x.strip()
    ]

    for i, token in enumerate(parts):
        if token.lower() in SEASONS:
            if i < 1:
                return None

            occasion = parts[i - 1].strip()

            return occasion or None

    return None


def extract_categories(
    tags,
    alias_to_category,
):
    result = set()

    for phrase in str(tags).split(","):

        words = re.findall(
            r"[a-z]+",
            phrase.lower(),
        )

        for token in reversed(words):

            category = (
                alias_to_category.get(
                    token
                )
            )

            if category:
                result.add(category)
                break

    return sorted(result)


def build_counts(
    df,
    alias_to_category,
):
    category_counts = defaultdict(
        Counter
    )

    global_counts = Counter()

    usable_posts = 0

    for row in df.itertuples(
        index=False
    ):

        occasion = extract_occasion(
            row.styles
        )

        if not occasion:
            continue

        categories = extract_categories(
            row.tags,
            alias_to_category,
        )

        if not categories:
            continue

        usable_posts += 1

        # Global occasion frequency is counted once per post.
        global_counts[
            occasion
        ] += 1

        # Category conditional frequency.
        for category in categories:
            category_counts[
                category
            ][occasion] += 1

    return (
        category_counts,
        global_counts,
        usable_posts,
    )


def recommend(
    category,
    usage,
    category_counts,
    global_counts,
    top_k=2,
):
    distribution = (
        category_counts.get(
            category
        )
    )

    if not distribution:
        return []

    compatibility = (
        USAGE_OCCASION_WEIGHTS.get(
            usage,
            {}
        )
    )

    if not compatibility:
        return []

    category_total = sum(
        distribution.values()
    )

    global_total = sum(
        global_counts.values()
    )

    scored = []

    for occasion, usage_weight in compatibility.items():

        if occasion in HIDDEN_OCCASIONS:
            continue

        count = distribution.get(
            occasion,
            0,
        )

        if count <= 0:
            continue

        # P(occasion | category)
        category_prior = (
            count
            / category_total
        )

        # P(occasion) across SFS.
        global_prior = (
            global_counts.get(
                occasion,
                0
            )
            / global_total
        )

        if global_prior <= 0:
            continue

        # How unusually common this occasion is for the category.
        lift = (
            category_prior
            / global_prior
        )

        # Prevent tiny rare labels with extreme lift from dominating.
        support = (
            count
            / (count + 50.0)
        )

        # Recommendation score:
        # - sqrt(prior): retain empirical support
        # - lift^0.75: reward category specificity
        # - support: damp tiny counts
        # - usage_weight: respect Task3 primary usage
        score = (
            math.sqrt(
                category_prior
            )
            * (
                min(lift, 8.0)
                ** 0.75
            )
            * support
            * usage_weight
        )

        scored.append(
            {
                "occasion":
                    occasion,

                "count":
                    count,

                "prior":
                    category_prior,

                "global_prior":
                    global_prior,

                "lift":
                    lift,

                "usage_weight":
                    usage_weight,

                "score":
                    score,
            }
        )

    scored.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    if not scored:
        return []

    best = scored[0]["score"]

    selected = []

    for row in scored:

        if len(selected) >= top_k:
            break

        # Weak second labels are hidden.
        if (
            selected
            and row["score"]
            < best * 0.60
        ):
            break

        selected.append(row)

    return selected


def load_task1_mapping():

    data = json.loads(
        TASK1_TO_SFS.read_text(
            encoding="utf-8"
        )
    )

    if isinstance(data, dict):

        if "mapping" in data:
            data = data[
                "mapping"
            ]

        elif "task1_to_sfs" in data:
            data = data[
                "task1_to_sfs"
            ]

    return data


def resolve_sfs_category(
    article_type,
    mapping,
):
    value = mapping.get(
        article_type
    )

    if value is None:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, dict):

        for key in (
            "sfs_category",
            "category",
            "mapped_category",
            "target",
        ):
            if value.get(key):
                return value[key]

    return None


def main():

    print(
        "Loading Task2B taxonomy..."
    )

    bundle = joblib.load(
        TASK2B_BUNDLE
    )

    alias_to_category = dict(
        bundle[
            "alias_to_category"
        ]
    )

    print(
        "taxonomy categories:",
        len(bundle["categories"]),
    )

    print(
        "Loading SFS..."
    )

    df = pd.read_csv(
        SFS_CSV,
        dtype=str,
        keep_default_na=False,
    )

    (
        category_counts,
        global_counts,
        usable,
    ) = build_counts(
        df,
        alias_to_category,
    )

    print(
        "usable SFS posts:",
        usable,
    )

    print(
        "categories with occasion data:",
        len(category_counts),
    )

    mapping = load_task1_mapping()

    tests = [
        ("Shirts", "Casual"),
        ("Jeans", "Casual"),
        ("Shorts", "Casual"),
        ("Dresses", "Formal"),
        ("Heels", "Formal"),
        ("Formal Shoes", "Formal"),
        ("Sports Shoes", "Sports"),
        ("Swimwear", "Sports"),
        ("Kurtas", "Ethnic"),
        ("Sarees", "Ethnic"),
        ("Bra", "Casual"),
        ("Watches", "Casual"),
    ]

    print()
    print("=" * 100)

    print(
        "TASK3B LIFT-BASED RECOMMENDATION TEST"
    )

    print("=" * 100)

    for article_type, usage in tests:

        category = (
            resolve_sfs_category(
                article_type,
                mapping,
            )
        )

        print()

        print(
            f"{article_type:18s}"
            f" | Usage={usage:7s}"
            f" | SFS={category}"
        )

        if not category:
            print(
                "  -> unavailable"
            )
            continue

        rows = recommend(
            category,
            usage,
            category_counts,
            global_counts,
            top_k=2,
        )

        if not rows:
            print(
                "  -> unavailable"
            )
            continue

        label = " / ".join(
            row["occasion"]
            for row in rows
        )

        print(
            "  recommend:",
            label,
        )

        for row in rows:

            print(
                "   ",
                f"{row['occasion']:28s}",
                f"count={row['count']:6d}",
                f"prior={row['prior']:.4f}",
                f"lift={row['lift']:.2f}",
                f"usage_w={row['usage_weight']:.2f}",
                f"score={row['score']:.6f}",
            )

        print(
            "  next candidates:"
        )

        # Show extra candidates for diagnosis only.
        all_rows = []

        distribution = (
            category_counts[
                category
            ]
        )

        for occasion, count in distribution.items():

            if (
                occasion
                in HIDDEN_OCCASIONS
            ):
                continue

            if (
                occasion
                not in
                USAGE_OCCASION_WEIGHTS[
                    usage
                ]
            ):
                continue

            category_prior = (
                count
                / sum(
                    distribution.values()
                )
            )

            global_prior = (
                global_counts[
                    occasion
                ]
                / sum(
                    global_counts.values()
                )
            )

            lift = (
                category_prior
                / global_prior
            )

            all_rows.append(
                (
                    occasion,
                    count,
                    category_prior,
                    lift,
                )
            )

        all_rows.sort(
            key=lambda x: (
                x[2] * x[3]
            ),
            reverse=True,
        )

        for row in all_rows[:5]:
            print(
                "    ",
                f"{row[0]:28s}",
                f"count={row[1]:6d}",
                f"prior={row[2]:.4f}",
                f"lift={row[3]:.2f}",
            )


if __name__ == "__main__":
    main()