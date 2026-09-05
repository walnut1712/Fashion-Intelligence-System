from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupShuffleSplit


SEASONS = ["spring", "summer", "fall", "winter"]


# ============================================================
# SFS TAG TAXONOMY
#
# This is ONLY taxonomy normalisation.
# It does NOT contain any season rules.
# ============================================================

ALIAS_TO_CATEGORY = {
    "dress": "dress",
    "dresses": "dress",

    "bag": "bag",
    "bags": "bag",
    "handbag": "bag",
    "handbags": "bag",

    "shirt": "shirt",
    "shirts": "shirt",

    "skirt": "skirt",
    "skirts": "skirt",

    "boot": "boots",
    "boots": "boots",

    "top": "top",
    "tops": "top",

    "shoe": "shoes",
    "shoes": "shoes",

    "jean": "jeans",
    "jeans": "jeans",

    "jacket": "jacket",
    "jackets": "jacket",

    "short": "shorts",
    "shorts": "shorts",

    "pant": "pants",
    "pants": "pants",
    "trouser": "pants",
    "trousers": "pants",

    "heel": "heels",
    "heels": "heels",

    "sweater": "sweater",
    "sweaters": "sweater",

    "hat": "hat",
    "hats": "hat",
    "cap": "hat",
    "caps": "hat",

    "blouse": "blouse",
    "blouses": "blouse",

    "blazer": "blazer",
    "blazers": "blazer",

    "coat": "coat",
    "coats": "coat",

    "sunglasses": "sunglasses",
    "glasses": "sunglasses",

    "necklace": "necklace",
    "necklaces": "necklace",

    "belt": "belt",
    "belts": "belt",

    "cardigan": "cardigan",
    "cardigans": "cardigan",

    "sandal": "sandals",
    "sandals": "sandals",

    "scarf": "scarf",
    "scarves": "scarf",

    "tight": "tights",
    "tights": "tights",
    "stocking": "tights",
    "stockings": "tights",

    "legging": "leggings",
    "leggings": "leggings",

    "wedge": "wedges",
    "wedges": "wedges",

    "vest": "vest",
    "vests": "vest",
    "waistcoat": "vest",
    "waistcoats": "vest",

    "flat": "flats",
    "flats": "flats",

    "purse": "purse",
    "purses": "purse",

    "sneaker": "sneakers",
    "sneakers": "sneakers",

    "bracelet": "bracelet",
    "bracelets": "bracelet",
    "bangle": "bracelet",
    "bangles": "bracelet",

    "pump": "pumps",
    "pumps": "pumps",

    "sock": "socks",
    "socks": "socks",

    "romper": "romper",
    "rompers": "romper",

    "watch": "watch",
    "watches": "watch",

    "clutch": "clutch",
    "clutches": "clutch",

    "earring": "earrings",
    "earrings": "earrings",

    "ring": "ring",
    "rings": "ring",

    "sweatshirt": "sweatshirt",
    "sweatshirts": "sweatshirt",
    "hoodie": "sweatshirt",
    "hoodies": "sweatshirt",

    "swimwear": "swimwear",
    "bikini": "swimwear",
    "bikinis": "swimwear",

    "bra": "bra",
    "bras": "bra",

    "backpack": "backpack",
    "backpacks": "backpack",
    "rucksack": "backpack",
    "rucksacks": "backpack",

    "tunic": "tunic",
    "tunics": "tunic",

    "jumpsuit": "jumpsuit",
    "jumpsuits": "jumpsuit",

    "wallet": "wallet",
    "wallets": "wallet",

    "tie": "tie",
    "ties": "tie",

    "glove": "gloves",
    "gloves": "gloves",

    "suspender": "suspenders",
    "suspenders": "suspenders",

    "pendant": "pendant",
    "pendants": "pendant",

    "camisole": "camisole",
    "camisoles": "camisole",
}


# ============================================================
# PARSING
# ============================================================

def extract_season(styles: object) -> str | None:
    if pd.isna(styles):
        return None

    tokens = [
        x.strip().lower()
        for x in str(styles).split(",")
    ]

    for token in tokens:
        if token in SEASONS:
            return token

    return None


def extract_categories(tags: object) -> list[str]:
    if pd.isna(tags):
        return []

    categories: set[str] = set()

    for phrase in str(tags).split(","):
        words = re.findall(
            r"[A-Za-z]+(?:'[A-Za-z]+)?",
            phrase.lower(),
        )

        if not words:
            continue

        category = ALIAS_TO_CATEGORY.get(
            words[-1]
        )

        if category is not None:
            categories.add(category)

    return sorted(categories)


def make_multihot(
    category_lists: pd.Series,
    categories: list[str],
) -> np.ndarray:

    index = {
        name: i
        for i, name in enumerate(categories)
    }

    x = np.zeros(
        (
            len(category_lists),
            len(categories),
        ),
        dtype=np.float32,
    )

    for row, items in enumerate(
        category_lists
    ):
        for item in items:
            x[row, index[item]] = 1.0

    return x


# ============================================================
# METRICS
# ============================================================

def metrics(
    y_true,
    y_pred,
    probs,
    classes,
) -> dict:

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "log_loss": float(
            log_loss(
                y_true,
                probs,
                labels=classes,
            )
        ),
    }


def majority_metrics(
    y_true,
    majority_label,
) -> dict:

    pred = np.full(
        len(y_true),
        majority_label,
        dtype=object,
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                pred,
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                pred,
                average="macro",
                zero_division=0,
            )
        ),

        "weighted_f1": float(
            f1_score(
                y_true,
                pred,
                average="weighted",
                zero_division=0,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                pred,
            )
        ),
    }


# ============================================================
# LEAKAGE-SAFE USER SPLIT
# ============================================================

def split_by_user(
    df: pd.DataFrame,
    seed: int = 42,
):

    indices = np.arange(len(df))

    y = df["season"].to_numpy()

    groups = df[
        "user_name"
    ].to_numpy()

    first = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=seed,
    )

    train_idx, temp_idx = next(
        first.split(
            indices,
            y,
            groups,
        )
    )

    second = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=seed + 1,
    )

    val_rel, test_rel = next(
        second.split(
            temp_idx,
            y[temp_idx],
            groups[temp_idx],
        )
    )

    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    return (
        train_idx,
        val_idx,
        test_idx,
    )


# ============================================================
# CATEGORY → SEASON STATISTICS
# ============================================================

def category_prior_table(
    df: pd.DataFrame,
    categories: list[str],
) -> pd.DataFrame:

    rows = []

    for category in categories:

        mask = df[
            "categories"
        ].map(
            lambda xs, c=category:
            c in xs
        )

        part = df.loc[
            mask,
            "season",
        ]

        counts = part.value_counts()

        n = int(len(part))

        # Laplace smoothing
        denom = (
            n +
            len(SEASONS)
        )

        row = {
            "category": category,
            "n": n,
        }

        for season in SEASONS:

            row[
                f"count_{season}"
            ] = int(
                counts.get(
                    season,
                    0,
                )
            )

            row[
                f"p_{season}"
            ] = float(
                (
                    counts.get(
                        season,
                        0,
                    ) + 1
                )
                / denom
            )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            "n",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    csv_path = (
        project_root
        / "external_data"
        / "sfs"
        / "SFS_metadata.csv"
    )

    out_dir = (
        project_root
        / "artifacts"
        / "task2b_sfs"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"SFS metadata not found: "
            f"{csv_path}"
        )

    print(
        "Loading:",
        csv_path,
    )

    raw = pd.read_csv(
        csv_path
    )

    print(
        "Raw shape:",
        raw.shape,
    )

    required = {
        "user_name",
        "pic_name",
        "styles",
        "tags",
    }

    missing = required.difference(
        raw.columns
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    df = raw[
        [
            "user_name",
            "pic_name",
            "styles",
            "tags",
        ]
    ].copy()

    df["season"] = (
        df["styles"]
        .map(extract_season)
    )

    df["categories"] = (
        df["tags"]
        .map(extract_categories)
    )

    valid_season = int(
        df["season"]
        .notna()
        .sum()
    )

    df = df[
        df["season"].notna()
        &
        df["categories"].map(bool)
    ].reset_index(drop=True)

    categories = sorted({
        category
        for items in df["categories"]
        for category in items
    })

    x = make_multihot(
        df["categories"],
        categories,
    )

    y = df[
        "season"
    ].to_numpy()

    print(
        "Valid season rows:",
        valid_season,
    )

    print(
        "Usable rows:",
        len(df),
    )

    print(
        "Usable proportion: "
        "{:.2%}".format(
            len(df)
            / len(raw)
        )
    )

    print(
        "Garment categories:",
        len(categories),
    )

    print(
        "Season distribution:"
    )

    print(
        df["season"]
        .value_counts()
        .to_string()
    )

    (
        train_idx,
        val_idx,
        test_idx,
    ) = split_by_user(
        df,
        seed=42,
    )

    print(
        "\nSplit sizes"
    )

    print(
        "train:",
        len(train_idx),
        "users:",
        df.iloc[
            train_idx
        ]["user_name"].nunique(),
    )

    print(
        "val:  ",
        len(val_idx),
        "users:",
        df.iloc[
            val_idx
        ]["user_name"].nunique(),
    )

    print(
        "test: ",
        len(test_idx),
        "users:",
        df.iloc[
            test_idx
        ]["user_name"].nunique(),
    )

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    train_users = set(
        df.iloc[
            train_idx
        ]["user_name"]
    )

    val_users = set(
        df.iloc[
            val_idx
        ]["user_name"]
    )

    test_users = set(
        df.iloc[
            test_idx
        ]["user_name"]
    )

    assert train_users.isdisjoint(
        val_users
    )

    assert train_users.isdisjoint(
        test_users
    )

    assert val_users.isdisjoint(
        test_users
    )

    # --------------------------------------------------------
    # Majority baseline
    # --------------------------------------------------------

    majority_label = (
        df.iloc[
            train_idx
        ]["season"]
        .value_counts()
        .idxmax()
    )

    baseline = (
        majority_metrics(
            y[test_idx],
            majority_label,
        )
    )

    # --------------------------------------------------------
    # Candidate models
    # --------------------------------------------------------

    candidates = {

        "logreg_unweighted":
            LogisticRegression(
                max_iter=400,
                solver="lbfgs",
                C=1.0,
            ),

        "logreg_balanced":
            LogisticRegression(
                max_iter=400,
                solver="lbfgs",
                C=1.0,
                class_weight="balanced",
            ),
    }

    candidate_results = {}
    fitted = {}

    for name, model in (
        candidates.items()
    ):

        print(
            "\nTraining",
            name,
        )

        model.fit(
            x[train_idx],
            y[train_idx],
        )

        val_pred = model.predict(
            x[val_idx]
        )

        val_prob = (
            model.predict_proba(
                x[val_idx]
            )
        )

        val_metrics = metrics(
            y[val_idx],
            val_pred,
            val_prob,
            model.classes_,
        )

        candidate_results[
            name
        ] = {
            "val":
                val_metrics
        }

        fitted[
            name
        ] = model

        print(
            "VAL:",
            json.dumps(
                val_metrics,
                indent=2,
            ),
        )

    # --------------------------------------------------------
    # Select using validation Macro-F1
    # --------------------------------------------------------

    best_name = max(
        candidate_results,
        key=lambda name:
        candidate_results[
            name
        ]["val"]["macro_f1"],
    )

    best_model = fitted[
        best_name
    ]

    test_pred = (
        best_model.predict(
            x[test_idx]
        )
    )

    test_prob = (
        best_model.predict_proba(
            x[test_idx]
        )
    )

    test_metrics = metrics(
        y[test_idx],
        test_pred,
        test_prob,
        best_model.classes_,
    )

    candidate_results[
        best_name
    ]["test"] = (
        test_metrics
    )

    print(
        "\nSelected:",
        best_name,
    )

    print(
        "TEST:",
        json.dumps(
            test_metrics,
            indent=2,
        ),
    )

    print(
        "Majority baseline:",
        json.dumps(
            baseline,
            indent=2,
        ),
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y[test_idx],
        test_pred,
        labels=best_model.classes_,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(
        report
    ).T.to_csv(
        out_dir
        / "classification_report.csv"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y[test_idx],
        test_pred,
        labels=best_model.classes_,
    )

    pd.DataFrame(
        cm,
        index=best_model.classes_,
        columns=best_model.classes_,
    ).to_csv(
        out_dir
        / "confusion_matrix.csv"
    )

    # --------------------------------------------------------
    # TRAIN-only category statistics
    # --------------------------------------------------------

    priors = category_prior_table(
        df.iloc[
            train_idx
        ].reset_index(
            drop=True
        ),
        categories,
    )

    priors.to_csv(
        out_dir
        / "category_season_priors.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Held-out test predictions
    # --------------------------------------------------------

    predictions = pd.DataFrame({
        "user_name":
            df.iloc[
                test_idx
            ]["user_name"]
            .to_numpy(),

        "pic_name":
            df.iloc[
                test_idx
            ]["pic_name"]
            .to_numpy(),

        "true_season":
            y[test_idx],

        "predicted_season":
            test_pred,
    })

    for i, class_name in enumerate(
        best_model.classes_
    ):

        predictions[
            f"p_{class_name}"
        ] = test_prob[:, i]

    predictions.to_csv(
        out_dir
        / "test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save model bundle
    # --------------------------------------------------------

    bundle = {
        "model":
            best_model,

        "categories":
            categories,

        "classes":
            best_model.classes_.tolist(),

        "alias_to_category":
            ALIAS_TO_CATEGORY,
    }

    joblib.dump(
        bundle,
        out_dir
        / "task2b_sfs_logreg.joblib",
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {

        "task":
            "Task 2B - external-data season recommendation",

        "source":
            "SFS_metadata.csv",

        "raw_rows":
            int(len(raw)),

        "valid_season_rows":
            valid_season,

        "usable_rows":
            int(len(df)),

        "usable_fraction":
            float(
                len(df)
                / len(raw)
            ),

        "num_users":
            int(
                df[
                    "user_name"
                ].nunique()
            ),

        "num_garment_categories":
            int(
                len(categories)
            ),

        "garment_categories":
            categories,

        "season_counts": {
            k: int(v)
            for k, v in (
                df["season"]
                .value_counts()
                .items()
            )
        },

        "split": {
            "method":
                "GroupShuffleSplit by user_name",

            "train_rows":
                int(len(train_idx)),

            "val_rows":
                int(len(val_idx)),

            "test_rows":
                int(len(test_idx)),

            "train_users":
                int(
                    df.iloc[
                        train_idx
                    ]["user_name"]
                    .nunique()
                ),

            "val_users":
                int(
                    df.iloc[
                        val_idx
                    ]["user_name"]
                    .nunique()
                ),

            "test_users":
                int(
                    df.iloc[
                        test_idx
                    ]["user_name"]
                    .nunique()
                ),
        },

        "selection_metric":
            "validation macro_f1",

        "selected_model":
            best_name,

        "candidate_metrics":
            candidate_results,

        "test_metrics":
            test_metrics,

        "majority_baseline_label":
            majority_label,

        "majority_baseline_test_metrics":
            baseline,

        "important_note":
            (
                "This model learns season occurrence "
                "from SFS street-fashion metadata. "
                "It is an auxiliary recommendation "
                "layer and does not replace the "
                "official Task 2A CNN trained on "
                "the assignment season labels."
            ),
    }

    (
        out_dir
        / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Useful examples
    # --------------------------------------------------------

    print(
        "\nTop category priors "
        "learned from TRAIN only:"
    )

    show = [
        "sandals",
        "shorts",
        "sweater",
        "coat",
        "jeans",
        "watch",
        "swimwear",
    ]

    available = priors[
        priors[
            "category"
        ].isin(show)
    ]

    print(
        available.to_string(
            index=False
        )
    )

    print(
        "\nSaved artifacts to:",
        out_dir,
    )

    for path in sorted(
        out_dir.iterdir()
    ):
        print(
            " -",
            path.name,
        )


if __name__ == "__main__":
    main()