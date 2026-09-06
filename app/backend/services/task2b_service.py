from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "task2b_sfs"
    / "task2b_sfs_logreg.joblib"
)

DEFAULT_MAPPING_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "task2b_sfs"
    / "task1_to_sfs_mapping.json"
)

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "task2b_sfs"
    / "manifest.json"
)


# ============================================================
# Suitable-season display policy
# ============================================================

SEASON_CYCLE = [
    "spring",
    "summer",
    "fall",
    "winter",
]

ALL_SEASON_SPREAD = 0.10
RELATIVE_THRESHOLD = 0.45
ABSOLUTE_THRESHOLD = 0.22


# ============================================================
# Explicit season-neutral categories
#
# These are semantic suitability rules, NOT labels learned
# from SFS and NOT Task2A catalogue-season predictions.
# ============================================================

SEASON_NEUTRAL_TASK1_TYPES = {
    # Underwear / sleepwear
    "Boxers",
    "Briefs",
    "Bra",
    "Camisoles",
    "Innerwear Vests",
    "Night suits",
    "Nightdress",

    # Cosmetics / personal care
    "Deodorant",
    "Foundation and Primer",
    "Fragrance Gift Set",
    "Kajal and Eyeliner",
    "Lip Liner",
    "Lipstick",
    "Nail Polish",
    "Perfume and Body Mist",
}

SEASON_NEUTRAL_SFS_CATEGORIES = {
    "bra",
    "camisole",
}


class Task2BService:
    """
    Auxiliary suitable-season recommendation service.

    Main learned path:
        Task1 top-1 articleType
            -> taxonomy mapping
            -> one-hot SFS garment category
            -> SFS-trained logistic regression
            -> season probabilities
            -> display policy

    Additional transparent fallback:
        intrinsically season-neutral category
            -> All Season

    Important:
    - Task2B does NOT replace Task2A.
    - Task2A remains catalogue/collection season.
    - Unsupported non-neutral categories remain unavailable.
    """

    METHOD = "hard_top1_sfs"

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        mapping_path: Path | str = DEFAULT_MAPPING_PATH,
        manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    ):
        self.model_path = Path(model_path)
        self.mapping_path = Path(mapping_path)
        self.manifest_path = Path(manifest_path)

        self._validate_files()

        self.bundle = joblib.load(
            self.model_path
        )

        with self.mapping_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            mapping_payload = json.load(file)

        if self.manifest_path.exists():
            with self.manifest_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                self.manifest = json.load(file)
        else:
            self.manifest = {}

        self.model = self.bundle["model"]

        self.categories = list(
            self.bundle["categories"]
        )

        self.category_to_index = {
            category: index
            for index, category
            in enumerate(self.categories)
        }

        self.mapping = mapping_payload["mapping"]

        self.task1_class_count = int(
            mapping_payload.get(
                "task1_class_count",
                len(self.mapping),
            )
        )

        self.supported_count = int(
            mapping_payload.get(
                "supported_count",
                sum(
                    1
                    for info in self.mapping.values()
                    if info.get("sfs_category") is not None
                ),
            )
        )

        self.supported_fraction = float(
            mapping_payload.get(
                "supported_fraction",
                (
                    self.supported_count
                    / self.task1_class_count
                    if self.task1_class_count
                    else 0.0
                ),
            )
        )

        raw_classes = getattr(
            self.model,
            "classes_",
            None,
        )

        if raw_classes is None:
            raw_classes = self.bundle.get(
                "classes"
            )

        if raw_classes is None:
            raise RuntimeError(
                "Task2B model does not expose season classes."
            )

        self.class_names = [
            str(value)
            for value in raw_classes
        ]

        self.num_classes = len(
            self.class_names
        )

        expected_seasons = {
            "fall",
            "spring",
            "summer",
            "winter",
        }

        if (
            self.num_classes != 4
            or set(self.class_names)
            != expected_seasons
        ):
            raise RuntimeError(
                "Unexpected Task2B season vocabulary: "
                f"{self.class_names}"
            )

        self.device = "cpu"

        self._validate_mapping()

    # ========================================================
    # Validation
    # ========================================================

    def _validate_files(self) -> None:
        missing = [
            path
            for path in (
                self.model_path,
                self.mapping_path,
            )
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing Task2B artifact(s): "
                + ", ".join(
                    str(path)
                    for path in missing
                )
            )

    def _validate_mapping(self) -> None:
        invalid = []

        for article_type, info in self.mapping.items():
            category = info.get(
                "sfs_category"
            )

            if (
                category is not None
                and category not in self.category_to_index
            ):
                invalid.append(
                    (
                        article_type,
                        category,
                    )
                )

        if invalid:
            text = ", ".join(
                f"{source}->{target}"
                for source, target
                in invalid
            )

            raise RuntimeError(
                "Task1-to-SFS mapping contains "
                "categories absent from Task2B: "
                + text
            )

    # ========================================================
    # Projection
    # ========================================================

    def _build_one_hot(
        self,
        category: str,
    ) -> np.ndarray:

        if category not in self.category_to_index:
            raise ValueError(
                f"Unknown SFS category: {category}"
            )

        vector = np.zeros(
            len(self.categories),
            dtype=np.float32,
        )

        vector[
            self.category_to_index[
                category
            ]
        ] = 1.0

        return vector

    # ========================================================
    # Raw SFS probabilities
    # ========================================================

    def _predict_vector(
        self,
        vector: np.ndarray,
    ) -> list[dict[str, Any]]:

        probabilities = (
            self.model.predict_proba(
                vector.reshape(1, -1)
            )[0]
        )

        if (
            len(probabilities)
            != len(self.class_names)
        ):
            raise RuntimeError(
                "Task2B probability dimension "
                "does not match class vocabulary."
            )

        ranked = [
            {
                "label": season,
                "p": round(
                    float(probability),
                    6,
                ),
            }
            for season, probability
            in zip(
                self.class_names,
                probabilities,
            )
        ]

        ranked.sort(
            key=lambda item: item["p"],
            reverse=True,
        )

        return ranked

    # ========================================================
    # Display policy
    # ========================================================

    @staticmethod
    def _cyclic_display_order(
        selected: list[str],
    ) -> list[str]:

        selected_set = set(
            selected
        )

        if len(selected) <= 1:
            return selected.copy()

        if len(selected) == 4:
            return SEASON_CYCLE.copy()

        if len(selected) == 3:
            omitted = [
                season
                for season in SEASON_CYCLE
                if season not in selected_set
            ][0]

            omitted_index = (
                SEASON_CYCLE.index(
                    omitted
                )
            )

            ordered = []

            for offset in range(
                1,
                4,
            ):
                season = (
                    SEASON_CYCLE[
                        (
                            omitted_index
                            + offset
                        )
                        % 4
                    ]
                )

                if season in selected_set:
                    ordered.append(
                        season
                    )

            return ordered

        if len(selected) == 2:
            for index, season in enumerate(
                SEASON_CYCLE
            ):
                next_season = (
                    SEASON_CYCLE[
                        (index + 1) % 4
                    ]
                )

                if (
                    season in selected_set
                    and next_season in selected_set
                ):
                    return [
                        season,
                        next_season,
                    ]

        return [
            season
            for season in SEASON_CYCLE
            if season in selected_set
        ]

    @classmethod
    def _display_policy(
        cls,
        ranked: list[dict[str, Any]],
    ) -> dict[str, Any]:

        probability_by_season = {
            item["label"]:
                float(item["p"])
            for item in ranked
        }

        values = np.asarray(
            [
                probability_by_season[
                    season
                ]
                for season
                in SEASON_CYCLE
            ],
            dtype=np.float64,
        )

        max_probability = float(
            values.max()
        )

        min_probability = float(
            values.min()
        )

        spread = (
            max_probability
            - min_probability
        )

        if spread <= ALL_SEASON_SPREAD:
            return {
                "display_label":
                    "All Season",

                "selected_seasons":
                    SEASON_CYCLE.copy(),

                "display_rule":
                    "flat_distribution",

                "probability_spread":
                    round(
                        spread,
                        6,
                    ),

                "effective_threshold":
                    None,
            }

        effective_threshold = max(
            ABSOLUTE_THRESHOLD,
            RELATIVE_THRESHOLD
            * max_probability,
        )

        selected = [
            season
            for season in SEASON_CYCLE
            if (
                probability_by_season[
                    season
                ]
                >= effective_threshold
            )
        ]

        if not selected:
            selected = [
                max(
                    probability_by_season,
                    key=probability_by_season.get,
                )
            ]

        if len(selected) == 4:
            return {
                "display_label":
                    "All Season",

                "selected_seasons":
                    SEASON_CYCLE.copy(),

                "display_rule":
                    "all_four_threshold",

                "probability_spread":
                    round(
                        spread,
                        6,
                    ),

                "effective_threshold":
                    round(
                        effective_threshold,
                        6,
                    ),
            }

        ordered = (
            cls._cyclic_display_order(
                selected
            )
        )

        return {
            "display_label":
                " / ".join(
                    season.title()
                    for season in ordered
                ),

            "selected_seasons":
                ordered,

            "display_rule":
                "multi_season_threshold",

            "probability_spread":
                round(
                    spread,
                    6,
                ),

            "effective_threshold":
                round(
                    effective_threshold,
                    6,
                ),
        }

    # ========================================================
    # Public prediction
    # ========================================================

    def predict(
        self,
        article_type: str,
    ) -> dict[str, Any]:

        if not isinstance(
            article_type,
            str,
        ):
            raise TypeError(
                "article_type must be a string"
            )

        article_type = article_type.strip()

        if not article_type:
            raise ValueError(
                "article_type cannot be empty"
            )

        if article_type not in self.mapping:
            raise ValueError(
                "Unknown Task1 articleType: "
                f"{article_type}"
            )

        info = self.mapping[
            article_type
        ]

        category = info.get(
            "sfs_category"
        )

        mapping_status = info.get(
            "status",
            "unknown",
        )

        task1_neutral = (
            article_type
            in SEASON_NEUTRAL_TASK1_TYPES
        )

        sfs_neutral = (
            category
            in SEASON_NEUTRAL_SFS_CATEGORIES
            if category is not None
            else False
        )

        # ----------------------------------------------------
        # Unsupported but semantically season-neutral
        # ----------------------------------------------------

        if (
            category is None
            and task1_neutral
        ):
            return {
                "supported":
                    False,

                "recommendation_available":
                    True,

                "article_type":
                    article_type,

                "sfs_category":
                    None,

                "mapping_status":
                    mapping_status,

                "method":
                    self.METHOD,

                "recommendation_source":
                    "season_neutral_rule",

                "top_season":
                    None,

                "season_probabilities":
                    [],

                "display_label":
                    "All Season",

                "selected_seasons":
                    SEASON_CYCLE.copy(),

                "display_rule":
                    "season_neutral_rule",

                "probability_spread":
                    None,

                "effective_threshold":
                    None,

                "reason": (
                    "This item type is treated as "
                    "intrinsically season-neutral."
                ),
            }

        # ----------------------------------------------------
        # Unsupported and not clearly neutral
        # ----------------------------------------------------

        if category is None:
            return {
                "supported":
                    False,

                "recommendation_available":
                    False,

                "article_type":
                    article_type,

                "sfs_category":
                    None,

                "mapping_status":
                    mapping_status,

                "method":
                    self.METHOD,

                "recommendation_source":
                    None,

                "top_season":
                    None,

                "season_probabilities":
                    [],

                "display_label":
                    None,

                "selected_seasons":
                    [],

                "display_rule":
                    "unsupported",

                "probability_spread":
                    None,

                "effective_threshold":
                    None,

                "reason": (
                    "No defensible SFS garment "
                    "taxonomy mapping exists for "
                    "this Task1 article type."
                ),
            }

        # ----------------------------------------------------
        # Supported SFS category
        # ----------------------------------------------------

        vector = self._build_one_hot(
            category
        )

        ranked = self._predict_vector(
            vector
        )

        # ----------------------------------------------------
        # Supported but semantically season-neutral
        #
        # Keep the raw SFS probabilities for transparency,
        # but override the user-facing suitability label.
        # ----------------------------------------------------

        if (
            task1_neutral
            or sfs_neutral
        ):
            return {
                "supported":
                    True,

                "recommendation_available":
                    True,

                "article_type":
                    article_type,

                "sfs_category":
                    category,

                "mapping_status":
                    mapping_status,

                "method":
                    self.METHOD,

                "recommendation_source":
                    "sfs_model_plus_season_neutral_rule",

                "top_season":
                    ranked[0]["label"],

                "season_probabilities":
                    ranked,

                "display_label":
                    "All Season",

                "selected_seasons":
                    SEASON_CYCLE.copy(),

                "display_rule":
                    "season_neutral_override",

                "probability_spread":
                    round(
                        (
                            max(
                                item["p"]
                                for item in ranked
                            )
                            -
                            min(
                                item["p"]
                                for item in ranked
                            )
                        ),
                        6,
                    ),

                "effective_threshold":
                    None,

                "reason": (
                    "Raw SFS probabilities are retained, "
                    "but this item is treated as "
                    "intrinsically season-neutral."
                ),
            }

        # ----------------------------------------------------
        # Normal learned recommendation
        # ----------------------------------------------------

        display = (
            self._display_policy(
                ranked
            )
        )

        return {
            "supported":
                True,

            "recommendation_available":
                True,

            "article_type":
                article_type,

            "sfs_category":
                category,

            "mapping_status":
                mapping_status,

            "method":
                self.METHOD,

            "recommendation_source":
                "sfs_model",

            "top_season":
                ranked[0]["label"],

            "season_probabilities":
                ranked,

            "display_label":
                display[
                    "display_label"
                ],

            "selected_seasons":
                display[
                    "selected_seasons"
                ],

            "display_rule":
                display[
                    "display_rule"
                ],

            "probability_spread":
                display[
                    "probability_spread"
                ],

            "effective_threshold":
                display[
                    "effective_threshold"
                ],

            "reason":
                None,
        }

    # ========================================================
    # Task1 adapter
    # ========================================================

    def predict_from_task1(
        self,
        task1_prediction: Any,
    ) -> dict[str, Any]:

        article_type = (
            self.extract_top1(
                task1_prediction
            )
        )

        result = self.predict(
            article_type
        )

        result["task1_top1"] = (
            article_type
        )

        return result

    @staticmethod
    def extract_top1(
        task1_prediction: Any,
    ) -> str:

        if isinstance(
            task1_prediction,
            list,
        ):
            if not task1_prediction:
                raise ValueError(
                    "Empty Task1 prediction list"
                )

            first = task1_prediction[0]

            if (
                isinstance(first, dict)
                and first.get("label")
            ):
                return str(
                    first["label"]
                )

        if isinstance(
            task1_prediction,
            dict,
        ):
            label = task1_prediction.get(
                "label"
            )

            if label:
                return str(label)

            top3 = task1_prediction.get(
                "top3"
            )

            if (
                isinstance(top3, list)
                and top3
                and isinstance(
                    top3[0],
                    dict,
                )
                and top3[0].get(
                    "label"
                )
            ):
                return str(
                    top3[0]["label"]
                )

        raise ValueError(
            "Cannot extract Task1 top-1 "
            "articleType from prediction."
        )

    # ========================================================
    # Health
    # ========================================================

    def health_info(
        self,
    ) -> dict[str, Any]:

        return {
            "loaded":
                True,

            "method":
                self.METHOD,

            "classes":
                self.class_names,

            "num_classes":
                self.num_classes,

            "sfs_categories":
                len(
                    self.categories
                ),

            "task1_classes":
                self.task1_class_count,

            "supported_task1_classes":
                self.supported_count,

            "supported_fraction":
                round(
                    self.supported_fraction,
                    4,
                ),

            "display_policy": {
                "all_season_spread":
                    ALL_SEASON_SPREAD,

                "relative_threshold":
                    RELATIVE_THRESHOLD,

                "absolute_threshold":
                    ABSOLUTE_THRESHOLD,

                "season_neutral_task1_types":
                    len(
                        SEASON_NEUTRAL_TASK1_TYPES
                    ),
            },

            "device":
                self.device,
        }