"""Task 3B - auxiliary fine-grained occasion recommendation.

This service does NOT replace Task 3.

Primary Task 3 remains:
    gender + usage
    usage classes = Casual / Ethnic / Formal / Sports

Task 3B adds a lightweight recommendation layer using:
    Task 1 articleType
    + Task 3 primary usage
    + SFS category/occasion priors

Example:
    Jeans + Casual
        -> Shopping / Brunch

The recommendation is descriptive and auxiliary. It does not alter
Task 3 predictions, confidence scores, metrics, or model outputs.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


class Task3BService:
    """SFS-based fine-grained occasion recommendation."""

    HIDDEN_OCCASIONS = {
        "Everyday",
        "Other",
    }

    # Product types where a fine-grained "where to wear it" recommendation
    # is either weak or semantically unhelpful.
    OCCASION_NEUTRAL_TASK1_TYPES = {
        "Boxers",
        "Briefs",
        "Bra",
        "Camisoles",
        "Innerwear Vests",
        "Night suits",
        "Nightdress",

        "Deodorant",
        "Foundation and Primer",
        "Fragrance Gift Set",
        "Kajal and Eyeliner",
        "Lip Liner",
        "Lipstick",
        "Nail Polish",
        "Perfume and Body Mist",
    }

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

    def __init__(
        self,
        priors_path=None,
        mapping_path=None,
    ):
        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[3]
        )

        self.priors_path = (
            Path(priors_path)
            if priors_path
            else (
                self.project_root
                / "artifacts"
                / "task3b_recommendation"
                / "task3b_recommendation_priors.json"
            )
        )

        self.mapping_path = (
            Path(mapping_path)
            if mapping_path
            else (
                self.project_root
                / "artifacts"
                / "task2b_sfs"
                / "task1_to_sfs_mapping.json"
            )
        )

        if not self.priors_path.exists():
            raise FileNotFoundError(
                "Task 3B priors not found: {}".format(
                    self.priors_path
                )
            )

        if not self.mapping_path.exists():
            raise FileNotFoundError(
                "Task 1 -> SFS mapping not found: {}".format(
                    self.mapping_path
                )
            )

        self._load_priors()
        self._load_mapping()

        print(
            "Task 3B loaded:"
            f" {len(self.category_counts)} SFS categories"
            f" | {len(self.global_counts)} occasions"
            f" | usable posts={self.usable_posts}"
        )

    # --------------------------------------------------------
    # Loading
    # --------------------------------------------------------

    def _load_priors(self):

        payload = json.loads(
            self.priors_path.read_text(
                encoding="utf-8"
            )
        )

        self.usable_posts = int(
            payload.get(
                "usable_posts",
                0,
            )
        )

        self.category_counts = {
            category:
                Counter(
                    {
                        occasion: int(count)
                        for occasion, count
                        in values.items()
                    }
                )

            for category, values
            in payload[
                "category_counts"
            ].items()
        }

        self.global_counts = Counter(
            {
                occasion: int(count)
                for occasion, count
                in payload[
                    "global_counts"
                ].items()
            }
        )

        policy = payload.get(
            "production_policy",
            {},
        )

        self.min_top_score = float(
            policy.get(
                "min_top_score",
                0.10,
            )
        )

        self.min_top_count = int(
            policy.get(
                "min_top_count",
                30,
            )
        )

        self.second_ratio = 0.60

    def _load_mapping(self):

        data = json.loads(
            self.mapping_path.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, dict):

            if "mapping" in data:
                data = data["mapping"]

            elif "task1_to_sfs" in data:
                data = data[
                    "task1_to_sfs"
                ]

        if not isinstance(data, dict):
            raise ValueError(
                "Unexpected Task1 -> SFS mapping format"
            )

        self.mapping = data

    # --------------------------------------------------------
    # Mapping
    # --------------------------------------------------------

    def resolve_sfs_category(
        self,
        article_type,
    ):
        value = self.mapping.get(
            article_type
        )

        if value is None:
            return None, None

        if isinstance(value, str):
            return value, "mapped"

        if isinstance(value, dict):

            category = None

            for key in (
                "sfs_category",
                "category",
                "mapped_category",
                "target",
            ):
                if value.get(key):
                    category = value[key]
                    break

            mapping_type = (
                value.get("mapping_type")
                or value.get("type")
                or value.get("status")
                or "mapped"
            )

            return (
                category,
                mapping_type,
            )

        return None, None

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    def _score_candidates(
        self,
        category,
        usage,
    ):

        distribution = (
            self.category_counts.get(
                category
            )
        )

        if not distribution:
            return []

        compatibility = (
            self.USAGE_OCCASION_WEIGHTS.get(
                usage,
                {},
            )
        )

        if not compatibility:
            return []

        category_total = float(
            sum(
                distribution.values()
            )
        )

        global_total = float(
            sum(
                self.global_counts.values()
            )
        )

        if (
            category_total <= 0
            or global_total <= 0
        ):
            return []

        scored = []

        for (
            occasion,
            usage_weight,
        ) in compatibility.items():

            if (
                occasion
                in self.HIDDEN_OCCASIONS
            ):
                continue

            count = int(
                distribution.get(
                    occasion,
                    0,
                )
            )

            if count <= 0:
                continue

            category_prior = (
                count
                / category_total
            )

            global_count = int(
                self.global_counts.get(
                    occasion,
                    0,
                )
            )

            if global_count <= 0:
                continue

            global_prior = (
                global_count
                / global_total
            )

            lift = (
                category_prior
                / global_prior
            )

            support = (
                count
                / (
                    count
                    + 50.0
                )
            )

            score = (
                math.sqrt(
                    category_prior
                )
                * (
                    min(
                        lift,
                        8.0,
                    )
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

                    "category_prior":
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
            key=lambda row:
                row["score"],
            reverse=True,
        )

        return scored

    # --------------------------------------------------------
    # Public recommendation
    # --------------------------------------------------------

    def recommend(
        self,
        article_type,
        usage,
        top_k=2,
    ):
        """
        Return a fine-grained auxiliary occasion recommendation.

        No recommendation is returned when:
        - Task1 article type is occasion-neutral
        - no SFS taxonomy mapping exists
        - Task3 usage is unsupported
        - SFS evidence is below production thresholds
        """

        article_type = str(
            article_type
        ).strip()

        usage = str(
            usage
        ).strip()

        base = {
            "recommendation_available":
                False,

            "display_label":
                None,

            "article_type":
                article_type,

            "usage":
                usage,

            "sfs_category":
                None,

            "mapping_type":
                None,

            "recommendations":
                [],

            "recommendation_source":
                "sfs_category_usage_prior",
        }

        # -------------------------------- neutral products

        if (
            article_type
            in self.OCCASION_NEUTRAL_TASK1_TYPES
        ):
            base[
                "reason"
            ] = "occasion_neutral_item"

            return base

        # -------------------------------- category mapping

        (
            category,
            mapping_type,
        ) = self.resolve_sfs_category(
            article_type
        )

        base[
            "sfs_category"
        ] = category

        base[
            "mapping_type"
        ] = mapping_type

        if not category:

            base[
                "reason"
            ] = "unsupported_article_type"

            return base

        # -------------------------------- usage

        if (
            usage
            not in
            self.USAGE_OCCASION_WEIGHTS
        ):

            base[
                "reason"
            ] = "unsupported_usage"

            return base

        # -------------------------------- score

        scored = self._score_candidates(
            category,
            usage,
        )

        if not scored:

            base[
                "reason"
            ] = "no_compatible_occasion"

            return base

        best = scored[0]

        # Require meaningful support.
        if (
            best["score"]
            < self.min_top_score
            or best["count"]
            < self.min_top_count
        ):

            base[
                "reason"
            ] = "insufficient_evidence"

            base[
                "top_score"
            ] = round(
                float(
                    best["score"]
                ),
                6,
            )

            base[
                "top_count"
            ] = int(
                best["count"]
            )

            return base

        selected = []

        for row in scored:

            if (
                len(selected)
                >= max(
                    1,
                    min(
                        int(top_k),
                        2,
                    ),
                )
            ):
                break

            if (
                row["count"]
                < self.min_top_count
            ):
                continue

            if (
                selected
                and row["score"]
                < best["score"]
                * self.second_ratio
            ):
                continue

            selected.append(
                row
            )

        if not selected:

            base[
                "reason"
            ] = "insufficient_evidence"

            return base

        display_label = " / ".join(
            row["occasion"]
            for row in selected
        )

        base.update(
            {
                "recommendation_available":
                    True,

                "display_label":
                    display_label,

                "recommendations":
                    [
                        {
                            "label":
                                row[
                                    "occasion"
                                ],

                            "score":
                                round(
                                    float(
                                        row[
                                            "score"
                                        ]
                                    ),
                                    6,
                                ),

                            "count":
                                int(
                                    row[
                                        "count"
                                    ]
                                ),

                            "lift":
                                round(
                                    float(
                                        row[
                                            "lift"
                                        ]
                                    ),
                                    3,
                                ),
                        }

                        for row
                        in selected
                    ],

                "reason":
                    "recommended",
            }
        )

        return base

    def health(self):

        return {
            "available":
                True,

            "method":
                "sfs_category_usage_lift_prior",

            "usable_posts":
                self.usable_posts,

            "categories":
                len(
                    self.category_counts
                ),

            "occasions":
                len(
                    self.global_counts
                ),

            "min_top_score":
                self.min_top_score,

            "min_top_count":
                self.min_top_count,

            "second_ratio":
                self.second_ratio,
        }