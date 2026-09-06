from io import BytesIO
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageOps, UnidentifiedImageError


# ============================================================
# DEVICE
# ============================================================

def choose_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# ORIGINAL TASK 2 CNN
# ============================================================

class SeasonCNN(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.40
            ),

            nn.Linear(
                128,
                num_classes
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# TASK 2 SERVICE
# ============================================================

class Task2Service:

    """
    Task 2 has TWO interpretations in the application.

    1. Catalogue season
       ----------------
       Original four-class CNN trained using the dataset's
       original `season` label:

           Fall
           Spring
           Summer
           Winter

       This remains the official Task 2 ML output.

    2. Suitable season
       ---------------
       Application-level interpretation using product semantics
       from Task 1's predicted article type / family.

       Examples:

           Sandals
               suitable = Spring / Summer
               catalogue = Winter 94%

           Watches
               suitable = All Season
               catalogue = Winter 97%

    The original CNN output is never discarded.
    """


    # ========================================================
    # STANDARD SEASON ORDER
    # ========================================================

    ALL_SEASONS = [
        "Spring",
        "Summer",
        "Fall",
        "Winter",
    ]


    # ========================================================
    # ARTICLE-TYPE SUITABILITY RULES
    #
    # Values are LISTS, not display strings.
    #
    # This allows frontend to distinguish:
    #
    #   4 seasons -> All Season
    #   3 seasons -> Spring / Summer / Fall
    #   2 seasons -> Spring / Summer
    #   1 season  -> Winter
    # ========================================================

    SUITABILITY_BY_TYPE = {

        # ----------------------------------------------------
        # ALL SEASON
        # ----------------------------------------------------

        "Watches": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Wallets": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Belts": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Backpacks": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Duffel Bag": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Messenger Bag": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Handbags": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Clutches": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Sunglasses": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Spectacle Frames": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Pendant": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Necklace and Chains": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Earrings": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Ring": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Bracelet": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Lipstick": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Lip Gloss": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Nail Polish": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Deodorant": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Perfume and Body Mist": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Fragrance": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Makeup": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Socks": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Jeans": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Trousers": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Track Pants": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Leggings": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Briefs": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Bra": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Sports Shoes": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],


        # ----------------------------------------------------
        # SPRING + SUMMER
        # ----------------------------------------------------

        "Tshirts": [
            "Spring",
            "Summer",
        ],

        "Tops": [
            "Spring",
            "Summer",
        ],

        "Shorts": [
            "Spring",
            "Summer",
        ],

        "Sandals": [
            "Spring",
            "Summer",
        ],

        "Flip Flops": [
            "Spring",
            "Summer",
        ],


        # ----------------------------------------------------
        # SUMMER
        # ----------------------------------------------------

        "Swimwear": [
            "Summer",
        ],


        # ----------------------------------------------------
        # SPRING + SUMMER + FALL
        # ----------------------------------------------------

        "Shirts": [
            "Spring",
            "Summer",
            "Fall",
        ],

        "Skirts": [
            "Spring",
            "Summer",
            "Fall",
        ],

        "Dresses": [
            "Spring",
            "Summer",
            "Fall",
        ],

        "Casual Shoes": [
            "Spring",
            "Summer",
            "Fall",
        ],

        "Formal Shoes": [
            "Spring",
            "Summer",
            "Fall",
        ],

        "Heels": [
            "Spring",
            "Summer",
            "Fall",
        ],

        "Flats": [
            "Spring",
            "Summer",
            "Fall",
        ],


        # ----------------------------------------------------
        # FALL + WINTER
        # ----------------------------------------------------

        "Sweatshirts": [
            "Fall",
            "Winter",
        ],

        "Sweaters": [
            "Fall",
            "Winter",
        ],

        "Jackets": [
            "Fall",
            "Winter",
        ],

        "Blazers": [
            "Fall",
            "Winter",
        ],

        "Boots": [
            "Fall",
            "Winter",
        ],

        "Scarves": [
            "Fall",
            "Winter",
        ],


        # ----------------------------------------------------
        # WINTER
        # ----------------------------------------------------

        "Coats": [
            "Winter",
        ],

        "Thermal": [
            "Winter",
        ],

        "Gloves": [
            "Winter",
        ],

        "Mufflers": [
            "Winter",
        ],
    }


    # ========================================================
    # FAMILY FALLBACK
    #
    # Used only if exact Task 1 articleType does not have a
    # dedicated rule.
    # ========================================================

    SUITABILITY_BY_FAMILY = {

        "Watches": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Jewellery": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Fragrance": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Lips": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Nails": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Makeup": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Personal Care": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Bags": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Wallets": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Belts": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Eyewear": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Innerwear": [
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],

        "Sandal": [
            "Spring",
            "Summer",
        ],

        "Flip Flops": [
            "Spring",
            "Summer",
        ],

        "Shoes": [
            "Spring",
            "Summer",
            "Fall",
        ],

        "Winter Wear": [
            "Fall",
            "Winter",
        ],
    }


    MIN_FAMILY_CONFIDENCE = 0.30


    # ========================================================
    # INIT
    # ========================================================

    def __init__(
        self,
        model_path=None,
        mapping_path=None,
    ):

        self.device = choose_device()

        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[3]
        )

        self.artifact_dir = (
            self.project_root
            / "artifacts"
            / "task2"
        )

        self.model_path = (
            Path(model_path)
            if model_path
            else
            self.artifact_dir
            / "task2_season_best_pytorch.pth"
        )

        self.mapping_path = (
            Path(mapping_path)
            if mapping_path
            else
            self.artifact_dir
            / "task2_season_class_mapping.json"
        )


        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Task 2 model not found: {self.model_path}"
            )


        if not self.mapping_path.exists():

            raise FileNotFoundError(
                f"Task 2 mapping not found: {self.mapping_path}"
            )


        # ----------------------------------------------------
        # CLASS MAPPING
        # ----------------------------------------------------

        with open(
            self.mapping_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.class_to_index = json.load(
                file
            )


        self.index_to_class = {

            int(index): name

            for name, index
            in self.class_to_index.items()
        }


        self.num_classes = len(
            self.class_to_index
        )


        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        self.model = SeasonCNN(
            num_classes=self.num_classes
        )


        checkpoint = torch.load(
            self.model_path,
            map_location=self.device
        )


        self.image_size = (
            60,
            80
        )


        if isinstance(
            checkpoint,
            dict
        ):

            if (
                "state_dict"
                in checkpoint
            ):

                state_dict = checkpoint[
                    "state_dict"
                ]

            elif (
                "model_state_dict"
                in checkpoint
            ):

                state_dict = checkpoint[
                    "model_state_dict"
                ]

            else:

                state_dict = checkpoint


            if (
                "image_size_pil"
                in checkpoint
            ):

                self.image_size = tuple(
                    checkpoint[
                        "image_size_pil"
                    ]
                )

        else:

            state_dict = checkpoint


        self.model.load_state_dict(
            state_dict
        )


        self.model.to(
            self.device
        )


        self.model.eval()


    # ========================================================
    # PREPROCESS
    # ========================================================

    def preprocess(
        self,
        image_bytes
    ):

        """
        Match the original 60x80 Task 2 training/evaluation:

            Resize((80, 60))
            ToTensor()

        PIL resize uses:

            (width, height)

        therefore:

            (60, 80)

        No letterbox.
        No normalization.
        """

        try:

            with Image.open(
                BytesIO(
                    image_bytes
                )
            ) as image:

                image = (
                    ImageOps
                    .exif_transpose(
                        image
                    )
                    .convert(
                        "RGB"
                    )
                )


                image = image.resize(
                    self.image_size,
                    Image.Resampling.BILINEAR
                )


                array = np.asarray(
                    image,
                    dtype=np.float32
                ) / 255.0


        except (
            UnidentifiedImageError,
            OSError
        ) as exc:

            raise ValueError(
                "Cannot decode uploaded image"
            ) from exc


        tensor = torch.from_numpy(

            array.transpose(
                2,
                0,
                1
            )

        ).float().unsqueeze(
            0
        )


        return tensor.to(
            self.device
        )


    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _confidence_at_least(
        value,
        threshold
    ):

        if value is None:
            return False

        try:

            return (
                float(value)
                >=
                float(threshold)
            )

        except (
            TypeError,
            ValueError
        ):

            return False


    @classmethod
    def _make_display_label(
        cls,
        seasons
    ):

        """
        Convert structured season list into UI label.

        4 seasons:
            All Season

        3 seasons:
            Spring / Summer / Fall

        2 seasons:
            Spring / Summer

        1 season:
            Winter
        """

        if not seasons:
            return None


        ordered = [

            season

            for season
            in cls.ALL_SEASONS

            if season
            in seasons
        ]


        if set(ordered) == set(
            cls.ALL_SEASONS
        ):

            return "All Season"


        return " / ".join(
            ordered
        )


    # ========================================================
    # GENERAL SUITABILITY
    # ========================================================

    def _general_suitability(
        self,
        article_type=None,
        article_type_confidence=None,
        article_family=None,
        article_family_confidence=None,
        catalogue_label=None
    ):

        # ----------------------------------------------------
        # 1. EXACT ARTICLE TYPE
        # ----------------------------------------------------

        if (
            article_type
            in self.SUITABILITY_BY_TYPE
        ):

            seasons = list(
                self.SUITABILITY_BY_TYPE[
                    article_type
                ]
            )


            return {

                "label":
                    self._make_display_label(
                        seasons
                    ),

                "seasons":
                    seasons,

                "source":
                    "article_type_semantics",

                "reason":
                    f"articleType={article_type}"
            }


        # ----------------------------------------------------
        # 2. ARTICLE FAMILY
        # ----------------------------------------------------

        if (
            article_family
            in self.SUITABILITY_BY_FAMILY

            and

            self._confidence_at_least(
                article_family_confidence,
                self.MIN_FAMILY_CONFIDENCE
            )
        ):

            seasons = list(
                self.SUITABILITY_BY_FAMILY[
                    article_family
                ]
            )


            return {

                "label":
                    self._make_display_label(
                        seasons
                    ),

                "seasons":
                    seasons,

                "source":
                    "family_semantics",

                "reason":
                    f"family={article_family}"
            }


        # ----------------------------------------------------
        # 3. FINAL FALLBACK
        #
        # "Varies by styling" has been removed.
        #
        # If no semantic rule exists, preserve the model's
        # catalogue class rather than inventing a new season.
        # ----------------------------------------------------

        fallback_seasons = (
            [catalogue_label]
            if catalogue_label
            else []
        )


        return {

            "label":
                catalogue_label
                if catalogue_label
                else "All Season",

            "seasons":
                fallback_seasons
                if fallback_seasons
                else list(
                    self.ALL_SEASONS
                ),

            "source":
                "catalogue_fallback",

            "reason":
                "no article-type or family suitability rule"
        }


    # ========================================================
    # PREDICT
    # ========================================================

    @torch.no_grad()
    def predict(
        self,
        image_bytes,
        article_type=None,
        article_type_confidence=None,
        article_family=None,
        article_family_confidence=None,
    ):

        # ----------------------------------------------------
        # ORIGINAL TASK 2 CNN
        # ----------------------------------------------------

        tensor = self.preprocess(
            image_bytes
        )


        logits = self.model(
            tensor
        )


        probabilities = F.softmax(
            logits,
            dim=1
        )[0]


        top_probs, top_indices = torch.topk(
            probabilities,
            k=self.num_classes
        )


        ranked = []


        for probability, index in zip(
            top_probs.cpu().tolist(),
            top_indices.cpu().tolist()
        ):

            ranked.append(
                {
                    "label":
                        self.index_to_class[
                            int(index)
                        ],

                    "confidence":
                        round(
                            float(probability),
                            4
                        )
                }
            )


        # ----------------------------------------------------
        # CATALOGUE RESULT
        # ----------------------------------------------------

        catalogue_label = (
            ranked[0][
                "label"
            ]
        )


        catalogue_confidence = (
            ranked[0][
                "confidence"
            ]
        )


        # ----------------------------------------------------
        # GENERAL SUITABILITY
        # ----------------------------------------------------

        suitable = self._general_suitability(

            article_type=
                article_type,

            article_type_confidence=
                article_type_confidence,

            article_family=
                article_family,

            article_family_confidence=
                article_family_confidence,

            catalogue_label=
                catalogue_label
        )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {

            # =================================================
            # ORIGINAL OFFICIAL TASK 2 CNN OUTPUT
            # =================================================

            "label":
                catalogue_label,

            "confidence":
                catalogue_confidence,

            "top3":
                ranked[:3],


            # =================================================
            # CATALOGUE / COLLECTION INTERPRETATION
            # =================================================

            "catalogue_label":
                catalogue_label,

            "catalogue_confidence":
                catalogue_confidence,

            "catalogue_note":
                "dataset catalogue / collection season",


            # =================================================
            # USER-FACING SUITABILITY
            # =================================================

            "suitable_label":
                suitable[
                    "label"
                ],

            "suitable_seasons":
                suitable[
                    "seasons"
                ],

            "suitable_source":
                suitable[
                    "source"
                ],

            "suitable_reason":
                suitable[
                    "reason"
                ],


            # =================================================
            # FRONTEND COMPATIBILITY
            # =================================================

            "display_label":
                suitable[
                    "label"
                ],

            "display_badge":
                (
                    "general"
                    if suitable["source"]
                    != "catalogue_fallback"
                    else "model"
                ),


            # IMPORTANT:
            #
            # This is intentionally the catalogue CNN
            # confidence so the frontend can draw the
            # percentage bar.
            #
            # It is NOT confidence in the semantic
            # suitable-season label.
            "display_confidence":
                catalogue_confidence,


            # =================================================
            # TASK 1 CONTEXT USED
            # =================================================

            "article_type_used":
                article_type,

            "article_type_confidence":
                (
                    round(
                        float(
                            article_type_confidence
                        ),
                        4
                    )

                    if
                    article_type_confidence
                    is not None

                    else None
                ),

            "article_family_used":
                article_family,

            "article_family_confidence":
                (
                    round(
                        float(
                            article_family_confidence
                        ),
                        4
                    )

                    if
                    article_family_confidence
                    is not None

                    else None
                )
        }