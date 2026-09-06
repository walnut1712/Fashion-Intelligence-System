"""Task 1 serving wrapper - fashion item type (articleType) from an image.

This service uses the shipped Task 1 catalogue checkpoint only.

For ingestion:
- catalogue-like images -> "squash" (historical catalogue preprocessing)
- real/user photos     -> "nobg"   (background removal before inference)

A previously referenced ``candidate_webphoto.pt`` is not part of the repository,
so the serving path must not depend on it.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import family_matrix  # noqa: E402
from src.data.user_image import (  # noqa: E402
    PREPROCESS_MODES,
    load_user_image,
    looks_like_catalogue,
)
from src.models.item_type_classifier import (  # noqa: E402
    apply_logit_adjustment,
    choose_device,
    load_item_type_model,
    preprocess_arrays,
    preprocess_image,
)

INGEST_MODES = ("auto", "squash") + tuple(PREPROCESS_MODES)
DEFAULT_INGEST = "auto"


class Task1Service:
    def __init__(self, model_path=None, tta=None, ingest=None):
        self.device = choose_device()
        self.project_root = PROJECT_ROOT

        self.model_path = (
            Path(model_path)
            if model_path
            else self.project_root / "artifacts" / "task1" / "task1_cnn.pt"
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Task 1 model not found: {}".format(self.model_path)
            )

        self.model, self.checkpoint = load_item_type_model(
            self.model_path,
            self.device,
        )

        self.tta = (
            bool(self.checkpoint.get("tta", False))
            if tta is None
            else bool(tta)
        )

        self.ingest = self._validate_ingest(ingest or DEFAULT_INGEST)

        self.class_names = list(self.checkpoint["class_names"])
        self.num_classes = len(self.class_names)
        self.image_size = tuple(self.checkpoint["image_size_pil"])
        self.model_name = self.checkpoint.get("model_name", "ItemTypeCNN")
        self.run_id = self.checkpoint.get("run_id")
        self.test_metrics = dict(self.checkpoint.get("test_metrics") or {})

        if self.num_classes != int(self.checkpoint["num_classes"]):
            raise ValueError(
                "Checkpoint disagrees with itself: "
                "num_classes={} but {} class names".format(
                    self.checkpoint["num_classes"],
                    self.num_classes,
                )
            )

    def model_card(self):
        """Return the published Task 1 metrics stored in the checkpoint."""
        metrics = self.test_metrics

        accuracy = float(metrics.get("accuracy", 0.0))
        weighted_f1 = float(metrics.get("weighted_f1", 0.0))
        macro_f1 = float(metrics.get("macro_f1", 0.0))
        top3 = float(metrics.get("top3_acc", 0.0))
        top5 = float(metrics.get("top5_acc", 0.0))

        return {
            "id": "Task 1",
            "name": "Item type",
            "headline": round(accuracy, 1),
            "headlineLabel": "top-1 accuracy",
            "detail": (
                "Top-3 <b>{:.1f}%</b> &middot; "
                "top-5 <b>{:.1f}%</b> &middot; "
                "weighted F1 <b>{:.1f}</b> over {} classes."
            ).format(top3, top5, weighted_f1, self.num_classes),
            "flag": "ok" if weighted_f1 >= 75 else "warn",
            "flagText": "Held-out test split, run {}".format(
                self.run_id or "unknown"
            ),
            "note": (
                "Macro F1 is <b>{:.1f}</b>: the long tail of rare "
                "item types is much weaker than the headline suggests."
            ).format(macro_f1),
        }

    @staticmethod
    def _validate_ingest(mode):
        mode = str(mode).lower()

        if mode not in INGEST_MODES:
            raise ValueError(
                "ingest must be one of {}, got {!r}".format(
                    INGEST_MODES,
                    mode,
                )
            )

        return mode

    def resolve_ingest(self, image_bytes, ingest=None):
        """Resolve a requested ingestion mode into the concrete mode used."""
        return self.route(image_bytes, ingest)[0]

    def route(self, image_bytes, ingest=None):
        """Choose preprocessing for one image.

        Returns ``(mode, which)`` for backward compatibility with the existing
        API response shape. ``which`` is always ``"catalogue"`` because the
        repository currently ships only ``task1_cnn.pt``.

        Auto routing:
        - catalogue-like image -> squash
        - photograph           -> nobg
        """
        mode = self._validate_ingest(ingest or self.ingest)

        if mode == "auto":
            mode = (
                "squash"
                if looks_like_catalogue(image_bytes, self.image_size)
                else "nobg"
            )

        return mode, "catalogue"

    def preprocess(self, image_bytes, ingest=None, checkpoint=None):
        """Convert image bytes to the normalised model input tensor."""
        mode, _ = self.route(image_bytes, ingest)
        checkpoint = checkpoint or self.checkpoint

        if mode == "squash":
            return preprocess_image(
                image_bytes,
                checkpoint,
                self.device,
            )

        array = load_user_image(
            image_bytes,
            size=self.image_size,
            mode=mode,
        )

        return preprocess_arrays(array[None], checkpoint).to(self.device)

    @torch.no_grad()
    def predict(self, image_bytes, ingest=None):
        mode, which = self.route(image_bytes, ingest)

        tensor = self.preprocess(
            image_bytes,
            ingest=mode,
            checkpoint=self.checkpoint,
        )

        temperature = float(self.checkpoint.get("temperature") or 1.0)

        probabilities = F.softmax(
            self.model(tensor).float() / temperature,
            dim=1,
        )

        if self.tta:
            mirrored = F.softmax(
                self.model(torch.flip(tensor, dims=[3])).float() / temperature,
                dim=1,
            )
            probabilities = (probabilities + mirrored) / 2

        probabilities = apply_logit_adjustment(
            probabilities,
            self.checkpoint,
        )[0]

        matrix, families = family_matrix(tuple(self.class_names))
        family_probs = probabilities.cpu().numpy() @ matrix
        family_rank = int(family_probs.argmax())

        top_probs, top_indices = torch.topk(
            probabilities,
            k=min(3, self.num_classes),
        )

        results = [
            {
                "label": self.class_names[index],
                "confidence": round(float(probability), 4),
            }
            for probability, index in zip(
                top_probs.cpu().tolist(),
                top_indices.cpu().tolist(),
            )
        ]

        return {
            "label": results[0]["label"],
            "confidence": results[0]["confidence"],
            "top3": results,
            "family": {
                "label": families[family_rank],
                "confidence": round(float(family_probs[family_rank]), 4),
            },
            "ingest": mode,
            "model": which,
        }
