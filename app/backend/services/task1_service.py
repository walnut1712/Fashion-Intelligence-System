"""Task 1 serving wrapper - fashion item type (articleType) from an image.

The architecture lives in ``src/models/item_type_classifier.py`` and is
imported, never re-declared. This module used to carry its own copy of
``ConvBlock`` / ``ItemTypeCNN``, which silently drifted from the notebook and
left the shipped checkpoint unloadable.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# The repo is not pip-installed, so make the project root importable before
# reaching into src/. parents[3] == the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.models.item_type_classifier import (  # noqa: E402
	choose_device,
	load_item_type_model,
	preprocess_image,
)


class Task1Service:
	def __init__(self, model_path=None, tta=None):
		self.device = choose_device()
		self.project_root = PROJECT_ROOT
		self.model_path = Path(model_path) if model_path else (
			self.project_root / "artifacts" / "task1" / "task1_cnn.pt"
		)
		if not self.model_path.exists():
			raise FileNotFoundError("Task 1 model not found: {}".format(self.model_path))

		self.model, self.checkpoint = load_item_type_model(self.model_path, self.device)
		# The notebook records whether it evaluated with horizontal-flip TTA, so
		# the served prediction matches the one the reported metrics describe.
		self.tta = bool(self.checkpoint.get("tta", False)) if tta is None else bool(tta)

		self.class_names = list(self.checkpoint["class_names"])
		self.num_classes = len(self.class_names)
		self.image_size = tuple(self.checkpoint["image_size_pil"])
		self.model_name = self.checkpoint.get("model_name", "ItemTypeCNN")
		self.run_id = self.checkpoint.get("run_id")
		self.test_metrics = dict(self.checkpoint.get("test_metrics") or {})

		if self.num_classes != int(self.checkpoint["num_classes"]):
			raise ValueError(
				"Checkpoint disagrees with itself: num_classes={} but {} class names".format(
					self.checkpoint["num_classes"], self.num_classes
				)
			)

	def model_card(self):
		"""The published-metrics row the frontend shows, straight from the checkpoint.

		Matches the shape of an ``FI.metrics.tasks[]`` entry in
		``app/frontend/demo-data.js``; the frontend merges this over its
		hardcoded copy so the UI can never advertise a stale run.
		"""
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
				"Top-3 <b>{:.1f}%</b> &middot; top-5 <b>{:.1f}%</b> &middot; "
				"weighted F1 <b>{:.1f}</b> over {} classes."
			).format(top3, top5, weighted_f1, self.num_classes),
			"flag": "ok" if weighted_f1 >= 75 else "warn",
			"flagText": "Held-out test split, run {}".format(self.run_id or "unknown"),
			"note": (
				"Macro F1 is <b>{:.1f}</b>: the long tail of rare item types is much "
				"weaker than the headline suggests."
			).format(macro_f1),
		}

	def preprocess(self, image_bytes):
		return preprocess_image(image_bytes, self.checkpoint, self.device)

	@torch.no_grad()
	def predict(self, image_bytes):
		tensor = self.preprocess(image_bytes)
		probabilities = F.softmax(self.model(tensor).float(), dim=1)
		if self.tta:
			mirrored = F.softmax(self.model(torch.flip(tensor, dims=[3])).float(), dim=1)
			probabilities = (probabilities + mirrored) / 2
		probabilities = probabilities[0]

		top_probs, top_indices = torch.topk(probabilities, k=min(3, self.num_classes))
		results = [
			{"label": self.class_names[index], "confidence": round(float(probability), 4)}
			for probability, index in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist())
		]
		return {
			"label": results[0]["label"],
			"confidence": results[0]["confidence"],
			"top3": results,
		}
