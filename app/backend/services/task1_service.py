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

# "squash" is the historical path - resize straight to 60x80 and let the aspect
# ratio distort - kept so the old behaviour stays reachable and comparable. The
# other three coerce an upload towards catalogue framing first.
#
# "auto" is the default because the measurement (src/evaluation/ood_benchmark.py)
# says no fixed mode can be: on a clean catalogue tile squash beats nobg 88.32 to
# 76.90, and on a shift-synthesised photograph nobg beats squash 38.07 to 10.15.
# Routing per image gets both, and the router agrees with the truth on 31/31 real
# web photos and 599/600 catalogue tiles.
INGEST_MODES = ("auto", "squash") + tuple(PREPROCESS_MODES)
DEFAULT_INGEST = "auto"

# Two checkpoints, because the two input populations are genuinely different and
# no single model is best on both. Measured accuracy, each at its best ingestion:
#
#                       catalogue tile   photograph (mild / moderate / severe)
#   task1_cnn.pt            87.85          31.90 / 19.62 / 10.63
#   candidate_webphoto.pt   85.32          50.00 / 43.54 / 19.24
#
# The graded submission set (A2_FashionDataset/.../images_test) is catalogue
# format - 73.4% of its pixels are near-white against the training set's 67.5% -
# so swapping wholesale would give up ~4 weighted-F1 on the marked deliverable
# for nothing. A real upload is the opposite: 14.3% near-white. Routing per image
# takes the better model on each and gives up nothing on either.
ROBUST_MODEL_PATH = "artifacts/task1/candidate_webphoto.pt"


class Task1Service:
	def __init__(self, model_path=None, tta=None, ingest=None, robust_path=None):
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
		self.ingest = self._validate_ingest(ingest or DEFAULT_INGEST)

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

		self._load_robust(robust_path)

	def _load_robust(self, robust_path=None):
		"""Load the photograph model, if it is present.

		Optional on purpose: the service must still start on a checkout that has
		only the catalogue checkpoint, and then simply routes everything to it.
		A robust model whose class order differs is rejected rather than used,
		because the label index is what the response is built from.
		"""
		self.robust_model = self.robust_checkpoint = None
		self.robust_path = Path(robust_path) if robust_path else (
			self.project_root / ROBUST_MODEL_PATH
		)
		self.robust_error = None
		if not self.robust_path.exists():
			self.robust_error = "not present: {}".format(self.robust_path.name)
			return
		try:
			model, checkpoint = load_item_type_model(self.robust_path, self.device)
			if list(checkpoint["class_names"]) != self.class_names:
				raise ValueError("class order differs from the catalogue checkpoint")
			self.robust_model, self.robust_checkpoint = model, checkpoint
		except Exception as error:
			self.robust_error = "{}: {}".format(type(error).__name__, error)

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

	@staticmethod
	def _validate_ingest(mode):
		mode = str(mode).lower()
		if mode not in INGEST_MODES:
			raise ValueError("ingest must be one of {}, got {!r}".format(INGEST_MODES, mode))
		return mode

	def preprocess(self, image_bytes, ingest=None, checkpoint=None):
		"""Bytes -> a normalised 1xCxHxW tensor, through the chosen ingestion mode.

		The catalogue this model was trained on is 60x80 cutouts on white, and it
		leaned on that: composite the same held-out garments onto a textured
		background and accuracy falls from 87.92 to 25.80. Coercing an upload back
		towards catalogue framing before it reaches the model is therefore part of
		inference, not a preprocessing detail.
		"""
		mode, _ = self.route(image_bytes, ingest)
		# The checkpoint is an explicit argument, not inferred from the route: the
		# two checkpoints carry different channel statistics, so normalising with
		# one and running the other silently mismatches the model to its input.
		# Defaults to the catalogue checkpoint, which is what self.model is.
		checkpoint = checkpoint or self.checkpoint
		if mode == "squash":
			return preprocess_image(image_bytes, checkpoint, self.device)
		array = load_user_image(image_bytes, size=self.image_size, mode=mode)
		return preprocess_arrays(array[None], checkpoint).to(self.device)

	def resolve_ingest(self, image_bytes, ingest=None):
		"""Turn the requested mode into a concrete one, resolving "auto"."""
		return self.route(image_bytes, ingest)[0]

	def route(self, image_bytes, ingest=None):
		"""Pick the ingestion mode and the model for one image.

		Returns ``(mode, which)`` where ``which`` is "catalogue" or "photo".
		A catalogue-shaped tile goes to the catalogue checkpoint unsquashed; a
		photograph goes to the background-randomised checkpoint through nobg. An
		explicitly requested mode still selects the model, so ``?ingest=squash``
		on a photo is a like-for-like comparison against the shipped behaviour.
		"""
		mode = self._validate_ingest(ingest or self.ingest)
		if mode == "auto":
			mode = "squash" if looks_like_catalogue(image_bytes, self.image_size) else "nobg"
		if mode == "squash" or self.robust_model is None:
			return mode, "catalogue"
		return mode, "photo"

	@torch.no_grad()
	def predict(self, image_bytes, ingest=None):
		mode, which = self.route(image_bytes, ingest)
		model = self.robust_model if which == "photo" else self.model
		checkpoint = self.robust_checkpoint if which == "photo" else self.checkpoint
		tta = bool(checkpoint.get("tta", False)) if which == "photo" else self.tta

		tensor = self.preprocess(image_bytes, ingest=mode, checkpoint=checkpoint)
		# Temperature scaling, when the checkpoint carries one. Divides the logits
		# by a positive constant, so it cannot change the predicted label - it only
		# makes the confidence the UI displays mean what it says.
		temperature = float(checkpoint.get("temperature") or 1.0)
		logits = model(tensor).float() / temperature
		probabilities = F.softmax(logits, dim=1)
		if tta:
			mirrored = F.softmax(
				model(torch.flip(tensor, dims=[3])).float() / temperature, dim=1)
			probabilities = (probabilities + mirrored) / 2
		# Post-hoc logit adjustment, when the checkpoint carries a tau. Serving
		# must sit on the same operating point the reported metrics describe.
		probabilities = apply_logit_adjustment(probabilities, checkpoint)[0]

		# The coarse answer. articleType is right 54.86% of the time on a shifted
		# photograph; its subCategory is right 66.95%, because most errors are
		# within-family (Casual Shoes for Sports Shoes) and rolling up absorbs
		# them. Marginalised rather than read off the argmax - that is worth
		# another half point, and it is the honest quantity.
		matrix, families = family_matrix(tuple(self.class_names))
		family_probs = probabilities.cpu().numpy() @ matrix
		family_rank = int(family_probs.argmax())

		top_probs, top_indices = torch.topk(probabilities, k=min(3, self.num_classes))
		results = [
			{"label": self.class_names[index], "confidence": round(float(probability), 4)}
			for probability, index in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist())
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
