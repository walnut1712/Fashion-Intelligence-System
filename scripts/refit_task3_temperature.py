"""Refit Task 3's confidence temperature for a fine-tuned checkpoint.

Why this exists
---------------
``Task3Service`` divides each head's logits by a per-attribute temperature before
the softmax. The deployed ``base`` checkpoint carries ``{gender: 5.83, usage:
5.94}``, fitted on validation in notebook 04: as trained the model reported 98.6%
mean confidence against 90.3% accuracy on gender (ECE 8.24), and the temperature
brings that to ECE 1.23 without changing a single prediction.

``train_task3_background_adaptation.py`` starts from that checkpoint and copies
its metadata forward, **including the temperature**. After eight epochs of
fine-tuning the logit scale is no longer the one that temperature was fitted for,
so an adapted checkpoint promoted as-is serves calibrated-looking confidences
that are not calibrated. Predictions are unaffected - dividing by a positive
scalar is monotonic, so argmax and therefore the graded submission cannot move -
which is exactly what makes the staleness easy to miss.

What it fits on, and why clean
------------------------------
Clean validation, the same distribution notebook 04 used. The temperature is then
comparable to the baseline's, and the sentence already in the repo - "it is
fitted on catalogue images, so it does not fix out-of-distribution
overconfidence" - stays true of both. Fitting a background-adapted model on
composited frames instead would calibrate it for uploads and decalibrate it for
the catalogue, which is a different decision and not one to make silently.

    python scripts/refit_task3_temperature.py \
        --checkpoint artifacts/task3_bgaug_lr3e-4/task3_bgadapt_best.pt

Writes the temperature into the checkpoint in place (``--dry-run`` to inspect
first) and records ``temperature_fitted_on`` so a later reader can tell which
weights it belongs to.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from train_task3_background_adaptation import (  # noqa: E402
    Task3Dataset,
    prepare_parts,
)
from train_t12_background_adaptation import (  # noqa: E402
    CacheIndex,
    get_device,
    resolve_project_root,
)

try:
    from app.backend.services.task3_service import EarlyBranchCNN  # noqa: E402
except ImportError:  # pragma: no cover - older module name
    from app.backend.services.task3_service import (  # noqa: E402
        EarlyBranchNet as EarlyBranchCNN,
    )

TARGETS = ("gender", "usage")


def expected_calibration_error(confidence, correct, bins=15):
    """ECE in percentage points, equal-width bins over [0, 1]."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if not mask.any():
            continue
        total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return 100.0 * total


def fit_one(logits, labels, max_iter=200):
    """One temperature, minimising NLL with LBFGS. Returns a float."""
    log_t = torch.zeros(1, requires_grad=True)      # optimise log T, so T > 0
    optimiser = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        optimiser.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_t.exp().item())


def collect_logits(model, loader, device):
    """Per-head logits and labels over a loader, on CPU."""
    model.eval()
    out = {t: [] for t in TARGETS}
    labels = {t: [] for t in TARGETS}
    with torch.no_grad():
        for images, gender, usage in loader:
            logits = model(images.to(device, non_blocking=True))
            truth = {"gender": gender, "usage": usage}
            for target in TARGETS:
                out[target].append(logits[target].detach().cpu())
                labels[target].append(truth[target].detach().cpu())
    return ({t: torch.cat(v) for t, v in out.items()},
            {t: torch.cat(v) for t, v in labels.items()})


def report(logits, labels, temperature):
    """Accuracy, mean confidence and ECE at a given temperature."""
    probabilities = F.softmax(logits / temperature, dim=1)
    confidence, predicted = probabilities.max(dim=1)
    correct = (predicted == labels).numpy()
    return {
        "accuracy": 100.0 * correct.mean(),
        "confidence": 100.0 * confidence.numpy().mean(),
        "ece": expected_calibration_error(confidence.numpy(), correct),
        "predictions": predicted,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true",
                        help="fit and report, but do not write the checkpoint")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    project = resolve_project_root()
    device = get_device()

    path = args.checkpoint
    if not path.exists():
        raise SystemExit("checkpoint not found: %s" % path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    class_names = {t: list(v) for t, v in checkpoint["class_names"].items()}
    num_classes = checkpoint.get(
        "num_classes", {k: len(v) for k, v in class_names.items()})
    image_size = tuple(checkpoint.get("image_size_pil", [60, 80]))
    arch = checkpoint.get("architecture", {})

    model = EarlyBranchCNN(
        num_classes,
        input_shape=(3, image_size[1], image_size[0]),
        shared_widths=tuple(arch.get("shared_widths", (32, 64))),
        branch_widths=tuple(arch.get("branch_widths", (128, 256))),
        hidden=int(arch.get("hidden", 256)),
        dropout=float(arch.get("dropout", 0.5)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    _, val_frame, _ = prepare_parts(class_names)
    cache = CacheIndex(project)
    dataset = Task3Dataset(
        val_frame, cache, image_size,
        checkpoint["channel_mean"], checkpoint["channel_std"],
        backgrounds=None, p_bg=0.0, deterministic=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    print("checkpoint :", path)
    print("validation :", len(dataset), "clean rows")
    inherited = checkpoint.get("temperature", {})
    print("inherited  :", {k: round(float(v), 4) for k, v in inherited.items()}
          or "none")
    print()

    logits, labels = collect_logits(model, loader, device)

    fitted, rows = {}, []
    for target in TARGETS:
        before = report(logits[target], labels[target], 1.0)
        temperature = fit_one(logits[target], labels[target])
        after = report(logits[target], labels[target], temperature)
        old = float(inherited.get(target, 1.0))
        stale = report(logits[target], labels[target], old)

        # Temperature is monotonic, so it must not move a single prediction.
        assert torch.equal(before["predictions"], after["predictions"]), (
            "temperature changed a prediction on %s, which is impossible for a "
            "positive scalar - check the fit" % target)

        # Stored to 4 decimal places, which is how notebook 04 wrote the
        # baseline's temperature. The checkpoint and artifacts/task3/
        # task3_cnn_summary.json are compared exactly by
        # tests/test_task3_service.py, so both must round identically.
        fitted[target] = round(temperature, 4)
        rows.append((target, before, stale, after, old, temperature))

    header = ("head", "acc", "conf T=1", "ECE T=1", "ECE inherited", "ECE refit",
              "T inherited", "T refit")
    print("%-8s %7s %10s %9s %14s %10s %12s %8s" % header)
    for target, before, stale, after, old, temperature in rows:
        print("%-8s %7.2f %10.2f %9.2f %14.2f %10.2f %12.4f %8.4f"
              % (target, before["accuracy"], before["confidence"], before["ece"],
                 stale["ece"], after["ece"], old, temperature))

    print()
    print("No prediction moved at any temperature, so the graded submission is")
    print("unaffected by this step; only reported confidence changes.")

    if args.dry_run:
        print("\n--dry-run: checkpoint not written")
        return 0

    checkpoint["temperature"] = fitted
    checkpoint["temperature_fitted_on"] = (
        "clean validation, %d rows, refit for these weights by "
        "scripts/refit_task3_temperature.py" % len(dataset))
    torch.save(checkpoint, path)
    print("\nwrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
