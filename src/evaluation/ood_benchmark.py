"""Task 1 - does the classifier survive a photograph it did not pose for?

Why this exists
---------------
Every Task 1 number in the repository is measured on catalogue tiles: 60x80
product cutouts, one item, centred, on white. The deployed checkpoint scores
87.80 accuracy / 87.44 weighted-F1 there. Nothing measured what happens to a
photo someone actually uploads, and the answer turns out to decide the whole
design. Re-scoring the same held-out rows with the garment composited onto a
textured background drops the model to 25.80 accuracy and 5.19 macro-F1, while
squashing the aspect ratio costs about one point. The model learned "garment on
white", not "garment".

So this module builds the missing measurement. It takes held-out rows whose
labels are known, synthesises a plausible *upload* from each one - a bigger
image, on a background, rotated, off-centre, at the wrong aspect ratio, JPEG'd -
and runs it through the real serving path including ingestion. That makes four
things comparable on one axis: the checkpoint, the ingestion mode, the shift
severity, and (later) a retrained candidate.

Methodology guard
-----------------
The corruption families here are deliberately **disjoint** from the ones Phase 2
trains on. Training randomises solid, gradient, gaussian-noise and in-batch-crop
backgrounds; this benchmark uses checkerboards, stripes and smooth blob fields.
A benchmark that reused the training families would be grading a model on its own
augmentation and would report a win that does not exist. The RNG stream is
separate and seeded per severity, so the benchmark is reproducible but not
something a training run can memorise.

Usage
-----
    python -m src.evaluation.ood_benchmark
    python -m src.evaluation.ood_benchmark --rows 800 --modes squash nobg
    python -m src.evaluation.ood_benchmark --checkpoints artifacts/task1/*.pt
"""

import argparse
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.user_image import catalogue_alpha, load_user_image  # noqa: E402
from src.models.item_type_classifier import (  # noqa: E402
    apply_logit_adjustment,
    choose_device,
    load_item_type_model,
    predict_proba,
    preprocess_arrays,
)
from src.training.train_item_type import load_splits  # noqa: E402

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "task1"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "evaluation"
DEFAULT_CHECKPOINT = ARTIFACT_DIR / "task1_cnn.pt"

# "squash" is not an ingestion mode, it is the *absence* of one: resize straight
# to 60x80 and let the aspect ratio distort. It is what Task 1 shipped, so it is
# the control every other mode has to beat.
INGEST_MODES = ("squash", "letterbox", "crop", "nobg")

# Severity ladder. "clean" passes the catalogue tile through untouched and must
# reproduce the published metrics - if it does not, the harness is wrong before
# any comparison is worth reading.
SEVERITIES = {
    "clean":    None,
    "mild":     dict(angle=5,  scale=(0.85, 1.10), aspect=(0.95, 1.20), fill=(0.62, 0.85),
                     blur=0.0, quality=92, families=("checker",)),
    "moderate": dict(angle=12, scale=(0.75, 1.15), aspect=(0.85, 1.45), fill=(0.50, 0.85),
                     blur=0.4, quality=80, families=("checker", "stripes")),
    "severe":   dict(angle=20, scale=(0.60, 1.20), aspect=(0.70, 1.80), fill=(0.38, 0.85),
                     blur=0.8, quality=65, families=("checker", "stripes", "blobs")),
}

CANVAS_HEIGHT = 400  # a synthesised upload is ~5x the catalogue tile, like a real one


# ------------------------------------------------------------- backgrounds ----
def _smooth(array, size):
    """Low-resolution noise blown up to full size - cheap, band-limited texture."""
    return np.asarray(
        Image.fromarray(array.astype(np.uint8)).resize(size, Image.BICUBIC),
        dtype=np.uint8)


def _background(family, size, rng):
    """One evaluation-only background. Never a family Phase 2 trains on."""
    width, height = size

    if family == "checker":
        cell = int(rng.integers(12, 48))
        a = rng.integers(20, 240, 3)
        b = np.clip(a + rng.integers(-90, 90, 3), 0, 255)
        ys, xs = np.mgrid[0:height, 0:width]
        pick = ((ys // cell) + (xs // cell)) % 2
        board = np.where(pick[..., None] == 0, a, b).astype(np.uint8)
        return np.asarray(Image.fromarray(board).filter(
            ImageFilter.GaussianBlur(rng.uniform(0.5, 2.5))), dtype=np.uint8)

    if family == "stripes":
        period = rng.uniform(14, 60)
        angle = rng.uniform(0, np.pi)
        a = rng.integers(20, 240, 3).astype(np.float32)
        b = np.clip(a + rng.integers(-80, 80, 3), 0, 255).astype(np.float32)
        ys, xs = np.mgrid[0:height, 0:width]
        wave = 0.5 + 0.5 * np.sin(2 * np.pi * (xs * np.cos(angle) + ys * np.sin(angle)) / period)
        stripes = a + (b - a) * wave[..., None]
        grain = rng.normal(0, 6, stripes.shape)
        return np.clip(stripes + grain, 0, 255).astype(np.uint8)

    # "blobs" - a smooth multi-colour field, standing in for an out-of-focus room
    coarse = rng.integers(0, 255, (max(2, height // 40), max(2, width // 40), 3))
    return _smooth(coarse, size)


# -------------------------------------------------------- upload synthesis ----
def _synthesise_upload(tile, alpha, params, rng):
    """One catalogue tile -> a plausible photograph of the same garment.

    Deliberately built at ~400px and handed back as a PIL image rather than a
    60x80 array, so that ingestion is inside the measurement. A benchmark that
    pre-resized to 60x80 could not tell letterbox from crop from nobg, which is
    the comparison this exists to make.
    """
    subject = Image.fromarray(tile).convert("RGBA")
    subject.putalpha(Image.fromarray(alpha * 255))

    aspect = rng.uniform(*params["aspect"])
    canvas_w = max(64, int(round(CANVAS_HEIGHT * 0.75 * aspect)))
    canvas = Image.fromarray(_background(
        rng.choice(params["families"]), (canvas_w, CANVAS_HEIGHT), rng))

    # Scale the garment to fill a variable share of the frame, then rotate it.
    fill = rng.uniform(*params["fill"]) * rng.uniform(*params["scale"])
    height = max(8, int(round(CANVAS_HEIGHT * fill)))
    width = max(6, int(round(height * 0.75)))
    subject = subject.resize((width, height), Image.BILINEAR)
    if params["angle"]:
        subject = subject.rotate(rng.uniform(-params["angle"], params["angle"]),
                                 resample=Image.BICUBIC, expand=True)

    # Off-centre placement, clamped so the subject stays inside the frame.
    max_x = max(0, canvas_w - subject.width)
    max_y = max(0, CANVAS_HEIGHT - subject.height)
    x = int(rng.integers(0, max_x + 1)) if max_x else (canvas_w - subject.width) // 2
    y = int(rng.integers(0, max_y + 1)) if max_y else (CANVAS_HEIGHT - subject.height) // 2
    canvas.paste(subject, (x, y), subject)

    if params["blur"]:
        canvas = canvas.filter(ImageFilter.GaussianBlur(params["blur"]))

    # A real upload has been through a lossy encoder at least once.
    buffer = BytesIO()
    canvas.convert("RGB").save(buffer, format="JPEG", quality=params["quality"])
    buffer.seek(0)
    return buffer


def build_shift_set(tiles, alphas, severity, seed=1_000):
    """Every row of a split, corrupted at one severity. Returns PIL-ready buffers."""
    params = SEVERITIES[severity]
    if params is None:
        return [Image.fromarray(tile) for tile in tiles]
    # Offset by position in the ladder, not hash(severity): string hashing is
    # salted per process, so hashing here would make the benchmark unreproducible
    # across runs - the one property it cannot be allowed to lose.
    rng = np.random.default_rng(seed + 101 * list(SEVERITIES).index(severity))
    return [_synthesise_upload(tile, alpha, params, rng)
            for tile, alpha in zip(tiles, alphas)]


# ------------------------------------------------------------------ scoring ----
def _ingest(sources, mode, size):
    """Run the serving ingestion over synthesised uploads -> uint8 NHWC stack.

    The same source list is scored once per ingestion mode, so every buffer is
    rewound first; without it the second mode reads an exhausted stream.
    """
    size = tuple(size)
    out = []
    for source in sources:
        if isinstance(source, BytesIO):
            source.seek(0)
        if mode == "squash":
            # Exactly what load_image_array does today: resize, aspect be damned.
            img = source if isinstance(source, Image.Image) else Image.open(source)
            out.append(np.asarray(img.convert("RGB").resize(size, Image.BILINEAR),
                                  dtype=np.uint8))
        else:
            out.append(load_user_image(source, size=size, mode=mode))
    return np.stack(out)


@torch.no_grad()
def _probabilities(model, checkpoint, arrays, device, batch_size=512):
    """Serving-identical scoring: flip-TTA when the checkpoint asks, then tau."""
    tta = bool(checkpoint.get("tta", False))
    tensor = preprocess_arrays(arrays, checkpoint).to(device)
    chunks = []
    for start in range(0, len(tensor), batch_size):
        batch = tensor[start:start + batch_size]
        probabilities = F.softmax(model(batch).float(), dim=1)
        if tta:
            mirrored = F.softmax(model(torch.flip(batch, dims=[3])).float(), dim=1)
            probabilities = (probabilities + mirrored) / 2
        chunks.append(apply_logit_adjustment(probabilities, checkpoint).cpu())
    return torch.cat(chunks).numpy()


def _metrics(y_true, probabilities):
    predicted = probabilities.argmax(1)
    order = np.argsort(-probabilities, axis=1)
    # Pin the label set to the classes actually present, as notebook 02 cell 32
    # does. Left unpinned, sklearn takes the union with y_pred, so every class the
    # model wrongly reaches for injects a 0.0 into macro-F1 - which would make a
    # shifted model look worse for being wrong in more directions, and would not
    # be comparable with the published macro-F1.
    labels = np.unique(y_true)
    return {
        "accuracy": accuracy_score(y_true, predicted) * 100,
        "weighted_f1": f1_score(y_true, predicted, average="weighted", labels=labels,
                                zero_division=0) * 100,
        "macro_f1": f1_score(y_true, predicted, average="macro", labels=labels,
                             zero_division=0) * 100,
        "balanced_acc": balanced_accuracy_score(y_true, predicted) * 100,
        "top3_acc": float((order[:, :3] == y_true[:, None]).any(axis=1).mean()) * 100,
        "mean_conf": float(probabilities.max(axis=1).mean()),
    }


def score_variant(checkpoint_path, modes=INGEST_MODES, severities=tuple(SEVERITIES),
                  rows=1500, seed=7, device=None, verbose=True):
    """One checkpoint across every (severity, ingestion mode). Returns rows of metrics."""
    device = device or choose_device()
    model, checkpoint = load_item_type_model(checkpoint_path, device)
    size = checkpoint["image_size_pil"]

    _, _, test_df, class_names, images = _cached_splits()
    if list(checkpoint["class_names"]) != class_names:
        raise ValueError(f"{Path(checkpoint_path).name}: class order differs from the split")

    subset = np.random.default_rng(seed).choice(len(test_df), min(rows, len(test_df)),
                                                replace=False)
    tiles = np.asarray(images[test_df["cache_position"].to_numpy()])[subset]
    y_true = test_df["label"].to_numpy()[subset]

    alphas, usable = catalogue_alpha(tiles)
    if verbose:
        print(f"  {usable.sum()}/{len(tiles)} rows have a usable matte", flush=True)
    tiles, alphas, y_true = tiles[usable], alphas[usable], y_true[usable]

    name = Path(checkpoint_path).name
    results = []
    for severity in severities:
        sources = build_shift_set(tiles, alphas, severity)
        for mode in modes:
            # A clean tile is already 60x80 and already 3:4, so letterbox and crop
            # collapse to the same resize as squash - scoring them would print
            # identical rows and imply a comparison that was not made. nobg does
            # NOT collapse: it re-crops tight to the subject and re-pads, changing
            # the framing the model was trained on. That row has to be measured,
            # because it decides whether nobg can be the default everywhere or
            # only for uploads - predict.py --submission runs over catalogue tiles.
            if severity == "clean" and mode in ("letterbox", "crop"):
                continue
            started = time.time()
            # Ingestion depends only on (severity, mode, rows, seed), never on the
            # checkpoint, and nobg segmentation is the slowest thing here by an
            # order of magnitude. Memoised so comparing N checkpoints costs one
            # segmentation pass rather than N.
            key = (severity, mode, len(y_true), seed, tuple(size))
            arrays = _INGEST_CACHE.get(key)
            if arrays is None:
                arrays = _ingest(sources, mode, size)
                _INGEST_CACHE[key] = arrays
            metrics = _metrics(y_true, _probabilities(model, checkpoint, arrays, device))
            results.append({"checkpoint": name, "severity": severity, "ingest": mode,
                            "rows": len(y_true), **{k: round(v, 2) for k, v in metrics.items()}})
            if verbose:
                print(f"  {severity:<9} {mode:<9} acc {metrics['accuracy']:6.2f}  "
                      f"wF1 {metrics['weighted_f1']:6.2f}  mF1 {metrics['macro_f1']:6.2f}  "
                      f"top3 {metrics['top3_acc']:6.2f}  conf {metrics['mean_conf']:.3f}  "
                      f"({time.time() - started:.0f}s)", flush=True)
    return results


# ------------------------------------------------------ the real-photo set ----
INPUT_IMAGES = PROJECT_ROOT / "A2_FashionDataset" / "input_images"
TEST_IMAGES = PROJECT_ROOT / "A2_FashionDataset" / "FashionDataset" / "test" / "images_test"
LABEL_SHEET = RESULTS_DIR / "task1_realphoto_labels.csv"


def _retrieval_neighbours(paths, k=3):
    """Task 4's opinion on each photo, as a second view for whoever labels it.

    Retrieval and classification fail differently, so a disagreement is a useful
    flag on the row. Degrades to empty strings when the Task 4 artefacts are
    missing - this is a labelling aid, not a dependency.
    """
    try:
        from src.visual_search.search_engine import SearchEngine
        engine = SearchEngine.load(PROJECT_ROOT / "artifacts" / "task4")
        # search() takes a list and returns a long DataFrame - one row per
        # (query, rank) - so one batched call, not one call per photo.
        hits = engine.search([str(p) for p in paths], k=k, mode="nobg")
    except Exception as error:
        print(f"  (no retrieval neighbours: {type(error).__name__}: {error})")
        return ["" for _ in paths]

    grouped = (hits.sort_values("rank")
               .groupby("query")["articleType"]
               .apply(lambda names: " | ".join(str(n) for n in names)))
    return [grouped.get(p.name, "") for p in paths]


def build_label_sheet(test_sample=150, seed=11, out=LABEL_SHEET, ingest="nobg"):
    """Write a CSV of proposed labels for a human to confirm.

    Two sources, and they answer different questions. The 31 ``input_images`` are
    genuine web photographs - the deployment case the user described, a dress
    found on the internet. The ``images_test`` sample is the graded set, which
    carries a 44.6% distribution shift from train and so is not catalogue-like
    either. Neither has ground truth, and without it none of the shift numbers in
    this module can be checked against a real photograph rather than a simulated
    one.

    Rows are stratified over the *predicted* class, since that is the only
    stratifier available with no labels, and deliberately include the
    low-confidence tail, where the proposals are most likely wrong and a human
    eye is worth the most.
    """
    device = choose_device()
    model, checkpoint = load_item_type_model(DEFAULT_CHECKPOINT, device)
    class_names = list(checkpoint["class_names"])

    from src.data.user_image import list_images
    paths = list(list_images(INPUT_IMAGES))
    sources = ["input_images"] * len(paths)

    predictions_csv = ARTIFACT_DIR / "task1_predictions.csv"
    if predictions_csv.exists() and TEST_IMAGES.exists():
        frame = pd.read_csv(predictions_csv)
        rng = np.random.default_rng(seed)
        # Half stratified over predicted class for coverage, half from the
        # low-confidence tail where the model is least likely to be right.
        per_class = max(1, test_sample // (2 * max(1, frame["articleType"].nunique())))
        spread = (frame.groupby("articleType", group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), per_class), random_state=seed)))
        tail = frame.nsmallest(test_sample // 2, "articleType_confidence")
        picked = pd.concat([spread, tail]).drop_duplicates("id")
        if len(picked) > test_sample:
            picked = picked.iloc[rng.choice(len(picked), test_sample, replace=False)]
        for row_id in picked["id"]:
            candidate = TEST_IMAGES / f"{row_id}.jpg"
            if candidate.exists():
                paths.append(candidate)
                sources.append("images_test")

    print(f"scoring {len(paths)} photos with ingest={ingest} ...", flush=True)
    probabilities = predict_proba(model, checkpoint, paths, device=device,
                                  tta=bool(checkpoint.get("tta", False)), ingest=ingest)
    order = np.argsort(-probabilities, axis=1)[:, :3]

    sheet = pd.DataFrame({
        "source": sources,
        "id": [p.stem for p in paths],
        "image_path": [str(p.relative_to(PROJECT_ROOT)) for p in paths],
        "proposed_articleType": [class_names[row[0]] for row in order],
        "proposed_confidence": probabilities.max(axis=1).round(4),
        "top3": [" | ".join(f"{class_names[i]} {probabilities[r, i]:.2f}" for i in row)
                 for r, row in enumerate(order)],
        "retrieval_neighbours": _retrieval_neighbours(paths),
        # The one column a human fills in. Left blank rather than pre-filled with
        # the proposal, so an unreviewed row is distinguishable from a confirmed
        # one - a pre-filled sheet returned untouched would silently score the
        # model against its own predictions.
        "CONFIRM_articleType": "",
    })
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out, index=False)
    print(f"wrote {out} ({len(sheet)} rows: "
          f"{sum(s == 'input_images' for s in sources)} web photos, "
          f"{sum(s == 'images_test' for s in sources)} test images)")
    print("Fill CONFIRM_articleType for as many rows as you can, then re-run with "
          "--score-realphoto.")
    return sheet


def score_realphoto(checkpoint_path, modes=INGEST_MODES, sheet=LABEL_SHEET, device=None):
    """Score confirmed rows of the label sheet. Skips rows nobody has confirmed."""
    sheet = Path(sheet)
    if not sheet.exists():
        print(f"  (no {sheet}; run --label-sheet first)")
        return []
    frame = pd.read_csv(sheet).fillna({"CONFIRM_articleType": ""})
    frame = frame[frame["CONFIRM_articleType"].astype(str).str.strip() != ""]
    if frame.empty:
        print(f"  ({sheet.name} has no confirmed rows yet)")
        return []

    device = device or choose_device()
    model, checkpoint = load_item_type_model(checkpoint_path, device)
    class_names = list(checkpoint["class_names"])
    lookup = {name: i for i, name in enumerate(class_names)}

    known = frame["CONFIRM_articleType"].isin(lookup)
    if not known.all():
        print(f"  ({(~known).sum()} rows name a class outside the 92; excluded)")
        frame = frame[known]
    paths = [PROJECT_ROOT / p for p in frame["image_path"]]
    y_true = frame["CONFIRM_articleType"].map(lookup).to_numpy()

    results = []
    for mode in modes:
        probabilities = predict_proba(model, checkpoint, paths, device=device,
                                      tta=bool(checkpoint.get("tta", False)), ingest=mode)
        metrics = _metrics(y_true, probabilities)
        results.append({"checkpoint": Path(checkpoint_path).name, "severity": "real-photo",
                        "ingest": mode, "rows": len(y_true),
                        **{k: round(v, 2) for k, v in metrics.items()}})
        print(f"  real-photo {mode:<9} acc {metrics['accuracy']:6.2f}  "
              f"wF1 {metrics['weighted_f1']:6.2f}  top3 {metrics['top3_acc']:6.2f}", flush=True)
    return results


# ------------------------------------------------------- operating points ----
def write_operating_points(checkpoint_path=DEFAULT_CHECKPOINT,
                           out=RESULTS_DIR / "task1_operating_points.csv",
                           device=None):
    """Regenerate the operating-point table from code, on the full test split.

    ``artifacts/task1/operating_points.csv`` has no generator anywhere in the
    repository - the committed five-row version was produced by an ad-hoc session
    and cannot be reproduced or re-derived for a new checkpoint. This rebuilds
    the reproducible rows.

    The committed table's "snapshot ensemble (3 epochs)" row is deliberately not
    reproduced: it averages three per-epoch snapshots the notebook held in memory
    and never wrote to disk, so there is nothing to load. Written to
    ``outputs/evaluation/`` rather than over the committed artefact, so the
    published numbers stay put until a checkpoint is actually promoted.
    """
    device = device or choose_device()
    model, checkpoint = load_item_type_model(checkpoint_path, device)
    train_df, _, test_df, class_names, images = _cached_splits()

    tiles = np.asarray(images[test_df["cache_position"].to_numpy()])
    y_true = test_df["label"].to_numpy()
    tensor = preprocess_arrays(tiles, checkpoint).to(device)

    counts = np.bincount(train_df["label"].to_numpy(), minlength=len(class_names))
    log_prior = torch.tensor(np.log(np.maximum(counts, 1) / counts.sum()),
                             dtype=torch.float32, device=device)
    tau = float(checkpoint.get("logit_adjustment_tau") or 0.0)

    with torch.no_grad():
        plain, flipped = [], []
        for start in range(0, len(tensor), 512):
            batch = tensor[start:start + 512]
            plain.append(F.softmax(model(batch).float(), dim=1).cpu())
            flipped.append(F.softmax(model(torch.flip(batch, dims=[3])).float(), dim=1).cpu())
    plain, flipped = torch.cat(plain), torch.cat(flipped)
    tta = (plain + flipped) / 2

    def adjusted(probabilities):
        shifted = torch.log(probabilities.clamp_min(1e-12)) - tau * log_prior.cpu()
        return F.softmax(shifted, dim=1)

    rows = []
    for name, probabilities, note in [
        ("plain argmax", plain, ""),
        ("horizontal-flip TTA", tta, ""),
        (f"plain argmax + logit adjustment (tau={tau:.2f})", adjusted(plain),
         "validation weighted-F1"),
        (f"horizontal-flip TTA + logit adjustment (tau={tau:.2f})", adjusted(tta),
         "validation weighted-F1"),
    ]:
        array = probabilities.numpy()
        order = np.argsort(-array, axis=1)
        metrics = _metrics(y_true, array)
        rows.append({
            "Model": name + (" (deployed)" if name.startswith("horizontal") and note else ""),
            "accuracy": round(metrics["accuracy"], 2),
            "macro_f1": round(metrics["macro_f1"], 2),
            "weighted_f1": round(metrics["weighted_f1"], 2),
            "balanced_acc": round(metrics["balanced_acc"], 2),
            "top3_acc": round(float((order[:, :3] == y_true[:, None]).any(axis=1).mean()) * 100, 2),
            "top5_acc": round(float((order[:, :5] == y_true[:, None]).any(axis=1).mean()) * 100, 2),
            "tau_selected_on": note,
        })

    frame = pd.DataFrame(rows)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(frame.to_string(index=False))
    print(f"wrote {out}")
    return frame


# --------------------------------------------- tau for the serving distribution ----
def calibrate_for_photos(checkpoint_path, rows=1200, severity="moderate", ingest="nobg",
                         seed=555, tau_grid=(0.0, 0.1, 0.2, 0.3, 0.5, 0.75),
                         metric="weighted_f1", device=None, write=True):
    """Re-select the logit-adjustment tau on the distribution the model serves.

    ``train_item_type.calibrate_checkpoint`` sweeps tau on *clean* validation
    tiles, which is right for the catalogue checkpoint and wrong for this one:
    the photograph model is only ever asked about photographs. Selected on clean
    tiles it took tau=0.50, and tau=0.50 is worse than tau=0 on every
    distribution measured - 84.52 vs 86.04 clean, 48.73 vs 51.02 shifted -
    because a large tau inflates exactly the rare classes ("Trunk", "Gloves",
    "Mufflers") that a low-confidence photograph already drifts towards.

    Uses the VALIDATION split with its own shift seed, so the tau is not chosen
    on the rows or the corruptions the benchmark later reports.
    """
    device = device or choose_device()
    model, checkpoint = load_item_type_model(checkpoint_path, device)
    train_df, val_df, _, class_names, images = _cached_splits()
    if list(checkpoint["class_names"]) != class_names:
        raise ValueError(f"{Path(checkpoint_path).name}: class order differs from the split")

    subset = np.random.default_rng(seed).choice(len(val_df), min(rows, len(val_df)),
                                                replace=False)
    tiles = np.asarray(images[val_df["cache_position"].to_numpy()])[subset]
    y_true = val_df["label"].to_numpy()[subset]
    alphas, usable = catalogue_alpha(tiles)
    tiles, alphas, y_true = tiles[usable], alphas[usable], y_true[usable]

    sources = build_shift_set(tiles, alphas, severity, seed=seed)
    arrays = _ingest(sources, ingest, checkpoint["image_size_pil"])
    tensor = preprocess_arrays(arrays, checkpoint).to(device)

    with torch.no_grad():
        chunks = []
        for start in range(0, len(tensor), 512):
            batch = tensor[start:start + 512]
            probabilities = F.softmax(model(batch).float(), dim=1)
            if checkpoint.get("tta"):
                probabilities = (probabilities + F.softmax(
                    model(torch.flip(batch, dims=[3])).float(), dim=1)) / 2
            chunks.append(probabilities.cpu())
    log_probabilities = torch.log(torch.cat(chunks).clamp_min(1e-12))

    counts = np.bincount(train_df["label"].to_numpy(), minlength=len(class_names))
    log_prior = np.log(np.maximum(counts, 1) / counts.sum())
    prior = torch.tensor(log_prior, dtype=torch.float32)

    best_tau, best = 0.0, None
    for tau in tau_grid:
        scores = _metrics(y_true, F.softmax(log_probabilities - tau * prior, dim=1).numpy())
        print(f"  tau={tau:<5.2f} acc {scores['accuracy']:6.2f}  "
              f"wF1 {scores['weighted_f1']:6.2f}  mF1 {scores['macro_f1']:6.2f}", flush=True)
        if best is None or scores[metric] > best[metric]:
            best_tau, best = tau, scores

    if write:
        checkpoint["class_log_prior"] = log_prior.tolist()
        checkpoint["logit_adjustment_tau"] = float(best_tau)
        checkpoint["logit_adjustment_selected_on"] = (
            f"shift-synthesised validation ({severity}, ingest={ingest}), {metric}")
        torch.save(checkpoint, Path(checkpoint_path))
    print(f"{Path(checkpoint_path).name}: tau={best_tau:.2f} selected on shifted "
          f"validation {metric} ({best[metric]:.2f})")
    return best_tau


# ----------------------------------------------------- confidence calibration ----
def expected_calibration_error(probabilities, y_true, bins=15):
    """ECE: mean gap between stated confidence and observed accuracy.

    The number that says whether a displayed confidence means anything. The
    frontend colours its badge at 0.85 and 0.60 (``app/frontend/app.js:74``), so
    a model that says 0.45 while being right 20% of the time is not merely
    imprecise, it is actively misleading the person looking at the badge.
    """
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def fit_temperature(checkpoint_path, severity="moderate", ingest="nobg", rows=1200,
                    seed=777, device=None, write=True):
    """Fit a single temperature on the distribution this checkpoint serves.

    Temperature scaling (Guo et al. 2017): divide the logits by one scalar chosen
    to minimise validation NLL. It cannot change any prediction - dividing by a
    positive constant preserves the argmax - so it is free of accuracy risk and
    only changes what the confidence number means.

    Fitted on the serving distribution for the same reason tau is: a temperature
    chosen on clean tiles describes a confidence the photograph path never
    produces. Validation rows only, with its own seed.
    """
    device = device or choose_device()
    model, checkpoint = load_item_type_model(checkpoint_path, device)
    _, val_df, _, class_names, images = _cached_splits()

    subset = np.random.default_rng(seed).choice(len(val_df), min(rows, len(val_df)),
                                                replace=False)
    tiles = np.asarray(images[val_df["cache_position"].to_numpy()])[subset]
    y_true = val_df["label"].to_numpy()[subset]
    alphas, usable = catalogue_alpha(tiles)
    tiles, alphas, y_true = tiles[usable], alphas[usable], y_true[usable]

    sources = build_shift_set(tiles, alphas, severity, seed=seed)
    arrays = _ingest(sources, ingest, checkpoint["image_size_pil"])
    tensor = preprocess_arrays(arrays, checkpoint).to(device)

    with torch.no_grad():
        chunks = []
        for start in range(0, len(tensor), 512):
            batch = tensor[start:start + 512]
            logits = model(batch).float()
            if checkpoint.get("tta"):
                logits = (logits + model(torch.flip(batch, dims=[3])).float()) / 2
            chunks.append(logits.cpu())
    logits = torch.cat(chunks)
    target = torch.from_numpy(y_true).long()

    # Coarse grid then a local refine - two lines instead of an optimiser, and the
    # objective is one-dimensional and smooth.
    def nll(temperature):
        return float(F.cross_entropy(logits / temperature, target))

    grid = np.concatenate([np.arange(0.5, 3.01, 0.1), np.arange(3.0, 8.01, 0.25)])
    best = min(grid, key=nll)
    refine = np.linspace(max(0.4, best - 0.1), best + 0.1, 21)
    best = float(min(refine, key=nll))

    before = F.softmax(logits, dim=1).numpy()
    after = F.softmax(logits / best, dim=1).numpy()
    ece_before = expected_calibration_error(before, y_true)
    ece_after = expected_calibration_error(after, y_true)
    print(f"  temperature {best:.2f}  NLL {nll(1.0):.3f} -> {nll(best):.3f}  "
          f"ECE {ece_before:.3f} -> {ece_after:.3f}  "
          f"mean conf {before.max(1).mean():.3f} -> {after.max(1).mean():.3f}  "
          f"(accuracy {(before.argmax(1) == y_true).mean():.3f}, unchanged)")

    if write:
        checkpoint["temperature"] = best
        checkpoint["temperature_fitted_on"] = (
            f"shift-synthesised validation ({severity}, ingest={ingest}), NLL")
        torch.save(checkpoint, Path(checkpoint_path))
    return best


# ------------------------------------------------- BatchNorm target adaptation ----
def adapt_batchnorm(checkpoint_path, out, rows=2500, severity="moderate",
                    ingest="nobg", seed=99, batch_size=256, device=None):
    """Recompute BatchNorm running statistics on the shifted domain, label-free.

    The cheapest thing that can be done to a trained model under covariate shift:
    its convolutions may still be fine, but every BatchNorm carries the mean and
    variance of catalogue tiles on white, and a photograph does not have those
    statistics. Re-estimating them needs no labels and no gradients.

    Two guards keep this honest:

    * it adapts on **train-split** rows, never on the held-out rows the benchmark
      scores, so the numbers afterwards are still measured on unseen data;
    * it uses its own RNG seed, so the exact corrupted images it sees are not the
      exact corrupted images it is later graded on.

    Written to a new checkpoint path, never in place - this is an A/B candidate,
    not an edit to the deployed artefact.
    """
    device = device or choose_device()
    model, checkpoint = load_item_type_model(checkpoint_path, device)
    size = checkpoint["image_size_pil"]

    train_df, _, _, class_names, images = _cached_splits()
    subset = np.random.default_rng(seed).choice(len(train_df), min(rows, len(train_df)),
                                                replace=False)
    tiles = np.asarray(images[train_df["cache_position"].to_numpy()])[subset]
    alphas, usable = catalogue_alpha(tiles)
    tiles, alphas = tiles[usable], alphas[usable]

    sources = build_shift_set(tiles, alphas, severity, seed=seed)
    arrays = _ingest(sources, ingest, size)
    tensor = preprocess_arrays(arrays, checkpoint).to(device)

    modules = [m for m in model.modules()
               if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    for module in modules:
        module.reset_running_stats()
        module.momentum = None      # None means cumulative average, not EMA

    model.train()
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start:start + batch_size]
            if batch.shape[0] < 2:
                continue            # BatchNorm1d in the head needs at least two rows
            model(batch)
    model.eval()

    checkpoint["state_dict"] = model.state_dict()
    checkpoint["model_name"] = f"{checkpoint.get('model_name', 'ItemTypeCNN')}_bnadapt"
    checkpoint["bn_adaptation"] = {
        "rows": int(len(tensor)), "severity": severity, "ingest": ingest, "seed": seed,
        "source": "train split, shift-synthesised, disjoint seed from the benchmark",
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)
    print(f"adapted {len(modules)} BatchNorm layers on {len(tensor)} shifted rows -> {out}")
    return out


_SPLITS = None
_INGEST_CACHE = {}


def _cached_splits():
    """load_splits() reads a 554 MB cache; do it once per process."""
    global _SPLITS
    if _SPLITS is None:
        _SPLITS = load_splits(verbose=False)
    return _SPLITS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", default=[str(DEFAULT_CHECKPOINT)])
    parser.add_argument("--modes", nargs="+", default=list(INGEST_MODES))
    parser.add_argument("--severities", nargs="+", default=list(SEVERITIES))
    parser.add_argument("--rows", type=int, default=1500,
                        help="held-out rows to sample (nobg segmentation is the slow part)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(RESULTS_DIR / "task1_ood_results.csv"))
    parser.add_argument("--label-sheet", action="store_true",
                        help="write the real-photo labelling sheet and exit")
    parser.add_argument("--score-realphoto", action="store_true",
                        help="also score the confirmed rows of the labelling sheet")
    parser.add_argument("--operating-points", action="store_true",
                        help="regenerate the operating-point table and exit")
    parser.add_argument("--calibrate-photos", action="store_true",
                        help="re-select the logit-adjustment tau for the first "
                             "checkpoint on shift-synthesised validation, i.e. the "
                             "distribution a photograph model actually serves")
    parser.add_argument("--fit-temperature", action="store_true",
                        help="fit a confidence temperature for the first checkpoint "
                             "on the distribution it serves, and write it back")
    parser.add_argument("--bn-adapt", metavar="OUT",
                        help="write a BatchNorm-adapted copy of the first "
                             "checkpoint to OUT and exit")
    args = parser.parse_args()

    if args.label_sheet:
        build_label_sheet()
        return

    if args.operating_points:
        write_operating_points(args.checkpoints[0])
        return

    if args.calibrate_photos:
        calibrate_for_photos(args.checkpoints[0])
        return

    if args.fit_temperature:
        fit_temperature(args.checkpoints[0])
        return

    if args.bn_adapt:
        adapt_batchnorm(args.checkpoints[0], args.bn_adapt)
        return

    rows = []
    for checkpoint_path in args.checkpoints:
        print(f"=== {Path(checkpoint_path).name} ===", flush=True)
        rows.extend(score_variant(checkpoint_path, modes=tuple(args.modes),
                                  severities=tuple(args.severities),
                                  rows=args.rows, seed=args.seed))
        if args.score_realphoto:
            rows.extend(score_realphoto(checkpoint_path, modes=tuple(args.modes)))

    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(frame)} rows)")

    pivot = frame.pivot_table(index=["checkpoint", "severity"], columns="ingest",
                              values="accuracy")
    print("\naccuracy by severity x ingestion mode")
    print(pivot.round(2).to_string())


if __name__ == "__main__":
    main()
