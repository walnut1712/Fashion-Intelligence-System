"""Both Task 4 encoders against both kinds of test image.

The background 2x2 in notebook 05 section 10d grades the two encoders on
*synthetic* out-of-domain frames: catalogue items cut out and pasted onto
Places365 scenes. That is a proxy, and a generous one - the garment is still
centred, still studio-lit, still the same photograph underneath.

``A2_FashionDataset/input_images/`` holds 31 photographs that are not a proxy.
They are real uploads: phone pictures, web screenshots, several garments in one
frame, a few things that are not clothing at all. This module scores both
encoders on both kinds of image so the four cells of the table are

    ================  =========================  =========================
                      catalogue test set         outside set (real photos)
    ================  =========================  =========================
    encoder, white    measured, labelled         measured, see below
    encoder, +bg      measured, labelled         measured, see below
    ================  =========================  =========================

What can and cannot be measured on the outside set
--------------------------------------------------
``A2_FashionDataset/input_images_labels.csv`` is an **empty template** - 31 rows,
every ``articleType`` blank. Nobody has filled it in. Without a ground-truth
type there is no P@10 to compute on these photographs, and inventing one would
be worse than not having it.

So the outside column reports what is honestly available without labels:

``top1`` / ``top10``
    mean cosine similarity of the returned items. An encoder out of its depth
    returns lower similarity, and the catalogue-style reference row makes the
    drop readable rather than a bare number.
``agreement``
    how much the two arms' top-10 sets overlap on the same photograph, and how
    often they agree on the single best match. This needs no labels at all and
    answers the question directly: did the training data change the answer?
``declined``
    how often the ingestion path could not segment the subject and fell back to
    a centre crop, which is a property of the photograph rather than the model.

If the label sheet is ever filled in, ``score_arms`` picks it up automatically
and adds real ``P@1``/``P@10`` columns. Nothing else has to change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: The two encoders of the background comparison. Arm C is the catalogue-only
#: control and is deliberately NOT the served checkpoint - see CLAUDE.md.
ARMS = {
    "white only": "task4_encoder_none_seed42.pt",
    "white + backgrounds": "task4_encoder_mixed_seed42.pt",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".gif"}


def real_photo_paths(project_root=None):
    """The 31 real uploads, sorted so the row order is stable across runs."""
    root = Path(project_root or PROJECT_ROOT)
    folder = root / "A2_FashionDataset" / "input_images"
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def real_photo_labels(project_root=None):
    """The hand-label sheet, or None when it carries no usable rows.

    Returns None rather than an empty frame when every ``articleType`` is blank,
    so a caller can say "unlabelled" once instead of every consumer testing the
    column itself.
    """
    root = Path(project_root or PROJECT_ROOT)
    path = root / "A2_FashionDataset" / "input_images_labels.csv"
    if not path.exists():
        return None
    labels = pd.read_csv(path)
    if "articleType" not in labels.columns:
        return None
    labels["articleType"] = labels["articleType"].fillna("").astype(str).str.strip()
    usable = labels[labels["articleType"] != ""]
    return usable if len(usable) else None


def build_engine(checkpoint_path, gallery, images, device=None, batch_size=256):
    """A ``SearchEngine`` over ``images`` using one arm's encoder.

    Both arms are indexed from the same 38,571-row training cache rather than
    from the served index, which covers 38,612. Mixing the two would compare the
    arms over different catalogues; the point here is the encoder, so the
    catalogue is held fixed.
    """
    from src.visual_search.search_engine import SearchEngine, build_encoder

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder = build_encoder(checkpoint)
    encoder.load_state_dict(checkpoint["state_dict"])
    encoder.to(device).eval()

    manifest = {
        "channel_mean": checkpoint["channel_mean"],
        "channel_std": checkpoint["channel_std"],
        "image_size_pil": checkpoint["image_size_pil"],
        "use_tta": True,
        "background_augmented": bool(checkpoint.get("background_augmented", True)),
        "background_source": checkpoint.get("background_source", "unknown"),
    }

    mean = np.asarray(checkpoint["channel_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["channel_std"], dtype=np.float32)
    vectors = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            chunk = np.asarray(images[start:start + batch_size],
                               dtype=np.float32) / 255.0
            tensor = torch.from_numpy(chunk.transpose(0, 3, 1, 2))
            tensor = ((tensor - torch.as_tensor(mean).view(1, 3, 1, 1))
                      / torch.as_tensor(std).view(1, 3, 1, 1)).to(device)
            batch = encoder.embed(tensor)
            batch = torch.nn.functional.normalize(
                batch + encoder.embed(torch.flip(tensor, dims=[3])), p=2, dim=1)
            vectors.append(batch.float().cpu().numpy())

    index = np.vstack(vectors)
    index /= np.clip(np.linalg.norm(index, axis=1, keepdims=True), 1e-8, None)
    return SearchEngine(index, gallery.reset_index(drop=True), manifest,
                        encoder, device)


def _retrieve(engine, paths, k, mode):
    """Top-k positions and similarities for a list of image paths."""
    vectors, infos = engine.embed(paths, mode=mode, return_info=True)
    similarity = vectors @ engine.index.T
    order = np.argsort(-similarity, axis=1)[:, :k]
    scores = np.take_along_axis(similarity, order, axis=1)
    declined = [bool(info.get("fell_back", False)) for info in infos]
    return order, scores, declined


def score_arms(engines, paths, k=10, mode="nobg", labels=None, reference=None):
    """Both arms on one set of photographs, plus what they agree on.

    ``reference`` is an optional list of catalogue-style image paths scored the
    same way, so the similarity drop on real uploads is readable against an
    in-domain number instead of standing alone.
    """
    names = list(engines)
    retrieved, rows = {}, []

    for name in names:
        order, scores, declined = _retrieve(engines[name], paths, k, mode)
        retrieved[name] = order
        row = {
            "arm": name,
            "images": len(paths),
            "top1 similarity": round(float(scores[:, 0].mean()), 4),
            "top%d similarity" % k: round(float(scores.mean()), 4),
            "ingestion declined": int(sum(declined)),
        }
        if reference:
            _, ref_scores, _ = _retrieve(engines[name], reference, k, mode)
            row["top1 on catalogue photos"] = round(float(ref_scores[:, 0].mean()), 4)
            row["similarity drop"] = round(
                float(ref_scores[:, 0].mean() - scores[:, 0].mean()), 4)
        rows.append(row)

    summary = pd.DataFrame(rows)

    # Agreement needs no labels, and is the measurement that answers "did the
    # training data change the answer" directly.
    if len(names) == 2:
        a, b = retrieved[names[0]], retrieved[names[1]]
        overlap = np.array([len(set(x) & set(y)) / k for x, y in zip(a, b)])
        summary["top%d overlap with other arm" % k] = round(float(overlap.mean()), 4)
        summary["same top-1 as other arm"] = round(float((a[:, 0] == b[:, 0]).mean()), 4)

    per_photo = pd.DataFrame({"file": [p.name for p in paths]})
    if len(names) == 2:
        per_photo["top10 overlap"] = overlap
        per_photo["same top-1"] = a[:, 0] == b[:, 0]

    if labels is not None:
        types = engines[names[0]].metadata["articleType"].to_numpy()
        by_file = dict(zip(labels["file"], labels["articleType"]))
        keep = [i for i, p in enumerate(paths) if by_file.get(p.name)]
        if keep:
            truth = np.array([by_file[paths[i].name] for i in keep])
            for position, name in enumerate(names):
                hits = types[retrieved[name][keep]] == truth[:, None]
                summary.loc[position, "P@1"] = round(float(hits[:, 0].mean()) * 100, 2)
                summary.loc[position, "P@%d" % k] = round(float(hits.mean()) * 100, 2)
                summary.loc[position, "labelled"] = len(keep)

    return summary, per_photo


def catalogue_test_scores(project_root=None):
    """The catalogue-test-set column, read from the committed comparison."""
    root = Path(project_root or PROJECT_ROOT)
    path = root / "outputs" / "evaluation" / "task4_background_arms.csv"
    if not path.exists():
        return None
    scores = pd.read_csv(path)
    rename = {"C_encoder_clean": "white only",
              "D_encoder_bgaug": "white + backgrounds"}
    scores["arm"] = scores["arm"].map(rename).fillna(scores["arm"])
    return scores


def describe_label_state(labels, paths):
    """One honest sentence about whether the outside column can be scored."""
    if labels is None:
        return ("input_images_labels.csv carries no filled-in articleType, so "
                "P@10 cannot be computed on the %d real photographs. The columns "
                "below need no labels; fill the sheet in and this cell adds "
                "P@1/P@10 by itself." % len(paths))
    return ("%d of %d real photographs are labelled, so P@1 and P@10 below are "
            "measured on those." % (len(labels), len(paths)))
