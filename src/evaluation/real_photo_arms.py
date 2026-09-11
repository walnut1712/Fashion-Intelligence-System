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

#: The arms of the background comparison, named exactly as notebook 05 Part 6b
#: names them so one vocabulary covers the whole task. Value is
#: (checkpoint filename, --backgrounds value, required).
#:
#: The control arm C is deliberately NOT the served checkpoint - see CLAUDE.md.
#: Arm P is optional: the project keeps one encoder, so its checkpoint is
#: routinely deleted after measuring, and a caller must cope with its absence
#: rather than fail. The first entry is the control that agreement is measured
#: against, so keep it first.
ARMS = {
    "C  catalogue only": ("task4_encoder_none_seed42.pt", "none", True),
    "P  procedural backdrops": ("task4_encoder_procedural_seed42.pt", "procedural", False),
    "D  mixed backdrops (deployed)": ("task4_encoder_mixed_seed42.pt", "mixed", True),
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".gif"}

# Written in the label sheet's ``articleType`` for a photograph that is not one
# catalogue garment - a sticker, a six-item flat-lay, an outfit where two
# garments are co-equal. It is a deliberate exclusion, not an unfilled row, and
# it must never reach the scorer: no catalogue item has articleType "none", so
# such a row would match nothing and silently count as a miss.
EXCLUDED_ARTICLE_TYPES = {"none", "n/a"}


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

    Returns None rather than an empty frame when no row carries a scorable
    ``articleType``, so a caller can say "unlabelled" once instead of every
    consumer testing the column itself.

    Rows marked with an ``EXCLUDED_ARTICLE_TYPES`` value are dropped here rather
    than downstream. They are photographs a human decided are not one catalogue
    garment, so they are outside what P@10 can mean; left in, they would be
    compared against a class that does not exist and scored as misses.
    """
    root = Path(project_root or PROJECT_ROOT)
    path = root / "A2_FashionDataset" / "input_images_labels.csv"
    if not path.exists():
        return None
    labels = pd.read_csv(path)
    if "articleType" not in labels.columns:
        return None
    labels["articleType"] = labels["articleType"].fillna("").astype(str).str.strip()
    scorable = ~labels["articleType"].str.lower().isin(EXCLUDED_ARTICLE_TYPES)
    usable = labels[(labels["articleType"] != "") & scorable]
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


def _ingest_once(engine, paths, mode):
    """Ingest each photograph exactly once, for every arm to share.

    This must NOT be done per arm. ``nobg`` falls back to ``cv2.grabCut``
    whenever ``rembg`` is absent, and grabCut seeds its GMM from OpenCV's
    process-global RNG, so segmenting the same photograph twice gives two
    different cutouts. Ingesting inside the arm loop therefore hands each arm a
    different picture and the comparison silently stops being paired - adding a
    third arm was enough to move the deployed arm's real-photo P@1 by four
    points, which is how this was found. One ingestion, shared, removes it.

    Sharing the ingestion repairs the *comparison*. The absolute scores used to
    depend on grabCut's RNG as well, which ``rembg`` fixes: it is the ladder's
    top tier, is deterministic, and is now a declared requirement, so two
    invocations of this harness produce byte-identical tables. Keep this
    function even so - the ladder still falls back to grabCut wherever ``rembg``
    is unavailable, and the pairing must not depend on which tier ran.
    """
    from src.data.user_image import load_user_image

    loaded = [load_user_image(path, engine.size, mode=mode, return_info=True)
              for path in paths]
    arrays = np.stack([array for array, _ in loaded])
    declined = [bool(info.get("fell_back", False)) for _, info in loaded]
    return arrays, declined


def _retrieve(engine, arrays, k):
    """Top-k positions and similarities for frames that are already ingested."""
    vectors = engine.embed_arrays(arrays)
    similarity = vectors @ engine.index.T
    order = np.argsort(-similarity, axis=1)[:, :k]
    scores = np.take_along_axis(similarity, order, axis=1)
    return order, scores


def score_arms(engines, paths, k=10, mode="nobg", labels=None, reference=None):
    """Every arm on one set of photographs, plus what they agree on.

    ``reference`` is an optional list of catalogue-style image paths scored the
    same way, so the similarity drop on real uploads is readable against an
    in-domain number instead of standing alone.
    """
    names = list(engines)
    retrieved, rows = {}, []

    # One ingestion for every arm - see _ingest_once. If the arms disagreed on
    # input size a single ingestion could not serve them, so say so rather than
    # feeding one of them the wrong shape.
    sizes = {tuple(engines[name].size) for name in names}
    if len(sizes) != 1:
        raise ValueError("arms disagree on input size %s, so they cannot share "
                         "one ingestion" % sorted(sizes))
    frames, declined = _ingest_once(engines[names[0]], paths, mode)
    reference_frames = (_ingest_once(engines[names[0]], reference, mode)[0]
                        if reference else None)

    for name in names:
        order, scores = _retrieve(engines[name], frames, k)
        retrieved[name] = order
        row = {
            "arm": name,
            "images": len(paths),
            "top1 similarity": round(float(scores[:, 0].mean()), 4),
            "top%d similarity" % k: round(float(scores.mean()), 4),
            "ingestion declined": int(sum(declined)),
        }
        if reference_frames is not None:
            _, ref_scores = _retrieve(engines[name], reference_frames, k)
            row["top1 on catalogue photos"] = round(float(ref_scores[:, 0].mean()), 4)
            row["similarity drop"] = round(
                float(ref_scores[:, 0].mean() - scores[:, 0].mean()), 4)
        rows.append(row)

    summary = pd.DataFrame(rows)

    # Agreement needs no labels, and is the measurement that answers "did the
    # training data change the answer" directly. Every arm is compared against
    # the first, which is the catalogue-only control, so the column reads as
    # "how far did this training distribution move the answer". The control's
    # own row is left blank rather than filled with a trivial 1.0.
    per_photo = pd.DataFrame({"file": [p.name for p in paths]})
    if len(names) >= 2:
        control = retrieved[names[0]]
        for position, name in enumerate(names):
            if position == 0:
                continue
            other = retrieved[name]
            overlap = np.array([len(set(x) & set(y)) / k
                                for x, y in zip(control, other)])
            same_top1 = control[:, 0] == other[:, 0]
            summary.loc[position, "top%d overlap with control" % k] = round(
                float(overlap.mean()), 4)
            summary.loc[position, "same top-1 as control"] = round(
                float(same_top1.mean()), 4)
            per_photo["top%d overlap, %s" % (k, name)] = overlap
            per_photo["same top-1, %s" % name] = same_top1

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
    rename = {"C_encoder_clean": "C  catalogue only",
              "P_encoder_procedural": "P  procedural backdrops",
              "D_encoder_bgaug": "D  mixed backdrops (deployed)"}
    scores["arm"] = scores["arm"].map(rename).fillna(scores["arm"])
    return scores


def describe_label_state(labels, paths):
    """One honest sentence about whether the outside column can be scored."""
    if labels is None:
        return ("input_images_labels.csv carries no filled-in articleType, so "
                "P@10 cannot be computed on the %d real photographs. The columns "
                "below need no labels; fill the sheet in and this cell adds "
                "P@1/P@10 by itself." % len(paths))
    return ("%d of %d real photographs carry a catalogue articleType, so P@1 and "
            "P@10 below are measured on those. The rest are marked %s: not one "
            "catalogue garment, and excluded rather than counted as misses."
            % (len(labels), len(paths), "/".join(sorted(EXCLUDED_ARTICLE_TYPES))))
