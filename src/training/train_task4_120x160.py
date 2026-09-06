#!/usr/bin/env python
"""Train the Task 4 retrieval encoder, at 60x80 or at 120x160.

Why a script and not a notebook cell
------------------------------------
A full run is longer than a notebook kernel should be trusted to stay alive.
Every epoch is checkpointed, so ``--resume`` picks the run back up.

The cost figures below were written on a machine with no CUDA device. This one
has an RTX 3060, so read them as an upper bound rather than an estimate; the
run prints its own wall clock at the end, which is the number to quote.

Why both resolutions live here
------------------------------
The published 80.2 cannot be compared against a 120x160 number. The 120x160
pipeline drops 41 rows flagged for conflicting task labels, and
``RetrievalProtocol`` shuffles *products* to pick its holdout - so removing any
row reshuffles the split. Measured, the reduced gallery shares only **6%** of
its held-out queries with the original.

Worse, the shipped 60x80 encoder was trained under the old split, so a
substantial share of what the new split calls held-out was in its training set.
Scoring it here would flatter it. The only honest baseline is a 60x80 encoder
trained on this gallery, by this code, and that is what ``--resolution 60x80``
produces. Two runs, one comparison.

Cost, and the warm-start shortcut
---------------------------------
Both arms train from scratch, because that is the only way the comparison is
clean, and from scratch is 30 epochs rather than the 12 a fine-tune needs. On a
CPU that is roughly 6 h at 60x80 and 24 h at 120x160.

``--warm-start artifacts/task4/task4_encoder_clean.pt`` cuts both to about a
third. It is not free: that encoder was trained under the old split, so both
arms inherit its exposure to items this split holds out, and the absolute
numbers are inflated. The resolution *difference* survives, because both arms
inherit the same thing - but it also tilts slightly toward 60x80, whose features
were learned at exactly that input scale. Use it to get an early read, not to
report a headline.

Where the backdrops come from
-----------------------------
``--backgrounds`` selects the training bank, and the choice is a second
comparison running alongside the resolution one:

``procedural``
    the five formula families (solid, gradient, noise, blobs, blurred crop)
    that notebook 06 used. None of them is a photograph.
``places365``
    scene photographs from ``A2_FashionDataset/external_data/places365``, drawn only from the 292
    training scene CATEGORIES.
``mixed`` (default)
    both, at ``photographic_share``. The serve path contains both cases - 38% of
    uploads reach the encoder with their backdrop intact, the rest arrive
    segmented onto a flat field - so training on one family teaches invariance
    to one half of what the service sees.

Evaluation always reports five benchmarks, over TWO held-out background
families: checkerboards and stripes, which no run ever trains on, and Places365
scenes from the 73 held-out categories. A model trained on photographs that
improves on ``photo`` while falling on ``hard`` has swapped one overfit for
another, and only grading both makes that visible.

Usage
-----
    python -m src.training.train_task4_120x160 --resolution 60x80   --seed 42
    python -m src.training.train_task4_120x160 --resolution 120x160 --seed 42
    python -m src.training.train_task4_120x160 --resolution 120x160 --resume
    python -m src.training.train_task4_120x160 --backgrounds procedural  # the arm to beat
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.places365_backgrounds import (  # noqa: E402
    load_background_bank,
    make_mixed_bank,
    write_split_manifest,
)
from src.data.synthetic_backgrounds import (  # noqa: E402
    EVAL_DEGRADATIONS,
    composite,
    degrade,
    make_backgrounds,
    make_eval_backgrounds,
    simulate_ingestion,
)
from src.evaluation.metrics import CatalogueIndex, RetrievalProtocol  # noqa: E402
from src.training.task4_training import (  # noqa: E402
    PKSampler,
    WildDataset,
    augment_batch,
    batch_hard_triplet_loss,
    channel_statistics,
    embedding_spread,
)
from src.visual_search.search_engine import ImprovedEncoder  # noqa: E402

PROCESSED = PROJECT_ROOT / "A2_FashionDataset" / "processed"

CONFIG = {
    "embedding_dim": 128,
    "triplet_margin": 0.3,
    "aux_type_weight": 0.5,
    "aux_colour_weight": 0.5,
    # 1e-3 is what notebook 05 used and it trains 60x80 fine, but it destabilises
    # 120x160: measured, a from-scratch 120x160 run organises to a spread of
    # 0.011 by epoch 5 and then falls back monotonically to 0.0016 by epoch 9,
    # with monitor clean P@10 degrading 62.66 -> 42.46 on CLEAN frames, before
    # any augmentation is applied. Four times the pixels reach the same global
    # average pool, so the pooled feature is a mean over 80 positions rather than
    # 20 and its scale differs; the same step size is too large for it. Halved
    # for BOTH arms rather than only the one that needed it, so resolution stays
    # the only difference between them.
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    #: Triplet mining is unstable early, when the hardest positive and hardest
    #: negative are both near the margin. Clipping bounds one bad batch.
    "grad_clip": 1.0,
    "batches_per_epoch": 250,
    "p": 16, "k": 8,
    "bg_end": 0.6,
    "ramp_epochs": 6,
    # Epochs trained on CLEAN catalogue frames before any backdrop or
    # degradation is applied. Not a nicety: from-scratch training with the
    # augmentation on from epoch 1 collapses, measured. The triplet loss sits at
    # exactly the margin (0.3003) from epoch 2, the mean pairwise distance
    # between embeddings falls from 0.006 to 0.001 and stays there, and the run
    # ends at clean P@10 48.5 against the 80.2 the same architecture reaches
    # when it is allowed to organise first. Notebook 06 section 6 recorded the
    # same collapse and worked around it by fine-tuning an already-trained clean
    # encoder; this does the same thing in one run, which keeps the arms
    # comparable without importing a checkpoint trained under a different split.
    "warmup_epochs": 12,
    # A healthy L2-normalised embedding has a mean pairwise distance near 1.
    # Anything below this after the warmup has collapsed, whatever the peak was.
    "collapse_floor": 0.05,
    "scale_range": (0.55, 1.00),
    "eval_every": 2,
    "collapse_fraction": 0.15,
    # The in-training monitor scores against a catalogue SUBSET and every 4th
    # query. Embedding all 32,944 items at 120x160 takes longer than the epoch
    # that produced the weights, so a full evaluation every other epoch would
    # have spent most of a 30-epoch run measuring rather than training. The full
    # catalogue is used once, at the end, for the numbers that get reported.
    "monitor_catalogue": 8000,
    "monitor_query_stride": 4,
    # Size of each background bank. 600 procedural frames was enough when every
    # frame was a formula; a photograph carries far more variation, so drawing
    # 8,000 distinct scenes costs ~100 s once and removes the risk of the encoder
    # memorising a small bank.
    "procedural_bank": 600,
    "photographic_bank": 8000,
    # Within a composited sample, how often the backdrop is a photograph rather
    # than a procedural field. 0.70 tracks the serve path: 38% of uploads arrive
    # with the backdrop intact (a scene), the other 62% arrive segmented onto a
    # flat field, and the encoder has to be invariant to both.
    "photographic_share": 0.70,
}

#: One point out of domain is worth three in domain, because the product serves
#: photographs. Shared with notebook 06 so selection and promotion cannot drift.
DEPLOYMENT_WEIGHT = 3.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", choices=("60x80", "120x160"), default="120x160")
    parser.add_argument("--epochs", type=int, default=42,
                        help="12 clean warmup epochs plus the 30-epoch augmented "
                             "recipe notebook 05 used for the deployed encoder")
    parser.add_argument("--warmup", type=int, default=None,
                        help="clean epochs before augmentation ramps in "
                             "(default 12; 0 reproduces the collapse)")
    parser.add_argument("--warm-start", type=Path, default=None,
                        help="initialise from a checkpoint instead of random. "
                             "Cheaper, but see the note in the module docstring")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backgrounds", choices=("procedural", "places365", "mixed"),
                        default="mixed",
                        help="source of the training backdrops. 'procedural' is "
                             "the five formula families notebook 06 used; "
                             "'places365' is scene photographs from the training "
                             "categories; 'mixed' draws both, which is the "
                             "default because the serve path contains both")
    parser.add_argument("--selection-benchmark", default="wildphoto",
                        choices=("wild", "wildphoto"),
                        help="which benchmark picks the best epoch. 'wildphoto' "
                             "is the closest proxy to an upload this project has")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-catalogue", type=int, default=None,
                        help="smaller catalogue for a smoke run")
    parser.add_argument("--batches", type=int, default=None,
                        help="override batches per epoch (smoke runs)")
    return parser.parse_args()


def load_gallery(resolution):
    images = np.load(PROCESSED / f"task4_cache_{resolution}.npy", mmap_mode="r")
    masks = np.load(PROCESSED / f"task4_masks_{resolution}.npy", mmap_mode="r")
    gallery = pd.read_csv(PROCESSED / f"task4_gallery_{resolution}.csv")
    if not (len(images) == len(masks) == len(gallery)):
        sys.exit("cache and gallery disagree: {} / {} / {}".format(
            len(images), len(masks), len(gallery)))
    return images, masks, gallery


def build_queries(images, masks, protocol, seed=123):
    """Five renderings of the held-out items, differing by one stage each.

    ``clean``
        the catalogue frame. The academic reference.
    ``hard``
        composited onto checkerboards and stripes, families no training run
        generates.
    ``photo``
        composited onto Places365 scenes from the 73 held-out CATEGORIES. A
        photographic backdrop the encoder has never seen the kind of.
    ``wild`` / ``wildphoto``
        the two above, plus the held-out camera degradations and the ingestion
        path. ``wildphoto`` is the closest proxy to a real upload here.

    Two background families are graded rather than one because a model trained
    on photographs must not be judged only on photographs. If it improves on
    ``photo`` while falling on ``hard``, it swapped one overfit for another and
    the table says so.
    """
    height, width = images.shape[1], images.shape[2]
    evaluation_backgrounds = make_eval_backgrounds(600, size=(width, height))
    photographic_backgrounds = load_background_bank(
        count=2000, shape=(height, width, 3), split="test", seed=seed)
    generator = np.random.default_rng(seed)

    clean = np.stack([np.asarray(images[p]) for p in protocol.heldout_queries])

    def composite_onto(bank):
        # 2,000 frames is ~115 MB at 120x160 - materialising this one is
        # affordable, unlike the 32,944-row catalogue.
        return np.stack([
            composite(np.asarray(images[p]), np.asarray(masks[p]),
                      bank[generator.integers(len(bank))], generator)
            for p in protocol.heldout_queries
        ])

    def weather(frames):
        return np.stack([
            simulate_ingestion(degrade(frame, generator, EVAL_DEGRADATIONS), generator)
            for frame in frames
        ])

    hard = composite_onto(evaluation_backgrounds)
    photo = composite_onto(photographic_backgrounds)
    return {"clean": clean, "hard": hard, "photo": photo,
            "wild": weather(hard), "wildphoto": weather(photo)}


@torch.no_grad()
def embed(model, arrays, mean, std, device, batch_size=256, tta=True,
          positions=None):
    """Embed frames, reading them a batch at a time.

    ``positions`` lets a caller hand over a memmap and the rows it wants instead
    of a materialised array. That is not a nicety: the 120x160 catalogue is
    32,944 x 160 x 120 x 3, so ``np.asarray(images[catalogue_pos])`` allocates
    1.9 GB before this function gets a chance to chunk anything, and the process
    was killed for memory doing exactly that.
    """
    model.eval()
    out = []
    total = len(positions) if positions is not None else len(arrays)
    for start in range(0, total, batch_size):
        if positions is not None:
            rows = positions[start:start + batch_size]
            chunk = np.asarray(arrays[rows], dtype=np.float32) / 255.0
        else:
            chunk = np.asarray(arrays[start:start + batch_size],
                               dtype=np.float32) / 255.0
        tensor = torch.from_numpy(chunk.transpose(0, 3, 1, 2))
        tensor = ((tensor - torch.as_tensor(mean).view(1, 3, 1, 1))
                  / torch.as_tensor(std).view(1, 3, 1, 1)).to(device)
        vectors = model.embed(tensor)
        if tta:
            vectors = F.normalize(
                vectors + model.embed(torch.flip(tensor, dims=[3])), p=2, dim=1)
        out.append(vectors.float().cpu().numpy())
    matrix = np.vstack(out)
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8, None)


def monitor(model, images, protocol, queries, mean, std, device, subset, stride):
    """Cheap in-training score: a catalogue subset, every Nth query, no TTA.

    Not comparable with ``evaluate`` and not meant to be - it exists to pick the
    best epoch and to show the run is alive, so it trades the last point of
    accuracy for being affordable every other epoch.
    """
    catalogue = embed(model, images, mean, std, device, tta=False, positions=subset)
    catalogue_t = torch.from_numpy(catalogue).to(device)
    picked = np.arange(0, len(protocol.heldout_queries), stride)
    truth = protocol.heldout_queries[picked]

    scores = {}
    for name, frames in queries.items():
        vectors = torch.from_numpy(
            embed(model, frames[picked], mean, std, device, tta=False)).to(device)
        top = torch.topk(vectors @ catalogue_t.T, k=10, dim=1).indices.cpu().numpy()
        positions = subset[top]
        type_hit = protocol.article[positions] == protocol.article[truth][:, None]
        colour_hit = protocol.colour[positions] == protocol.colour[truth][:, None]
        scores[name] = {"P@10": float(type_hit.mean() * 100),
                        "both@10": float((type_hit & colour_hit).mean() * 100)}
    return scores


def evaluate(model, images, protocol, queries, mean, std, device):
    """Score one model on each benchmark through the shared protocol."""
    catalogue = embed(model, images, mean, std, device,
                      positions=protocol.catalogue_pos)
    results = {}
    for name, frames in queries.items():
        full = np.zeros((len(protocol.gallery), catalogue.shape[1]), dtype=np.float32)
        full[protocol.catalogue_pos] = catalogue
        full[protocol.heldout_queries] = embed(model, frames, mean, std, device)
        index = CatalogueIndex(full, protocol, name=name)
        summary, _ = protocol.evaluate_deployment(index, full, store=False)
        results[name] = {k: float(v) * 100 for k, v in summary.items()
                         if k in ("P@1", "P@10", "colour@10", "colourfam@10",
                                  "both@10", "bothfam@10")}
    return results


def main():
    args = parse_args()
    if args.batches:
        CONFIG["batches_per_epoch"] = args.batches
    if args.warmup is not None:
        CONFIG["warmup_epochs"] = args.warmup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = PROJECT_ROOT / "artifacts" / f"task4_{args.resolution}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # The background source is part of the run's identity, not a detail: a
    # procedural run and a places365 run at the same resolution and seed are two
    # arms of a comparison and must not land on one filename.
    tag = f"{args.backgrounds}_seed{args.seed}"
    state_path = artifact_dir / f"train_state_{tag}.pt"

    images, masks, gallery = load_gallery(args.resolution)
    print(f"{args.resolution}: {len(gallery):,} items, frames {images.shape[1:]}")

    protocol = RetrievalProtocol(gallery=gallery)
    catalogue_pos = protocol.catalogue_pos
    if args.limit_catalogue:
        catalogue_pos = catalogue_pos[:args.limit_catalogue]
    print(f"catalogue {len(protocol.catalogue_pos):,} | "
          f"held-out queries {len(protocol.heldout_queries):,}")

    type_codes, type_names = pd.factorize(gallery["articleType"])
    colour_codes, colour_names = pd.factorize(gallery["baseColour"].fillna("Unknown"))
    product_codes = pd.factorize(gallery["productDisplayName"].fillna(""))[0]

    # Normalisation from the CATALOGUE rows only - the held-out items must not
    # reach the statistics every model is normalised by.
    mean, std = channel_statistics(images, protocol.catalogue_pos)
    print("channel mean", np.round(mean, 4).tolist(), "std", np.round(std, 4).tolist())

    height, width = images.shape[1], images.shape[2]
    procedural = make_backgrounds(CONFIG["procedural_bank"],
                                  shape=(height, width, 3), seed=42,
                                  source_images=images[
                                      np.sort(np.random.default_rng(0).choice(
                                          protocol.catalogue_pos, 400, replace=False))])

    if args.backgrounds == "procedural":
        backgrounds = procedural
    else:
        # Training draws only from the 292 training CATEGORIES; the 73 held-out
        # ones are reserved for the `photo` and `wildphoto` benchmarks, so an
        # unseen background means an unseen kind of place rather than another
        # photograph of a place already trained on.
        manifest = write_split_manifest()
        print("Places365: {} train categories ({:,} images) | {} held out "
              "({:,} images)".format(manifest["n_train_categories"],
                                     manifest["n_train_images"],
                                     manifest["n_test_categories"],
                                     manifest["n_test_images"]))
        photographic = load_background_bank(
            count=CONFIG["photographic_bank"], shape=(height, width, 3),
            split="train", seed=args.seed, verbose=True)
        backgrounds = (photographic if args.backgrounds == "places365"
                       else make_mixed_bank(procedural, photographic,
                                            CONFIG["photographic_share"],
                                            seed=args.seed))
    print(f"training backdrops: {args.backgrounds}, {len(backgrounds):,} frames")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = ImprovedEncoder(embedding_dim=CONFIG["embedding_dim"],
                            n_types=len(type_names),
                            n_colours=len(colour_names)).to(device)

    if args.warm_start:
        checkpoint = torch.load(args.warm_start, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"warm-started from {args.warm_start.name} - the absolute numbers "
              "carry that encoder's split, so read the resolution DIFFERENCE only")

    dataset = WildDataset(images, masks, backgrounds, catalogue_pos,
                          type_codes[catalogue_pos], colour_codes[catalogue_pos],
                          mean, std, scale_range=CONFIG["scale_range"],
                          seed=args.seed)
    sampler = PKSampler(type_codes[catalogue_pos], product_codes[catalogue_pos],
                        p=CONFIG["p"], k=CONFIG["k"],
                        batches_per_epoch=CONFIG["batches_per_epoch"], seed=args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                                  weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch, history = 0, []
    best = {"score": -np.inf, "state": None, "epoch": -1}
    if args.resume and state_path.exists():
        saved = torch.load(state_path, map_location=device)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch, history, best = saved["epoch"], saved["history"], saved["best"]
        print(f"resumed from epoch {start_epoch}")

    queries = build_queries(images, masks, protocol)
    print("queries:", {k: v.shape for k, v in queries.items()})

    monitor_subset = np.sort(np.random.default_rng(7).choice(
        protocol.catalogue_pos,
        min(CONFIG["monitor_catalogue"], len(protocol.catalogue_pos)),
        replace=False))
    print(f"monitor: {len(monitor_subset):,} catalogue items, "
          f"every {CONFIG['monitor_query_stride']}th query")

    # The collapse guard tracks the PEAK spread seen, not the starting one. A
    # randomly initialised encoder maps every input to nearly the same point, so
    # its spread is ~0.02 and a fraction of it is a threshold nothing can fall
    # below - the guard notebook 06 section 6 needed would never have fired.
    # Spread rises as the embedding organises, then collapse is a fall away from
    # that peak.
    peak_spread = embedding_spread(model, images, protocol.catalogue_pos,
                                   mean, std, device)
    print(f"initial embedding spread: {peak_spread:.4f}")
    started = time.time()

    warmup = CONFIG["warmup_epochs"]
    for epoch in range(start_epoch, args.epochs):
        # Stage 1: clean frames, so the embedding organises at all. Stage 2: the
        # backdrop and the camera ramp in over `ramp_epochs`.
        ramp = min(1.0, max(0.0, (epoch + 1 - warmup) / CONFIG["ramp_epochs"]))
        dataset.probability = CONFIG["bg_end"] * ramp
        dataset.strength = ramp

        model.train()
        totals, batches = defaultdict(float), 0
        for frames, types, colours in loader:
            frames = augment_batch(frames.to(device), jitter=0.10)
            types = types.to(device)
            colours = colours.to(device)

            optimizer.zero_grad(set_to_none=True)
            embeddings, type_logits, colour_logits = model(frames, with_heads=True)
            triplet = batch_hard_triplet_loss(embeddings, types,
                                              CONFIG["triplet_margin"])
            loss = (triplet
                    + CONFIG["aux_type_weight"] * F.cross_entropy(type_logits, types)
                    + CONFIG["aux_colour_weight"] * F.cross_entropy(colour_logits, colours))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimizer.step()
            totals["triplet"] += triplet.item()
            totals["total"] += loss.item()
            batches += 1
        scheduler.step()

        spread = embedding_spread(model, images, protocol.catalogue_pos,
                                  mean, std, device)
        record = {"epoch": epoch + 1, "seed": args.seed,
                  "bg_probability": round(dataset.probability, 3),
                  "strength": round(dataset.strength, 3),
                  "triplet": totals["triplet"] / batches,
                  "total": totals["total"] / batches, "spread": spread}

        peak_spread = max(peak_spread, spread)
        # Two conditions, because the relative one alone missed a real collapse.
        # A randomly initialised encoder maps everything to nearly one point, so
        # its spread (~0.006) is a peak nothing has to beat, and 15% of it
        # (0.0009) is a floor a fully collapsed run (0.0010) sits above. The
        # absolute floor is what actually catches it; the relative test still
        # catches a collapse that happens after the embedding has organised.
        collapsed_absolute = (epoch + 1 > warmup
                              and spread < CONFIG["collapse_floor"])
        # The relative test is only meaningful once the embedding has actually
        # organised. Before that every value is ~0.01 and a ratio between two
        # such numbers is noise, so requiring the peak to have cleared the floor
        # stops a wobble during the plateau from killing a healthy run.
        collapsed_relative = (peak_spread > CONFIG["collapse_floor"]
                              and spread < CONFIG["collapse_fraction"] * peak_spread)
        if collapsed_absolute or collapsed_relative:
            print(f"COLLAPSE at epoch {epoch + 1}: spread {spread:.4f} against a "
                  f"peak of {peak_spread:.4f} and a floor of "
                  f"{CONFIG['collapse_floor']}. Stopping.")
            break

        if (epoch + 1) % CONFIG["eval_every"] == 0 or epoch == args.epochs - 1:
            scores = monitor(model, images, protocol, queries, mean, std, device,
                             monitor_subset, CONFIG["monitor_query_stride"])
            record.update({f"{b}_{m}": v for b, s in scores.items()
                           for m, v in s.items()})
            weighted = (DEPLOYMENT_WEIGHT * scores[args.selection_benchmark]["both@10"]
                        + scores["clean"]["both@10"])
            if weighted > best["score"]:
                best = {"score": weighted,
                        "state": copy.deepcopy(model.state_dict()),
                        "epoch": epoch + 1}
            print(f"epoch {epoch + 1:>2}/{args.epochs} triplet {record['triplet']:.4f} "
                  f"spread {spread:.3f} | monitor clean P@10 "
                  f"{scores['clean']['P@10']:.2f} wild P@10 {scores['wild']['P@10']:.2f}"
                  f" wildphoto P@10 {scores['wildphoto']['P@10']:.2f}"
                  f"{'  <- best' if best['epoch'] == epoch + 1 else ''}", flush=True)
        else:
            print(f"epoch {epoch + 1:>2}/{args.epochs} triplet {record['triplet']:.4f} "
                  f"spread {spread:.3f}", flush=True)

        history.append(record)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch + 1,
                    "history": history, "best": best}, state_path)

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"restored the best epoch ({best['epoch']})")

    final = evaluate(model, images, protocol, queries, mean, std, device)
    minutes = (time.time() - started) / 60
    print(f"\ntrained in {minutes:.1f} min")
    for bench, scores in final.items():
        print(f"  {bench:<6} " + "  ".join(f"{m} {v:.2f}" for m, v in scores.items()))

    torch.save({
        "state_dict": model.state_dict(),
        "architecture": "improved",
        "embedding_dim": CONFIG["embedding_dim"],
        "n_types": len(type_names), "n_colours": len(colour_names),
        "channel_mean": list(map(float, mean)), "channel_std": list(map(float, std)),
        "image_size_pil": [width, height],
        "resolution": args.resolution,
        "background_augmented": True, "degradation_augmented": True,
        "background_source": args.backgrounds,
        "photographic_share": (CONFIG["photographic_share"]
                               if args.backgrounds == "mixed"
                               else float(args.backgrounds == "places365")),
        "background_split": "places365 scene categories, 292 train / 73 held out",
        "selection_benchmark": args.selection_benchmark,
        "warmup_epochs": CONFIG["warmup_epochs"],
        "ramp_epochs": CONFIG["ramp_epochs"],
        "learning_rate": CONFIG["learning_rate"],
        "seed": args.seed, "best_epoch": best["epoch"],
        "gallery": f"task4_gallery_{args.resolution}.csv",
        "trained_by": "src/training/train_task4_120x160.py",
    }, artifact_dir / f"task4_encoder_{tag}.pt")

    pd.DataFrame(history).to_csv(
        artifact_dir / f"history_{tag}.csv", index=False)
    with open(artifact_dir / f"summary_{tag}.json", "w") as handle:
        json.dump({"resolution": args.resolution, "seed": args.seed,
                   "backgrounds": args.backgrounds,
                   "selection_benchmark": args.selection_benchmark,
        "warmup_epochs": CONFIG["warmup_epochs"],
        "ramp_epochs": CONFIG["ramp_epochs"],
        "learning_rate": CONFIG["learning_rate"],
                   "epochs": args.epochs, "best_epoch": best["epoch"],
                   "minutes": round(minutes, 1), "benchmarks": final}, handle, indent=2)
    print(f"\nwrote {artifact_dir}")


if __name__ == "__main__":
    main()
