#!/usr/bin/env python
"""Train the Task 4 retrieval encoder, at 60x80 or at 120x160.

Why a script and not a notebook cell
------------------------------------
At 120x160 a run is roughly 10 hours on a machine with no CUDA device, which is
longer than a notebook kernel should be trusted to stay alive. Every epoch is
checkpointed, so ``--resume`` picks the run back up.

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

Usage
-----
    python -m src.training.train_task4_120x160 --resolution 60x80   --seed 42
    python -m src.training.train_task4_120x160 --resolution 120x160 --seed 42
    python -m src.training.train_task4_120x160 --resolution 120x160 --resume
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
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batches_per_epoch": 250,
    "p": 16, "k": 8,
    "bg_end": 0.6,
    "ramp_epochs": 4,
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
}

#: One point out of domain is worth three in domain, because the product serves
#: photographs. Shared with notebook 06 so selection and promotion cannot drift.
DEPLOYMENT_WEIGHT = 3.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", choices=("60x80", "120x160"), default="120x160")
    parser.add_argument("--epochs", type=int, default=30,
                        help="30 is the from-scratch recipe notebook 05 used for "
                             "the deployed encoder; 12 is the warm-started one")
    parser.add_argument("--warm-start", type=Path, default=None,
                        help="initialise from a checkpoint instead of random. "
                             "Cheaper, but see the note in the module docstring")
    parser.add_argument("--seed", type=int, default=42)
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


def build_queries(images, masks, protocol, resolution, seed=123):
    """Clean, hard and wild renderings of the held-out items.

    ``wild`` is ``hard`` plus the held-out camera degradations and the ingestion
    path, so the three differ by exactly one stage each and a movement can be
    attributed.
    """
    height, width = images.shape[1], images.shape[2]
    evaluation_backgrounds = make_eval_backgrounds(600, size=(width, height))
    generator = np.random.default_rng(seed)

    clean = np.stack([np.asarray(images[p]) for p in protocol.heldout_queries])
    # 2,000 frames is ~115 MB at 120x160 - materialising this one is affordable,
    # unlike the 32,944-row catalogue.
    hard = np.stack([
        composite(np.asarray(images[p]), np.asarray(masks[p]),
                  evaluation_backgrounds[generator.integers(len(evaluation_backgrounds))],
                  generator)
        for p in protocol.heldout_queries
    ])
    wild = np.stack([
        simulate_ingestion(degrade(frame, generator, EVAL_DEGRADATIONS), generator)
        for frame in hard
    ])
    return {"clean": clean, "hard": hard, "wild": wild}


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = PROJECT_ROOT / "artifacts" / f"task4_{args.resolution}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    state_path = artifact_dir / f"train_state_seed{args.seed}.pt"

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
    backgrounds = make_backgrounds(600, shape=(height, width, 3), seed=42,
                                   source_images=images[
                                       np.sort(np.random.default_rng(0).choice(
                                           protocol.catalogue_pos, 400, replace=False))])

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

    queries = build_queries(images, masks, protocol, args.resolution)
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

    for epoch in range(start_epoch, args.epochs):
        ramp = min(1.0, (epoch + 1) / CONFIG["ramp_epochs"])
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
        if spread < CONFIG["collapse_fraction"] * peak_spread:
            print(f"COLLAPSE at epoch {epoch + 1}: spread {spread:.4f} against a "
                  f"peak of {peak_spread:.4f}. Stopping.")
            break

        if (epoch + 1) % CONFIG["eval_every"] == 0 or epoch == args.epochs - 1:
            scores = monitor(model, images, protocol, queries, mean, std, device,
                             monitor_subset, CONFIG["monitor_query_stride"])
            record.update({f"{b}_{m}": v for b, s in scores.items()
                           for m, v in s.items()})
            weighted = (DEPLOYMENT_WEIGHT * scores["wild"]["both@10"]
                        + scores["clean"]["both@10"])
            if weighted > best["score"]:
                best = {"score": weighted,
                        "state": copy.deepcopy(model.state_dict()),
                        "epoch": epoch + 1}
            print(f"epoch {epoch + 1:>2}/{args.epochs} triplet {record['triplet']:.4f} "
                  f"spread {spread:.3f} | monitor clean P@10 "
                  f"{scores['clean']['P@10']:.2f} wild P@10 {scores['wild']['P@10']:.2f}"
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
        "seed": args.seed, "best_epoch": best["epoch"],
        "gallery": f"task4_gallery_{args.resolution}.csv",
        "trained_by": "src/training/train_task4_120x160.py",
    }, artifact_dir / f"task4_encoder_seed{args.seed}.pt")

    pd.DataFrame(history).to_csv(
        artifact_dir / f"history_seed{args.seed}.csv", index=False)
    with open(artifact_dir / f"summary_seed{args.seed}.json", "w") as handle:
        json.dump({"resolution": args.resolution, "seed": args.seed,
                   "epochs": args.epochs, "best_epoch": best["epoch"],
                   "minutes": round(minutes, 1), "benchmarks": final}, handle, indent=2)
    print(f"\nwrote {artifact_dir}")


if __name__ == "__main__":
    main()
