"""Paired comparison of the catalogue-only and background-augmented encoders.

Arms C and D of the Task 4 2x2 - the same architecture, split, recipe, seed and
learning rate, differing in one thing: whether any training frame ever left the
white catalogue background. Arm C is ``--backgrounds none``, arm D is the
deployed ``mixed`` encoder.

Both are scored on the query frames ``build_queries(seed=123)`` produces, which
is the same call the training script makes, so the comparison is paired per
query and ``RetrievalProtocol.compare`` can put a bootstrap interval on each
difference rather than leaving two column of numbers side by side.

    python scripts/compare_task4_background_arms.py

Writes ``outputs/evaluation/task4_background_arms.csv`` (the scores) and
``..._significance.csv`` (the paired differences with 95% intervals).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import CatalogueIndex, RetrievalProtocol  # noqa: E402
from src.training.train_task4_120x160 import (  # noqa: E402
    build_queries,
    embed,
    load_gallery,
)
from src.visual_search.search_engine import build_encoder  # noqa: E402

BENCHMARKS = ("clean", "hard", "photo", "wild", "wildphoto")
REPORTED = ("P@1", "P@10", "colour@10", "colourfam@10", "both@10", "bothfam@10")
METRICS = ("both@10", "type@10", "colour@10")


def score_encoder(path, images, protocol, queries, device):
    """Every benchmark for one checkpoint, keeping the per-query vectors."""
    checkpoint = torch.load(path, map_location=device)
    model = build_encoder(checkpoint).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    mean = np.asarray(checkpoint["channel_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["channel_std"], dtype=np.float32)

    catalogue = embed(model, images, mean, std, device,
                      positions=protocol.catalogue_pos)
    full = np.zeros((len(protocol.gallery), catalogue.shape[1]), dtype=np.float32)
    full[protocol.catalogue_pos] = catalogue

    summaries = {}
    for name in BENCHMARKS:
        full[protocol.heldout_queries] = embed(model, queries[name], mean, std, device)
        index = CatalogueIndex(full, protocol, name=name)
        summaries[name], _ = protocol.evaluate_deployment(index, full, store=False)
    return summaries, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", default="120x160")
    parser.add_argument("--query-seed", type=int, default=123)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()

    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = PROJECT_ROOT / "artifacts" / f"task4_{args.resolution}"
    arms = {"C_encoder_clean": artifacts / "task4_encoder_none_seed42.pt",
            "D_encoder_bgaug": artifacts / "task4_encoder_mixed_seed42.pt"}
    for label, path in arms.items():
        if not path.exists():
            sys.exit(f"{label}: {path} is missing - train that arm first")

    images, masks, gallery = load_gallery(args.resolution)
    protocol = RetrievalProtocol(gallery=gallery)
    queries = build_queries(images, masks, protocol, seed=args.query_seed)

    scored = {}
    for label, path in arms.items():
        print(f"scoring {label} ...")
        summaries, checkpoint = score_encoder(path, images, protocol, queries, device)
        scored[label] = summaries
        print("  trained with backgrounds={} augmented={}".format(
            checkpoint.get("background_source"),
            checkpoint.get("background_augmented")))

    rows = [{"arm": label, "benchmark": name,
             **{m: round(float(summary[m]) * 100, 2) for m in REPORTED}}
            for label, summaries in scored.items()
            for name, summary in summaries.items()]
    table = pd.DataFrame(rows)

    # D minus C: what the background augmentation bought, per benchmark, paired
    # on the query. A positive difference means the augmented arm is ahead.
    differences = []
    for name in BENCHMARKS:
        for metric in METRICS:
            # paired_bootstrap returns b - a, so C goes first for D minus C.
            stats = protocol.compare(scored["C_encoder_clean"][name],
                                     scored["D_encoder_bgaug"][name],
                                     metric=metric, n_resamples=args.resamples)
            differences.append({
                "benchmark": name, "metric": metric,
                "D_bgaug": round(float(scored["D_encoder_bgaug"][name][metric]) * 100, 2),
                "C_clean": round(float(scored["C_encoder_clean"][name][metric]) * 100, 2),
                "delta": round(stats["delta"] * 100, 2),
                "ci_low": round(stats["ci_low"] * 100, 2),
                "ci_high": round(stats["ci_high"] * 100, 2),
                "p_value": round(stats["p_value"], 4),
                "significant": stats["significant"],
                "n_queries": stats["n_queries"],
            })
    significance = pd.DataFrame(differences)

    out_dir = PROJECT_ROOT / "outputs" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "task4_background_arms.csv", index=False)
    significance.to_csv(out_dir / "task4_background_arms_significance.csv", index=False)

    print("\nP@10 and both@10 by arm\n")
    print(table.pivot_table(index="benchmark", columns="arm",
                            values=["P@10", "both@10"]).reindex(BENCHMARKS).to_string())
    print("\nD (augmented) minus C (catalogue only), paired, 95% interval\n")
    print(significance[significance.metric == "both@10"].to_string(index=False))
    print("\nwrote {} in {:.1f} min".format(out_dir, (time.time() - started) / 60))


if __name__ == "__main__":
    main()
