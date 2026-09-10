"""Paired comparison of the Task 4 background arms.

One architecture, three training distributions - the same split, recipe, seed
and learning rate throughout, differing only in what the network was shown:

    C  ``--backgrounds none``        the catalogue exactly as it ships
    P  ``--backgrounds procedural``  composited onto formulae only
    D  ``--backgrounds mixed``       composited onto 70% Places365 scenes

C and D are required. **P is optional**, because the project keeps one encoder
and its checkpoint is routinely deleted after measuring; when it is absent the
script scores the other two and says so, rather than failing.

Every arm is scored on the query frames ``build_queries(seed=123)`` produces,
which is the same call the training script makes, so the comparison is paired
per query and ``RetrievalProtocol.compare`` can put a bootstrap interval on each
difference rather than leaving columns of numbers side by side. Each augmented
arm is contrasted against C, the control, and the ``contrast`` column of the
significance table says which pair a row describes.

    python scripts/compare_task4_background_arms.py

Writes ``outputs/evaluation/task4_background_arms.csv`` (the scores) and
``..._significance.csv`` (the paired differences with 95% intervals).
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import product
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

CONTROL = "C_encoder_clean"
# label -> (checkpoint filename, required). Arm P is optional: see the module
# docstring. Ordered control first so the printed table reads C, P, D.
ARMS = {
    "C_encoder_clean": ("task4_encoder_none_seed42.pt", True),
    "P_encoder_procedural": ("task4_encoder_procedural_seed42.pt", False),
    "D_encoder_bgaug": ("task4_encoder_mixed_seed42.pt", True),
}


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
    arms = {}
    for label, (filename, required) in ARMS.items():
        path = artifacts / filename
        if path.exists():
            arms[label] = path
        elif required:
            sys.exit(f"{label}: {path} is missing - train that arm first")
        else:
            print(f"{label}: {filename} not on disk, skipping this arm")

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

    # Each augmented arm minus the control, per benchmark, paired on the query.
    # A positive difference means the augmented arm is ahead. `contrast` names
    # the pair, so a consumer must say which comparison it wants rather than
    # assuming the file holds only one.
    differences = []
    augmented = [label for label in arms if label != CONTROL]
    for arm, name, metric in product(augmented, BENCHMARKS, METRICS):
        contrast = "%s_minus_%s" % (arm.split("_")[0], CONTROL.split("_")[0])
        # paired_bootstrap returns b - a, so the control goes first.
        stats = protocol.compare(scored[CONTROL][name], scored[arm][name],
                                 metric=metric, n_resamples=args.resamples)
        differences.append({
            "contrast": contrast, "arm": arm, "control": CONTROL,
            "benchmark": name, "metric": metric,
            "arm_score": round(float(scored[arm][name][metric]) * 100, 2),
            "control_score": round(float(scored[CONTROL][name][metric]) * 100, 2),
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
    print("\nEach augmented arm minus C (catalogue only), paired, 95% interval\n")
    print(significance[significance.metric == "both@10"].to_string(index=False))
    print("\nwrote {} in {:.1f} min".format(out_dir, (time.time() - started) / 60))


if __name__ == "__main__":
    main()
