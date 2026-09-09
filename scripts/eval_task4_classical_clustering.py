"""k-Means over hand-built descriptors: the unsupervised arm of the Task 4 2x2.

Why this exists
---------------
Notebook 06 clusters the *encoder's* embeddings, which makes it an index layer
over the encoder rather than a second model. Comparing the two answers only
"how much does approximation cost", because the clustering arm cannot be better
than the thing it approximates. To ask whether the learned representation is
what earns the result, the clustering has to run on features the encoder had no
hand in - so this scores k-Means over ``classical_features`` (128-bin HSV
histogram + 108-bin gradient histogram, 236 dims).

That descriptor was measured and retired as a *retrieval* method. It comes back
here in a narrower role, as the substrate for the unsupervised arm, and its
retrieval numbers are not re-entered in notebook 05 section 5.

The 2x2
-------
The data question is whether the catalogue's uniformity - every one of the
38,571 frames is a centred garment on white - is what limits retrieval on a real
upload, and whether background augmentation fixes it. Crossed with model family:

===================  =========================  ==========================
                     catalogue only             + background augmentation
===================  =========================  ==========================
k-Means / classical  arm A (this script)        arm B (this script)
triplet encoder      arm C (``--backgrounds     arm D (the deployed
                     none``)                    ``mixed`` encoder)
===================  =========================  ==========================

What "augmented" means for a descriptor that is not trained
-----------------------------------------------------------
There are no weights to fit, so the intervention cannot be "train on augmented
data". The equivalent is to build each catalogue item's descriptor from several
renderings of it - the clean frame plus ``--views`` composites onto the same
mixed backdrop bank the encoder trained against - and average them. Averaging
suppresses the components that move with the backdrop and keeps the ones that do
not, which is the same invariance augmentation buys a network, obtained without
learning anything.

The averaging is applied to the GALLERY only, never to the query. That asymmetry
is deliberate and it is the deployment shape: the catalogue images and their
masks are ours to re-render offline, while a user's upload arrives once, already
against whatever backdrop it has.

Both arms are scored two ways - exact full-scan over the descriptor, and the
k-Means routed search - so a loss can be attributed to the representation rather
than to the clustering, or the other way round.

Usage
-----
    python scripts/eval_task4_classical_clustering.py
    python scripts/eval_task4_classical_clustering.py --k 120 --views 3

Queries come from ``build_queries(seed=123)``, the same call the training script
makes, so these rows are scored on byte-identical frames to arms C and D and the
four are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.cluster import KMeans  # noqa: E402

from src.data.places365_backgrounds import (  # noqa: E402
    load_background_bank,
    make_mixed_bank,
)
from src.data.synthetic_backgrounds import composite, make_backgrounds  # noqa: E402
from src.evaluation.metrics import CatalogueIndex, RetrievalProtocol  # noqa: E402
from src.features.image_features import classical_features  # noqa: E402
from src.training.train_task4_120x160 import (  # noqa: E402
    CONFIG,
    build_queries,
    load_gallery,
)

BENCHMARKS = ("clean", "hard", "photo", "wild", "wildphoto")
REPORTED = ("P@1", "P@10", "colour@10", "colourfam@10", "both@10", "bothfam@10")


class ClusteredIndex(CatalogueIndex):
    """Search restricted to the nearest ``probes`` k-Means clusters.

    Same contract as ``CatalogueIndex`` - it takes gallery-shaped embeddings and
    returns gallery positions - so ``RetrievalProtocol.evaluate_deployment``
    cannot tell the two apart and scores them identically.
    """

    def __init__(self, embeddings, protocol, k=120, probes=3, seed=42,
                 name="clustered", catalogue=None, device=None):
        super().__init__(embeddings, protocol, name=name, catalogue=catalogue,
                         device=device)
        model = KMeans(n_clusters=k, n_init=10, random_state=seed)
        self.labels = model.fit_predict(self.vectors)
        centroids = model.cluster_centers_.astype(np.float32)
        # Cosine routing, to match the ranking metric. An unnormalised centroid
        # would route by Euclidean distance and rank by cosine, which is two
        # different notions of "near" in one query.
        self.centroids = centroids / np.clip(
            np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8, None)
        self.members = [np.flatnonzero(self.labels == c) for c in range(k)]
        self.probes = probes
        self.inertia = float(model.inertia_)

    def search(self, query_vectors, k=10, exclude=None, batch_size=512):
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        queries = queries / np.clip(
            np.linalg.norm(queries, axis=1, keepdims=True), 1e-8, None)

        probed = np.argsort(-(queries @ self.centroids.T), axis=1)[:, :self.probes]
        scores = np.zeros((len(queries), k), dtype=np.float32)
        positions = np.zeros((len(queries), k), dtype=np.int64)

        for row, (query, clusters) in enumerate(zip(queries, probed)):
            local = np.concatenate([self.members[c] for c in clusters])
            if len(local) < k:
                # A query whose probed clusters cannot fill the top-k. Padding
                # with the global nearest would hide the failure, so the row is
                # short and the missing slots score as misses, which is what a
                # router that cannot answer actually costs.
                local = np.arange(len(self.vectors))
            similarity = self.vectors[local] @ query
            top = np.argpartition(-similarity, k - 1)[:k]
            top = top[np.argsort(-similarity[top])]
            scores[row] = similarity[top]
            positions[row] = local[top]

        return scores, self.catalogue[positions]

    def touched(self):
        """Mean fraction of the catalogue one query scans. The point of a router."""
        sizes = np.array([len(m) for m in self.members], dtype=np.float64)
        return float(np.sort(sizes)[::-1][:self.probes].sum() / len(self.vectors))


def descriptors(frames, batch_size=512):
    """``classical_features`` over a frame array, chunked. Expects uint8."""
    out = []
    for start in range(0, len(frames), batch_size):
        out.append(classical_features(np.asarray(frames[start:start + batch_size],
                                                 dtype=np.uint8)))
    return np.vstack(out)


def catalogue_descriptors(images, masks, positions, bank, views, seed,
                          batch_size=256):
    """Descriptors for the catalogue, optionally averaged over rendered views.

    ``views=0`` is arm A: the clean frame alone. Anything higher is arm B, which
    adds that many composites onto ``bank`` and averages. The images are read a
    chunk at a time - the 120x160 catalogue is 32,944 x 160 x 120 x 3, and
    materialising four views of it at once is 7.6 GB.
    """
    generator = np.random.default_rng(seed)
    out = np.zeros((len(positions), 236), dtype=np.float32)

    for start in range(0, len(positions), batch_size):
        rows = positions[start:start + batch_size]
        clean = np.asarray(images[rows], dtype=np.uint8)
        total = descriptors(clean)

        for _ in range(views):
            rendered = np.stack([
                composite(frame, np.asarray(masks[row]),
                          bank[generator.integers(len(bank))], generator,
                          scale_range=CONFIG["scale_range"])
                for frame, row in zip(clean, rows)
            ])
            total = total + descriptors(rendered)

        out[start:start + len(rows)] = total / (views + 1)

    return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-8, None)


def score(catalogue_features, query_features, protocol, k, probes, seed):
    """Both index shapes, on every benchmark, through the shared protocol."""
    dim = catalogue_features.shape[1]
    full = np.zeros((len(protocol.gallery), dim), dtype=np.float32)
    full[protocol.catalogue_pos] = catalogue_features

    exact = CatalogueIndex(full, protocol, name="exact")
    clustered = ClusteredIndex(full, protocol, k=k, probes=probes, seed=seed)
    print("    k-Means: {} clusters, inertia {:.1f}, a query scans "
          "<= {:.1%} of the catalogue".format(k, clustered.inertia,
                                              clustered.touched()))

    results = {}
    for name in BENCHMARKS:
        full[protocol.heldout_queries] = query_features[name]
        for shape, index in (("exact", exact), ("kmeans", clustered)):
            summary, _ = protocol.evaluate_deployment(index, full, store=False)
            results[(name, shape)] = {
                metric: float(summary[metric]) * 100 for metric in REPORTED}
            results[(name, shape)]["_per_query_both"] = summary["_per_query_both"]
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", default="120x160",
                        choices=("60x80", "120x160"))
    parser.add_argument("--k", type=int, default=120,
                        help="clusters, matching notebook 06 so the two are readable "
                             "against each other")
    parser.add_argument("--probes", type=int, default=3,
                        help="clusters searched per query, matching notebook 06")
    parser.add_argument("--views", type=int, default=3,
                        help="rendered backdrops averaged into arm B's gallery "
                             "descriptor, on top of the clean frame")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-seed", type=int, default=123,
                        help="must match the training script, or these arms are "
                             "scored on different frames to the encoders")
    args = parser.parse_args()

    started = time.time()
    images, masks, gallery = load_gallery(args.resolution)
    protocol = RetrievalProtocol(gallery=gallery)
    print("{}: {:,} items | catalogue {:,} | held-out queries {:,}".format(
        args.resolution, len(gallery), len(protocol.catalogue_pos),
        len(protocol.heldout_queries)))

    queries = build_queries(images, masks, protocol, seed=args.query_seed)
    print("query descriptors ...")
    query_features = {name: descriptors(frames) for name, frames in queries.items()}

    height, width = images.shape[1], images.shape[2]
    procedural = make_backgrounds(
        CONFIG["procedural_bank"], shape=(height, width, 3), seed=42,
        source_images=images[np.sort(np.random.default_rng(0).choice(
            protocol.catalogue_pos, 400, replace=False))])
    photographic = load_background_bank(
        count=CONFIG["photographic_bank"], shape=(height, width, 3),
        split="train", seed=args.seed)
    bank = make_mixed_bank(procedural, photographic,
                           CONFIG["photographic_share"], seed=args.seed)
    print("backdrop bank for arm B: {:,} frames, {:.0%} photographic".format(
        len(bank), CONFIG["photographic_share"]))

    arms = {}
    for label, views in (("A_classical_clean", 0), ("B_classical_bgaug", args.views)):
        print("\n{} - {} view{} per catalogue item".format(
            label, views + 1, "" if views == 0 else "s"))
        features = catalogue_descriptors(images, masks, protocol.catalogue_pos,
                                         bank, views, args.seed)
        arms[label] = score(features, query_features, protocol, args.k,
                            args.probes, args.seed)

    rows = []
    for label, results in arms.items():
        for (benchmark, shape), scores in results.items():
            rows.append({"arm": label, "benchmark": benchmark, "index": shape,
                         **{m: round(scores[m], 2) for m in REPORTED}})
    table = pd.DataFrame(rows)

    out_dir = PROJECT_ROOT / "outputs" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    # The view count is arm B's only knob, so it belongs in the filename - a
    # fixed name silently overwrites the previous setting's results.
    stem = f"task4_classical_clustering_arms_v{args.views}"
    table.to_csv(out_dir / f"{stem}.csv", index=False)

    print("\nP@10 by arm and index shape\n")
    print(table.pivot_table(index="benchmark", columns=["arm", "index"],
                            values="P@10").reindex(BENCHMARKS).to_string())

    with open(out_dir / f"{stem}.json", "w") as handle:
        json.dump({"resolution": args.resolution, "k": args.k,
                   "probes": args.probes, "views": args.views,
                   "seed": args.seed, "query_seed": args.query_seed,
                   "minutes": round((time.time() - started) / 60, 1),
                   "rows": rows}, handle, indent=2)
    print("\nwrote {} in {:.1f} min".format(out_dir, (time.time() - started) / 60))


if __name__ == "__main__":
    main()
