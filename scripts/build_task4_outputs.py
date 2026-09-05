#!/usr/bin/env python
"""Produce Task 4's own deliverable over the whole test set.

Why this exists
---------------
Task 4 contributes nothing to ``styles_prediction.csv`` - the graded submission
is classification only, and the spec defines no format for retrieval. That is
exactly why Task 4's evidence has to be produced deliberately: nothing else in
the pipeline forces it.

Until now the only retrieval output over ``images_test`` was a 16-image sample
plotted inside a notebook, which is not something a marker can inspect or a
reader can check. This walks all 5,829 test images and writes:

    outputs/task4_test_retrieval.csv    top-K neighbours per image, with the
                                        confidence signals the API now returns
    outputs/task4_test_clusters.csv     one row per image: nearest cluster, its
                                        dominant articleType, and the margin
    outputs/task4_test_summary.json     counts, confidence rates, timings

The clustering pass is skipped with a warning rather than failing when the
cluster artefacts are missing or stale, because retrieval is the deliverable and
clustering is the extra.

``--rebuild-index`` regenerates the two search indexes from the encoder the
manifest names. It exists because nothing did: the manifest claims this script
built ``search_index_full.npy``, but both indexes were in fact produced ad hoc
while moving Task 4 from serving the evaluation split to serving the whole
catalogue. An encoder that cannot be turned into a deployable index by running a
command cannot be honestly promoted.

Usage
-----
    python scripts/build_task4_outputs.py
    python scripts/build_task4_outputs.py --k 20 --limit 200 --mode letterbox
    python scripts/build_task4_outputs.py --rebuild-index          # after retraining
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import RetrievalProtocol  # noqa: E402
from src.visual_search.search_engine import SearchEngine  # noqa: E402

TEST_DIR = PROJECT_ROOT / "A2_FashionDataset" / "FashionDataset" / "test" / "images_test"
TRAIN_DIR = PROJECT_ROOT / "A2_FashionDataset" / "FashionDataset" / "train" / "images_train"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "task4"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=TEST_DIR,
                        help="directory of test images (default: images_test)")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR,
                        help="directory to write the CSVs into")
    parser.add_argument("--k", type=int, default=10, help="neighbours per image")
    parser.add_argument("--mode", default="nobg",
                        help="ingestion mode: letterbox, crop or nobg")
    parser.add_argument("--batch", type=int, default=256,
                        help="images per search call")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N images (for a smoke test)")
    parser.add_argument("--no-clusters", action="store_true",
                        help="skip the cluster assignment pass")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="re-embed the catalogue and rewrite both search "
                             "indexes from the encoder the manifest names, then exit")
    parser.add_argument("--catalogue-images", type=Path, default=TRAIN_DIR,
                        help="directory of catalogue images for --rebuild-index")
    return parser.parse_args()


def rebuild_index(engine, image_dir, artifact_dir=ARTIFACT_DIR, batch_size=256):
    """Re-embed every catalogue photograph and rewrite both index files.

    Two indexes, and conflating them would misreport the system:

    ``search_index_full.npy``
        every item the shop stocks, because a customer must be able to find
        anything in the catalogue. This is what the API serves.
    ``search_index_eval.npy``
        the same embeddings minus the 15% product-level holdout, so that a query
        drawn from the holdout cannot retrieve itself. Every published metric is
        measured against this one.

    The split is not stored anywhere - it is regenerated from
    ``RetrievalProtocol``, which is seeded, so the same gallery always yields the
    same holdout and the evaluation index stays comparable across rebuilds.
    """
    metadata_path = artifact_dir / "gallery_metadata.csv"
    if not metadata_path.exists():
        sys.exit("Cannot rebuild without {}".format(metadata_path))
    gallery = pd.read_csv(metadata_path)
    print("Gallery: {:,} rows".format(len(gallery)))

    height, width = engine.size[1], engine.size[0]
    ids = gallery["id"].to_numpy()
    missing = [i for i in ids[:64] if not (image_dir / "{}.jpg".format(i)).exists()]
    if missing:
        sys.exit("Catalogue images not found in {} (e.g. {})".format(image_dir, missing[:3]))

    vectors = []
    started = time.perf_counter()
    for start in range(0, len(ids), batch_size):
        chunk = ids[start:start + batch_size]
        # A plain resize, not the upload ingestion path: these are the frames the
        # encoder was trained on, and letterboxing would pad the handful of
        # square catalogue images it learnt stretched.
        arrays = np.stack([
            np.asarray(Image.open(image_dir / "{}.jpg".format(i)).convert("RGB")
                       .resize((width, height)))
            for i in chunk
        ])
        vectors.append(engine.embed_arrays(arrays))
        print("  {:>6,}/{:,}".format(min(start + batch_size, len(ids)), len(ids)),
              flush=True)
    full = np.vstack(vectors).astype(np.float32)
    print("Embedded {:,} items in {:.1f} min".format(
        len(full), (time.perf_counter() - started) / 60))

    catalogue_pos = RetrievalProtocol(gallery=gallery).catalogue_pos
    evaluation = full[catalogue_pos]

    # Report drift rather than overwriting in silence: a rebuild that quietly
    # changes what the published metrics were measured on is how a stale number
    # ends up on the model card.
    previous_path = artifact_dir / "search_index_full.npy"
    if previous_path.exists():
        previous = np.load(previous_path)
        if previous.shape == full.shape:
            cosine = (previous * full).sum(1)
            print("Agreement with the existing index: mean {:.6f}, min {:.6f}".format(
                float(cosine.mean()), float(cosine.min())))
        else:
            print("Existing index has shape {} against {} - replacing".format(
                previous.shape, full.shape))

    np.save(previous_path, full)
    np.save(artifact_dir / "search_index_eval.npy", evaluation)
    np.save(artifact_dir / "gallery_ids.npy", ids)
    print("Wrote search_index_full.npy {} and search_index_eval.npy {}".format(
        full.shape, evaluation.shape))

    manifest_path = artifact_dir / "search_manifest.json"
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    manifest["catalogue_size"] = int(len(full))
    manifest["evaluation_catalogue_size"] = int(len(evaluation))
    manifest.setdefault("provenance", {})["deployment_index_built_by"] = (
        "scripts/build_task4_outputs.py --rebuild-index")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    print("Manifest sizes updated: {:,} served, {:,} scored".format(
        len(full), len(evaluation)))


def main():
    args = parse_args()

    if args.rebuild_index:
        engine = SearchEngine.load()
        print("Encoder: {} | {}".format(
            engine.manifest.get("encoder_file", "unknown"), engine.device))
        rebuild_index(engine, args.catalogue_images, batch_size=args.batch)
        return

    if not args.images.is_dir():
        sys.exit("Test image directory not found: {}".format(args.images))
    paths = sorted(args.images.glob("*.jpg"), key=lambda p: int(p.stem)
                   if p.stem.isdigit() else 0)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        sys.exit("No .jpg images found in {}".format(args.images))
    print("Test images: {:,}".format(len(paths)))

    engine = SearchEngine.load()
    print("Engine: {} | {:,} catalogue items | {}-d | {}".format(
        engine.manifest.get("best_method", "unknown"), len(engine.index),
        engine.index.shape[1], engine.device))

    args.out.mkdir(parents=True, exist_ok=True)

    # -- retrieval -----------------------------------------------------
    frames = []
    started = time.perf_counter()
    for start in range(0, len(paths), args.batch):
        chunk = paths[start:start + args.batch]
        frames.append(engine.search(chunk, k=args.k, mode=args.mode,
                                    with_diagnostics=True))
        done = min(start + args.batch, len(paths))
        print("  {:>5}/{:,}".format(done, len(paths)), flush=True)
    retrieval = pd.concat(frames, ignore_index=True)
    elapsed = time.perf_counter() - started

    retrieval.insert(1, "test_id", retrieval["query"].str.replace(r"\.jpg$", "",
                                                                  regex=True))
    retrieval_path = args.out / "task4_test_retrieval.csv"
    retrieval.to_csv(retrieval_path, index=False)
    print("\nWrote {} ({:,} rows)".format(retrieval_path.name, len(retrieval)))

    per_image = retrieval[retrieval["rank"] == 1]
    if len(per_image) != len(paths):
        raise AssertionError(
            "expected one rank-1 row per image, got {} for {} images".format(
                len(per_image), len(paths)))

    summary = {
        "images": len(paths),
        "k": args.k,
        "mode": args.mode,
        "method": engine.manifest.get("best_method"),
        "catalogue_size": int(len(engine.index)),
        "seconds": round(elapsed, 1),
        "ms_per_image": round(elapsed / len(paths) * 1000, 2),
        "confident_share": round(float(per_image["confident"].mean()), 4),
        "mean_top1_similarity": round(float(per_image["top1_similarity"].mean()), 4),
        "mean_coherence": round(float(per_image["coherence"].mean()), 4),
        "ingest_fell_back_share": round(float(per_image["ingest_fell_back"].mean()), 4),
        "top_predicted_types": per_image["articleType"].value_counts().head(10).to_dict(),
    }
    print("  confident: {:.1%} | mean top-1 similarity {:.3f} | {:.1f} ms/image".format(
        summary["confident_share"], summary["mean_top1_similarity"],
        summary["ms_per_image"]))

    # -- clustering ----------------------------------------------------
    if not args.no_clusters:
        try:
            from src.visual_search.cluster_engine import ClusterEngine

            clusters = ClusterEngine.load()
            rows = []
            for path in paths:
                prediction = clusters.predict(path, mode=args.mode)
                best = prediction.get("best", {})
                rows.append({
                    "test_id": path.stem,
                    "cluster": best.get("cluster"),
                    "dominant_type": best.get("dominant_type"),
                    "purity": best.get("purity"),
                    "distance": best.get("distance"),
                    # Margin to the runner-up cluster. Small means the item sits
                    # between two clusters and the assignment is a coin toss -
                    # web photos average 0.055 against 0.160 for catalogue shots.
                    "margin": prediction.get("margin"),
                    "confident": prediction.get("confident"),
                })
            cluster_df = pd.DataFrame(rows)
            cluster_path = args.out / "task4_test_clusters.csv"
            cluster_df.to_csv(cluster_path, index=False)
            print("Wrote {} ({:,} rows)".format(cluster_path.name, len(cluster_df)))
            summary["cluster_confident_share"] = round(
                float(cluster_df["confident"].mean()), 4)
        except Exception as error:                       # noqa: BLE001
            # Clustering is the extra, not the deliverable. A stale centroid file
            # must not cost us the retrieval output.
            print("Skipped clustering: {}: {}".format(type(error).__name__, error))
            summary["cluster_error"] = "{}: {}".format(type(error).__name__, error)

    summary_path = args.out / "task4_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("Wrote {}".format(summary_path.name))


if __name__ == "__main__":
    main()
