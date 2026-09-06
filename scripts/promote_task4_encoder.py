#!/usr/bin/env python
"""Promote a trained Task 4 encoder to the served artefacts.

The encoder trained by ``src/training/train_task4_120x160.py`` uses a different
gallery from the one previously served: 38,571 rows flagged ``use_for_supervised``
rather than 38,612, and a product holdout drawn from that reduced set. So the
index, the metadata and the ids all have to be rebuilt together - swapping only
the checkpoint would leave ``SearchEngine.load`` raising on a row-count mismatch,
which is the check that exists precisely to catch this.

    python scripts/promote_task4_encoder.py \
        --encoder artifacts/task4_120x160/task4_encoder_mixed_seed42.pt

Writes ``search_index_full.npy`` (every catalogue row, because a shopper must be
able to find anything the shop stocks), ``search_index_eval.npy`` (the same minus
the held-out products, which is what every reported metric is measured against),
``gallery_metadata.csv``, ``gallery_ids.npy`` and ``search_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import RetrievalProtocol  # noqa: E402
from src.visual_search.search_engine import build_encoder  # noqa: E402

PROCESSED = PROJECT_ROOT / "A2_FashionDataset" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "task4"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None,
                        help="summary_*.json from the same run; defaults to the "
                             "sibling file matching the checkpoint's tag")
    parser.add_argument("--batch", type=int, default=256)
    return parser.parse_args()


@torch.no_grad()
def embed_all(encoder, images, mean, std, device, batch_size=256):
    """Every row, with the mirror-average TTA the manifest advertises."""
    mean_t = torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    out = []
    for start in range(0, len(images), batch_size):
        chunk = np.asarray(images[start:start + batch_size], dtype=np.float32) / 255.0
        tensor = torch.from_numpy(chunk.transpose(0, 3, 1, 2))
        tensor = ((tensor - mean_t) / std_t).to(device)
        vectors = encoder.embed(tensor)
        vectors = F.normalize(vectors + encoder.embed(torch.flip(tensor, dims=[3])),
                              p=2, dim=1)
        out.append(vectors.float().cpu().numpy())
        if (start // batch_size) % 20 == 0:
            print(f"  {min(start + batch_size, len(images)):>6,}/{len(images):,}",
                  flush=True)
    return np.vstack(out)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.encoder, map_location=device, weights_only=False)
    resolution = checkpoint["resolution"]
    print(f"Encoder: {args.encoder.name} | {resolution} | "
          f"backgrounds={checkpoint.get('background_source')}")

    summary_path = args.summary or (
        args.encoder.parent
        / args.encoder.name.replace("task4_encoder_", "summary_").replace(".pt", ".json"))
    summary = json.loads(Path(summary_path).read_text()) if Path(summary_path).exists() else {}

    # Two galleries, and conflating them makes 41 products unfindable.
    #
    # The TRAINING gallery is task4_gallery_*.csv: 38,571 rows, because the
    # pipeline drops 41 images whose duplicates carry conflicting labels. That
    # exclusion is right for training and for measurement - a row with an
    # unreliable label cannot referee anything.
    #
    # The SERVED catalogue is every row the shop stocks, all 38,612. A conflicting
    # gender label is no reason a customer cannot find the product. Serving the
    # training gallery instead leaves those 41 items unreachable by search, which
    # `test_the_served_index_covers_the_whole_catalogue` exists to catch.
    train_gallery = pd.read_csv(PROCESSED / f"task4_gallery_{resolution}.csv")
    train_images = np.load(PROCESSED / f"task4_cache_{resolution}.npy", mmap_mode="r")
    if len(train_gallery) != len(train_images):
        sys.exit(f"gallery {len(train_gallery)} != cache {len(train_images)}")

    columns = list(train_gallery.columns)
    served = pd.read_csv(PROCESSED / "clean_train_metadata.csv")[columns]
    served = served.sort_values("id").reset_index(drop=True)
    image_dir = PROCESSED / f"images_train_{resolution}"
    missing = [i for i in served["id"] if not (image_dir / f"{i}.jpg").exists()]
    if missing:
        sys.exit(f"{len(missing)} served ids have no {resolution} image, e.g. {missing[:3]}")
    print(f"Training gallery: {len(train_gallery):,} rows | "
          f"served catalogue: {len(served):,} rows "
          f"(+{len(served) - len(train_gallery)} the training split drops)")

    encoder = build_encoder(checkpoint)
    encoder.load_state_dict(checkpoint["state_dict"])
    encoder.to(device).eval()

    height, width = train_images.shape[1], train_images.shape[2]
    served_frames = np.zeros((len(served), height, width, 3), dtype=np.uint8)
    for position, item in enumerate(served["id"]):
        with Image.open(image_dir / f"{item}.jpg") as opened:
            frame = opened.convert("RGB")
            if frame.size != (width, height):
                frame = frame.resize((width, height), Image.BILINEAR)
            served_frames[position] = np.asarray(frame, dtype=np.uint8)

    index = embed_all(encoder, served_frames, checkpoint["channel_mean"],
                      checkpoint["channel_std"], device, args.batch)
    print("Served index:", index.shape)

    # The evaluation index is built from the TRAINING gallery and drops the
    # held-out products, so a reported metric is never measured against a
    # catalogue containing the query's own product.
    protocol = RetrievalProtocol(gallery=train_gallery)
    evaluation = embed_all(encoder, train_images[protocol.catalogue_pos],
                           checkpoint["channel_mean"], checkpoint["channel_std"],
                           device, args.batch)
    gallery = served

    previous = ARTIFACT_DIR / "search_manifest.json"
    if previous.exists():
        shutil.copy(previous, ARTIFACT_DIR / "search_manifest_prev.json")

    np.save(ARTIFACT_DIR / "search_index_full.npy", index)
    np.save(ARTIFACT_DIR / "search_index_eval.npy", evaluation)
    np.save(ARTIFACT_DIR / "gallery_ids.npy", gallery["id"].to_numpy())
    gallery.to_csv(ARTIFACT_DIR / "gallery_metadata.csv", index=False)
    shutil.copy(args.encoder, ARTIFACT_DIR / "task4_improved_encoder.pt")

    manifest = {
        "best_method": "Improved+TTA+places365",
        "index_file": "search_index_full.npy",
        "encoder_file": "task4_improved_encoder.pt",
        "catalogue_size": int(len(gallery)),
        "embedding_dim": int(index.shape[1]),
        "image_size_pil": [int(width), int(height)],
        "channel_mean": [float(v) for v in checkpoint["channel_mean"]],
        "channel_std": [float(v) for v in checkpoint["channel_std"]],
        "use_tta": True,
        "uses_reranking": False,
        "background_augmented": True,
        "architecture": checkpoint.get("architecture", "improved"),
        "resolution": resolution,
        "evaluation_index_file": "search_index_eval.npy",
        "evaluation_catalogue_size": int(len(evaluation)),
        "benchmarks": summary.get("benchmarks", {}),
        "training_gallery_size": int(len(train_gallery)),
        "index_scope": (
            "The served index covers all 38,612 catalogue rows, because a customer must be "
            "able to find anything the shop stocks. The 15% product-level holdout is a "
            "property of the measurement, not of the shop, so the evaluation index "
            "excludes it and every reported metric is computed against that."),
        "provenance": {
            "trained_by": "src/training/train_task4_120x160.py",
            "promoted_by": "scripts/promote_task4_encoder.py",
            "source_checkpoint": str(args.encoder.resolve().relative_to(PROJECT_ROOT)
                                     ).replace("\\", "/"),
            "gallery": f"task4_gallery_{resolution}.csv",
            "background_source": checkpoint.get("background_source"),
            "photographic_share": checkpoint.get("photographic_share"),
            "background_split": checkpoint.get("background_split"),
            "warmup_epochs": checkpoint.get("warmup_epochs"),
            "learning_rate": summary.get("learning_rate"),
            "seed": checkpoint.get("seed"),
            "best_epoch": checkpoint.get("best_epoch"),
            "benchmark_note": (
                "`photo` and `wildphoto` composite onto Places365 scenes from the 73 "
                "HELD-OUT categories; `hard` and `wild` use checkerboards and stripes. "
                "Both families are reported because a procedurally trained encoder "
                "scores 59.50 on the first and 42.86 on the second, so one family "
                "alone cannot support a claim of background invariance."),
        },
    }
    with open(ARTIFACT_DIR / "search_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nPromoted. index {index.shape} | eval {evaluation.shape}")
    print(f"  {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
