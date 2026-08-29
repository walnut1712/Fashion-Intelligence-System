"""Clustering model for Task 4 - loadable, testable on any image.

Wraps the k-Means clustering built in ``notebooks/09_task4_clustering.ipynb``
into something that can be pointed at a photograph:

    engine = ClusterEngine.load()
    engine.predict("some_photo.jpg")     # which cluster, and what is in it
    engine.search("some_photo.jpg")      # cluster-accelerated similar items

Required artefacts
------------------
artifacts/task4/search_manifest.json        encoder + preprocessing settings
artifacts/task4/task4_improved_encoder.pt   the trained encoder
artifacts/task4/gallery_metadata.csv        one row per catalogue item
outputs/kmeans_centroids.npy                cluster centres
outputs/cluster_assignments.csv             item -> cluster
outputs/cluster_summary.csv                 cluster -> dominant articleType
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .search_engine import ImprovedEncoder, load_user_image

__all__ = ["ClusterEngine"]


class ClusterEngine:
    """k-Means over CNN embeddings, applied to new images."""

    def __init__(self, encoder, centroids, metadata, assignments, summary,
                 manifest, device):
        self.encoder = encoder
        self.centroids = centroids            # (k, d) float32
        self.metadata = metadata              # catalogue rows, aligned to `assignments`
        self.assignments = assignments        # cluster id per catalogue row
        self.summary = summary                # cluster -> dominant type, purity, size
        self.manifest = manifest
        self.device = device

        self.mean = np.array(manifest["channel_mean"], dtype=np.float32)
        self.std = np.array(manifest["channel_std"], dtype=np.float32)
        self.size = tuple(manifest["image_size_pil"])
        self.use_tta = bool(manifest.get("use_tta", True))

        # embeddings are needed to rank within a cluster
        index_path = Path(manifest["_artifact_dir"]) / manifest["index_file"]
        vectors = np.load(index_path).astype(np.float32)
        self.vectors = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True),
                                         1e-8, None)
        self.members = {int(c): np.where(assignments == c)[0]
                        for c in np.unique(assignments)}

    # ------------------------------------------------------------ load ----
    @classmethod
    def load(cls, project_dir=None, device=None):
        project_dir = Path(project_dir or Path(__file__).resolve().parents[2])
        artifact_dir = project_dir / "artifacts" / "task4"
        output_dir = project_dir / "outputs"

        missing = [str(p) for p in [
            artifact_dir / "search_manifest.json",
            output_dir / "kmeans_centroids.npy",
            output_dir / "cluster_assignments.csv",
            output_dir / "cluster_summary.csv",
        ] if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing artefacts: {}. Run notebooks/09_task4_clustering.ipynb."
                .format(", ".join(missing))
            )

        with open(artifact_dir / "search_manifest.json") as handle:
            manifest = json.load(handle)
        manifest["_artifact_dir"] = str(artifact_dir)

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(
            artifact_dir / manifest.get("encoder_file", "task4_improved_encoder.pt"),
            map_location=device)
        encoder = ImprovedEncoder(
            embedding_dim=checkpoint["embedding_dim"],
            n_types=checkpoint.get("n_types", 0),
            n_colours=checkpoint.get("n_colours", 0),
        )
        encoder.load_state_dict(checkpoint["state_dict"])
        encoder.to(device).eval()

        centroids = np.load(output_dir / "kmeans_centroids.npy").astype(np.float32)
        assignments_df = pd.read_csv(output_dir / "cluster_assignments.csv")
        summary = pd.read_csv(output_dir / "cluster_summary.csv").set_index("cluster")
        metadata = pd.read_csv(artifact_dir / "gallery_metadata.csv")

        if len(assignments_df) != len(metadata):
            raise ValueError("cluster_assignments.csv does not match gallery_metadata.csv")

        return cls(encoder, centroids, metadata,
                   assignments_df["kmeans_cluster"].to_numpy(), summary,
                   manifest, device)

    # --------------------------------------------------------- embedding ----
    @torch.no_grad()
    def embed(self, paths, mode="letterbox"):
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        arrays = np.stack([load_user_image(p, self.size, mode=mode) for p in paths])
        tensor = torch.from_numpy(arrays.astype(np.float32).transpose(0, 3, 1, 2) / 255.0)
        tensor = ((tensor - torch.tensor(self.mean).view(1, 3, 1, 1))
                  / torch.tensor(self.std).view(1, 3, 1, 1)).to(self.device)

        vectors = self.encoder.embed(tensor)
        if self.use_tta:
            vectors = F.normalize(vectors + self.encoder.embed(torch.flip(tensor, dims=[3])),
                                  p=2, dim=1)
        vectors = vectors.float().cpu().numpy()
        return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8, None)

    # ---------------------------------------------------------- predict ----
    def predict(self, path, top_clusters=3, mode="letterbox"):
        """Which cluster does this image belong to, and what is that cluster?"""
        vector = self.embed(path, mode=mode)[0]
        distances = np.linalg.norm(self.centroids - vector, axis=1)
        order = np.argsort(distances)[:top_clusters]

        ranked = []
        for cluster in order:
            row = self.summary.loc[int(cluster)] if int(cluster) in self.summary.index else {}
            ranked.append({
                "cluster": int(cluster),
                "distance": round(float(distances[cluster]), 4),
                "dominant_type": row.get("dominant articleType", "unknown"),
                "purity": float(row.get("purity", np.nan)),
                "size": int(row.get("size", len(self.members.get(int(cluster), [])))),
            })

        # A confident assignment is much closer to its own centroid than the next.
        margin = float(distances[order[1]] - distances[order[0]]) if len(order) > 1 else 0.0
        return {"best": ranked[0], "alternatives": ranked[1:],
                "margin": round(margin, 4),
                "confident": bool(margin > 0.05)}

    def search(self, path, k=10, probes=3, mode="letterbox"):
        """Cluster-accelerated retrieval: rank only the nearest clusters' members."""
        vector = self.embed(path, mode=mode)[0]
        nearest = np.argsort(np.linalg.norm(self.centroids - vector, axis=1))[:probes]
        candidates = np.concatenate([self.members[int(c)] for c in nearest
                                     if int(c) in self.members])
        scores = self.vectors[candidates] @ vector
        order = np.argsort(-scores)[:k]
        chosen = candidates[order]

        results = self.metadata.iloc[chosen][
            ["id", "articleType", "baseColour", "gender", "usage", "productDisplayName"]
        ].copy()
        results.insert(0, "rank", np.arange(1, len(results) + 1))
        results["similarity"] = scores[order].round(4)
        results["cluster"] = self.assignments[chosen]
        return results.reset_index(drop=True), int(len(candidates))

    def cluster_members(self, cluster, n=12):
        """Sample catalogue rows from one cluster - useful for showing what it holds."""
        indices = self.members.get(int(cluster), np.array([], dtype=int))
        if len(indices) == 0:
            return self.metadata.iloc[[]]
        picked = indices[:n] if len(indices) <= n else np.random.default_rng(0).choice(
            indices, n, replace=False)
        return self.metadata.iloc[picked]
