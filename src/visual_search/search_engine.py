"""Task 4 visual search - loadable engine for user-supplied images.

Loads the artefacts produced by ``05_task4_visual_search.ipynb`` and
answers "which catalogue items look like this photo?" for images that were
never part of the dataset.

Nothing here retrains anything. Required artefacts, all in ``artifacts/task4``:

    search_manifest.json          which model, which index, preprocessing stats
    search_index_*.npy            catalogue embeddings (32,837 x 128)
    gallery_metadata.csv          one row per catalogue item, aligned to the index
    task4_improved_encoder.pt     encoder weights

Usage
-----
>>> engine = SearchEngine.load()
>>> results = engine.search("some_photo.webp", k=10)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ingestion lives in src/data/user_image.py so Task 1 serving, Task 4 retrieval
# and the offline benchmark all coerce an upload the same way. Re-exported here
# because callers (cluster_engine, task4_service) import these names from this
# module, and because they are part of this module's published __all__.
from src.data.user_image import (  # noqa: F401
    PREPROCESS_MODES,
    SUPPORTED_SUFFIXES,
    _fit_to_size,
    _to_rgb_on_white,
    foreground_mask,
    list_images,
    load_user_image,
)

__all__ = ["SearchEngine", "load_user_image", "ImprovedEncoder",
           "foreground_mask", "list_images", "PREPROCESS_MODES"]


# ---------------------------------------------------------------- model ----
class ConvBlock(nn.Module):
    """Conv-BN-ReLU x2 then max-pool. Must match the training definition."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class ImprovedEncoder(nn.Module):
    """Embedding network with the auxiliary heads used during training.

    The heads are never used at inference; they are declared only so the saved
    ``state_dict`` loads without ``strict=False``.
    """

    def __init__(self, embedding_dim=128, widths=(32, 64, 128, 256),
                 n_types=0, n_colours=0):
        super().__init__()
        channels = 3
        blocks = []
        for width in widths:
            blocks.append(ConvBlock(channels, width))
            channels = width
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(channels, embedding_dim)
        self.type_head = nn.Linear(embedding_dim, n_types) if n_types else None
        self.colour_head = nn.Linear(embedding_dim, n_colours) if n_colours else None

    def embed(self, x):
        features = self.pool(self.backbone(x)).flatten(1)
        return F.normalize(self.project(features), p=2, dim=1)

    def forward(self, x, with_heads=False):
        """`with_heads=True` also returns the auxiliary logits, which retraining
        needs. Inference only ever uses the embedding."""
        z = self.embed(x)
        if not with_heads:
            return z
        return (z,
                self.type_head(z) if self.type_head is not None else None,
                self.colour_head(z) if self.colour_head is not None else None)



# --------------------------------------------------------------- engine ----
class SearchEngine:
    """Cosine-similarity search over the saved catalogue embeddings."""

    def __init__(self, index, metadata, manifest, encoder, device):
        self.index = index                # (N, D) float32, L2-normalised
        self.metadata = metadata          # N rows, aligned to `index`
        self.manifest = manifest
        self.encoder = encoder
        self.device = device
        self.mean = np.array(manifest["channel_mean"], dtype=np.float32)
        self.std = np.array(manifest["channel_std"], dtype=np.float32)
        self.size = tuple(manifest["image_size_pil"])
        self.use_tta = bool(manifest.get("use_tta", True))
        self._tensor = torch.from_numpy(index).to(device)

    # -- construction --------------------------------------------------
    @classmethod
    def load(cls, artifact_dir=None, device=None):
        artifact_dir = Path(artifact_dir or
                            Path(__file__).resolve().parents[2] / "artifacts" / "task4")
        with open(artifact_dir / "search_manifest.json") as handle:
            manifest = json.load(handle)

        index = np.load(artifact_dir / manifest["index_file"]).astype(np.float32)
        index /= np.clip(np.linalg.norm(index, axis=1, keepdims=True), 1e-8, None)

        metadata = pd.read_csv(artifact_dir / "gallery_metadata.csv")
        if len(metadata) != len(index):
            raise ValueError(
                f"Index has {len(index)} rows but metadata has {len(metadata)}. "
                "Re-run the final cells of 05_task4_visual_search.ipynb."
            )

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(artifact_dir / manifest.get(
            "encoder_file", "task4_improved_encoder.pt"), map_location=device)

        encoder = ImprovedEncoder(
            embedding_dim=checkpoint["embedding_dim"],
            widths=tuple(checkpoint.get("widths", (32, 64, 128, 256))),
            n_types=checkpoint.get("n_types", 0),
            n_colours=checkpoint.get("n_colours", 0),
        )
        encoder.load_state_dict(checkpoint["state_dict"])
        encoder.to(device).eval()

        return cls(index, metadata, manifest, encoder, device)

    # -- embedding -----------------------------------------------------
    @torch.no_grad()
    def embed(self, paths, batch_size=64, mode="letterbox"):
        """Embed user images with the exact pipeline used for the catalogue."""
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        vectors = []
        for start in range(0, len(paths), batch_size):
            arrays = np.stack([load_user_image(p, self.size, mode=mode)
                               for p in paths[start:start + batch_size]])
            tensor = torch.from_numpy(arrays.astype(np.float32).transpose(0, 3, 1, 2) / 255.0)
            tensor = ((tensor - torch.tensor(self.mean).view(1, 3, 1, 1))
                      / torch.tensor(self.std).view(1, 3, 1, 1)).to(self.device)

            embedded = self.encoder.embed(tensor)
            if self.use_tta:                      # average with the mirror image
                mirrored = self.encoder.embed(torch.flip(tensor, dims=[3]))
                embedded = F.normalize(embedded + mirrored, p=2, dim=1)
            vectors.append(embedded.float().cpu().numpy())
        return np.vstack(vectors)

    # -- search --------------------------------------------------------
    def search(self, paths, k=10, mode="letterbox"):
        """Return a DataFrame of the k most similar catalogue items per query."""
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        queries = torch.from_numpy(self.embed(paths, mode=mode)).to(self.device)
        similarity = queries @ self._tensor.T
        scores, indices = torch.topk(similarity, k=min(k, similarity.shape[1]), dim=1)
        scores, indices = scores.cpu().numpy(), indices.cpu().numpy()

        frames = []
        for row, path in enumerate(paths):
            frame = self.metadata.iloc[indices[row]].copy()
            frame.insert(0, "rank", np.arange(1, len(frame) + 1))
            frame.insert(0, "query", Path(path).name)
            frame["similarity"] = scores[row].round(4)
            frame["mode"] = mode
            frames.append(frame.reset_index(drop=True))
        return pd.concat(frames, ignore_index=True)
