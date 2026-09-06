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
           "ImprovedEncoderV2", "GeM", "CosineHead", "build_encoder", "ARCHITECTURES",
           "NON_WEARABLE_CATEGORIES", "BAND_NAMES",
           "foreground_mask", "list_images", "PREPROCESS_MODES",
           "DEFAULT_CONFIDENCE"]


# Provisional gate for "should this answer be shown as confident?".
#
# Measured on the two populations the system actually sees (notebook 06 §10):
#
#     catalogue photographs   mean top-1 similarity 0.837, coherence 0.833
#     real user uploads       mean top-1 similarity 0.664, coherence 0.489
#
# The thresholds sit between those two clusters, so a flat-lay product photo
# passes and a hard upload is flagged rather than answered with false
# confidence. They are deliberately conservative and are re-derived whenever the
# encoder changes - never hand-tune them here without re-running that section.
DEFAULT_CONFIDENCE = {"min_top1_similarity": 0.70, "min_coherence": 0.50}


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


# ------------------------------------------------- improved components ----
class GeM(nn.Module):
    """Generalised-mean pooling.

    ``AdaptiveAvgPool2d`` weights every spatial cell equally, which suits
    classification but blurs retrieval: the garment occupies roughly a third of
    a 60x80 tile and the rest is white. GeM learns an exponent ``p`` that
    interpolates between average pooling (p=1) and max pooling (p -> inf), so
    the network can concentrate on the cells that carry the item.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x):
        clamped = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(clamped, 1).pow(1.0 / self.p)

    def extra_repr(self):
        return f"p={float(self.p):.3f}"


class CosineHead(nn.Module):
    """Cosine classifier with a learnable scale and an optional CosFace margin.

    ``ImprovedEncoder`` applies ``Linear`` directly to an L2-normalised
    embedding, which makes it a cosine classifier whose logits are bounded by
    ``||W||``. With the input on the unit sphere the cross-entropy can never
    become confident, so the auxiliary losses contributed only a weak gradient -
    the heads were ArcFace without the scale or the margin.

    Adding the scale restores that gradient; the additive margin pushes classes
    apart in the same space the retrieval metric uses. Unlike the PK-sampled
    triplet loss this also reaches classes with fewer than K images, which
    previously received no metric gradient at all.
    """

    def __init__(self, in_features, n_classes, scale=30.0, margin=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.log_scale = nn.Parameter(torch.tensor(float(np.log(scale))))
        self.margin = float(margin)

    @property
    def scale(self):
        return self.log_scale.exp()

    def forward(self, x, target=None):
        cosine = F.normalize(x, p=2, dim=1) @ F.normalize(self.weight, p=2, dim=1).T
        if self.margin > 0 and target is not None:
            one_hot = torch.zeros_like(cosine).scatter_(1, target.view(-1, 1), 1.0)
            cosine = cosine - one_hot * self.margin
        return self.scale * cosine


class ImprovedEncoderV2(nn.Module):
    """Two-branch encoder: deep semantics plus a mid-level colour pathway.

    ``ImprovedEncoder`` reads both articleType and baseColour off the final
    block. Colour is a low-level property, and after four stride-2 blocks and a
    global pool very little of it survives - which is the most likely reason
    ``colour@10`` (~54) trails ``P@10`` (~80) so badly, and why the unsupervised
    autoencoder beat the triplet encoder on the colour control (26.55 vs 18.63).

    Here the embedding is the concatenation of a deep branch (block 4, carrying
    shape and category) and a shallow branch tapped at ``colour_block``
    (carrying colour and texture). Both are supervised, and both are inside the
    embedding, so colour can actually influence retrieval rather than only the
    auxiliary loss.

    The total width is unchanged at 128, so the index and the served artefacts
    are the same size as before.
    """

    architecture = "improved_v2"

    def __init__(self, embedding_dim=128, widths=(32, 64, 128, 256),
                 n_types=0, n_colours=0, colour_dim=32, colour_block=2,
                 pool="gem", bnneck=True, scale=30.0, margin=0.2):
        super().__init__()
        if not 0 < colour_dim < embedding_dim:
            raise ValueError("colour_dim must be inside (0, embedding_dim)")
        if not 1 <= colour_block <= len(widths):
            raise ValueError("colour_block must index one of the backbone blocks")

        channels = 3
        blocks = []
        for width in widths:
            blocks.append(ConvBlock(channels, width))
            channels = width
        self.blocks = nn.ModuleList(blocks)

        self.colour_block = colour_block
        self.deep_dim = embedding_dim - colour_dim
        self.colour_dim = colour_dim

        self.pool = GeM() if pool == "gem" else nn.AdaptiveAvgPool2d(1)
        self.colour_pool = nn.AdaptiveAvgPool2d(1)      # colour is a mean, not a peak

        self.project = nn.Linear(channels, self.deep_dim)
        self.colour_project = nn.Linear(widths[colour_block - 1], colour_dim)

        # BNNeck: the triplet loss wants the raw feature, cross-entropy wants a
        # centred one. Forcing both onto a single L2-normalised vector makes them
        # pull against each other.
        self.bottleneck = nn.BatchNorm1d(embedding_dim) if bnneck else None
        if self.bottleneck is not None:
            self.bottleneck.bias.requires_grad_(False)

        self.type_head = CosineHead(embedding_dim, n_types, scale, margin) if n_types else None
        self.colour_head = CosineHead(colour_dim, n_colours, scale, margin) if n_colours else None

    def warm_start(self, state_dict, strict_shapes=True):
        """Lift a trained ``ImprovedEncoder``'s backbone into this model.

        The colour-branch architecture was specified in notebook 06 as a ladder
        of three candidates x three seeds x thirty epochs *from scratch*, which
        on a CPU-only machine is days rather than a night, and that cost is why
        it was never run.

        It does not need to be from scratch. The two models share their
        convolutional stack exactly - the same four ``ConvBlock`` widths, the
        same tensors - and differ only in the key prefix (``backbone.N`` against
        ``blocks.N``) because one uses ``Sequential`` and the other a
        ``ModuleList``. All 48 backbone tensors transfer, which is 96.5% of the
        parameters, leaving 14 to train: the two projections, the BNNeck and the
        cosine heads.

        Be clear about what this does and does not buy. The features transfer;
        the embedding space does not, because ``project`` changes shape
        (256->96 here against 256->128 there) and is necessarily fresh. So this
        is a shorter run, not a free one - budget more epochs than the twelve a
        pure fine-tune needs.

        Returns ``(loaded, skipped)`` tensor-name lists so a caller can assert
        on what actually transferred rather than trusting a silent load.
        """
        own = self.state_dict()
        loaded, skipped = [], []
        for key, value in state_dict.items():
            target = key.replace("backbone.", "blocks.", 1)
            if target in own and (not strict_shapes or own[target].shape == value.shape):
                own[target] = value.clone()
                loaded.append(target)
            else:
                skipped.append(key)
        self.load_state_dict(own)
        return loaded, skipped

    def features(self, x):
        """Return (embedding_before_bnneck, colour_branch_features)."""
        colour_feature = None
        for depth, block in enumerate(self.blocks, start=1):
            x = block(x)
            if depth == self.colour_block:
                colour_feature = self.colour_project(
                    self.colour_pool(x).flatten(1))
        deep = self.project(self.pool(x).flatten(1))
        return torch.cat([deep, colour_feature], dim=1), colour_feature

    def embed(self, x):
        combined, _ = self.features(x)
        if self.bottleneck is not None:
            combined = self.bottleneck(combined)
        return F.normalize(combined, p=2, dim=1)

    def forward(self, x, with_heads=False, target_type=None, target_colour=None):
        """Training reads the heads; inference only ever uses the embedding.

        The triplet loss is applied to ``metric`` (pre-BNNeck) and the
        classification losses to the post-BNNeck vector, which is the split
        BNNeck exists to provide.
        """
        combined, colour_feature = self.features(x)
        normalised = self.bottleneck(combined) if self.bottleneck is not None else combined
        z = F.normalize(normalised, p=2, dim=1)
        if not with_heads:
            return z
        return (
            z,
            F.normalize(combined, p=2, dim=1),                       # metric space
            self.type_head(normalised, target_type) if self.type_head is not None else None,
            self.colour_head(colour_feature, target_colour) if self.colour_head is not None else None,
        )


#: Every Task 4 encoder, keyed by the string a checkpoint records in
#: ``architecture``. Checkpoints written before that field existed are
#: ``ImprovedEncoder`` by definition, so the default preserves them.
ARCHITECTURES = {
    "improved": ImprovedEncoder,
    "improved_v2": ImprovedEncoderV2,
}


def build_encoder(checkpoint):
    """Rebuild the encoder a checkpoint describes.

    Task 3 already stores its architecture in the checkpoint and rebuilds from
    that, so changing the notebook cannot orphan a trained model. Task 4 did
    not, and ``SearchEngine.load`` and ``ClusterEngine.load`` disagreed about
    ``widths`` as a result. This is the one place that decision now lives.
    """
    name = checkpoint.get("architecture", "improved")
    if name not in ARCHITECTURES:
        raise ValueError(
            f"Unknown Task 4 architecture {name!r}. Known: {sorted(ARCHITECTURES)}")

    kwargs = dict(
        embedding_dim=checkpoint["embedding_dim"],
        widths=tuple(checkpoint.get("widths", (32, 64, 128, 256))),
        n_types=checkpoint.get("n_types", 0),
        n_colours=checkpoint.get("n_colours", 0),
    )
    if name == "improved_v2":
        for key in ("colour_dim", "colour_block", "pool", "bnneck", "scale", "margin"):
            if key in checkpoint:
                kwargs[key] = checkpoint[key]
    return ARCHITECTURES[name](**kwargs)


# ------------------------------------------------------ region proposal ----
#
# The encoder produces one vector per image, so a photograph of a person wearing
# a shirt, jeans and shoes is answered with a single guess. Notebook 06 section
# 11 measured the fix: propose regions, search each, keep the ones that beat the
# whole image. 53 of 186 regions were accepted across the 31 real uploads, and
# 21 of 31 found a closer match than the whole-image query (mean similarity
# 0.687 -> 0.719).
#
# Implemented with scipy rather than OpenCV, because the backend requirements do
# not include opencv and the notebook version's ``import cv2`` would take the
# whole API down on a machine without it.

#: Horizontal bands of the subject: upper/lower body, then thirds.
BAND_LAYOUT = (("upper", 0.00, 0.55), ("lower", 0.45, 1.00),
               ("top3", 0.00, 0.38), ("mid3", 0.31, 0.69), ("low3", 0.62, 1.00))

# A thin strip becomes a smear once stretched to 60x80, and smears land in
# whichever dense cluster happens to be nearby - which is how an early version
# matched a third of its regions to Socks, Handbags and Backpacks.
#: Expressed as a share of the frame, not an absolute pixel count. At 60x80 this
#: is the 48x48 it has always been; at 120x160 the same fraction is 96x96, where
#: a fixed 2,304 pixels would have accepted crops four times thinner.
MIN_CROP_FRACTION = (48 * 48) / (60 * 80)
MIN_ASPECT, MAX_ASPECT = 0.25, 4.0

#: Ranking is by SIMILARITY, never coherence. Ranking by coherence backfired:
#: coherence rose 0.61 -> 0.92 while similarity actually fell, because thin
#: crops on white are perfectly self-consistent and completely wrong.
REGION_ACCEPTANCE = {"min_similarity": 0.62, "min_coherence": 0.50,
                     "similarity_tolerance": 0.01}

#: Bands split a photograph of a *worn* outfit, so a band is a garment or an
#: accessory on a body - never a bottle of perfume, a lipstick or a cushion
#: cover. Measured on the 31 real uploads, 2 of 36 accepted band matches were
#: these: `600_google-pattern-socks.jpg` returned "Perfume and Body Mist" at
#: 0.741 from its top third, outscoring every real garment in the frame.
#:
#: Only masterCategory, and only for bands. A positional rule - footwear cannot
#: appear in an upper band - was considered and rejected: it assumes the frame
#: is a full-body shot, and many uploads are a single product, where the top
#: third of a shoe is legitimately footwear. Whole-image and component regions
#: are left alone, because a photograph of a perfume bottle should still find
#: perfume.
NON_WEARABLE_CATEGORIES = frozenset(
    {"Personal Care", "Home", "Sporting Goods", "Free Items"})
BAND_NAMES = frozenset(name for name, _, _ in BAND_LAYOUT)


def propose_regions(rgb, min_cover=0.10, min_component=0.03):
    """Whole subject, separate components, and horizontal bands of the subject."""
    from scipy import ndimage

    mask, method = foreground_mask(rgb)
    height, width = mask.shape
    ys, xs = np.where(mask)
    if len(ys) < 20:
        return ([{"name": "whole", "bbox": (0, 0, width, height), "cover": 1.0}],
                mask, method)

    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    regions = [{"name": "whole", "bbox": (x0, y0, x1 + 1, y1 + 1),
                "cover": float(mask[y0:y1 + 1, x0:x1 + 1].mean())}]

    # Genuinely separate objects - flat-lays of several items. Worn garments
    # touch, so this splits only about one upload in thirty.
    closed = ndimage.binary_closing(mask, np.ones((5, 5), bool), iterations=2)
    labels, count = ndimage.label(closed)
    for index in range(1, count + 1):
        component = labels == index
        if component.sum() / (height * width) <= min_component:
            continue
        cys, cxs = np.where(component)
        regions.append({
            "name": "part{}".format(index),
            "bbox": (cxs.min(), cys.min(), cxs.max() + 1, cys.max() + 1),
            "cover": float(component[cys.min():cys.max() + 1,
                                     cxs.min():cxs.max() + 1].mean()),
        })

    # Horizontal bands - upper body, lower body, footwear for a standing figure.
    span = y1 - y0 + 1
    for name, start, end in BAND_LAYOUT:
        ya, yb = int(y0 + start * span), int(y0 + end * span)
        if yb - ya < 12:
            continue
        band = mask[ya:yb, x0:x1 + 1]
        if band.mean() < min_cover:
            continue
        columns = np.where(band.any(axis=0))[0]
        if len(columns) < 5:
            continue
        regions.append({"name": name,
                        "bbox": (x0 + columns.min(), ya, x0 + columns.max() + 1, yb),
                        "cover": float(band.mean())})

    keep = []
    frame_pixels = rgb.shape[0] * rgb.shape[1]
    minimum_pixels = max(1, int(round(MIN_CROP_FRACTION * frame_pixels)))
    for region in regions:
        bx0, by0, bx1, by1 = region["bbox"]
        w, h = bx1 - bx0, by1 - by0
        if w * h < minimum_pixels:
            continue
        if not (MIN_ASPECT <= w / max(h, 1) <= MAX_ASPECT):
            continue
        keep.append(region)
    return (keep or regions[:1]), mask, method


def crop_region(rgb, mask, bbox, size, margin=0.04):
    """Cut the region out and place it on white, matching catalogue presentation."""
    from PIL import Image

    x0, y0, x1, y1 = bbox
    mx = int((x1 - x0) * margin) + 1
    my = int((y1 - y0) * margin) + 1
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(rgb.shape[1], x1 + mx), min(rgb.shape[0], y1 + my)
    patch, patch_mask = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    on_white = np.where(patch_mask[..., None], patch, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(on_white).resize(size, Image.BILINEAR),
                      dtype=np.uint8)


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
        # Which catalogue rows could plausibly be a band of a worn outfit.
        # Precomputed because ``search_regions`` needs it per band. Metadata
        # without a masterCategory column keeps every row: an index that cannot
        # answer the question should not have the filter applied to it silently.
        if "masterCategory" in self.metadata.columns:
            self._wearable = ~self.metadata["masterCategory"].isin(
                NON_WEARABLE_CATEGORIES).to_numpy()
        else:
            self._wearable = np.ones(len(self.metadata), dtype=bool)

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

        encoder = build_encoder(checkpoint)
        encoder.load_state_dict(checkpoint["state_dict"])
        encoder.to(device).eval()

        return cls(index, metadata, manifest, encoder, device)

    # -- embedding -----------------------------------------------------
    @torch.no_grad()
    def embed(self, paths, batch_size=64, mode="letterbox", return_info=False):
        """Embed user images with the exact pipeline used for the catalogue.

        ``return_info`` also yields, per image, which segmentation tier ran and
        whether it fell back. The ingestion layer has always produced this; the
        service simply never asked, so there was no way to tell a clean
        background removal from a silent fallback to a centre crop.
        """
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        vectors, infos = [], []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            if return_info:
                loaded = [load_user_image(p, self.size, mode=mode, return_info=True)
                          for p in batch_paths]
                arrays = np.stack([a for a, _ in loaded])
                infos.extend(info for _, info in loaded)
            else:
                arrays = np.stack([load_user_image(p, self.size, mode=mode)
                                   for p in batch_paths])
            vectors.append(self.embed_arrays(arrays))
        stacked = np.vstack(vectors)
        return (stacked, infos) if return_info else stacked

    @torch.no_grad()
    def embed_arrays(self, arrays):
        """Embed already-loaded ``uint8`` frames: normalise, encode, TTA.

        Split out of ``embed`` so that rebuilding the catalogue index does not
        have to restate the normalisation and the mirror average. Catalogue
        photographs are already the target size and do not want the ingestion
        pipeline user uploads get - letterboxing pads the handful of square
        catalogue images the encoder was trained on stretched - so the index
        builder supplies frames directly and everything downstream stays shared.
        """
        tensor = torch.from_numpy(
            np.asarray(arrays).astype(np.float32).transpose(0, 3, 1, 2) / 255.0)
        tensor = ((tensor - torch.tensor(self.mean).view(1, 3, 1, 1))
                  / torch.tensor(self.std).view(1, 3, 1, 1)).to(self.device)

        embedded = self.encoder.embed(tensor)
        if self.use_tta:                          # average with the mirror image
            mirrored = self.encoder.embed(torch.flip(tensor, dims=[3]))
            embedded = F.normalize(embedded + mirrored, p=2, dim=1)
        return embedded.float().cpu().numpy()

    # -- result shaping ------------------------------------------------
    def _dedupe(self, positions, scores):
        """Keep the best-scoring photo of each distinct product.

        The catalogue holds several shots of the same item, and they are each
        other's nearest neighbours, so an undeduplicated top-10 could be ten
        photographs of one product - a technically perfect result that shows a
        shopper a single thing.
        """
        if "productDisplayName" not in self.metadata.columns:
            return positions, scores
        names = self.metadata["productDisplayName"].to_numpy()
        seen, keep = set(), []
        for slot, position in enumerate(positions):
            name = names[position]
            key = name if isinstance(name, str) and name else f"__row_{position}"
            if key in seen:
                continue
            seen.add(key)
            keep.append(slot)
        keep = np.asarray(keep, dtype=int)
        return positions[keep], scores[keep]

    def _diversify(self, positions, scores, k, diversity):
        """Maximal Marginal Relevance over the candidate pool.

        ``diversity`` is the weight on "unlike what is already shown". At 0 this
        is plain relevance ranking; raising it trades a little similarity for a
        result grid that spans more of the catalogue.
        """
        if diversity <= 0 or len(positions) <= 1:
            return positions[:k], scores[:k]

        candidates = self.index[positions]                # already unit-norm
        chosen = [0]
        while len(chosen) < min(k, len(positions)):
            selected = candidates[chosen]                 # (m, D)
            redundancy = (candidates @ selected.T).max(axis=1)
            mmr = (1.0 - diversity) * scores - diversity * redundancy
            mmr[chosen] = -np.inf
            chosen.append(int(np.argmax(mmr)))
        order = np.asarray(chosen, dtype=int)
        return positions[order], scores[order]

    def _confidence(self, frame, scores, thresholds):
        """Label-free signals for "is this answer worth showing?".

        ``coherence`` is the share of the returned items agreeing with the top
        result's articleType. A confident match pulls a tight, consistent
        neighbourhood; a query the encoder cannot place returns a scattered one.
        """
        top1 = float(scores[0]) if len(scores) else 0.0
        coherence = 1.0
        if "articleType" in frame.columns and len(frame):
            types = frame["articleType"].to_numpy()
            coherence = float((types == types[0]).mean())
        return {
            "top1_similarity": round(top1, 4),
            "coherence": round(coherence, 4),
            "confident": bool(top1 >= thresholds["min_top1_similarity"]
                              and coherence >= thresholds["min_coherence"]),
        }

    # -- search --------------------------------------------------------
    def search(self, paths, k=10, mode="letterbox", dedupe=True, diversity=0.0,
               with_diagnostics=False, confidence=None):
        """Return a DataFrame of the k most similar catalogue items per query.

        ``dedupe`` collapses repeated photographs of one product (on by
        default - it is what a shopper wants). ``diversity`` in (0, 1] applies
        MMR re-ranking. ``with_diagnostics`` adds the ingestion tier and the
        confidence signals, so a caller can tell a solid match from a guess.
        """
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        thresholds = {**DEFAULT_CONFIDENCE, **(confidence or {})}

        if with_diagnostics:
            embedded, infos = self.embed(paths, mode=mode, return_info=True)
        else:
            embedded, infos = self.embed(paths, mode=mode), [None] * len(paths)

        queries = torch.from_numpy(embedded).to(self.device)
        similarity = queries @ self._tensor.T

        # Over-fetch so that dropping duplicate products still leaves k results.
        # A pool of 5k was not enough: some catalogue items carry a dozen photos
        # under one productDisplayName, and 115 of the 5,829 test images came
        # back with fewer than k results - one with only 4. Widening the pool
        # costs nothing measurable, since topk runs over all 32,837 rows either
        # way.
        pool = min(max(k * 12, 120) if (dedupe or diversity > 0) else k,
                   similarity.shape[1])
        scores, indices = torch.topk(similarity, k=pool, dim=1)
        scores, indices = scores.cpu().numpy(), indices.cpu().numpy()

        frames = []
        for row, path in enumerate(paths):
            positions, row_scores = indices[row], scores[row]
            if dedupe:
                positions, row_scores = self._dedupe(positions, row_scores)
            positions, row_scores = self._diversify(positions, row_scores, k, diversity)
            positions, row_scores = positions[:k], row_scores[:k]

            frame = self.metadata.iloc[positions].copy()
            frame.insert(0, "rank", np.arange(1, len(frame) + 1))
            frame.insert(0, "query", Path(path).name)
            frame["similarity"] = row_scores.round(4)
            frame["mode"] = mode

            if with_diagnostics:
                for key, value in self._confidence(frame, row_scores, thresholds).items():
                    frame[key] = value
                info = infos[row] or {}
                frame["ingest_method"] = info.get("method")
                frame["ingest_fell_back"] = info.get("fell_back")
            frames.append(frame.reset_index(drop=True))
        return pd.concat(frames, ignore_index=True)

    # -- per-garment search --------------------------------------------
    def search_regions(self, path, k=10, acceptance=None, **kwargs):
        """Search each proposed region of one photo, keep the ones worth showing.

        Returns a DataFrame with a ``region`` column. A photograph of a person
        wearing several items yields one group per garment instead of a single
        guess for the whole frame.

        A region is kept when it is a confident match in its own right, or when
        it is at least as close as the whole image was - the rule notebook 06
        settled on after ranking by coherence produced consistent nonsense.
        """
        from PIL import Image

        rules = {**REGION_ACCEPTANCE, **(acceptance or {})}
        with Image.open(path) as opened:
            rgb = np.asarray(_to_rgb_on_white(opened))

        regions, mask, method = propose_regions(rgb)
        tiles = np.stack([crop_region(rgb, mask, r["bbox"], self.size)
                          for r in regions])

        tensor = torch.from_numpy(tiles.astype(np.float32).transpose(0, 3, 1, 2) / 255.0)
        tensor = ((tensor - torch.tensor(self.mean).view(1, 3, 1, 1))
                  / torch.tensor(self.std).view(1, 3, 1, 1)).to(self.device)
        with torch.no_grad():
            vectors = self.encoder.embed(tensor)
            if self.use_tta:
                vectors = F.normalize(
                    vectors + self.encoder.embed(torch.flip(tensor, dims=[3])),
                    p=2, dim=1)
        vectors = vectors.float().cpu().numpy()

        similarity = torch.from_numpy(vectors).to(self.device) @ self._tensor.T
        pool = min(max(k * 12, 120), similarity.shape[1])
        scores, indices = torch.topk(similarity, k=pool, dim=1)
        scores, indices = scores.cpu().numpy(), indices.cpu().numpy()

        whole_similarity = float(scores[0][0])          # region 0 is always "whole"
        frames = []
        for row, region in enumerate(regions):
            candidates, candidate_scores = indices[row], scores[row]
            if region["name"] in BAND_NAMES:
                # Drop rather than merely refuse to accept: a band whose best
                # match is a perfume should return its best *garment*, not an
                # unusable answer flagged false.
                keep = self._wearable[candidates]
                if keep.any():
                    candidates, candidate_scores = candidates[keep], candidate_scores[keep]

            positions, row_scores = self._dedupe(candidates, candidate_scores)
            positions, row_scores = positions[:k], row_scores[:k]

            frame = self.metadata.iloc[positions].copy()
            frame.insert(0, "rank", np.arange(1, len(frame) + 1))
            frame.insert(0, "region", region["name"])
            frame.insert(0, "query", Path(path).name)
            frame["similarity"] = row_scores.round(4)

            signals = self._confidence(frame, row_scores, DEFAULT_CONFIDENCE)
            top = float(row_scores[0])
            accepted = (
                (top >= rules["min_similarity"]
                 and signals["coherence"] >= rules["min_coherence"])
                or top >= whole_similarity - rules["similarity_tolerance"]
            )
            frame["coherence"] = signals["coherence"]
            frame["accepted"] = bool(accepted)
            frame["segmentation"] = method
            frames.append(frame.reset_index(drop=True))

        return pd.concat(frames, ignore_index=True)
