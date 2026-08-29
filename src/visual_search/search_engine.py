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
from PIL import Image, ImageOps

__all__ = ["SearchEngine", "load_user_image", "ImprovedEncoder",
           "foreground_mask", "list_images", "PREPROCESS_MODES"]

# Optional decoder for .avif - Pillow >= 11 handles it natively.
try:  # pragma: no cover
    import pillow_avif  # noqa: F401
except ImportError:  # pragma: no cover
    pass

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".gif", ".tiff"}


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


# ---------------------------------------------------------- preprocessing ----
PREPROCESS_MODES = ("letterbox", "crop", "nobg")

_REMBG_SESSION = None


def _rembg_session():
    """Lazy rembg loader. Returns None when rembg is not installed."""
    global _REMBG_SESSION
    if _REMBG_SESSION is False:
        return None
    if _REMBG_SESSION is None:
        try:
            from rembg import new_session
            _REMBG_SESSION = new_session("u2netp")
        except Exception:
            _REMBG_SESSION = False
            return None
    return _REMBG_SESSION


def _to_rgb_on_white(img, background=(255, 255, 255)):
    """EXIF-correct, palette-aware, alpha flattened onto WHITE."""
    img = ImageOps.exif_transpose(img)
    if img.mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
    if img.mode in ("RGBA", "LA"):
        canvas = Image.new("RGB", img.size, background)
        canvas.paste(img, mask=img.split()[-1])
        return canvas
    return img.convert("RGB")


def _border_is_uniform(array, tolerance=12):
    border = np.concatenate([array[0, :], array[-1, :], array[:, 0], array[:, -1]])
    return border.astype(np.int16).std(axis=0).max() < tolerance


def _centre_kept(mask, fraction=0.45):
    """How much of the central region survived. Near zero means the subject was eaten."""
    height, width = mask.shape
    y0, y1 = int(height * (0.5 - fraction / 2)), int(height * (0.5 + fraction / 2))
    x0, x1 = int(width * (0.5 - fraction / 2)), int(width * (0.5 + fraction / 2))
    centre = mask[y0:y1, x0:x1]
    return float(centre.mean()) if centre.size else 0.0


def foreground_mask(array, max_side=512, min_centre=0.55):
    """Best-effort subject mask. Returns (mask, method_name).

    Tiered by quality: rembg (learned matting) -> GrabCut -> flood fill.

    Two guards stop the classical methods deleting the product itself, which is
    the failure mode when the garment is dark, or light on a light background:

    * GrabCut is told the central region is **definite foreground**, so it cannot
      classify the middle of the frame as background.
    * Any mask that removes most of the centre is rejected outright.
    """
    height, width = array.shape[:2]

    session = _rembg_session()
    if session is not None:
        try:
            from rembg import remove
            cut = remove(Image.fromarray(array), session=session)
            alpha = np.asarray(cut.split()[-1])
            mask = (alpha > 128).astype(np.uint8)
            if 0.03 < mask.mean() < 0.97 and _centre_kept(mask) > 0.25:
                return mask, "rembg"
        except Exception:
            pass

    try:
        import cv2
    except ImportError:
        return np.ones((height, width), np.uint8), "none (opencv missing)"

    scale = min(1.0, max_side / max(height, width))
    small = (cv2.resize(array, (int(width * scale), int(height * scale)),
                        interpolation=cv2.INTER_AREA) if scale < 1 else array)
    small_h, small_w = small.shape[:2]

    candidates = []

    # --- GrabCut with a definite-foreground core -----------------------------
    mask = np.zeros((small_h, small_w), np.uint8)
    inset_x, inset_y = int(small_w * 0.05) + 1, int(small_h * 0.05) + 1
    rect = (inset_x, inset_y, small_w - 2 * inset_x, small_h - 2 * inset_y)
    try:
        cv2.grabCut(small, mask, rect, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), 3, cv2.GC_INIT_WITH_RECT)
        # pin the middle as foreground, the extreme border as background
        cy0, cy1 = int(small_h * 0.35), int(small_h * 0.65)
        cx0, cx1 = int(small_w * 0.35), int(small_w * 0.65)
        mask[cy0:cy1, cx0:cx1] = cv2.GC_FGD
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = cv2.GC_BGD
        cv2.grabCut(small, mask, None, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), 3, cv2.GC_INIT_WITH_MASK)
        grab = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        candidates.append((grab, "grabcut+core"))
    except cv2.error:
        pass

    # --- flood fill, only when the border is genuinely uniform ---------------
    if _border_is_uniform(small):
        flood = small.copy()
        buffer = np.zeros((small_h + 2, small_w + 2), np.uint8)
        for seed in [(0, 0), (small_w - 1, 0), (0, small_h - 1), (small_w - 1, small_h - 1)]:
            cv2.floodFill(flood, buffer, seed, (0, 0, 0), (14,) * 3, (14,) * 3,
                          cv2.FLOODFILL_MASK_ONLY | (255 << 8) | 4)
        candidates.append(((buffer[1:-1, 1:-1] == 0).astype(np.uint8), "floodfill"))

    kernel = np.ones((3, 3), np.uint8)
    scored = []
    for candidate, name in candidates:
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
        area = candidate.mean()
        centre = _centre_kept(candidate)
        if not (0.03 < area < 0.97):
            continue
        if centre < min_centre:            # ate the subject - reject
            continue
        scored.append((centre, -area, candidate, name))

    if not scored:
        return np.ones((height, width), np.uint8), "none (subject would be removed)"

    # prefer the mask that keeps the centre intact while removing the most border
    scored.sort(reverse=True)
    _, _, best, name = scored[0]
    best = cv2.resize(best, (width, height), interpolation=cv2.INTER_NEAREST)
    return best, name


def _fit_to_size(img, size, background=(255, 255, 255), mode="letterbox"):
    """Pad ('letterbox') or centre-crop ('crop') to the target aspect, then resize."""
    target_w, target_h = size
    target_ratio = target_w / target_h
    width, height = img.size

    if mode == "crop":
        if width / height > target_ratio:                 # too wide -> trim sides
            new_width = int(round(height * target_ratio))
            left = (width - new_width) // 2
            img = img.crop((left, 0, left + new_width, height))
        else:                                             # too tall -> trim top/bottom
            new_height = int(round(width / target_ratio))
            top = (height - new_height) // 2
            img = img.crop((0, top, width, top + new_height))
    else:
        if width / height > target_ratio:
            new_height = int(round(width / target_ratio))
            canvas = Image.new("RGB", (width, new_height), background)
            canvas.paste(img, (0, (new_height - height) // 2))
            img = canvas
        else:
            new_width = int(round(height * target_ratio))
            canvas = Image.new("RGB", (new_width, height), background)
            canvas.paste(img, ((new_width - width) // 2, 0))
            img = canvas

    return np.asarray(img.resize(size, Image.BILINEAR), dtype=np.uint8)


def load_user_image(path, size=(60, 80), mode="letterbox", margin=0.06,
                    background=(255, 255, 255), return_info=False):
    """Read an arbitrary user image into the catalogue's format.

    The catalogue is 60x80 product shots on white, one item, filling roughly half
    the frame. A user upload is none of those, so it must be coerced first.

    mode
        ``"letterbox"`` pad to 3:4 and resize. Keeps everything, including the
        background, which the embedding then partly describes.
        ``"crop"`` centre-crop to 3:4. Discards the edges of the frame, which is
        often where the background lives, and preserves the subject's scale.
        ``"nobg"`` segment the subject, place it on white, crop tight to it, then
        pad and resize. Closest to catalogue framing when segmentation succeeds.

    ``"nobg"`` reverts to ``"crop"`` whenever the mask looks implausible - a
    light-coloured product on a light background is the case that breaks the
    classical methods, and deleting the product would be worse than keeping the
    background.
    """
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"mode must be one of {PREPROCESS_MODES}")

    with Image.open(path) as opened:
        img = _to_rgb_on_white(opened, background)

    info = {"mode": mode, "method": "none", "foreground_fraction": 1.0,
            "fell_back": False}

    if mode != "nobg":
        array = _fit_to_size(img, size, background, mode=mode)
        info["ink_fraction"] = float((array.min(axis=2) < 235).mean())
        return (array, info) if return_info else array

    array = np.asarray(img)
    mask, method = foreground_mask(array)
    info["method"] = method
    info["foreground_fraction"] = float(mask.mean())

    cut = np.where(mask[..., None].astype(bool), array, 255).astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs) and len(ys):
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        pad_x = int((x1 - x0) * margin) + 1
        pad_y = int((y1 - y0) * margin) + 1
        x0, x1 = max(0, x0 - pad_x), min(cut.shape[1] - 1, x1 + pad_x)
        y0, y1 = max(0, y0 - pad_y), min(cut.shape[0] - 1, y1 + pad_y)
        cut = cut[y0:y1 + 1, x0:x1 + 1]

    result = _fit_to_size(Image.fromarray(cut), size, background, mode="letterbox")

    # If almost nothing survived, the segmentation removed the product itself.
    ink = float((result.min(axis=2) < 235).mean())
    if ink < 0.05:
        info["fell_back"] = True
        info["method"] += " -> reverted to crop"
        result = _fit_to_size(img, size, background, mode="crop")
        ink = float((result.min(axis=2) < 235).mean())

    info["ink_fraction"] = ink
    return (result, info) if return_info else result


def list_images(folder):
    """Every readable image in a folder, sorted, ignoring other files."""
    folder = Path(folder)
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)


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
