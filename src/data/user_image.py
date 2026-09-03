"""Shared ingestion for images that were never part of the catalogue.

Why this exists
---------------
The catalogue is 60x80 product cutouts, one item, centred, on white. Sampled over
the training cache, 67% of all pixels are near-white, and the channel means the
models normalise with are ``[0.854, 0.834, 0.827]``. A user upload is none of
those, and the gap is not cosmetic. Measured on 2,500 held-out rows with the
deployed Task 1 checkpoint:

    clean catalogue                       87.92 acc / 71.98 macro-F1
    aspect squashed, still white          86.88 acc / 70.64 macro-F1
    composited onto a textured background 25.80 acc /  5.19 macro-F1

Squashing the aspect ratio costs about a point. The background costs sixty. The
model has learned "garment on white" rather than "garment", so coercing an upload
back onto white is the single highest-value thing this module does.

This code started inside ``src/visual_search/search_engine.py``, where only Task 4
could reach it - Task 1 served raw uploads straight into a 60x80 resize. It lives
here so Task 1 serving, Task 4 retrieval and the offline benchmark all ingest an
upload the same way, for the same reason ``src/models/item_type_classifier.py``
exists: duplicated definitions drift, and the drift is only discovered when a
shipped artefact stops loading.

Segmentation is tiered by quality, best first, and every tier is optional:

    rembg (u2netp)   learned matting, if installed
    GrabCut          OpenCV, seeded with a definite-foreground core
    border model     numpy + scipy, fits the background from the frame edge
    flood fill       OpenCV, only when the border is genuinely uniform

The border model is new, and it is what makes ``nobg`` work on a machine without
OpenCV. Before it, ``foreground_mask`` returned an all-ones mask on such a
machine and ``nobg`` silently degraded to "keep the entire background" - the one
case the classifier cannot survive. The frontend has been requesting
``search_mode=nobg`` on every upload and not getting it.
"""

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

__all__ = [
    "PREPROCESS_MODES",
    "SUPPORTED_SUFFIXES",
    "catalogue_alpha",
    "foreground_mask",
    "list_images",
    "load_user_image",
    "looks_like_catalogue",
]

# Optional decoder for .avif - Pillow >= 11 handles it natively.
try:  # pragma: no cover
    import pillow_avif  # noqa: F401
except ImportError:  # pragma: no cover
    pass

try:  # scipy powers the border-model tier; absent it, that tier is skipped.
    from scipy import ndimage
except ImportError:  # pragma: no cover
    ndimage = None

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".gif", ".tiff"}

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


def _cv2():
    """OpenCV if importable, else None. Two tiers and the morphology want it."""
    try:
        import cv2
        return cv2
    except ImportError:
        return None


# ---------------------------------------------------------------- decoding ----
def _open_source(source):
    """Path, bytes, file-like, ``PIL.Image`` or uint8 HWC array -> ``PIL.Image``.

    Mirrors the source contract of ``load_image_array`` in
    ``src/models/item_type_classifier.py``, plus arrays: the FastAPI services hold
    request bodies as bytes, and the benchmark and tests hold decoded arrays.
    """
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, np.ndarray):
        return Image.fromarray(np.ascontiguousarray(source.astype(np.uint8)))
    # Undecodable input surfaces as ValueError, exactly as load_image_array does:
    # the API maps ValueError to a 400 and anything else to a 500, so letting
    # Pillow's OSError escape turns a bad upload into a server error.
    try:
        if isinstance(source, (bytes, bytearray, memoryview)):
            return Image.open(BytesIO(bytes(source)))
        return Image.open(source)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Cannot decode image: {}".format(error))


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


# ------------------------------------------------------------ mask helpers ----
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


def _downscale(array, max_side):
    """Shrink so the longest side is ``max_side``. PIL, so it needs no OpenCV."""
    height, width = array.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return array
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return np.asarray(Image.fromarray(array).resize(size, Image.BILINEAR), dtype=np.uint8)


def _resize_mask(mask, width, height):
    """Nearest-neighbour resize of a 0/1 mask back to full resolution."""
    scaled = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (width, height), Image.NEAREST)
    return (np.asarray(scaled, dtype=np.uint8) > 127).astype(np.uint8)


def _mask_runs(mask):
    """Mean number of separate mask runs per row and per column.

    The shredding detector. A coherent object crossed by a scan line gives one
    run; a comb gives many. Measured over the real photos, good mattes sit at
    1.2-1.8 runs per column while the two failures sit far above: the striped
    dress at 4.16, where the border-colour model matched the *subject's own*
    light stripes and cut them out, and a shoe on a striped backdrop at 5.18,
    where it kept the backdrop.

    Connected-component counting cannot see either case - the stripes touch at
    the frame edge, so both masks are a single component covering 100% of the
    mask area. Run counting is what actually separates them.
    """
    mask = mask > 0
    if not mask.any():
        return 0.0, 0.0
    row_starts = (np.diff(mask.astype(np.int8), axis=1) == 1).sum(axis=1)
    col_starts = (np.diff(mask.astype(np.int8), axis=0) == 1).sum(axis=0)
    rows, cols = row_starts[mask.any(axis=1)], col_starts[mask.any(axis=0)]
    return (float(rows.mean()) if rows.size else 0.0,
            float(cols.mean()) if cols.size else 0.0)


def _clean(mask, cv2_module):
    """Open once then close twice, to drop specks and seal pinholes."""
    if cv2_module is not None:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2_module.morphologyEx(mask, cv2_module.MORPH_OPEN, kernel, iterations=1)
        return cv2_module.morphologyEx(mask, cv2_module.MORPH_CLOSE, kernel, iterations=2)
    if ndimage is None:
        return mask
    kernel = np.ones((3, 3), bool)
    opened = ndimage.binary_opening(mask.astype(bool), kernel, iterations=1)
    closed = ndimage.binary_closing(opened, kernel, iterations=2)
    return closed.astype(np.uint8)


def _modal_colours(ring, k=3):
    """A few representative border colours, so a two-tone scene still models.

    A single median describes a seamless studio backdrop, but not a photo shot
    against a wall above a floor, or a gradient. k-means over the border ring
    gives one centre per background region; a pixel is background when it is
    close to *any* of them.
    """
    if len(ring) < k * 8:
        return np.median(ring, axis=0)[None, :]
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:  # pragma: no cover
        return np.median(ring, axis=0)[None, :]
    fitted = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=0).fit(ring)
    return fitted.cluster_centers_.astype(np.float32)


def _border_colour_masks(small, tolerances=(1.0, 2.0, 3.5, 6.0)):
    """Background modelled from the frame edge. numpy + scipy (+ sklearn) only.

    Generalises the flood-fill tier. Rather than assuming the background is white,
    or uniform enough to flood from the corners, it fits a colour model to a ring
    of border pixels and calls everything close to that model background - which
    covers the plain studio backdrop, of whatever colour, that e-commerce
    photography actually uses.

    Unioned with a bright-and-unsaturated test, so a white backdrop still reads as
    background when the border itself carries a watermark or a logo.

    Returns one candidate per tolerance rather than committing to a single mask.
    A tight tolerance is right for a clean backdrop and leaves a busy one almost
    entirely "foreground"; a loose one is the reverse. Handing the whole ladder to
    the caller lets the existing scoring - keep the centre, remove the most border
    - choose, and lets every tolerance be rejected when none of them is plausible.
    """
    if ndimage is None:
        return []

    height, width = small.shape[:2]
    band = max(2, int(round(min(height, width) * 0.03)))
    ring = np.concatenate([
        small[:band].reshape(-1, 3), small[-band:].reshape(-1, 3),
        small[:, :band].reshape(-1, 3), small[:, -band:].reshape(-1, 3),
    ]).astype(np.float32)

    centres = _modal_colours(ring)
    # Median absolute deviation, scaled to a standard-deviation equivalent, so a
    # slightly busy border widens the tolerance instead of poisoning the estimate.
    # Capped, because an uncapped spread defeats the whole model: a photo shot in
    # a room has a border MAD near 90, which at any tolerance calls every pixel
    # background and leaves nothing to segment. Past the cap the border is not a
    # backdrop at all, and the k-means centres carry the model instead.
    median = np.median(ring, axis=0)
    spread = np.clip(1.4826 * np.median(np.abs(ring - median), axis=0), 4.0, 24.0)

    pixels = small.astype(np.float32)
    value = pixels.max(axis=2)
    near_white = (value >= 232) & ((value - pixels.min(axis=2)) <= 18)
    structure = np.ones((3, 3), bool)

    # A product photo puts its subject in the middle, so keep every component
    # reaching the centre box - a garment split by an occlusion is still one item.
    y0, y1 = int(height * 0.35), int(height * 0.65)
    x0, x1 = int(width * 0.35), int(width * 0.65)

    out = []
    for tolerance in tolerances:
        near_border = np.zeros(pixels.shape[:2], bool)
        for centre in centres:
            near_border |= (np.abs(pixels - centre) <= tolerance * spread).all(axis=2)

        foreground = ndimage.binary_fill_holes(
            ndimage.binary_opening(~(near_border | near_white), structure))
        labels, count = ndimage.label(foreground)
        if not count:
            continue
        keep = np.unique(labels[y0:y1, x0:x1])
        keep = keep[keep > 0]
        if not keep.size:
            continue
        mask = ndimage.binary_fill_holes(np.isin(labels, keep))
        out.append((mask.astype(np.uint8), f"border-model(t={tolerance})"))
    return out


def foreground_mask(array, max_side=256, min_centre=0.55, max_runs=2.5):
    """Best-effort subject mask. Returns ``(mask, method_name)``.

    Tiered by quality: rembg (learned matting) -> GrabCut -> border colour model
    -> flood fill. Every tier is optional, and the ladder is ordered so a machine
    with OpenCV or rembg installed keeps exactly the behaviour it had.

    ``max_side`` is the resolution the classical tiers work at. 256 rather than
    512: the mask is only ever used to cut a tile that ends up 60x80, so 256 is
    already four times the detail that survives, and it halves the per-upload
    cost (1108ms -> 510ms measured over the 31 real photos) without losing a
    single successful segmentation.

    Two guards stop a classical method deleting the product itself, which is the
    failure mode when the garment is dark, or light on a light background:

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

    cv2 = _cv2()
    small = _downscale(array, max_side)
    small_h, small_w = small.shape[:2]

    candidates = []

    # --- GrabCut with a definite-foreground core -----------------------------
    if cv2 is not None:
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

    # --- border colour model, the tier that needs no OpenCV -------------------
    candidates.extend(_border_colour_masks(small))

    # --- flood fill, only when the border is genuinely uniform ---------------
    if cv2 is not None and _border_is_uniform(small):
        flood = small.copy()
        buffer = np.zeros((small_h + 2, small_w + 2), np.uint8)
        for seed in [(0, 0), (small_w - 1, 0), (0, small_h - 1), (small_w - 1, small_h - 1)]:
            cv2.floodFill(flood, buffer, seed, (0, 0, 0), (14,) * 3, (14,) * 3,
                          cv2.FLOODFILL_MASK_ONLY | (255 << 8) | 4)
        candidates.append(((buffer[1:-1, 1:-1] == 0).astype(np.uint8), "floodfill"))

    scored = []
    for candidate, name in candidates:
        candidate = _clean(candidate, cv2)
        area = candidate.mean()
        centre = _centre_kept(candidate)
        if not (0.03 < area < 0.97):
            continue
        if centre < min_centre:            # ate the subject - reject
            continue
        if max(_mask_runs(candidate)) > max_runs:
            continue                       # a comb, not an object - reject
        # Centre coverage is rounded so that candidates which all keep the subject
        # intact tie, and the second key decides between them. Compared raw, it
        # never ties and the mask that removes the *least* background always wins
        # - which is the opposite of what this is for.
        scored.append((round(centre, 2), -area, candidate, name))

    if not scored:
        reason = "subject would be removed" if candidates else "no segmentation backend"
        return np.ones((height, width), np.uint8), f"none ({reason})"

    # prefer the mask that keeps the centre intact while removing the most border
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    _, _, best, name = scored[0]
    return _resize_mask(best, width, height), name


# ------------------------------------------------------------- fit to size ----
def _fit_to_size(img, size, background=(255, 255, 255), mode="letterbox"):
    """Pad ("letterbox") or centre-crop ("crop") to the target aspect, then resize."""
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


def load_user_image(source, size=(60, 80), mode="letterbox", margin=0.06,
                    background=(255, 255, 255), return_info=False):
    """Read an arbitrary image into the catalogue's format.

    ``source`` may be a path, raw bytes, a file-like object or a ``PIL.Image`` -
    the API services hold request bodies as bytes, the benchmark holds arrays.

    The catalogue is 60x80 product shots on white, one item, filling roughly half
    the frame. A user upload is none of those, so it must be coerced first.

    mode
        ``"letterbox"`` pad to 3:4 and resize. Keeps everything, including the
        background, which the model then partly describes.
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

    opened = _open_source(source)
    try:
        img = _to_rgb_on_white(opened, background)
    finally:
        if opened is not source and opened is not img:
            opened.close()

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

    # No tier produced a plausible mask. Keeping the whole frame is the worst
    # available answer - it is exactly the "garment on a real background" input
    # that drops the classifier from 88% to 26% - so degrade to a centre crop,
    # which at least discards the frame edges where the background lives.
    if method.startswith("none"):
        info["fell_back"] = True
        info["method"] = f"{method} -> crop"
        result = _fit_to_size(img, size, background, mode="crop")
        info["ink_fraction"] = float((result.min(axis=2) < 235).mean())
        return (result, info) if return_info else result

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


def looks_like_catalogue(source, size=(60, 80), aspect_tolerance=0.08,
                         white_fraction=0.35, white_level=235,
                         small_ratio=2.0, photo_ratio=2.75):
    """Is this image already in catalogue form, or is it a photograph?

    The two cases want opposite handling, and the gap is large in both
    directions. Measured on held-out rows with the deployed checkpoint:

        clean catalogue tile   squash 88.32   nobg 76.90    (-11.4 for nobg)
        shift-synthesised      squash 10.15   nobg 38.07    (+27.9 for nobg)

    So no single ingestion mode can be the default. ``nobg`` re-crops tight to
    the subject and re-frames it, which is a rescue for a photograph and damage
    to a tile that was already framed the way training framed it.

    Pixel size is the primary signal, because it separates the two populations
    almost perfectly: every image in the dataset is 60x80, while the smallest of
    the 31 real uploads is 225px on its long side.

    Whiteness is only a tiebreak for the ambiguous middle, and deliberately not
    the primary test - plenty of catalogue tiles are dark garments photographed
    full-bleed, with almost no white at all (id 52003 is 0.1% near-white), and
    routing those to the photograph model on a whiteness test alone measurably
    changed their predictions for the worse.
    """
    opened = _open_source(source)
    try:
        img = _to_rgb_on_white(opened)
    finally:
        if opened is not source and opened is not img:
            opened.close()

    width, height = img.size
    target = size[0] / size[1]
    if abs(width / height - target) / target > aspect_tolerance:
        return False

    longest, model_input = max(width, height), max(size)
    if longest <= small_ratio * model_input:
        return True                      # only a dataset tile is this small
    if longest >= photo_ratio * model_input:
        return False                     # nothing in the dataset is this large
    array = np.asarray(img)
    return float((array.min(axis=2) > white_level).mean()) >= white_fraction


def list_images(folder):
    """Every readable image in a folder, sorted, ignoring other files."""
    folder = Path(folder)
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)


# --------------------------------------------------------- catalogue mattes ----
def catalogue_alpha(images, threshold=235, min_centre=0.10, max_area=0.97):
    """Subject mattes for catalogue tiles, which need no segmentation at all.

    The inverse of ``foreground_mask``: those are already cutouts on white, so a
    brightness threshold *is* the matte. Sampled over the training cache, 94.5%
    of border pixels are above the threshold and 67% of all pixels are.

    Vectorised over a whole ``(N, H, W, 3)`` uint8 stack, because the callers -
    the shift benchmark and background-randomised training - need all 38k of them.

    Returns ``(alpha, usable)``: a uint8 0/1 matte per row, and a bool flag per
    row saying whether it is trustworthy. A tile whose matte covers almost the
    whole frame (a dark product photographed edge to edge) or almost none of the
    centre would produce a nonsense composite, so callers skip those rows rather
    than corrupt them.
    """
    images = np.asarray(images)
    alpha = (images.min(axis=3) < threshold).astype(np.uint8)

    height, width = alpha.shape[1:]
    y0, y1 = int(height * 0.275), int(height * 0.725)
    x0, x1 = int(width * 0.275), int(width * 0.725)
    centre = alpha[:, y0:y1, x0:x1].mean(axis=(1, 2))
    area = alpha.mean(axis=(1, 2))
    usable = (centre >= min_centre) & (area <= max_area) & (area > 0.02)
    return alpha, usable
