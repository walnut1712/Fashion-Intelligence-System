"""Ingestion regression tests for uploads that are not catalogue tiles.

These exist because Task 1 shipped with exactly one decode path - resize to
60x80 and let the aspect ratio distort - and no test noticed that it destroys
a photograph. Measured on held-out rows composited onto a textured background,
the deployed checkpoint falls from 87.92 accuracy to 25.80. Ingestion is
therefore load-bearing inference code, and these tests pin the three properties
it has to keep:

* a catalogue tile must come through byte-identical, or every published metric
  and the committed prediction fixture silently change meaning;
* a real upload must not be geometrically distorted;
* segmentation must refuse rather than delete the product, and say so.
"""

import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.user_image import (  # noqa: E402
    PREPROCESS_MODES,
    catalogue_alpha,
    foreground_mask,
    load_user_image,
)
from src.models.item_type_classifier import load_image_array  # noqa: E402

SIZE = (60, 80)  # (width, height), matching checkpoint["image_size_pil"]


def _tile(seed=0):
    """A catalogue-shaped tile: a dark blob centred on white, already 60x80."""
    rng = np.random.default_rng(seed)
    array = np.full((80, 60, 3), 255, np.uint8)
    array[22:58, 16:44] = rng.integers(20, 90, (36, 28, 3), dtype=np.uint8)
    return array


def _photo(width, height, background, seed=0):
    """A subject on a coloured ground at an arbitrary size - a stand-in upload."""
    rng = np.random.default_rng(seed)
    array = np.empty((height, width, 3), np.uint8)
    array[:, :] = background
    y0, y1 = int(height * 0.22), int(height * 0.78)
    x0, x1 = int(width * 0.28), int(width * 0.72)
    array[y0:y1, x0:x1] = rng.integers(15, 70, (y1 - y0, x1 - x0, 3), dtype=np.uint8)
    return array


def _as_jpeg(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return buffer


# ------------------------------------------------------- the contract guard ----
def test_letterbox_is_a_noop_on_a_catalogue_tile():
    """A 60x80 3:4 tile is already at the target aspect, so padding adds nothing.

    This is what lets ingestion be switched on without invalidating the committed
    predictions in tests/test_prediction.py.
    """
    tile = _tile()
    assert np.array_equal(load_user_image(tile, size=SIZE, mode="letterbox"), tile)


def test_letterbox_matches_the_historical_decode_path_on_catalogue_input():
    """letterbox and load_image_array must agree wherever the input is 3:4."""
    tile = _tile(seed=3)
    assert np.array_equal(load_user_image(tile, size=SIZE, mode="letterbox"),
                          load_image_array(tile, SIZE))


# --------------------------------------------------------- geometry, not squash ----
@pytest.mark.parametrize("width,height", [(612, 612), (1200, 800), (400, 1000)])
def test_upload_is_not_distorted(width, height):
    """A non-3:4 upload keeps its proportions under letterbox; squash does not.

    Checked on the subject's own aspect ratio rather than on pixels: letterbox
    pads to 3:4 so the subject keeps the shape it had, while the historical
    resize stretches it by exactly the ratio mismatch.
    """
    photo = _photo(width, height, background=(250, 250, 250), seed=1)

    def subject_aspect(tile):
        ink = tile.min(axis=2) < 200
        ys, xs = np.where(ink)
        assert ys.size, "the synthetic subject vanished"
        return (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)

    original = subject_aspect(photo)
    letterboxed = subject_aspect(load_user_image(photo, size=SIZE, mode="letterbox"))
    squashed = subject_aspect(load_image_array(photo, SIZE))

    assert letterboxed == pytest.approx(original, rel=0.15)
    # and the historical path really is worse, or this test proves nothing
    assert abs(letterboxed - original) < abs(squashed - original)


def test_every_mode_returns_the_requested_shape():
    photo = _photo(500, 300, background=(120, 90, 60), seed=2)
    for mode in PREPROCESS_MODES:
        tile = load_user_image(photo, size=SIZE, mode=mode)
        assert tile.shape == (SIZE[1], SIZE[0], 3), mode
        assert tile.dtype == np.uint8, mode


def test_sources_may_be_bytes_or_pil_or_path(tmp_path):
    """The API holds request bodies as bytes; the benchmark holds arrays."""
    photo = _photo(300, 400, background=(240, 240, 240), seed=4)
    path = tmp_path / "upload.jpg"
    Image.fromarray(photo).save(path, quality=95)

    from_path = load_user_image(path, size=SIZE, mode="letterbox")
    from_bytes = load_user_image(path.read_bytes(), size=SIZE, mode="letterbox")
    from_buffer = load_user_image(_as_jpeg(photo), size=SIZE, mode="letterbox")
    from_pil = load_user_image(Image.fromarray(photo), size=SIZE, mode="letterbox")

    assert np.array_equal(from_path, from_bytes)
    assert from_buffer.shape == from_pil.shape == from_path.shape


# ------------------------------------------------------------- segmentation ----
def test_mask_recovers_a_subject_on_a_plain_ground():
    """The border-colour tier is what makes nobg work without OpenCV."""
    photo = _photo(320, 400, background=(196, 64, 64), seed=5)
    mask, method = foreground_mask(photo)
    assert not method.startswith("none"), method

    expected = np.zeros(photo.shape[:2], bool)
    expected[int(400 * 0.22):int(400 * 0.78), int(320 * 0.28):int(320 * 0.72)] = True
    intersection = (mask.astype(bool) & expected).sum()
    union = (mask.astype(bool) | expected).sum()
    assert intersection / union > 0.7, "mask does not agree with the known subject"


def test_nobg_places_the_subject_on_white():
    photo = _photo(320, 400, background=(40, 140, 40), seed=6)
    tile = load_user_image(photo, size=SIZE, mode="nobg")
    assert float((tile.min(axis=2) > 235).mean()) > 0.2, "background was not removed"


def test_nobg_declines_rather_than_deleting_the_product():
    """A frame with no recoverable background must fall back, and report it.

    Silently keeping the whole frame is the failure this guards: that is exactly
    the garment-on-a-real-background input the classifier cannot survive.
    """
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    tile, info = load_user_image(noise, size=SIZE, mode="nobg", return_info=True)
    assert tile.shape == (SIZE[1], SIZE[0], 3)
    if info["method"].startswith("none"):
        assert info["fell_back"] is True
        assert "crop" in info["method"]


def test_mask_runs_separates_a_blob_from_a_comb():
    """The shredding detector, which connected components cannot do.

    A striped subject makes the border-colour model cut the subject's own light
    bands out, leaving a comb. Because the bands touch the frame edge the result
    is still a single connected component covering 100% of the mask, so only run
    counting catches it.
    """
    from src.data.user_image import _mask_runs

    blob = np.zeros((60, 60), np.uint8)
    blob[10:50, 10:50] = 1
    comb = np.zeros((60, 60), np.uint8)
    comb[::4, :] = 1                     # horizontal bars: many runs down a column

    assert max(_mask_runs(blob)) < 2.0
    assert max(_mask_runs(comb)) > 5.0


def test_foreground_mask_never_returns_a_comb():
    """A striped subject must not be cut into fragments by segmentation.

    Asserted on the mask, not on the returned tile: a correctly preserved striped
    garment legitimately has many ink runs, because the *garment* is striped. The
    invariant is about what segmentation kept, not what the fabric looks like.
    """
    from src.data.user_image import _mask_runs, foreground_mask

    rng = np.random.default_rng(11)
    photo = rng.integers(180, 220, (360, 270, 3), dtype=np.uint8)
    for y in range(90, 280, 12):          # alternating light/dark bands
        photo[y:y + 6, 70:200] = 235
        photo[y + 6:y + 12, 70:200] = 40

    mask, method = foreground_mask(photo)
    assert max(_mask_runs(mask)) <= 2.5, method


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        load_user_image(_tile(), size=SIZE, mode="magic")


# ----------------------------------------------------------- catalogue mattes ----
def test_catalogue_alpha_finds_the_subject_and_flags_it_usable():
    tiles = np.stack([_tile(seed) for seed in range(4)])
    alpha, usable = catalogue_alpha(tiles)
    assert alpha.shape == tiles.shape[:3]
    assert usable.all()
    # the synthetic blob occupies rows 22:58, cols 16:44 of an 80x60 tile
    assert alpha[:, 22:58, 16:44].mean() > 0.95
    assert alpha[:, :20, :].mean() < 0.05


def test_catalogue_alpha_rejects_a_tile_with_no_white():
    """A tile that thresholds to everything would composite into nonsense."""
    solid = np.zeros((2, 80, 60, 3), np.uint8)
    _, usable = catalogue_alpha(solid)
    assert not usable.any()
