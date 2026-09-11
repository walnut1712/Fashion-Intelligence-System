"""The graded path must never reach the segmentation ladder.

The assignment forbids pretrained systems trained on other datasets. Every model
that produces a prediction here is trained from scratch on the supplied data, but
``src/data/user_image.foreground_mask`` has a top tier that is not: ``rembg``
(u2netp) is a pretrained matting network. It is optional, the ladder degrades to
GrabCut without it, and as of 2026-09-11 it is installed on this machine and
listed in both requirements files.

That makes "the graded path never reaches the ladder" a property worth asserting
rather than assuming, because it did not always hold. ``looks_like_catalogue``
applied its aspect-ratio gate before its size test, and the graded set is not
perfectly uniform - 6 of the 5,829 tiles are 60x77, 60x76, 60x75, 60x60 or 53x80.
Four of them (52166, 56624, 59593, 59606) failed the gate, were classified as
photographs, and were routed to ``nobg``, which segments. With rembg installed
that put a pretrained network in the graded path for those four tiles, and it
also silently disagreed with the committed predictions: regenerating the
submission would have changed 52166 from ``Unisex`` to ``Men`` and 56624 from
``Women`` to ``Men``.

The fix was to make the size test absolute and run it first - a dataset tile is
at most 100px on its long side, while the smallest real upload in
``input_images/`` is 225px - so the aspect gate only ever sees images too large
to be tiles. These tests pin both directions of that, because a rule that called
everything a catalogue tile would pass the first test and break the app.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GRADED_DIR = (PROJECT_ROOT / "A2_FashionDataset" / "FashionDataset" / "test"
              / "images_test")
PHOTO_DIR = PROJECT_ROOT / "A2_FashionDataset" / "input_images"

# The four that used to fail, kept by name so a regression names itself.
PREVIOUSLY_MISROUTED = ["52166.jpg", "56624.jpg", "59593.jpg", "59606.jpg"]


@pytest.mark.skipif(not GRADED_DIR.exists(), reason="graded images not present")
def test_no_graded_tile_routes_to_the_photograph_branch():
    """All 5,829 graded tiles take the catalogue branch, which never segments."""
    from src.data.user_image import list_images, looks_like_catalogue

    tiles = list_images(GRADED_DIR)
    assert len(tiles) > 5000, "graded set looks incomplete: %d" % len(tiles)

    routed = [p.name for p in tiles if not looks_like_catalogue(p)]
    assert not routed, (
        "%d graded tile(s) route to the photograph branch and would be passed to "
        "foreground_mask, whose top tier is pretrained: %s"
        % (len(routed), ", ".join(sorted(routed)[:10])))


@pytest.mark.skipif(not GRADED_DIR.exists(), reason="graded images not present")
@pytest.mark.parametrize("name", PREVIOUSLY_MISROUTED)
def test_the_off_aspect_graded_tiles_are_catalogue(name):
    """The specific tiles the aspect gate used to reject."""
    from src.data.user_image import looks_like_catalogue

    path = GRADED_DIR / name
    if not path.exists():
        pytest.skip("%s not present" % name)
    assert looks_like_catalogue(path), (
        "%s is a catalogue tile with a non-standard aspect ratio; routing it to "
        "the photograph branch segments it and changes its graded prediction"
        % name)


@pytest.mark.skipif(not PHOTO_DIR.exists(), reason="input_images not present")
def test_real_photographs_still_route_to_the_photograph_branch():
    """The other direction, which a too-permissive tile test would break.

    Two of the 31 are 225x225. A size test written as a multiple of the model
    input (2.0 x 160 = 320px) swallows them, so the ceiling is absolute.
    """
    from src.data.user_image import list_images, looks_like_catalogue

    photos = list_images(PHOTO_DIR)
    assert photos, "no photographs found"

    routed = [p.name for p in photos if looks_like_catalogue(p)]
    assert not routed, (
        "%d real photograph(s) route to the catalogue branch and would be fed to "
        "the model un-segmented: %s" % (len(routed), ", ".join(sorted(routed))))
