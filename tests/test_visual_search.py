"""Task 4 artefact and engine regression tests.

Task 4 shipped with no tests at all, so a broken index surfaced only when
uvicorn started - and because ``/api/analyze`` returns 503 when *any* service
fails to load, a bad Task 4 index took the whole frontend down with it.

Two failure modes these guard against, both of which have already happened:

* the index and ``gallery_metadata.csv`` drifting out of alignment, so results
  carry the wrong item's metadata;
* a checkpoint being mistaken for something it is not. The background-augmented
  encoder was loaded as the "clean baseline" for three rounds because nothing
  asserted on ``background_augmented``.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    CatalogueIndex,
    colour_families,
    RetrievalProtocol,
    VisualSearchIndex,
    mcnemar,
    paired_bootstrap,
)
from src.data import synthetic_backgrounds  # noqa: E402
from src.data.synthetic_backgrounds import (  # noqa: E402
    DEFAULT_SEGMENT_PROBABILITY,
    EVAL_DEGRADATIONS,
    TRAINING_DEGRADATIONS,
    degrade,
    simulate_ingestion,
)
from src.visual_search.search_engine import (  # noqa: E402
    BAND_LAYOUT,
    BAND_NAMES,
    NON_WEARABLE_CATEGORIES,
    ImprovedEncoder,
)


def _noise_image(seed=0):
    """A frame with enough structure that segmentation has something to find."""
    generator = np.random.default_rng(seed)
    image = np.full((80, 60, 3), 240, dtype=np.uint8)
    image[18:62, 12:48] = generator.integers(0, 160, (44, 36, 3))
    return image

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "task4"
MANIFEST_PATH = ARTIFACT_DIR / "search_manifest.json"
CLEAN_CHECKPOINT = ARTIFACT_DIR / "task4_encoder_clean.pt"

REQUIRED_MANIFEST_KEYS = [
    "index_file",
    "encoder_file",
    "catalogue_size",
    "embedding_dim",
    "image_size_pil",
    "channel_mean",
    "channel_std",
]


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST_PATH.exists():
        pytest.skip("{} not present".format(MANIFEST_PATH))
    with open(MANIFEST_PATH) as handle:
        return json.load(handle)


# ----------------------------------------------------------- manifest ----
def test_manifest_carries_every_key_the_loaders_read(manifest):
    """SearchEngine.load and ClusterEngine.load read these with no default.

    A missing key is not a soft failure - it raises at uvicorn startup.
    """
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    assert not missing, "manifest is missing {}".format(missing)


def test_manifest_normalisation_stats_are_three_channel(manifest):
    assert len(manifest["channel_mean"]) == 3
    assert len(manifest["channel_std"]) == 3
    assert all(s > 0 for s in manifest["channel_std"]), "a zero std would divide by zero"


def test_manifest_image_size_matches_the_dataset(manifest):
    assert tuple(manifest["image_size_pil"]) == (60, 80), (
        "the catalogue is 60x80; a different size silently mis-scales every query"
    )


# -------------------------------------------------------------- index ----
def test_index_and_metadata_agree_in_length(manifest):
    """The index is positionally aligned to the metadata frame.

    If these drift, every result carries a different item's label and nothing
    raises - the search just returns confidently wrong rows.
    """
    import pandas as pd

    index_path = ARTIFACT_DIR / manifest["index_file"]
    metadata_path = ARTIFACT_DIR / "gallery_metadata.csv"
    if not (index_path.exists() and metadata_path.exists()):
        pytest.skip("index or metadata not present")

    index = np.load(index_path, mmap_mode="r")
    metadata = pd.read_csv(metadata_path)
    assert len(index) == len(metadata)
    assert index.shape[1] == manifest["embedding_dim"]
    assert len(index) == manifest["catalogue_size"]


def test_index_rows_are_unit_norm(manifest):
    """Cosine similarity via a plain matmul is only correct on unit vectors."""
    index_path = ARTIFACT_DIR / manifest["index_file"]
    if not index_path.exists():
        pytest.skip("index not present")
    index = np.load(index_path)
    norms = np.linalg.norm(index, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), (
        "index rows are not L2-normalised (min {:.4f}, max {:.4f})".format(
            norms.min(), norms.max())
    )


# --------------------------------------------------------- checkpoints ----
def test_shipped_encoder_loads_strictly(manifest):
    """A strict load is the point - a silently partial load is the bug."""
    path = ARTIFACT_DIR / manifest["encoder_file"]
    if not path.exists():
        pytest.skip("{} not present".format(path))

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    encoder = ImprovedEncoder(
        embedding_dim=checkpoint["embedding_dim"],
        widths=tuple(checkpoint.get("widths", (32, 64, 128, 256))),
        n_types=checkpoint.get("n_types", 0),
        n_colours=checkpoint.get("n_colours", 0),
    )
    encoder.load_state_dict(checkpoint["state_dict"])       # strict by default

    encoder.eval()
    with torch.no_grad():
        out = encoder.embed(torch.zeros(2, 3, 80, 60))
    assert out.shape == (2, checkpoint["embedding_dim"])
    assert torch.allclose(out.norm(dim=1), torch.ones(2), atol=1e-5), (
        "embed() must return L2-normalised vectors"
    )


def test_the_clean_baseline_is_actually_clean():
    """The regression that motivated this file.

    ``task4_encoder_clean.pt`` is the baseline every background-augmentation
    claim is measured against. If it ever carries ``background_augmented``, the
    comparison is augmented-vs-augmented and the reported delta is meaningless.
    """
    if not CLEAN_CHECKPOINT.exists():
        pytest.skip("{} not present".format(CLEAN_CHECKPOINT))
    checkpoint = torch.load(CLEAN_CHECKPOINT, map_location="cpu", weights_only=False)
    assert not checkpoint.get("background_augmented", False)


def test_clean_and_deployed_checkpoints_are_not_the_same_weights(manifest):
    """Guards the in-place overwrite that collapsed the two files into one."""
    deployed = ARTIFACT_DIR / manifest["encoder_file"]
    if not (CLEAN_CHECKPOINT.exists() and deployed.exists()):
        pytest.skip("both checkpoints not present")
    if CLEAN_CHECKPOINT.resolve() == deployed.resolve():
        pytest.skip("this build deploys the clean encoder directly")

    a = torch.load(CLEAN_CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"]
    b = torch.load(deployed, map_location="cpu", weights_only=False)["state_dict"]
    identical = all(torch.equal(a[k], b[k]) for k in a) if a.keys() == b.keys() else False
    assert not identical, (
        "the clean baseline and the deployed encoder hold identical weights - "
        "the augmented run has overwritten its own baseline again"
    )


# ------------------------------------------------------------ protocol ----
@pytest.fixture(scope="module")
def toy_gallery():
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 400
    types = rng.choice(["Tshirts", "Watches", "Heels", "Socks"], n, p=[.4, .3, .2, .1])
    sub = {"Tshirts": "Topwear", "Watches": "Watches", "Heels": "Shoes", "Socks": "Socks"}
    master = {"Tshirts": "Apparel", "Watches": "Accessories",
              "Heels": "Footwear", "Socks": "Apparel"}
    return pd.DataFrame({
        "id": np.arange(n),
        "articleType": types,
        "subCategory": [sub[t] for t in types],
        "masterCategory": [master[t] for t in types],
        "baseColour": rng.choice(["Black", "Blue"], n),
        "productDisplayName": ["product-{}".format(i // 4) for i in range(n)],
    })


def test_split_never_lets_a_product_straddle_it(toy_gallery):
    """Two photos of one item on opposite sides would leak the answer."""
    protocol = RetrievalProtocol(toy_gallery, n_queries=60, max_queries_per_class=20)
    catalogue = set(protocol.product_key[protocol.catalogue_pos])
    heldout = set(protocol.product_key[protocol.heldout_pos])
    assert not (catalogue & heldout)


def test_protocol_is_deterministic(toy_gallery):
    a = RetrievalProtocol(toy_gallery, n_queries=60)
    b = RetrievalProtocol(toy_gallery, n_queries=60)
    assert np.array_equal(a.heldout_queries, b.heldout_queries)
    assert np.array_equal(a.query_positions, b.query_positions)


def test_stratified_sampling_reaches_rare_classes(toy_gallery):
    """Plain rng.choice is dominated by whichever types are most common."""
    stratified = RetrievalProtocol(toy_gallery, n_queries=60, stratified=True,
                                   max_queries_per_class=15)
    counts = np.unique(stratified.article[stratified.query_positions],
                       return_counts=True)[1]
    assert counts.max() <= 15


def test_a_perfect_encoder_scores_perfectly(toy_gallery):
    """Sanity anchor: if this drifts, the metrics are wrong, not the model."""
    onehot = np.eye(4)[
        [["Tshirts", "Watches", "Heels", "Socks"].index(t)
         for t in toy_gallery["articleType"]]
    ].astype(np.float32)

    protocol = RetrievalProtocol(toy_gallery, n_queries=60)
    index = CatalogueIndex(onehot, protocol, name="perfect")
    summary, _ = protocol.evaluate_deployment(index, onehot)
    assert summary["P@10"] == pytest.approx(1.0)
    assert summary["nDCG@10"] == pytest.approx(1.0)


def test_catalogue_index_never_returns_a_held_out_position(toy_gallery):
    protocol = RetrievalProtocol(toy_gallery, n_queries=60)
    embeddings = np.random.default_rng(1).normal(size=(len(toy_gallery), 8)).astype(np.float32)
    index = CatalogueIndex(embeddings, protocol, name="random")
    _, ranked = index.search(embeddings[protocol.heldout_queries], k=10)
    assert not set(ranked.ravel().tolist()) & set(protocol.heldout_pos.tolist())


# -------------------------------------------------------- significance ----
def test_a_model_compared_with_itself_is_never_significant():
    """The property the old unpaired binomial floor could not provide."""
    scores = np.random.default_rng(0).random(500)
    result = paired_bootstrap(scores, scores)
    assert result["delta"] == pytest.approx(0.0)
    assert not result["significant"]


def test_a_real_difference_is_detected():
    rng = np.random.default_rng(0)
    a = rng.random(1000)
    b = a + 0.05
    result = paired_bootstrap(a, b)
    assert result["significant"]
    assert result["ci_low"] > 0


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(10), np.zeros(11))


def test_mcnemar_ignores_agreements():
    """Only discordant pairs carry information."""
    a = [1, 1, 0, 0, 1]
    assert mcnemar(a, a)["p_value"] == 1.0


def test_visual_search_index_is_normalised_on_construction():
    raw = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    index = VisualSearchIndex(raw)
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0)


# ------------------------------------------------------- architecture v2 ----
def test_v2_embedding_is_unit_norm_and_the_expected_width():
    from src.visual_search.search_engine import ImprovedEncoderV2

    model = ImprovedEncoderV2(embedding_dim=128, n_types=10, n_colours=5).eval()
    with torch.no_grad():
        z = model.embed(torch.randn(4, 3, 80, 60))
    assert z.shape == (4, 128)
    assert torch.allclose(z.norm(dim=1), torch.ones(4), atol=1e-5)


def test_v2_keeps_the_index_the_same_size_as_v1():
    """The colour branch is carved out of the 128 dims, not added to them.

    If this ever grows, every stored index and the served artefact size change
    with it.
    """
    from src.visual_search.search_engine import ImprovedEncoderV2

    model = ImprovedEncoderV2(embedding_dim=128, colour_dim=32)
    assert model.deep_dim + model.colour_dim == 128


def test_v2_colour_branch_receives_gradient():
    """The whole point of the second branch - if it is starved, it is decoration."""
    from src.visual_search.search_engine import ImprovedEncoderV2

    model = ImprovedEncoderV2(embedding_dim=128, n_types=10, n_colours=5)
    _, metric, type_logits, colour_logits = model(
        torch.randn(4, 3, 80, 60), with_heads=True,
        target_type=torch.tensor([0, 1, 2, 3]), target_colour=torch.tensor([0, 1, 2, 3]))
    (type_logits.sum() + colour_logits.sum() + metric.sum()).backward()
    assert model.colour_project.weight.grad is not None
    assert model.colour_project.weight.grad.abs().sum() > 0


def test_cosine_head_scale_is_learnable_and_margin_lowers_the_target():
    """``ImprovedEncoder``'s heads had neither, which is why they barely trained."""
    from src.visual_search.search_engine import CosineHead

    head = CosineHead(16, 4, scale=30.0, margin=0.3)
    assert head.log_scale.requires_grad
    x = torch.randn(4, 16)
    target = torch.tensor([0, 1, 2, 3])
    with torch.no_grad():
        plain = head(x)
        margined = head(x, target)
    picked = torch.arange(4)
    assert torch.all(margined[picked, target] < plain[picked, target])


def test_build_encoder_defaults_to_the_original_architecture():
    """Checkpoints written before the field existed must keep loading."""
    from src.visual_search.search_engine import ImprovedEncoder, build_encoder

    encoder = build_encoder({"embedding_dim": 128, "n_types": 4, "n_colours": 2})
    assert isinstance(encoder, ImprovedEncoder)


def test_build_encoder_round_trips_v2():
    from src.visual_search.search_engine import ImprovedEncoderV2, build_encoder

    model = ImprovedEncoderV2(embedding_dim=128, n_types=10, n_colours=5).eval()
    checkpoint = {"state_dict": model.state_dict(), "architecture": "improved_v2",
                  "embedding_dim": 128, "n_types": 10, "n_colours": 5,
                  "colour_dim": 32, "colour_block": 2, "pool": "gem", "bnneck": True}
    rebuilt = build_encoder(checkpoint)
    rebuilt.load_state_dict(checkpoint["state_dict"])       # strict
    rebuilt.eval()
    x = torch.randn(2, 3, 80, 60)
    with torch.no_grad():
        assert torch.allclose(rebuilt.embed(x), model.embed(x), atol=1e-6)


def test_build_encoder_rejects_an_unknown_architecture():
    from src.visual_search.search_engine import build_encoder

    with pytest.raises(ValueError, match="Unknown Task 4 architecture"):
        build_encoder({"architecture": "not-a-real-encoder", "embedding_dim": 128})


# ------------------------------------------------------- result shaping ----
def _toy_engine(n=40):
    """A SearchEngine built on hand-made vectors - no artefacts required."""
    import pandas as pd

    from src.visual_search.search_engine import SearchEngine

    rng = np.random.default_rng(0)
    index = rng.normal(size=(n, 8)).astype(np.float32)
    index /= np.linalg.norm(index, axis=1, keepdims=True)
    metadata = pd.DataFrame({
        "id": np.arange(n),
        "articleType": ["Tshirts"] * (n // 2) + ["Watches"] * (n - n // 2),
        # every item appears four times under one product name
        "productDisplayName": ["product-{}".format(i // 4) for i in range(n)],
    })
    manifest = {"channel_mean": [0.5] * 3, "channel_std": [0.25] * 3,
                "image_size_pil": [60, 80], "use_tta": False}
    return SearchEngine(index, metadata, manifest, encoder=None, device="cpu")


def test_dedupe_returns_distinct_products():
    """Ten photographs of one item is a technically perfect, useless result."""
    engine = _toy_engine()
    positions = np.arange(12)
    scores = np.linspace(1.0, 0.5, 12)
    kept_positions, kept_scores = engine._dedupe(positions, scores)
    names = engine.metadata["productDisplayName"].to_numpy()[kept_positions]
    assert len(set(names)) == len(names)
    assert np.all(np.diff(kept_scores) <= 0), "dedupe must preserve score order"


def test_diversity_off_by_default_preserves_pure_relevance_order():
    engine = _toy_engine()
    positions = np.arange(10)
    scores = np.linspace(1.0, 0.1, 10)
    got_positions, _ = engine._diversify(positions, scores, k=5, diversity=0.0)
    assert list(got_positions) == list(positions[:5])


def test_diversity_changes_the_selection():
    engine = _toy_engine()
    positions = np.arange(20)
    scores = np.linspace(1.0, 0.2, 20)
    plain, _ = engine._diversify(positions, scores, k=5, diversity=0.0)
    diverse, _ = engine._diversify(positions, scores, k=5, diversity=0.7)
    assert diverse[0] == plain[0], "the top hit stays the top hit"
    assert list(diverse) != list(plain)


def test_confidence_flags_an_incoherent_neighbourhood():
    import pandas as pd

    engine = _toy_engine()
    tight = pd.DataFrame({"articleType": ["Tshirts"] * 6})
    scattered = pd.DataFrame(
        {"articleType": ["Tshirts", "Watches", "Heels", "Bra", "Socks", "Belts"]})

    from src.visual_search.search_engine import DEFAULT_CONFIDENCE

    good = engine._confidence(tight, np.full(6, 0.9), DEFAULT_CONFIDENCE)
    bad = engine._confidence(scattered, np.full(6, 0.62), DEFAULT_CONFIDENCE)
    assert good["confident"] and good["coherence"] == pytest.approx(1.0)
    assert not bad["confident"]


# ------------------------------------------------------- region proposal ----
def test_region_proposal_needs_no_opencv():
    """The backend requirements have no opencv; importing it would 503 the API.

    The notebook version of this code imports cv2 directly, so the port must
    stay on scipy. Checked with the AST rather than a text search, so a mention
    of cv2 in a comment does not trip it.
    """
    import ast as _ast
    import inspect

    from src.visual_search import search_engine

    tree = _ast.parse(inspect.getsource(search_engine))
    imported = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, _ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "cv2" not in imported, "search_engine must not depend on OpenCV"


def test_region_proposal_always_returns_the_whole_subject_first():
    """Rank-by-similarity compares every band against the whole image, so the
    whole image must be region zero."""
    from src.visual_search.search_engine import propose_regions

    rgb = np.full((200, 150, 3), 255, dtype=np.uint8)
    rgb[40:160, 40:110] = (30, 90, 160)             # a solid subject on white
    regions, mask, method = propose_regions(rgb)
    assert regions[0]["name"] == "whole"
    assert mask.shape == rgb.shape[:2]


def test_region_proposal_rejects_degenerate_boxes():
    """Thin strips become smears at 60x80 and match whatever cluster is nearby."""
    from src.visual_search.search_engine import (MAX_ASPECT, MIN_ASPECT,
                                                 MIN_CROP_PIXELS, propose_regions)

    rgb = np.full((200, 150, 3), 255, dtype=np.uint8)
    rgb[40:160, 40:110] = (30, 90, 160)
    regions, _, _ = propose_regions(rgb)
    for region in regions:
        x0, y0, x1, y1 = region["bbox"]
        width, height = x1 - x0, y1 - y0
        assert width * height >= MIN_CROP_PIXELS
        assert MIN_ASPECT <= width / max(height, 1) <= MAX_ASPECT


def test_crop_region_puts_the_subject_on_white():
    from src.visual_search.search_engine import crop_region

    rgb = np.full((100, 80, 3), 40, dtype=np.uint8)
    mask = np.zeros((100, 80), dtype=bool)
    mask[30:70, 20:60] = True
    tile = crop_region(rgb, mask, (20, 30, 60, 70), (60, 80))
    assert tile.shape == (80, 60, 3)
    assert tile.max() == 255, "the area outside the mask must be white"


# ---------------------------------------------------- classical features ----
def test_classical_descriptor_has_the_documented_layout():
    """236 dims, colour first - the fusion code slices on that assumption."""
    from src.features.image_features import CLASSICAL_COLOUR_DIMS, classical_features

    rng = np.random.default_rng(0)
    batch = rng.integers(0, 256, (4, 80, 60, 3), dtype=np.uint8)
    features = classical_features(batch)
    assert features.shape == (4, 236)
    assert CLASSICAL_COLOUR_DIMS == 128
    assert np.allclose(features[:, :CLASSICAL_COLOUR_DIMS].sum(axis=1), 1.0, atol=1e-5)


def test_colour_histogram_separates_two_flat_colours():
    """The point of keeping a hand-built descriptor: it cannot ignore colour."""
    from src.features.image_features import colour_histogram

    red = np.zeros((1, 20, 20, 3), dtype=np.uint8); red[..., 0] = 220
    blue = np.zeros((1, 20, 20, 3), dtype=np.uint8); blue[..., 2] = 220
    a, b = colour_histogram(red)[0], colour_histogram(blue)[0]
    assert float(a @ b) < 0.01, "flat red and flat blue must not share bins"


def test_colour_histogram_is_invariant_to_a_shift_of_the_subject():
    """A global histogram has no spatial term - that is the trade it makes."""
    from src.features.image_features import colour_histogram

    left = np.full((1, 40, 40, 3), 255, dtype=np.uint8)
    left[:, 10:30, 2:22] = (200, 40, 40)
    right = np.full((1, 40, 40, 3), 255, dtype=np.uint8)
    right[:, 10:30, 18:38] = (200, 40, 40)
    assert np.allclose(colour_histogram(left), colour_histogram(right), atol=1e-6)


def test_gradient_histogram_is_not_invariant_to_that_shift():
    """The grid is what puts layout back in, so this must differ."""
    from src.features.image_features import gradient_histogram

    left = np.full((1, 40, 40, 3), 255, dtype=np.uint8)
    left[:, 10:30, 2:22] = 40
    right = np.full((1, 40, 40, 3), 255, dtype=np.uint8)
    right[:, 10:30, 18:38] = 40
    assert not np.allclose(gradient_histogram(left), gradient_histogram(right), atol=1e-3)


def test_fuse_weight_zero_is_the_learned_embedding():
    from src.features.embeddings import fuse, l2

    rng = np.random.default_rng(0)
    learned = rng.normal(size=(6, 8)).astype(np.float32)
    colour = rng.random((6, 4)).astype(np.float32)
    fused = fuse(learned, colour, weight=0.0)
    assert fused.shape == (6, 12)
    assert np.allclose(fused[:, :8], l2(learned), atol=1e-6)
    assert np.allclose(fused[:, 8:], 0.0)


def test_fuse_rejects_a_weight_outside_the_unit_interval():
    from src.features.embeddings import fuse

    with pytest.raises(ValueError):
        fuse(np.zeros((2, 4)), np.zeros((2, 2)), weight=1.5)


def test_rerank_at_alpha_one_preserves_the_base_ranking(toy_gallery):
    """alpha=1 means colour is ignored, so it must match the plain index."""
    from src.evaluation.metrics import CatalogueIndex
    from src.features.embeddings import RerankIndex, fuse

    rng = np.random.default_rng(0)
    protocol = RetrievalProtocol(toy_gallery, n_queries=40)
    learned = rng.normal(size=(len(toy_gallery), 8)).astype(np.float32)
    colour = rng.random((len(toy_gallery), 4)).astype(np.float32)

    plain = CatalogueIndex(learned, protocol, name="plain")
    rerank = RerankIndex(learned, colour, alpha=1.0, protocol=protocol)

    queries = protocol.heldout_queries[:10]
    _, base_positions = plain.search(learned[queries], k=5)
    _, rerank_positions = rerank.search(fuse(learned, colour, 0.5)[queries], k=5)
    assert np.array_equal(base_positions, rerank_positions)


def test_pool_is_wide_enough_to_survive_dedup():
    """A grid asking for k must come back with k.

    The first full pass over the 5,829 test images returned fewer than k for 115
    of them - one with only 4 - because a 5k candidate pool was exhausted by
    catalogue items carrying a dozen photos under one productDisplayName.
    """
    engine = _toy_engine(n=60)
    k = 10
    pool = min(max(k * 12, 120), len(engine.index))
    # every item in the toy engine repeats four times under one product name,
    # so the pool must be at least 4k for dedup to still yield k
    assert pool >= 4 * k

    positions = np.arange(len(engine.index))
    scores = np.linspace(1.0, 0.1, len(engine.index))
    kept, _ = engine._dedupe(positions, scores)
    assert len(kept) >= k, "dedup left fewer than k distinct products"


# ------------------------------------------------- deployment index scope ----
def test_the_served_index_covers_the_whole_catalogue(manifest):
    """A shopper must be able to find anything the shop stocks.

    The 15% product-level holdout exists so evaluation queries are unseen. It is
    a property of the measurement, not of the catalogue. Building the *served*
    index from the evaluation split once left 5,775 items - one in seven -
    permanently unreachable, against a task that asks for retrieval over the
    full ~38,000.
    """
    import pandas as pd

    cleaned = (PROJECT_ROOT / "A2_FashionDataset" / "processed"
               / "clean_train_metadata.csv")
    metadata_path = ARTIFACT_DIR / "gallery_metadata.csv"
    if not (cleaned.exists() and metadata_path.exists()):
        pytest.skip("catalogue metadata not present")

    catalogue = pd.read_csv(cleaned)
    served = pd.read_csv(metadata_path)
    unreachable = set(catalogue["id"]) - set(served["id"])
    assert not unreachable, (
        "{} catalogue items cannot be retrieved by the served index".format(
            len(unreachable))
    )
    assert manifest["catalogue_size"] == len(catalogue)


def test_the_evaluation_index_is_kept_separate_and_smaller(manifest):
    """Metrics must stay computed against an index that excludes the queries.

    Serving the full catalogue is correct; measuring against it would not be, so
    the two indexes are distinct files and the manifest names both.
    """
    evaluation = manifest.get("evaluation_index_file")
    if evaluation is None:
        pytest.skip("no separate evaluation index recorded")
    assert evaluation != manifest["index_file"], (
        "the evaluation and deployment indexes must not be the same file"
    )
    assert manifest["evaluation_catalogue_size"] < manifest["catalogue_size"]

    path = ARTIFACT_DIR / evaluation
    if path.exists():
        index = np.load(path, mmap_mode="r")
        assert len(index) == manifest["evaluation_catalogue_size"]


# ------------------------------------------- photographic degradation ----
# Background augmentation took hard-benchmark P@10 from 12.2 to 60.6, and the
# residual named in notebook 06 is the flat-lay-versus-worn gap. These guard the
# second augmentation axis - what a camera does to a photograph - and above all
# the train/eval separation, without which a gain is measured on the model's own
# augmentation and means nothing.

def test_training_and_evaluation_degradations_are_disjoint():
    """The same discipline the background banks already follow.

    A model graded on a corruption its training loop generates is graded on
    itself. ``EVAL_FAMILIES`` and ``TRAINING_FAMILIES`` established this; the
    degradation families must not quietly break it.
    """
    assert not set(TRAINING_DEGRADATIONS) & set(EVAL_DEGRADATIONS)
    assert set(TRAINING_DEGRADATIONS) | set(EVAL_DEGRADATIONS) == set(
        synthetic_backgrounds._DEGRADATION_ORDER)


def test_degrade_is_deterministic_under_a_fixed_seed():
    """Two runs of a reported number must produce that number twice."""
    image = _noise_image()
    first = degrade(image, np.random.default_rng(11), TRAINING_DEGRADATIONS)
    second = degrade(image, np.random.default_rng(11), TRAINING_DEGRADATIONS)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, image), "nothing was applied"


def test_degrade_at_zero_strength_is_the_identity():
    """The ramp starts at zero, so zero must mean untouched.

    Notebook 06 ramps the background probability from 0 rather than starting at
    full strength, because a from-scratch run at full strength collapsed. The
    degradation strength is ramped the same way and needs the same property.
    """
    image = _noise_image()
    unchanged = degrade(image, np.random.default_rng(3), TRAINING_DEGRADATIONS,
                        strength=0.0)
    assert np.array_equal(unchanged, image)


def test_degrade_does_not_depend_on_the_order_families_are_listed():
    """Corruptions apply in photographic order, not in call order."""
    image = _noise_image()
    forward = degrade(image, np.random.default_rng(5), TRAINING_DEGRADATIONS)
    reversed_ = degrade(image, np.random.default_rng(5),
                        tuple(reversed(TRAINING_DEGRADATIONS)))
    assert np.array_equal(forward, reversed_)


def test_degrade_rejects_an_unknown_family():
    """A typo in a family name must fail loudly, not silently do nothing."""
    with pytest.raises(ValueError, match="Unknown degradation families"):
        degrade(_noise_image(), np.random.default_rng(0), ("rotate", "sepia"))


def test_rotation_fills_the_corners_with_the_backdrop_not_black():
    """Black corners would be a perfect rotation detector.

    A constant saturated wedge appears in no real photograph, so an encoder
    would learn to spot the augmentation rather than the invariance it is meant
    to teach.
    """
    image = np.full((80, 60, 3), 200, dtype=np.uint8)
    rotated = degrade(image, np.random.default_rng(2), ("rotate",),
                      probability=1.0)
    corners = np.array([rotated[0, 0], rotated[0, -1],
                        rotated[-1, 0], rotated[-1, -1]])
    assert corners.min() > 150, "corners were filled with something dark"


# ------------------------------------------------ serve-path simulation ----

def test_simulate_ingestion_never_returns_an_empty_frame():
    """Deleting the product is the failure mode this whole path guards against."""
    generator = np.random.default_rng(4)
    for seed in range(12):
        frame = simulate_ingestion(_noise_image(seed), generator)
        assert frame.shape == (80, 60, 3)
        assert frame.std() > 1.0, "segmentation returned a blank frame"


def test_simulate_ingestion_declines_rather_than_deleting_the_product():
    """No plausible mask means hand the image back, exactly as production does.

    ``load_user_image`` falls back to a centre crop rather than returning a white
    rectangle; a uniform frame has nothing to segment and must survive intact.
    """
    flat = np.full((80, 60, 3), 128, dtype=np.uint8)
    result, info = simulate_ingestion(flat, np.random.default_rng(0),
                                      p_segment=1.0, return_info=True)
    assert not info["segmented"]
    assert np.array_equal(result, flat)


def test_simulate_ingestion_follows_the_measured_fallback_rate():
    """The 62/38 split is measured, not chosen.

    ``outputs/task4_ingestion_fallback.csv``: 3,636 images segmented against
    2,193 that fell back to a centre crop.
    """
    assert DEFAULT_SEGMENT_PROBABILITY == pytest.approx(3636 / (3636 + 2193))

    generator = np.random.default_rng(0)
    attempted = sum(
        generator.random() < DEFAULT_SEGMENT_PROBABILITY for _ in range(4000))
    assert 0.58 < attempted / 4000 < 0.66


# ------------------------------------------------- published metrics ----

def test_the_manifest_records_an_out_of_domain_number_that_is_not_circular(manifest):
    """``hard_metrics`` was measured on the encoder's own augmentation.

    The manifest's provenance note says so itself: those figures predate the
    disjoint evaluation bank and grade the model against the background families
    its training loop generates. A separate, honestly-measured number has to
    exist for anything to publish.
    """
    disjoint = manifest.get("hard_metrics_disjoint")
    if disjoint is None:
        pytest.skip("no disjoint measurement recorded yet")
    assert "P@10" in disjoint
    assert disjoint.get("source"), "an out-of-domain number needs its provenance"
    assert disjoint["P@10"] > manifest["clean_metrics"]["P@10"] * 0.5, (
        "an encoder scoring below half its clean P@10 out of domain has collapsed"
    )


def test_the_model_card_publishes_the_disjoint_number(manifest):
    """The card used to print 52.8 - the circular figure - as the honest one."""
    if not manifest.get("hard_metrics_disjoint"):
        pytest.skip("no disjoint measurement recorded yet")
    service = pytest.importorskip(
        "app.backend.services.task4_service", reason="backend not importable")

    card = service.Task4Service().model_card()
    expected = manifest["hard_metrics_disjoint"]["P@10"] * 100
    assert "{:.1f}".format(expected) in card["note"]
    stale = manifest.get("hard_metrics", {}).get("P@10")
    if stale and abs(stale * 100 - expected) > 0.05:
        assert "{:.1f}".format(stale * 100) not in card["note"]


# ------------------------------------------------------ colour families ----
# colour@10 (~54) trails P@10 (~80) by 26 points. Part of that is an exact
# string match over 46 labels rather than a perceptual failure: 42% of the
# catalogue carries a colour with a same-family sibling. These pin the mapping
# to a lexical rule so it cannot quietly grow into "whatever makes the number
# better" - measured, it recovers 3.04 points, leaving 23 that are real.

def test_colour_families_merge_only_lexical_variants():
    """A multi-word colour joins the single-word colour it contains."""
    mapping = colour_families([
        "Blue", "Navy Blue", "Turquoise Blue", "White", "Off White",
        "Grey", "Grey Melange", "Teal", "Olive", "Green", "Sea Green"])
    assert mapping["Navy Blue"] == "Blue"
    assert mapping["Turquoise Blue"] == "Blue"
    assert mapping["Off White"] == "White"
    assert mapping["Grey Melange"] == "Grey"
    assert mapping["Sea Green"] == "Green"
    assert mapping["Blue"] == "Blue"


def test_colour_families_make_no_perceptual_claims():
    """Teal is not Blue and Olive is not Green under this rule.

    Folding them in would be a judgement about what counts as the same colour,
    which is exactly what this mapping must not do - it removes a naming
    artefact, it does not make the metric easier.
    """
    mapping = colour_families(["Blue", "Teal", "Green", "Olive", "Silver",
                               "Gold", "Bronze", "Copper", "Metallic"])
    assert mapping["Teal"] == "Teal"
    assert mapping["Olive"] == "Olive"
    for metallic in ("Silver", "Gold", "Bronze", "Copper"):
        assert mapping[metallic] == metallic


def test_colour_families_never_merge_two_single_word_colours():
    """The rule can only ever collapse a compound onto a base, never base onto base."""
    gallery_path = ARTIFACT_DIR / "gallery_metadata.csv"
    if not gallery_path.exists():
        pytest.skip("gallery metadata not present")
    import pandas as pd

    values = pd.read_csv(gallery_path)["baseColour"].dropna().unique()
    mapping = colour_families(values)
    for label, family in mapping.items():
        if " " not in label:
            assert family == label, "{} was merged into {}".format(label, family)
    assert len(set(mapping.values())) < len(set(mapping))


# ------------------------------------------------------- band plausibility ----

def test_band_names_cover_the_layout_and_exclude_whole():
    """The prior must apply to bands only, never to the whole image."""
    assert BAND_NAMES == {name for name, _, _ in BAND_LAYOUT}
    assert "whole" not in BAND_NAMES


def test_the_wearable_mask_excludes_only_unwearable_categories():
    """A band of a worn outfit is never a perfume or a cushion cover."""
    metadata = pd.DataFrame({
        "masterCategory": ["Apparel", "Footwear", "Accessories",
                           "Personal Care", "Home", "Sporting Goods", "Free Items"],
    })
    wearable = ~metadata["masterCategory"].isin(NON_WEARABLE_CATEGORIES).to_numpy()
    assert list(wearable) == [True, True, True, False, False, False, False]


def test_bands_do_not_return_unwearable_items():
    """Regression for the measured failure.

    ``600_google-pattern-socks.jpg`` returned "Perfume and Body Mist" at 0.741
    from its top third - accepted, and outscoring every real garment in a
    photograph of socks. Two of 36 accepted band matches were of this kind.
    """
    upload = (PROJECT_ROOT / "A2_FashionDataset" / "input_images"
              / "600_google-pattern-socks.jpg")
    if not upload.exists() or not MANIFEST_PATH.exists():
        pytest.skip("upload or Task 4 artefacts not present")

    from src.visual_search.search_engine import SearchEngine

    results = SearchEngine.load().search_regions(upload, k=5)
    bands = results[results["region"].isin(BAND_NAMES)]
    assert len(bands), "no band regions were proposed"
    offenders = bands[bands["masterCategory"].isin(NON_WEARABLE_CATEGORIES)]
    assert offenders.empty, "bands returned {}".format(
        sorted(offenders["articleType"].unique()))

    # The whole image keeps the unfiltered catalogue: a photograph of a perfume
    # bottle must still be able to find perfume.
    whole = results[results["region"] == "whole"]
    assert len(whole), "the whole-image region disappeared"


# --------------------------------------------------- V2 warm start ----
# The colour branch was costed as 9 runs x 30 epochs from scratch and shelved.
# The two models share their convolutional stack exactly, so it is a fine-tune.

def test_v2_warm_start_transfers_the_whole_backbone():
    """Every ConvBlock tensor must cross; a silent partial load would waste a run."""
    from src.visual_search.search_engine import ImprovedEncoderV2

    source = ImprovedEncoder(embedding_dim=128, n_types=124, n_colours=47)
    target = ImprovedEncoderV2(embedding_dim=128, n_types=124, n_colours=47)

    loaded, skipped = target.warm_start(source.state_dict())
    backbone = [k for k in source.state_dict() if k.startswith("backbone.")]
    assert len(backbone) == 48
    assert all(k.replace("backbone.", "blocks.", 1) in loaded for k in backbone)

    # project changes shape (256->96 against 256->128) and cannot transfer.
    assert "project.weight" in skipped and "project.bias" in skipped


def test_v2_warm_start_actually_copies_the_weights():
    """load_state_dict succeeding is not evidence the values arrived."""
    from src.visual_search.search_engine import ImprovedEncoderV2

    source = ImprovedEncoder(embedding_dim=128, n_types=124, n_colours=47)
    target = ImprovedEncoderV2(embedding_dim=128, n_types=124, n_colours=47)
    target.warm_start(source.state_dict())

    key = "backbone.0.block.0.weight"
    assert torch.equal(source.state_dict()[key],
                       target.state_dict()[key.replace("backbone.", "blocks.", 1)])


def test_v2_still_embeds_after_a_warm_start():
    """A warm-started model has to be usable, not merely loadable."""
    from src.visual_search.search_engine import ImprovedEncoderV2

    target = ImprovedEncoderV2(embedding_dim=128, n_types=124, n_colours=47)
    target.warm_start(
        ImprovedEncoder(embedding_dim=128, n_types=124, n_colours=47).state_dict())
    target.eval()

    vectors = target.embed(torch.randn(4, 3, 80, 60))
    assert vectors.shape == (4, 128)
    assert torch.allclose(vectors.norm(dim=1), torch.ones(4), atol=1e-5)


def test_v2_warm_start_from_the_shipped_clean_encoder():
    """The real checkpoint, not a synthetic one - shapes there are what matter."""
    if not CLEAN_CHECKPOINT.exists():
        pytest.skip("clean encoder not present")
    from src.visual_search.search_engine import ImprovedEncoderV2

    checkpoint = torch.load(CLEAN_CHECKPOINT, map_location="cpu")
    assert not checkpoint.get("background_augmented", False), (
        "the warm start must begin from the clean encoder, not an augmented one"
    )
    target = ImprovedEncoderV2(
        embedding_dim=checkpoint["embedding_dim"],
        n_types=checkpoint.get("n_types", 124),
        n_colours=checkpoint.get("n_colours", 47))

    loaded, _ = target.warm_start(checkpoint["state_dict"])
    carried = sum(target.state_dict()[k].numel() for k in loaded)
    total = sum(p.numel() for p in target.parameters())
    assert carried / total > 0.95, "only {:.1%} of parameters transferred".format(
        carried / total)


# ------------------------------------------------------- resolution ----
# The 120x160 catalogue lifts the ceiling every limitations section in this
# project ends at. These guard the two things that make the comparison a
# comparison: the data layer must not assume 60x80, and both arms must be built
# from one gallery.

@pytest.mark.parametrize("shape", [(80, 60, 3), (160, 120, 3)])
def test_the_augmentation_pipeline_follows_the_frame(shape):
    """composite, degrade and simulate_ingestion take size from their input."""
    from src.data.synthetic_backgrounds import (
        TRAINING_DEGRADATIONS, composite, degrade, make_backgrounds,
        make_eval_backgrounds, simulate_ingestion)

    height, width, _ = shape
    generator = np.random.default_rng(0)
    image = np.full(shape, 240, dtype=np.uint8)
    image[height // 4:3 * height // 4, width // 4:3 * width // 4] = 60
    mask = np.zeros(shape[:2], dtype=bool)
    mask[height // 4:3 * height // 4, width // 4:3 * width // 4] = True

    backgrounds = make_backgrounds(4, shape=shape, seed=1)
    assert backgrounds.shape[1:] == shape
    assert make_eval_backgrounds(4, size=(width, height)).shape[1:] == shape

    composited = composite(image, mask, backgrounds[0], generator)
    degraded = degrade(composited, generator, TRAINING_DEGRADATIONS)
    ingested = simulate_ingestion(degraded, generator)
    for stage in (composited, degraded, ingested):
        assert stage.shape == shape


def test_mask_morphology_scales_with_the_frame():
    """A 3x3 opening removes proportionally less of a taller frame.

    Left unscaled, the same item would keep visibly more mask at 120x160 than at
    60x80 and the two arms would not be segmenting alike.
    """
    from src.data.synthetic_backgrounds import _scaled_structure

    _, small = _scaled_structure(80)
    _, large = _scaled_structure(160)
    assert small == 1 and large == 2


def test_region_minimum_is_a_fraction_not_a_pixel_count():
    """At 120x160 a fixed 48x48 floor would accept crops four times thinner."""
    from src.visual_search.search_engine import MIN_CROP_FRACTION

    assert MIN_CROP_FRACTION == pytest.approx((48 * 48) / (60 * 80))
    assert round(MIN_CROP_FRACTION * 120 * 160) == 48 * 48 * 4


def test_the_two_resolution_caches_describe_the_same_gallery():
    """Position-for-position alignment is what lets the arms be compared."""
    processed = PROJECT_ROOT / "A2_FashionDataset" / "processed"
    paths = {r: (processed / f"task4_cache_{r}_ids.npy",
                 processed / f"task4_gallery_{r}.csv") for r in ("60x80", "120x160")}
    if not all(p.exists() for pair in paths.values() for p in pair):
        pytest.skip("resolution caches not built")

    import pandas as pd

    ids = {r: np.load(p[0]) for r, p in paths.items()}
    assert np.array_equal(ids["60x80"], ids["120x160"]), (
        "the two caches are not in the same row order"
    )
    for resolution, (_, gallery_path) in paths.items():
        gallery = pd.read_csv(gallery_path)
        assert np.array_equal(gallery["id"].to_numpy(), ids[resolution])


def test_the_task4_gallery_drops_the_unusable_rows():
    """The 41 conflicting-label rows must not reach training at either size."""
    processed = PROJECT_ROOT / "A2_FashionDataset" / "processed"
    supervised = processed / "train_metadata_120x160_supervised.csv"
    gallery_path = processed / "task4_gallery_120x160.csv"
    if not (supervised.exists() and gallery_path.exists()):
        pytest.skip("120x160 pipeline not present")

    import pandas as pd

    flags = pd.read_csv(supervised)
    excluded = set(flags.loc[~flags["use_for_supervised"].astype(bool), "id"])
    gallery = pd.read_csv(gallery_path)
    assert len(excluded) == 41
    assert not (set(gallery["id"]) & excluded)


def test_channel_statistics_survive_a_large_sample():
    """float32 accumulation silently destroyed these at 120x160.

    4,000 images at 120x160 is 76.8 million values per channel. A float32
    running sum passes its 24-bit mantissa near 16.7 million, after which adding
    0.85 changes nothing, and the reported mean for a catalogue of
    white-background product shots came back as 0.2185. The 60x80 arm sat just
    under the limit and looked fine, so a smaller test would not have caught it.
    """
    from src.training.task4_training import channel_statistics

    # Bright and constant, so the true answer is known exactly and any
    # accumulation error is unmissable.
    images = np.full((4000, 160, 120, 3), 217, dtype=np.uint8)
    mean, std = channel_statistics(images, np.arange(len(images)), sample=4000)

    assert np.allclose(mean, 217 / 255, atol=1e-4), mean
    assert np.allclose(std, 0.0, atol=1e-4), std


def test_channel_statistics_use_only_the_positions_given():
    """Held-out pixels must not reach the normalisation every model uses."""
    from src.training.task4_training import channel_statistics

    images = np.zeros((200, 16, 12, 3), dtype=np.uint8)
    images[:100] = 255                       # "catalogue"
    images[100:] = 0                         # "held out"
    mean, _ = channel_statistics(images, np.arange(100), sample=100)
    assert np.allclose(mean, 1.0, atol=1e-6)
