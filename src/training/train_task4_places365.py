from pathlib import Path
from collections import defaultdict
import copy
import json
import random
import time

import numpy as np
import pandas as pd
from scipy import ndimage
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.visual_search.search_engine import ImprovedEncoder


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "A2_FashionDataset"
PROCESSED = DATASET_ROOT / "processed"
TASK4_DIR = ROOT / "artifacts" / "task4"

BG_ROOT = ROOT / "external_data" / "places365"
TRAIN_BG_LIST = BG_ROOT / "train_backgrounds.txt"
TEST_BG_LIST = BG_ROOT / "test_backgrounds.txt"

OUT_DIR = ROOT / "artifacts" / "task4_places365"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WIDTH = 60
HEIGHT = 80
IMAGE_SIZE = (WIDTH, HEIGHT)
IMAGE_SHAPE = (HEIGHT, WIDTH, 3)

N_QUERIES = 2000
TOP_K = 10

EPOCHS = 10
LR = 5e-5
BG_END = 0.60
RAMP_EPOCHS = 4
SCALE_RANGE = (0.55, 1.00)

P = 16
K = 8
BATCHES_PER_EPOCH = 250


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_all()

print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD FASHION DATA
# ============================================================

gallery = pd.read_csv(PROCESSED / "clean_train_metadata.csv").reset_index(drop=True)
gallery["position"] = np.arange(len(gallery))

CACHE_X = PROCESSED / "search_cache_60x80.npy"
CACHE_IDS = PROCESSED / "search_cache_60x80_ids.npy"

def resolve_train_image(row):
    raw = Path(str(row["image_path"]))

    if raw.exists():
        return raw

    if not raw.is_absolute():
        candidate = ROOT / raw
        if candidate.exists():
            return candidate

    item_id = str(row["id"])

    folders = [
        DATASET_ROOT / "FashionDataset" / "train" / "images_train",
        DATASET_ROOT / "train" / "images_train",
        DATASET_ROOT / "images_train",
        DATASET_ROOT / "FashionDataset" / "images_train",
    ]

    for folder in folders:
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = folder / (item_id + ext)
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"Could not resolve image for id={item_id}, stored path={raw}"
    )


def build_image_cache():
    print("Building 60x80 image cache...")

    cache = np.lib.format.open_memmap(
        CACHE_X,
        mode="w+",
        dtype=np.uint8,
        shape=(len(gallery), HEIGHT, WIDTH, 3),
    )

    for i, (_, row) in enumerate(gallery.iterrows()):
        image_path = resolve_train_image(row)

        with Image.open(image_path) as img:
            img = img.convert("RGB").resize(
                (WIDTH, HEIGHT),
                Image.BILINEAR,
            )
            cache[i] = np.asarray(img, dtype=np.uint8)

        if (i + 1) % 5000 == 0:
            print(f"  cached {i + 1}/{len(gallery)}")

    cache.flush()
    np.save(CACHE_IDS, gallery["id"].to_numpy())

    print("Image cache complete:", CACHE_X)


if not CACHE_X.exists():
    build_image_cache()

IMAGES = np.load(CACHE_X, mmap_mode="r")

if CACHE_IDS.exists():
    ids = np.load(CACHE_IDS)
    assert np.array_equal(ids, gallery["id"].to_numpy())

print("Fashion images:", IMAGES.shape)

article = gallery["articleType"].to_numpy()
colour_values = gallery["baseColour"].fillna("Unknown").to_numpy()
product_key = gallery["productDisplayName"].fillna(
    gallery["id"].astype(str)
).to_numpy()

type_codes, type_names = pd.factorize(gallery["articleType"])
colour_codes, colour_names = pd.factorize(
    gallery["baseColour"].fillna("Unknown")
)
product_codes = pd.factorize(product_key)[0]


# ============================================================
# SAME PRODUCT-LEVEL SPLIT AS EXISTING TASK4 EXPERIMENT
# ============================================================

rng = np.random.default_rng(SEED)
unique_products = np.unique(product_key)
rng.shuffle(unique_products)

holdout_products = set(
    unique_products[: int(len(unique_products) * 0.15)]
)

is_holdout = np.array([
    p in holdout_products
    for p in product_key
])

CATALOGUE_POS = np.where(~is_holdout)[0]
HELDOUT_POS = np.where(is_holdout)[0]

catalogue_types = pd.Series(
    article[CATALOGUE_POS]
).value_counts()

HELDOUT_POS = HELDOUT_POS[
    [article[p] in catalogue_types.index for p in HELDOUT_POS]
]

HELDOUT_QUERIES = np.sort(
    rng.choice(
        HELDOUT_POS,
        min(N_QUERIES, len(HELDOUT_POS)),
        replace=False,
    )
)

assert not (
    set(product_key[CATALOGUE_POS])
    & set(product_key[HELDOUT_POS])
)

print(
    "Catalogue:",
    len(CATALOGUE_POS),
    "| Held-out queries:",
    len(HELDOUT_QUERIES),
)


# ============================================================
# ITEM MASKS
# ============================================================

MASK_CACHE = PROCESSED / "item_masks_60x80.npy"


def extract_item_masks(images, tolerance=20, block=4000):
    masks = np.zeros(
        (len(images), HEIGHT, WIDTH),
        dtype=bool,
    )

    for start in range(0, len(images), block):
        chunk = np.asarray(
            images[start:start + block],
            dtype=np.int16,
        )

        border = np.concatenate(
            [
                chunk[:, 0, :, :],
                chunk[:, -1, :, :],
                chunk[:, :, 0, :],
                chunk[:, :, -1, :],
            ],
            axis=1,
        )

        bg = np.median(
            border,
            axis=1,
        )[:, None, None, :]

        masks[start:start + block] = (
            np.abs(chunk - bg).max(axis=3) > tolerance
        )

    return masks



def clean_mask(mask, min_relative_area=0.15):
    """Pure SciPy replacement for the original OpenCV mask cleanup."""
    mask = mask.astype(bool)

    structure = np.ones((3, 3), dtype=bool)

    cleaned = ndimage.binary_opening(
        mask,
        structure=structure,
        iterations=1,
    )

    cleaned = ndimage.binary_closing(
        cleaned,
        structure=structure,
        iterations=2,
    )

    labels, count = ndimage.label(cleaned)

    if count <= 0:
        return mask

    areas = np.bincount(labels.ravel())
    if len(areas) <= 1:
        return mask

    areas[0] = 0
    largest = areas.max()

    keep = np.where(
        areas > min_relative_area * largest
    )[0]

    keep = keep[keep != 0]

    if len(keep) == 0:
        return mask

    return np.isin(labels, keep)


if MASK_CACHE.exists():
    ITEM_MASKS = np.load(
        MASK_CACHE,
        mmap_mode="r",
    )
else:
    print("Building fashion item masks...")
    raw = extract_item_masks(IMAGES)
    masks = np.stack([
        clean_mask(m)
        for m in raw
    ])
    np.save(MASK_CACHE, masks)
    ITEM_MASKS = np.load(
        MASK_CACHE,
        mmap_mode="r",
    )

print("Masks:", ITEM_MASKS.shape)


# ============================================================
# PLACES365
# ============================================================

def read_bg_list(path):
    lines = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    paths = []

    for line in lines:
        p = Path(line)
        if not p.is_absolute():
            p = ROOT / p

        if p.exists():
            paths.append(p)

    return paths


TRAIN_BG_PATHS = read_bg_list(
    TRAIN_BG_LIST
)
TEST_BG_PATHS = read_bg_list(
    TEST_BG_LIST
)

print("Places train backgrounds:", len(TRAIN_BG_PATHS))
print("Places test backgrounds :", len(TEST_BG_PATHS))

assert len(TRAIN_BG_PATHS) > 0
assert len(TEST_BG_PATHS) > 0

assert not (
    set(TRAIN_BG_PATHS)
    & set(TEST_BG_PATHS)
)


def load_background(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(
            IMAGE_SIZE,
            Image.BILINEAR,
        )
        return np.asarray(
            img,
            dtype=np.uint8,
        )


# ============================================================
# COMPOSITING
# ============================================================

def composite(
    image,
    mask,
    background,
    generator,
    scale_range=SCALE_RANGE,
):
    ys, xs = np.where(mask)

    if len(ys) < 10:
        return image.copy()

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    item = image[
        y0:y1 + 1,
        x0:x1 + 1,
    ]

    item_mask = mask[
        y0:y1 + 1,
        x0:x1 + 1,
    ]

    target_height = max(
        8,
        int(
            HEIGHT
            * generator.uniform(
                *scale_range
            )
        ),
    )

    ratio = (
        item.shape[1]
        / max(item.shape[0], 1)
    )

    target_width = max(
        6,
        min(
            WIDTH,
            int(
                target_height
                * ratio
            ),
        ),
    )

    item_img = Image.fromarray(
        item
    ).resize(
        (
            target_width,
            target_height,
        ),
        Image.BILINEAR,
    )

    mask_img = Image.fromarray(
        (
            item_mask * 255
        ).astype(np.uint8)
    ).resize(
        (
            target_width,
            target_height,
        ),
        Image.NEAREST,
    )

    canvas = Image.fromarray(
        background.copy()
    )

    offset_x = int(
        generator.integers(
            0,
            max(
                1,
                WIDTH - target_width + 1,
            ),
        )
    )

    offset_y = int(
        generator.integers(
            0,
            max(
                1,
                HEIGHT - target_height + 1,
            ),
        )
    )

    canvas.paste(
        item_img,
        (offset_x, offset_y),
        mask_img,
    )

    return np.asarray(
        canvas,
        dtype=np.uint8,
    )


# ============================================================
# LOAD CURRENT PRODUCTION MODEL
# ============================================================

checkpoint = torch.load(
    TASK4_DIR / "task4_improved_encoder.pt",
    map_location=DEVICE,
)

CURRENT_MODEL = ImprovedEncoder(
    embedding_dim=checkpoint["embedding_dim"],
    n_types=checkpoint.get(
        "n_types",
        len(type_names),
    ),
    n_colours=checkpoint.get(
        "n_colours",
        len(colour_names),
    ),
)

CURRENT_MODEL.load_state_dict(
    checkpoint["state_dict"]
)

CURRENT_MODEL.to(DEVICE).eval()

CHANNEL_MEAN = np.asarray(
    checkpoint["channel_mean"],
    dtype=np.float32,
)

CHANNEL_STD = np.asarray(
    checkpoint["channel_std"],
    dtype=np.float32,
)

MEAN_T = torch.tensor(
    CHANNEL_MEAN,
    dtype=torch.float32,
).view(1, 3, 1, 1)

STD_T = torch.tensor(
    CHANNEL_STD,
    dtype=torch.float32,
).view(1, 3, 1, 1)

print(
    "Current production background_augmented:",
    checkpoint.get(
        "background_augmented"
    ),
)


# ============================================================
# DATASET / SAMPLER / LOSS
# ============================================================

class PlacesDataset(Dataset):
    def __init__(
        self,
        positions,
        types,
        colours,
        probability=0.0,
        seed=SEED,
    ):
        self.positions = np.asarray(
            positions
        )
        self.types = np.asarray(
            types
        )
        self.colours = np.asarray(
            colours
        )
        self.probability = probability
        self.generator = (
            np.random.default_rng(seed)
        )

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        position = self.positions[index]

        image = np.asarray(
            IMAGES[position]
        )

        if (
            self.generator.random()
            < self.probability
        ):
            bg_path = TRAIN_BG_PATHS[
                self.generator.integers(
                    len(TRAIN_BG_PATHS)
                )
            ]

            background = load_background(
                bg_path
            )

            image = composite(
                image,
                np.asarray(
                    ITEM_MASKS[position]
                ),
                background,
                self.generator,
            )

        tensor = torch.from_numpy(
            image.astype(
                np.float32
            ).transpose(
                2, 0, 1
            )
            / 255.0
        )

        tensor = (
            tensor - MEAN_T[0]
        ) / STD_T[0]

        return (
            tensor,
            int(self.types[index]),
            int(self.colours[index]),
        )


class PKSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        labels,
        product_ids,
        p=P,
        k=K,
        batches_per_epoch=BATCHES_PER_EPOCH,
        seed=SEED,
    ):
        self.p = p
        self.k = k
        self.batches_per_epoch = (
            batches_per_epoch
        )

        self.rng = (
            np.random.default_rng(seed)
        )

        self.by_class = defaultdict(list)

        for index, label in enumerate(
            labels
        ):
            self.by_class[label].append(
                index
            )

        self.classes = [
            c
            for c, items
            in self.by_class.items()
            if len(items) >= k
        ]

        self.product_ids = np.asarray(
            product_ids
        )

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        for _ in range(
            self.batches_per_epoch
        ):
            batch = []

            classes = self.rng.choice(
                self.classes,
                min(
                    self.p,
                    len(self.classes),
                ),
                replace=False,
            )

            for cls in classes:
                candidates = (
                    self.by_class[cls]
                )

                picked = []
                seen = set()

                for index in (
                    self.rng.permutation(
                        candidates
                    )
                ):
                    product = (
                        self.product_ids[index]
                    )

                    if product in seen:
                        continue

                    picked.append(index)
                    seen.add(product)

                    if len(picked) == self.k:
                        break

                while len(picked) < self.k:
                    picked.append(
                        int(
                            self.rng.choice(
                                candidates
                            )
                        )
                    )

                batch.extend(picked)

            yield batch


def batch_hard_triplet_loss(
    embeddings,
    labels,
    margin=0.3,
):
    distances = torch.cdist(
        embeddings,
        embeddings,
        p=2,
    )

    same = (
        labels[:, None]
        == labels[None, :]
    )

    eye = torch.eye(
        len(labels),
        dtype=torch.bool,
        device=labels.device,
    )

    positive_mask = (
        same & ~eye
    )

    negative_mask = ~same

    hardest_positive = (
        distances
        * positive_mask
    ).max(dim=1).values

    hardest_negative = (
        distances
        + (
            ~negative_mask
        ).float()
        * 1e6
    ).min(dim=1).values

    valid = (
        positive_mask.any(dim=1)
        & negative_mask.any(dim=1)
    )

    loss = F.relu(
        hardest_positive
        - hardest_negative
        + margin
    )

    return (
        loss[valid].mean()
        if valid.any()
        else loss.sum() * 0.0
    )


def augment_batch(
    x,
    jitter=0.10,
):
    do_flip = (
        torch.rand(
            x.size(0),
            device=x.device,
        )
        < 0.5
    )

    x = torch.where(
        do_flip.view(
            -1, 1, 1, 1
        ),
        torch.flip(
            x,
            dims=[3],
        ),
        x,
    )

    brightness = (
        1.0
        + (
            torch.rand(
                x.size(0),
                1,
                1,
                1,
                device=x.device,
            )
            * 2
            - 1
        )
        * jitter
    )

    contrast = (
        1.0
        + (
            torch.rand(
                x.size(0),
                1,
                1,
                1,
                device=x.device,
            )
            * 2
            - 1
        )
        * jitter
    )

    mean = x.mean(
        dim=(1, 2, 3),
        keepdim=True,
    )

    return (
        x - mean
    ) * contrast + mean * brightness


# ============================================================
# EMBEDDING
# ============================================================

@torch.no_grad()
def embed_arrays(
    model,
    arrays,
    batch_size=256,
    tta=True,
):
    model.eval()
    output = []

    for start in range(
        0,
        len(arrays),
        batch_size,
    ):
        chunk = np.asarray(
            arrays[
                start:start + batch_size
            ],
            dtype=np.float32,
        ) / 255.0

        tensor = torch.from_numpy(
            chunk.transpose(
                0, 3, 1, 2
            )
        )

        tensor = (
            (
                tensor - MEAN_T
            )
            / STD_T
        ).to(DEVICE)

        vectors = model.embed(
            tensor
        )

        if tta:
            vectors = F.normalize(
                vectors
                + model.embed(
                    torch.flip(
                        tensor,
                        dims=[3],
                    )
                ),
                p=2,
                dim=1,
            )

        output.append(
            vectors.float()
            .cpu()
            .numpy()
        )

    matrix = np.vstack(output)

    return matrix / np.clip(
        np.linalg.norm(
            matrix,
            axis=1,
            keepdims=True,
        ),
        1e-8,
        None,
    )


# ============================================================
# BUILD FIXED CLEAN + PLACES365 HARD BENCHMARK
# ============================================================

print("\nBuilding benchmark...")

clean_queries = np.stack([
    np.asarray(IMAGES[p])
    for p in HELDOUT_QUERIES
])

hard_rng = np.random.default_rng(9876)

places_hard_queries = []

for position in HELDOUT_QUERIES:
    bg_path = TEST_BG_PATHS[
        hard_rng.integers(
            len(TEST_BG_PATHS)
        )
    ]

    bg = load_background(
        bg_path
    )

    places_hard_queries.append(
        composite(
            np.asarray(
                IMAGES[position]
            ),
            np.asarray(
                ITEM_MASKS[position]
            ),
            bg,
            hard_rng,
        )
    )

places_hard_queries = np.stack(
    places_hard_queries
)

print(
    "Clean:",
    clean_queries.shape,
    "| Places-hard:",
    places_hard_queries.shape,
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    queries,
    model_name,
    benchmark,
):
    catalogue = embed_arrays(
        model,
        np.asarray(
            IMAGES[CATALOGUE_POS]
        ),
    )

    query_vectors = embed_arrays(
        model,
        queries,
    )

    similarity = (
        torch.from_numpy(
            query_vectors
        ).to(DEVICE)
        @
        torch.from_numpy(
            catalogue
        ).to(DEVICE).T
    )

    top = torch.topk(
        similarity,
        k=TOP_K,
        dim=1,
    ).indices.cpu().numpy()

    positions = CATALOGUE_POS[top]

    truth_types = article[
        HELDOUT_QUERIES
    ][:, None]

    truth_colours = colour_values[
        HELDOUT_QUERIES
    ][:, None]

    type_hit = (
        article[positions]
        == truth_types
    )

    colour_hit = (
        colour_values[positions]
        == truth_colours
    )

    return {
        "model": model_name,
        "benchmark": benchmark,
        "P@10": round(
            type_hit.mean() * 100,
            2,
        ),
        "P@1": round(
            type_hit[:, 0].mean()
            * 100,
            2,
        ),
        "colour@10": round(
            colour_hit.mean() * 100,
            2,
        ),
        "both@10": round(
            (
                type_hit
                & colour_hit
            ).mean()
            * 100,
            2,
        ),
    }


# ============================================================
# CURRENT MODEL BEFORE TRAINING
# ============================================================

print("\n=== CURRENT PRODUCTION MODEL ===")

current_rows = []

for queries, name in [
    (
        clean_queries,
        "clean",
    ),
    (
        places_hard_queries,
        "places-hard",
    ),
]:
    row = evaluate(
        CURRENT_MODEL,
        queries,
        "current",
        name,
    )

    current_rows.append(row)

    print(row)


# ============================================================
# FINE-TUNE WITH PLACES365
# ============================================================

print("\n=== FINE-TUNE WITH PLACES365 ===")

candidate = copy.deepcopy(
    CURRENT_MODEL
).to(DEVICE)

dataset = PlacesDataset(
    CATALOGUE_POS,
    type_codes[CATALOGUE_POS],
    colour_codes[CATALOGUE_POS],
)

sampler = PKSampler(
    type_codes[CATALOGUE_POS],
    product_codes[CATALOGUE_POS],
)

loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=0,
    pin_memory=(
        DEVICE.type == "cuda"
    ),
)

optimizer = torch.optim.AdamW(
    candidate.parameters(),
    lr=LR,
    weight_decay=1e-4,
)

scheduler = (
    torch.optim.lr_scheduler
    .CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )
)

history = []
best_score = -np.inf
best_state = None
best_epoch = None

started = time.time()

for epoch in range(EPOCHS):
    ramp = min(
        1.0,
        (epoch + 1)
        / RAMP_EPOCHS,
    )

    dataset.probability = (
        BG_END * ramp
    )

    candidate.train()

    total_loss = 0.0
    total_triplet = 0.0
    batches = 0

    for (
        images,
        types,
        colours,
    ) in loader:

        images = augment_batch(
            images.to(
                DEVICE,
                non_blocking=True,
            )
        )

        types = types.to(
            DEVICE,
            non_blocking=True,
        )

        colours = colours.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        (
            embeddings,
            type_logits,
            colour_logits,
        ) = candidate(
            images,
            with_heads=True,
        )

        triplet = (
            batch_hard_triplet_loss(
                embeddings,
                types,
                margin=0.3,
            )
        )

        loss = (
            triplet
            + 0.5
            * F.cross_entropy(
                type_logits,
                types,
            )
            + 0.5
            * F.cross_entropy(
                colour_logits,
                colours,
            )
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_triplet += (
            triplet.item()
        )
        batches += 1

    scheduler.step()

    print(
        f"epoch {epoch + 1:02d}/{EPOCHS} "
        f"bg={dataset.probability:.2f} "
        f"triplet={total_triplet / batches:.4f} "
        f"loss={total_loss / batches:.4f}"
    )

    # full validation every 2 epochs
    if (
        (epoch + 1) % 2 == 0
        or epoch == EPOCHS - 1
    ):
        clean_result = evaluate(
            candidate,
            clean_queries,
            "places365",
            "clean",
        )

        hard_result = evaluate(
            candidate,
            places_hard_queries,
            "places365",
            "places-hard",
        )

        score = (
            clean_result["both@10"]
            + hard_result["both@10"]
        ) / 2.0

        print(
            "  clean both@10:",
            clean_result["both@10"],
            "| places-hard both@10:",
            hard_result["both@10"],
        )

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(
                candidate.state_dict()
            )
            best_epoch = epoch + 1

    history.append({
        "epoch": epoch + 1,
        "bg_probability":
            dataset.probability,
        "triplet":
            total_triplet / batches,
        "loss":
            total_loss / batches,
    })


print(
    "\nTraining time:",
    round(
        (
            time.time()
            - started
        ) / 60,
        1,
    ),
    "minutes",
)

if best_state is not None:
    candidate.load_state_dict(
        best_state
    )

print(
    "Best epoch:",
    best_epoch,
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n=== FINAL COMPARISON ===")

rows = []

for model, model_name in [
    (
        CURRENT_MODEL,
        "current",
    ),
    (
        candidate,
        "places365",
    ),
]:
    for queries, benchmark in [
        (
            clean_queries,
            "clean",
        ),
        (
            places_hard_queries,
            "places-hard",
        ),
    ]:
        row = evaluate(
            model,
            queries,
            model_name,
            benchmark,
        )

        rows.append(row)

        print(
            f"{model_name:10s} "
            f"{benchmark:12s} "
            f"P@10={row['P@10']:6.2f} "
            f"colour@10={row['colour@10']:6.2f} "
            f"both@10={row['both@10']:6.2f}"
        )


comparison = pd.DataFrame(
    rows
)

comparison.to_csv(
    OUT_DIR
    / "places365_comparison.csv",
    index=False,
)

pd.DataFrame(
    history
).to_csv(
    OUT_DIR
    / "training_history.csv",
    index=False,
)


# ============================================================
# SAVE CANDIDATE ONLY - DO NOT TOUCH PRODUCTION
# ============================================================

torch.save(
    {
        "state_dict":
            candidate.state_dict(),
        "embedding_dim":
            checkpoint[
                "embedding_dim"
            ],
        "n_types":
            checkpoint.get(
                "n_types",
                len(type_names),
            ),
        "n_colours":
            checkpoint.get(
                "n_colours",
                len(colour_names),
            ),
        "channel_mean":
            CHANNEL_MEAN.tolist(),
        "channel_std":
            CHANNEL_STD.tolist(),
        "image_size_pil":
            [WIDTH, HEIGHT],
        "use_tta": True,
        "background_augmented": True,
        "background_source":
            "Places365",
        "starting_model":
            "current_task4_production",
        "best_epoch":
            best_epoch,
    },
    OUT_DIR
    / "task4_places365_candidate.pt",
)

summary = {
    "current_checkpoint":
        str(
            TASK4_DIR
            / "task4_improved_encoder.pt"
        ),
    "train_backgrounds":
        len(TRAIN_BG_PATHS),
    "test_backgrounds":
        len(TEST_BG_PATHS),
    "heldout_queries":
        len(HELDOUT_QUERIES),
    "best_epoch":
        best_epoch,
    "results":
        rows,
}

with open(
    OUT_DIR
    / "summary.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        summary,
        handle,
        indent=2,
    )

print(
    "\nSaved experiment to:",
    OUT_DIR,
)

print(
    "\nIMPORTANT: production Task4 "
    "has NOT been overwritten."
)
