
from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.visual_search.search_engine import (
    ImprovedEncoder,
    load_user_image,
    list_images,
)

ROOT = Path.cwd()

DATASET_ROOT = ROOT / "A2_FashionDataset"
PROCESSED = DATASET_ROOT / "processed"

CURRENT_PT = ROOT / "artifacts" / "task4" / "task4_improved_encoder.pt"
PLACES_PT = ROOT / "artifacts" / "task4_places365" / "task4_places365_candidate.pt"

INPUT_DIR = DATASET_ROOT / "input_images"

OUTPUT_DIR = ROOT / "outputs" / "task4_real_ab"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WIDTH = 60
HEIGHT = 80
IMAGE_SIZE_PIL = (WIDTH, HEIGHT)

TOP_K = 6
SEED = 42


# ============================================================
# DATA
# ============================================================

gallery = pd.read_csv(
    PROCESSED / "clean_train_metadata.csv"
).reset_index(drop=True)

IMAGES = np.load(
    PROCESSED / "search_cache_60x80.npy",
    mmap_mode="r",
)

article = gallery["articleType"].to_numpy()
colour = gallery["baseColour"].fillna("Unknown").to_numpy()

product_key = gallery["productDisplayName"].fillna(
    gallery["id"].astype(str)
).to_numpy()


# exact same Task4 split
rng = np.random.default_rng(SEED)

unique_products = np.unique(product_key)
rng.shuffle(unique_products)

holdout_products = set(
    unique_products[:int(len(unique_products) * 0.15)]
)

is_holdout = np.array([
    p in holdout_products for p in product_key
])

CATALOGUE_POS = np.where(~is_holdout)[0]

print("Device:", DEVICE)
print("Catalogue:", len(CATALOGUE_POS))


# ============================================================
# MODEL
# ============================================================

def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)

    model = ImprovedEncoder(
        embedding_dim=ckpt["embedding_dim"],
        n_types=ckpt["n_types"],
        n_colours=ckpt["n_colours"],
    )

    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()

    mean = torch.tensor(
        ckpt["channel_mean"],
        dtype=torch.float32,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        ckpt["channel_std"],
        dtype=torch.float32,
    ).view(1, 3, 1, 1)

    return model, mean, std


current_model, current_mean, current_std = load_model(CURRENT_PT)
places_model, places_mean, places_std = load_model(PLACES_PT)

print("Models loaded.")


# ============================================================
# EMBEDDING
# ============================================================

@torch.no_grad()
def embed(model, arrays, mean, std, batch_size=512):
    model.eval()

    output = []

    for start in range(0, len(arrays), batch_size):
        chunk = np.asarray(
            arrays[start:start + batch_size],
            dtype=np.float32,
        ) / 255.0

        x = torch.from_numpy(
            chunk.transpose(0, 3, 1, 2)
        )

        x = ((x - mean) / std).to(DEVICE)

        v1 = model.embed(x)
        v2 = model.embed(torch.flip(x, dims=[3]))

        v = F.normalize(
            v1 + v2,
            p=2,
            dim=1,
        )

        output.append(v.cpu().numpy())

    result = np.vstack(output)

    return result / np.clip(
        np.linalg.norm(result, axis=1, keepdims=True),
        1e-8,
        None,
    )


# ============================================================
# USER IMAGES
# ============================================================

user_paths = list_images(INPUT_DIR)

if not user_paths:
    raise RuntimeError(
        f"No images found in {INPUT_DIR}"
    )

print("Real user images:", len(user_paths))
print("Preprocessing: NOBG (background removed)")

user_images = np.stack([
    load_user_image(
        path,
        IMAGE_SIZE_PIL,
        mode="nobg",
    )
    for path in user_paths
])


# ============================================================
# RETRIEVAL
# ============================================================

catalogue_images = np.asarray(
    IMAGES[CATALOGUE_POS]
)

print("\nEmbedding catalogue with CURRENT model...")
current_catalogue = embed(
    current_model,
    catalogue_images,
    current_mean,
    current_std,
)

print("Embedding catalogue with PLACES365 model...")
places_catalogue = embed(
    places_model,
    catalogue_images,
    places_mean,
    places_std,
)

print("Embedding user images...")

current_query = embed(
    current_model,
    user_images,
    current_mean,
    current_std,
)

places_query = embed(
    places_model,
    user_images,
    places_mean,
    places_std,
)


def retrieve(query_vectors, catalogue_vectors):
    q = torch.from_numpy(query_vectors).to(DEVICE)
    c = torch.from_numpy(catalogue_vectors).to(DEVICE)

    scores = q @ c.T

    values, indices = torch.topk(
        scores,
        k=TOP_K,
        dim=1,
    )

    positions = CATALOGUE_POS[
        indices.cpu().numpy()
    ]

    return positions, values.cpu().numpy()


current_pos, current_scores = retrieve(
    current_query,
    current_catalogue,
)

places_pos, places_scores = retrieve(
    places_query,
    places_catalogue,
)


# ============================================================
# TABLE
# ============================================================

rows = []

for i, path in enumerate(user_paths):

    cur_types = article[current_pos[i]]
    new_types = article[places_pos[i]]

    cur_colours = colour[current_pos[i]]
    new_colours = colour[places_pos[i]]

    cur_coherence = float(
        (cur_types == cur_types[0]).mean()
    )

    new_coherence = float(
        (new_types == new_types[0]).mean()
    )

    rows.append({
        "file": path.name,

        "current_top":
            cur_types[0],

        "current_colour":
            cur_colours[0],

        "current_similarity":
            round(float(current_scores[i, 0]), 3),

        "current_coherence":
            round(cur_coherence, 2),

        "places_top":
            new_types[0],

        "places_colour":
            new_colours[0],

        "places_similarity":
            round(float(places_scores[i, 0]), 3),

        "places_coherence":
            round(new_coherence, 2),

        "same_top":
            bool(cur_types[0] == new_types[0]),
    })


comparison = pd.DataFrame(rows)

comparison.to_csv(
    OUTPUT_DIR / "comparison.csv",
    index=False,
)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

print("\n=== REAL IMAGE A/B COMPARISON ===")
print(comparison.to_string(index=False))

print("\nCurrent mean coherence:",
      round(comparison["current_coherence"].mean(), 3))

print("Places mean coherence:",
      round(comparison["places_coherence"].mean(), 3))

print(
    "Top-1 changed on:",
    int((~comparison["same_top"]).sum()),
    "/",
    len(comparison),
    "images",
)


# ============================================================
# VISUAL A/B
# ============================================================

print("\nGenerating visual comparisons...")

for i, path in enumerate(user_paths):

    fig, axes = plt.subplots(
        2,
        TOP_K + 1,
        figsize=(14, 5),
    )

    # CURRENT
    axes[0, 0].imshow(user_images[i])
    axes[0, 0].set_title(
        "QUERY\n" + path.name[:18],
        fontsize=7,
    )

    for rank in range(TOP_K):
        p = current_pos[i, rank]

        axes[0, rank + 1].imshow(IMAGES[p])
        axes[0, rank + 1].set_title(
            f"OLD #{rank+1}\n"
            f"{article[p]}\n"
            f"{colour[p]}",
            fontsize=7,
        )

    # PLACES365
    axes[1, 0].imshow(user_images[i])
    axes[1, 0].set_title(
        "QUERY\nPlaces365",
        fontsize=7,
    )

    for rank in range(TOP_K):
        p = places_pos[i, rank]

        axes[1, rank + 1].imshow(IMAGES[p])
        axes[1, rank + 1].set_title(
            f"NEW #{rank+1}\n"
            f"{article[p]}\n"
            f"{colour[p]}",
            fontsize=7,
        )

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        "Current model (top) vs Places365 model (bottom)",
        fontsize=11,
    )

    plt.tight_layout()

    safe_name = (
        f"{i:02d}_"
        + "".join(
            c if c.isalnum() else "_"
            for c in path.stem
        )[:30]
        + ".png"
    )

    plt.savefig(
        OUTPUT_DIR / safe_name,
        dpi=140,
        bbox_inches="tight",
    )

    plt.close(fig)


print("\nDONE")
print("CSV:", OUTPUT_DIR / "comparison.csv")
print("Visuals:", OUTPUT_DIR)
