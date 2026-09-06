from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.visual_search.search_engine import (
    ImprovedEncoder,
    load_user_image,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path.cwd()

DATASET_ROOT = ROOT / "A2_FashionDataset"
PROCESSED = DATASET_ROOT / "processed"

GT_PATH = ROOT / "outputs" / "task4_real_gt" / "ground_truth_template.csv"

CURRENT_PT = ROOT / "artifacts" / "task4" / "task4_improved_encoder.pt"
PLACES_PT = ROOT / "artifacts" / "task4_places365" / "task4_places365_candidate.pt"

INPUT_DIR = DATASET_ROOT / "input_images"

OUT_DIR = ROOT / "outputs" / "task4_real_gt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WIDTH = 60
HEIGHT = 80
IMAGE_SIZE = (WIDTH, HEIGHT)

TOP_K = 6
SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

gallery = pd.read_csv(
    PROCESSED / "clean_train_metadata.csv"
).reset_index(drop=True)

IMAGES = np.load(
    PROCESSED / "search_cache_60x80.npy",
    mmap_mode="r",
)

article = gallery["articleType"].to_numpy()

product_key = gallery["productDisplayName"].fillna(
    gallery["id"].astype(str)
).to_numpy()


# same Task4 product split
rng = np.random.default_rng(SEED)

unique_products = np.unique(product_key)
rng.shuffle(unique_products)

holdout_products = set(
    unique_products[
        :int(len(unique_products) * 0.15)
    ]
)

is_holdout = np.array([
    p in holdout_products
    for p in product_key
])

CATALOGUE_POS = np.where(
    ~is_holdout
)[0]

print("Device:", DEVICE)
print("Catalogue:", len(CATALOGUE_POS))


# ============================================================
# GROUND TRUTH
# ============================================================

gt = pd.read_csv(GT_PATH)

gt = gt[
    gt["include_eval"] == True
].copy()

gt = gt.reset_index(drop=True)

assert len(gt) == 21, len(gt)

print("Real GT queries:", len(gt))


# ============================================================
# MODELS
# ============================================================

def load_model(path):

    ckpt = torch.load(
        path,
        map_location=DEVICE,
    )

    model = ImprovedEncoder(
        embedding_dim=ckpt["embedding_dim"],
        n_types=ckpt["n_types"],
        n_colours=ckpt["n_colours"],
    )

    model.load_state_dict(
        ckpt["state_dict"]
    )

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


current_model, current_mean, current_std = load_model(
    CURRENT_PT
)

places_model, places_mean, places_std = load_model(
    PLACES_PT
)

print("Models loaded.")


# ============================================================
# EMBEDDING
# ============================================================

@torch.no_grad()
def embed(
    model,
    arrays,
    mean,
    std,
    batch_size=512,
):

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

        x = torch.from_numpy(
            chunk.transpose(
                0, 3, 1, 2
            )
        )

        x = (
            (x - mean) / std
        ).to(DEVICE)

        a = model.embed(x)

        b = model.embed(
            torch.flip(
                x,
                dims=[3],
            )
        )

        vectors = F.normalize(
            a + b,
            p=2,
            dim=1,
        )

        output.append(
            vectors.cpu().numpy()
        )

    result = np.vstack(output)

    return result / np.clip(
        np.linalg.norm(
            result,
            axis=1,
            keepdims=True,
        ),
        1e-8,
        None,
    )


# ============================================================
# CATALOGUE
# ============================================================

catalogue_images = np.asarray(
    IMAGES[CATALOGUE_POS]
)

print("\nEmbedding CURRENT catalogue...")

current_catalogue = embed(
    current_model,
    catalogue_images,
    current_mean,
    current_std,
)

print("Embedding PLACES catalogue...")

places_catalogue = embed(
    places_model,
    catalogue_images,
    places_mean,
    places_std,
)


# ============================================================
# PREPARE 21 REAL IMAGES
# ============================================================

current_arrays = []
places_arrays = []

paths = []

for _, row in gt.iterrows():

    path = INPUT_DIR / row["file"]

    if not path.exists():
        raise FileNotFoundError(path)

    paths.append(path)

    # production route
    current_arrays.append(
        load_user_image(
            path,
            IMAGE_SIZE,
            mode="nobg",
        )
    )

    # Places365 route
    places_arrays.append(
        load_user_image(
            path,
            IMAGE_SIZE,
            mode="letterbox",
        )
    )


current_arrays = np.stack(
    current_arrays
)

places_arrays = np.stack(
    places_arrays
)

print(
    "Prepared:",
    len(paths),
    "real images"
)


# ============================================================
# QUERY EMBEDDINGS
# ============================================================

current_q = embed(
    current_model,
    current_arrays,
    current_mean,
    current_std,
)

places_q = embed(
    places_model,
    places_arrays,
    places_mean,
    places_std,
)


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(
    query_vectors,
    catalogue_vectors,
):

    similarity = (
        torch.from_numpy(
            query_vectors
        ).to(DEVICE)
        @
        torch.from_numpy(
            catalogue_vectors
        ).to(DEVICE).T
    )

    values, indices = torch.topk(
        similarity,
        k=TOP_K,
        dim=1,
    )

    positions = CATALOGUE_POS[
        indices.cpu().numpy()
    ]

    return (
        positions,
        values.cpu().numpy(),
    )


current_pos, current_sim = retrieve(
    current_q,
    current_catalogue,
)

places_pos, places_sim = retrieve(
    places_q,
    places_catalogue,
)


# ============================================================
# SCORE
# ============================================================

def score_system(
    positions,
    name,
):

    rows = []

    for i, row in gt.iterrows():

        truth = row[
            "ground_truth_articleType"
        ]

        retrieved = article[
            positions[i]
        ]

        hits = (
            retrieved == truth
        )

        top1 = bool(
            hits[0]
        )

        top6_hit = bool(
            hits.any()
        )

        relevant_count = int(
            hits.sum()
        )

        precision6 = (
            relevant_count
            / TOP_K
        )

        rank = None

        matches = np.where(
            hits
        )[0]

        if len(matches):
            rank = int(
                matches[0] + 1
            )

        rows.append({
            "index":
                int(row["index"]),

            "file":
                row["file"],

            "ground_truth":
                truth,

            f"{name}_top1":
                retrieved[0],

            f"{name}_top1_correct":
                top1,

            f"{name}_top6_hit":
                top6_hit,

            f"{name}_correct_in_top6":
                relevant_count,

            f"{name}_P@6":
                precision6,

            f"{name}_first_correct_rank":
                rank,

            f"{name}_top6":
                " | ".join(
                    retrieved.tolist()
                ),
        })

    return pd.DataFrame(rows)


a = score_system(
    current_pos,
    "current",
)

b = score_system(
    places_pos,
    "places",
)

result = a.merge(
    b,
    on=[
        "index",
        "file",
        "ground_truth",
    ],
)


# ============================================================
# WIN / LOSS / TIE
# ============================================================

def compare_row(row):

    a_hits = row[
        "current_correct_in_top6"
    ]

    b_hits = row[
        "places_correct_in_top6"
    ]

    if b_hits > a_hits:
        return "PLACES"

    if a_hits > b_hits:
        return "CURRENT"

    # tie-break with Top-1 correctness
    a_top1 = row[
        "current_top1_correct"
    ]

    b_top1 = row[
        "places_top1_correct"
    ]

    if b_top1 and not a_top1:
        return "PLACES"

    if a_top1 and not b_top1:
        return "CURRENT"

    return "TIE"


result["winner"] = result.apply(
    compare_row,
    axis=1,
)


# ============================================================
# SUMMARY
# ============================================================

def summarise(
    df,
    prefix,
):

    top1 = (
        df[
            f"{prefix}_top1_correct"
        ].mean()
        * 100
    )

    top6_hit = (
        df[
            f"{prefix}_top6_hit"
        ].mean()
        * 100
    )

    p6 = (
        df[
            f"{prefix}_P@6"
        ].mean()
        * 100
    )

    ranks = df[
        f"{prefix}_first_correct_rank"
    ].dropna()

    mrr = (
        np.mean(
            1.0 / ranks
        )
        if len(ranks)
        else 0.0
    )

    return {
        "Top-1 Accuracy":
            top1,

        "Top-6 Hit Rate":
            top6_hit,

        "P@6":
            p6,

        "MRR":
            mrr,
    }


current_summary = summarise(
    result,
    "current",
)

places_summary = summarise(
    result,
    "places",
)


print(
    "\n=== REAL-WORLD GROUND-TRUTH TEST ==="
)

print(
    f"{'Metric':22s}"
    f"{'CURRENT+nobg':>18s}"
    f"{'PLACES+letterbox':>20s}"
)

for metric in [
    "Top-1 Accuracy",
    "Top-6 Hit Rate",
    "P@6",
    "MRR",
]:

    a_value = current_summary[
        metric
    ]

    b_value = places_summary[
        metric
    ]

    if metric == "MRR":

        print(
            f"{metric:22s}"
            f"{a_value:18.3f}"
            f"{b_value:20.3f}"
        )

    else:

        print(
            f"{metric:22s}"
            f"{a_value:17.2f}%"
            f"{b_value:19.2f}%"
        )


wins = result[
    "winner"
].value_counts()

print(
    "\n=== QUERY-LEVEL RESULT ==="
)

print(
    "PLACES wins :",
    int(
        wins.get(
            "PLACES",
            0,
        )
    )
)

print(
    "CURRENT wins:",
    int(
        wins.get(
            "CURRENT",
            0,
        )
    )
)

print(
    "Ties        :",
    int(
        wins.get(
            "TIE",
            0,
        )
    )
)


print(
    "\n=== PER-QUERY ==="
)

print(
    result[
        [
            "index",
            "ground_truth",
            "current_top1",
            "current_correct_in_top6",
            "places_top1",
            "places_correct_in_top6",
            "winner",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUT_DIR
    / "real_gt_detailed_results.csv",
    index=False,
)

pd.DataFrame([
    {
        "system":
            "CURRENT+nobg",
        **current_summary,
    },
    {
        "system":
            "PLACES+letterbox",
        **places_summary,
    },
]).to_csv(
    OUT_DIR
    / "real_gt_summary.csv",
    index=False,
)

print(
    "\nSaved:",
    OUT_DIR
    / "real_gt_summary.csv"
)
