from __future__ import annotations

"""
Evaluate whether Task 1 semantic predictions improve Task 4 retrieval.

Compares on the SAME queries:
  A) Task 4 cosine only
  B) Task 4 + original final Task 1 (120x160) semantic reranking
  C) Task 4 + background-adapted Task 1 (120x160) semantic reranking

Two domains:
  - clean catalogue query
  - held-out Places365 composite query (OOD proxy)

Important:
  - Task 4 encoder is NOT retrained.
  - Task 1 is used only as a semantic reranker over a cosine shortlist.
  - The semantic weight is selected on a tuning split, then reported on a disjoint test split.
  - Production artefacts are never overwritten.
"""

import argparse
import json
import random
import re
import time
from pathlib import Path
import sys

# Allow direct execution from scripts/ while importing repo packages.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.places365_backgrounds import load_background_bank
from src.data.synthetic_backgrounds import composite
from src.models.item_type_classifier import load_item_type_model
from src.visual_search.search_engine import ImprovedEncoder


SEED = 42
SOURCE_SHAPE = (160, 120, 3)  # H, W, C


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def project_root() -> Path:
    if (REPO_ROOT / "A2_FashionDataset").is_dir():
        return REPO_ROOT
    raise FileNotFoundError(f"Cannot find A2_FashionDataset under {REPO_ROOT}")


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Task4Runtime:
    def __init__(self, project: Path, dev: torch.device):
        art = project / "artifacts" / "task4"
        manifest_path = art / "search_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.meta = pd.read_csv(art / "gallery_metadata.csv").reset_index(drop=True)
        self.index = np.load(
            art / self.manifest["index_file"], mmap_mode="r"
        ).astype(np.float32, copy=False)

        if len(self.meta) != len(self.index):
            raise RuntimeError(
                f"Task4 metadata/index mismatch: {len(self.meta)} vs {len(self.index)}"
            )

        ckpt_path = art / self.manifest.get("encoder_file", "task4_improved_encoder.pt")
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)

        self.model = ImprovedEncoder(
            embedding_dim=ckpt["embedding_dim"],
            n_types=ckpt.get("n_types", self.meta["articleType"].nunique()),
            n_colours=ckpt.get("n_colours", self.meta["baseColour"].fillna("Unknown").nunique()),
        ).to(dev)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.mean = torch.tensor(
            ckpt["channel_mean"], dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            ckpt["channel_std"], dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.dev = dev
        self.use_tta = bool(ckpt.get("use_tta", self.manifest.get("use_tta", True)))

        # The stored index should already be L2-normalised, but normalise in memory
        # to make reranking independent of historical save conventions.
        norms = np.linalg.norm(self.index, axis=1, keepdims=True)
        self.index_norm = np.asarray(self.index) / np.clip(norms, 1e-8, None)

    @torch.no_grad()
    def embed(self, arrays: np.ndarray, batch_size=128) -> np.ndarray:
        out = []
        for start in range(0, len(arrays), batch_size):
            x = arrays[start:start + batch_size].astype(np.float32) / 255.0
            x = torch.from_numpy(x.transpose(0, 3, 1, 2))
            x = ((x - self.mean) / self.std).to(self.dev)

            z = self.model.embed(x)
            if self.use_tta:
                z2 = self.model.embed(torch.flip(x, dims=[3]))
                z = F.normalize(z + z2, p=2, dim=1)
            else:
                z = F.normalize(z, p=2, dim=1)

            out.append(z.float().cpu().numpy())
        return np.vstack(out)


class Task1Runtime:
    def __init__(self, checkpoint: Path, dev: torch.device):
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        self.model, self.ckpt = load_item_type_model(checkpoint, dev)
        self.model.eval()
        self.dev = dev
        self.names = list(self.ckpt["class_names"])
        self.name_to_idx = {n: i for i, n in enumerate(self.names)}
        self.mean = np.asarray(self.ckpt["channel_mean"], dtype=np.float32)
        self.std = np.asarray(self.ckpt["channel_std"], dtype=np.float32)
        self.size = tuple(self.ckpt["image_size_pil"])
        if self.size != (120, 160):
            raise RuntimeError(
                f"Expected Task1 120x160 checkpoint, got PIL size={self.size}"
            )

    @torch.no_grad()
    def probabilities(self, arrays: np.ndarray, batch_size=128) -> np.ndarray:
        # arrays are already H=160,W=120, matching PIL width=120,height=160.
        probs = []
        for start in range(0, len(arrays), batch_size):
            x = arrays[start:start + batch_size].astype(np.float32) / 255.0
            x = (x - self.mean) / self.std
            x = torch.from_numpy(x.transpose(0, 3, 1, 2)).float().to(self.dev)
            logits = self.model(x)
            probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        return np.vstack(probs)


class Cache:
    def __init__(self, project: Path):
        p = project / "A2_FashionDataset" / "processed"
        gallery = p / "task4_gallery_120x160.csv"
        images = p / "task4_cache_120x160.npy"
        masks = p / "task4_masks_120x160.npy"

        missing = [x for x in (gallery, images, masks) if not x.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing Task4 120x160 cache. Build once with:\n"
                "  python scripts/build_task4_cache.py --resolution 120x160\n"
                f"Missing: {[x.name for x in missing]}"
            )

        g = pd.read_csv(gallery).reset_index(drop=True)
        if "position" not in g.columns:
            g["position"] = np.arange(len(g))
        self.id_to_pos = {
            str(i): int(pos) for i, pos in zip(g["id"], g["position"])
        }
        self.images = np.load(images, mmap_mode="r")
        self.masks = np.load(masks, mmap_mode="r")


def stratified_query_rows(
    meta: pd.DataFrame,
    supported_types: set[str],
    cache: Cache,
    n_total: int,
    seed: int,
) -> np.ndarray:
    eligible = meta[
        meta["articleType"].isin(supported_types)
        & meta["id"].astype(str).isin(cache.id_to_pos)
    ].copy()

    rng = np.random.default_rng(seed)
    groups = []
    for _, group in eligible.groupby("articleType"):
        idx = np.array(group.index, copy=True)
        rng.shuffle(idx)
        groups.append(idx.tolist())

    # Round-robin sampling prevents common classes from dominating.
    chosen = []
    depth = 0
    while len(chosen) < n_total:
        added = 0
        for group in groups:
            if depth < len(group):
                chosen.append(group[depth])
                added += 1
                if len(chosen) == n_total:
                    break
        if added == 0:
            break
        depth += 1

    if len(chosen) < n_total:
        raise RuntimeError(
            f"Only {len(chosen)} eligible stratified queries available; requested {n_total}."
        )

    chosen = np.asarray(chosen, dtype=np.int64)
    rng.shuffle(chosen)
    return chosen


def arrays_for_rows(meta, rows, cache):
    positions = [cache.id_to_pos[str(meta.loc[r, "id"])] for r in rows]
    images = np.stack([np.asarray(cache.images[p]) for p in positions])
    masks = np.stack([np.asarray(cache.masks[p]) for p in positions])
    return images, masks


def make_ood(clean: np.ndarray, masks: np.ndarray, n_backgrounds: int, seed: int):
    backgrounds = load_background_bank(
        n_backgrounds,
        shape=SOURCE_SHAPE,
        split="test",
        seed=seed,
    )
    out = np.empty_like(clean)
    for i in range(len(clean)):
        rng = np.random.default_rng(seed + 10007 * i)
        bg = backgrounds[int(rng.integers(len(backgrounds)))]
        out[i] = composite(
            clean[i],
            masks[i],
            bg,
            rng,
            scale_range=(0.55, 1.00),
        )
    return out


def candidate_semantic_matrix(
    task1_probs: np.ndarray,
    task1: Task1Runtime,
    candidate_types: np.ndarray,
    shortlist: np.ndarray,
) -> np.ndarray:
    """
    For each query and shortlisted candidate, use P_T1(candidate.articleType | query).
    Unsupported T4 articleTypes receive zero semantic bonus.
    """
    lut = np.full(len(candidate_types), -1, dtype=np.int32)
    for i, name in enumerate(candidate_types):
        lut[i] = task1.name_to_idx.get(str(name), -1)

    out = np.zeros(shortlist.shape, dtype=np.float32)
    for qi in range(shortlist.shape[0]):
        class_idx = lut[shortlist[qi]]
        valid = class_idx >= 0
        out[qi, valid] = task1_probs[qi, class_idx[valid]]
    return out


def p_at_k(ranked_types, true_types, k=10):
    vals = []
    for pred, truth in zip(ranked_types[:, :k], true_types):
        vals.append(np.mean(pred == truth))
    return float(np.mean(vals))


def ap_at_k(ranked_types, true_types, k=10):
    vals = []
    for pred, truth in zip(ranked_types[:, :k], true_types):
        hit = (pred == truth).astype(np.float32)
        if hit.sum() == 0:
            vals.append(0.0)
            continue
        prec = np.cumsum(hit) / (np.arange(k) + 1)
        vals.append(float((prec * hit).sum() / hit.sum()))
    return float(np.mean(vals))


def get_rankings(
    query_vectors: np.ndarray,
    index_vectors: np.ndarray,
    meta: pd.DataFrame,
    query_rows: np.ndarray,
    semantic_probs: np.ndarray | None,
    task1: Task1Runtime | None,
    semantic_weight: float,
    pool: int,
    k: int,
):
    """
    Exact cosine -> shortlist -> optional Task1 semantic bonus -> top-k.
    Excludes the query image and every candidate with the same productDisplayName.
    """
    candidate_types = meta["articleType"].astype(str).to_numpy()
    product = meta["productDisplayName"].fillna(meta["id"].astype(str)).astype(str).to_numpy()

    all_ranked = []
    batch = 64

    for start in range(0, len(query_vectors), batch):
        q = query_vectors[start:start + batch]
        sims = q @ index_vectors.T

        local_rows = query_rows[start:start + len(q)]
        for j, row in enumerate(local_rows):
            # remove exact query and same product identity
            same_product = product == product[row]
            sims[j, same_product] = -np.inf
            sims[j, row] = -np.inf

        take = min(pool, len(meta))
        shortlist = np.argpartition(-sims, kth=take - 1, axis=1)[:, :take]
        short_sims = np.take_along_axis(sims, shortlist, axis=1)

        # order shortlist by cosine before optional semantic rerank
        cos_order = np.argsort(-short_sims, axis=1)
        shortlist = np.take_along_axis(shortlist, cos_order, axis=1)
        short_sims = np.take_along_axis(short_sims, cos_order, axis=1)

        if semantic_probs is not None and task1 is not None and semantic_weight > 0:
            local_probs = semantic_probs[start:start + len(q)]
            semantic = candidate_semantic_matrix(
                local_probs, task1, candidate_types, shortlist
            )
            combined = short_sims + semantic_weight * semantic
            order = np.argsort(-combined, axis=1)
            shortlist = np.take_along_axis(shortlist, order, axis=1)

        all_ranked.append(shortlist[:, :k])

    ranked = np.vstack(all_ranked)
    return candidate_types[ranked]


def evaluate(
    name,
    vectors,
    index_vectors,
    meta,
    rows,
    probs,
    task1,
    weight,
    pool,
    k,
):
    ranked_types = get_rankings(
        vectors, index_vectors, meta, rows, probs, task1, weight, pool, k
    )
    truth = meta.loc[rows, "articleType"].astype(str).to_numpy()
    return {
        "system": name,
        "semantic_weight": weight,
        f"P@{k}": p_at_k(ranked_types, truth, k),
        f"mAP@{k}": ap_at_k(ranked_types, truth, k),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-queries", type=int, default=500)
    ap.add_argument("--test-queries", type=int, default=1000)
    ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--bg-bank-size", type=int, default=800)
    ap.add_argument(
        "--task1-baseline", type=Path,
        default=Path("artifacts/task1_120x160/task1_120x160_onecycle_best.pt"),
        help="Baseline checkpoint; relative paths are resolved from the repository root.",
    )
    ap.add_argument(
        "--task1-adapted", type=Path,
        default=Path("artifacts/task1_120x160/task1_120x160_background_adapted.pt"),
        help="Adapted checkpoint (defaults to the deployed model).",
    )
    ap.add_argument(
        "--tag", default="",
        help="Optional output suffix, e.g. _rerank_experiment, to keep experiment tables separate.",
    )
    ap.add_argument(
        "--weights",
        default="0,0.05,0.10,0.15,0.20,0.30,0.40",
        help="Semantic bonus grid added to cosine score.",
    )
    args = ap.parse_args()
    if min(args.tune_queries, args.test_queries, args.k, args.bg_bank_size) < 1:
        ap.error("Query counts, --k, and --bg-bank-size must be positive.")
    if args.pool < args.k:
        ap.error("--pool must be at least --k.")
    if args.tag and not re.fullmatch(r"_[A-Za-z0-9][A-Za-z0-9_-]*", args.tag):
        ap.error("--tag must start with '_' and contain only letters, digits, '_' or '-'.")

    seed_all()
    project = project_root()
    dev = device()
    print("Device:", dev)
    if dev.type == "cuda":
        print("GPU   :", torch.cuda.get_device_name(0))

    t4 = Task4Runtime(project, dev)
    cache = Cache(project)

    base_path = project / args.task1_baseline
    robust_path = project / args.task1_adapted

    t1_base = Task1Runtime(base_path, dev)
    t1_robust = Task1Runtime(robust_path, dev)

    if t1_base.names != t1_robust.names:
        raise RuntimeError("Baseline and adapted Task1 class order differ.")

    n_total = args.tune_queries + args.test_queries
    rows = stratified_query_rows(
        t4.meta, set(t1_base.names), cache, n_total=n_total, seed=SEED
    )
    tune_rows = rows[:args.tune_queries]
    test_rows = rows[args.tune_queries:]

    print(f"Task4 gallery/index: {len(t4.meta):,} x {t4.index_norm.shape[1]}")
    print(f"Task1 classes supported: {len(t1_base.names)}")
    print(f"Tune/Test queries: {len(tune_rows)} / {len(test_rows)}")
    print(f"Rerank pool: {args.pool} | top-k: {args.k}")

    clean_all, masks_all = arrays_for_rows(t4.meta, rows, cache)
    print("Building held-out Places365 OOD query set...")
    ood_all = make_ood(clean_all, masks_all, args.bg_bank_size, seed=SEED + 909)

    print("Embedding Task4 queries...")
    z_clean = t4.embed(clean_all)
    z_ood = t4.embed(ood_all)

    print("Running Task1 probabilities...")
    p_base_clean = t1_base.probabilities(clean_all)
    p_base_ood = t1_base.probabilities(ood_all)
    p_rob_clean = t1_robust.probabilities(clean_all)
    p_rob_ood = t1_robust.probabilities(ood_all)

    nt = args.tune_queries
    weights = [float(x) for x in args.weights.split(",")]

    # Tune only the robust semantic fusion; objective is average clean + OOD P@k.
    tune_rows_out = []
    for w in weights:
        clean_result = evaluate(
            "T4+T1adapt", z_clean[:nt], t4.index_norm, t4.meta, tune_rows,
            p_rob_clean[:nt], t1_robust, w, args.pool, args.k
        )
        ood_result = evaluate(
            "T4+T1adapt", z_ood[:nt], t4.index_norm, t4.meta, tune_rows,
            p_rob_ood[:nt], t1_robust, w, args.pool, args.k
        )
        objective = 0.5 * clean_result[f"P@{args.k}"] + 0.5 * ood_result[f"P@{args.k}"]
        tune_rows_out.append({
            "weight": w,
            f"clean_P@{args.k}": clean_result[f"P@{args.k}"],
            f"ood_P@{args.k}": ood_result[f"P@{args.k}"],
            "mean_objective": objective,
        })

    tune_df = pd.DataFrame(tune_rows_out).sort_values(
        ["mean_objective", "weight"], ascending=[False, True]
    )
    best_w = float(tune_df.iloc[0]["weight"])

    print("\nTUNING GRID")
    print(tune_df.to_string(index=False))
    print(f"\nSelected semantic weight = {best_w:.2f}")

    sl = slice(nt, None)
    test_systems = []

    for domain, vecs, pb, pr in [
        ("clean", z_clean[sl], p_base_clean[sl], p_rob_clean[sl]),
        ("heldout_places365", z_ood[sl], p_base_ood[sl], p_rob_ood[sl]),
    ]:
        r0 = evaluate(
            "T4 cosine only", vecs, t4.index_norm, t4.meta, test_rows,
            None, None, 0.0, args.pool, args.k
        )
        r1 = evaluate(
            "T4 + original T1", vecs, t4.index_norm, t4.meta, test_rows,
            pb, t1_base, best_w, args.pool, args.k
        )
        r2 = evaluate(
            "T4 + BG-adapted T1", vecs, t4.index_norm, t4.meta, test_rows,
            pr, t1_robust, best_w, args.pool, args.k
        )
        for r in (r0, r1, r2):
            r["domain"] = domain
            test_systems.append(r)

    result_df = pd.DataFrame(test_systems)
    pcol = f"P@{args.k}"
    mcol = f"mAP@{args.k}"

    # Add paired system deltas against T4-only within each domain.
    result_df["delta_P"] = 0.0
    result_df["delta_mAP"] = 0.0
    for domain in result_df["domain"].unique():
        mask = result_df["domain"] == domain
        base_p = float(result_df.loc[mask & (result_df["system"] == "T4 cosine only"), pcol].iloc[0])
        base_m = float(result_df.loc[mask & (result_df["system"] == "T4 cosine only"), mcol].iloc[0])
        result_df.loc[mask, "delta_P"] = result_df.loc[mask, pcol] - base_p
        result_df.loc[mask, "delta_mAP"] = result_df.loc[mask, mcol] - base_m

    out = project / "outputs" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    tune_path = out / f"task1_t4_rerank_tuning{args.tag}.csv"
    test_path = out / f"task1_t4_rerank_test{args.tag}.csv"
    json_path = out / f"task1_t4_rerank_summary{args.tag}.json"

    tune_df.to_csv(tune_path, index=False)
    result_df.to_csv(test_path, index=False)

    summary = {
        "protocol": {
            "task1_baseline": str(base_path),
            "task1_background_adapted": str(robust_path),
            "task4_manifest_method": t4.manifest.get("best_method"),
            "tune_queries": len(tune_rows),
            "test_queries": len(test_rows),
            "query_types_limited_to_task1_supported_92_classes": True,
            "ood_proxy": "garments composited onto held-out Places365 scene categories",
            "rerank_pool": args.pool,
            "top_k": args.k,
            "selection_objective": f"0.5*clean_P@{args.k} + 0.5*OOD_P@{args.k}",
            "selected_semantic_weight": best_w,
            "same_product_candidates_excluded": True,
        },
        "results": result_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 84)
    print("FINAL DISJOINT TEST RESULTS")
    print("=" * 84)
    show = result_df.copy()
    for c in [pcol, mcol, "delta_P", "delta_mAP"]:
        show[c] = (100 * show[c]).round(2)
    print(show.to_string(index=False))

    print("\nSaved:")
    print(" ", tune_path)
    print(" ", test_path)
    print(" ", json_path)
    print("\nNo production artefact was modified.")


if __name__ == "__main__":
    main()



