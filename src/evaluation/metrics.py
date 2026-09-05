"""Task 4 retrieval evaluation - the single protocol every notebook uses.

Before this module existed, notebooks ``05`` and ``06`` each carried their own
copy of the product-level split, the query draw and the retrieval metrics. The
two implementations happened to agree - same holdout fraction, same 2,000
queries, same ``rng.choice`` - but nothing enforced that, and they were free to
drift apart on any edit. They had already diverged in what they measured: ``05``
reports graded relevance, nDCG against an achievable ideal, and R-precision,
while ``06`` computed only P@10, colour@10 and both@10.

``05`` additionally samples an *in-gallery* query set stratified by
``articleType`` (capped at 40 per class, so rare types are actually measured).
That protocol is separate from the deployment one and ``06`` never used it.

A note for anyone comparing the two notebooks' headline figures: the familiar
``both@10`` 45.92 and 43.04 are **two different encoders**, not two protocols -
the clean baseline and the background-augmented one, which score 45.90 and 43.05
respectively when both are run through this module. See
``outputs/task4_disjoint_benchmark.csv``.

Everything here is deterministic given ``seed`` and the gallery frame.

Usage
-----
>>> protocol = RetrievalProtocol(gallery)
>>> index = CatalogueIndex(embeddings, protocol, name="Improved+TTA")
>>> summary, ranked = protocol.evaluate_deployment(index, embeddings)
>>> protocol.compare(summary_a, summary_b, "both@10")     # paired, with a CI
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:                                            # torch is optional
    import torch
except ImportError:                             # pragma: no cover
    torch = None

__all__ = [
    "DEFAULT_K_VALUES",
    "RetrievalProtocol",
    "VisualSearchIndex",
    "CatalogueIndex",
    "paired_bootstrap",
    "bootstrap_mean",
    "mcnemar",
    "evaluate_real_photos",
    "colour_families",
]

DEFAULT_K_VALUES = (1, 5, 10, 20)


# --------------------------------------------------------------- indexes ----
class VisualSearchIndex:
    """Exact cosine-similarity search over a matrix of embeddings."""

    def __init__(self, embeddings, name="index", device=None):
        vectors = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / np.clip(norms, 1e-8, None)
        self.name = name
        self.device = device
        self._tensor = None
        if torch is not None:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._tensor = torch.from_numpy(self.vectors).to(self.device)

    def __len__(self):
        return len(self.vectors)

    @property
    def dim(self):
        return self.vectors.shape[1]

    def search(self, query_vectors, k=10, exclude=None, batch_size=512):
        """Return (scores, positions) per query, most similar first.

        ``exclude`` is a per-query array of positions to suppress - the query
        itself, and in strict mode every other photo of the same product.
        """
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        queries = queries / np.clip(np.linalg.norm(queries, axis=1, keepdims=True),
                                    1e-8, None)

        k = min(k, len(self.vectors))
        all_scores, all_positions = [], []
        for start in range(0, len(queries), batch_size):
            chunk = queries[start:start + batch_size]
            if self._tensor is not None:
                similarity = torch.from_numpy(chunk).to(self.device) @ self._tensor.T
                if exclude is not None:
                    for row, positions in enumerate(exclude[start:start + batch_size]):
                        if len(positions):
                            similarity[row, torch.as_tensor(np.asarray(positions),
                                                            device=self.device)] = -2.0
                scores, indices = torch.topk(similarity, k=k, dim=1)
                scores, indices = scores.cpu().numpy(), indices.cpu().numpy()
            else:                                             # pragma: no cover
                similarity = chunk @ self.vectors.T
                if exclude is not None:
                    for row, positions in enumerate(exclude[start:start + batch_size]):
                        if len(positions):
                            similarity[row, np.asarray(positions)] = -2.0
                indices = np.argsort(-similarity, axis=1)[:, :k]
                scores = np.take_along_axis(similarity, indices, axis=1)
            all_scores.append(scores)
            all_positions.append(indices)

        return np.concatenate(all_scores), np.concatenate(all_positions)


class CatalogueIndex(VisualSearchIndex):
    """Searches only the catalogue split but returns GALLERY positions.

    This is the deployment shape: the user's photo is not in the index, so the
    query cannot retrieve itself or a sibling photo of the same product.
    """

    def __init__(self, embeddings, protocol, name="index", catalogue=None, device=None):
        self.catalogue = protocol.catalogue_pos if catalogue is None else catalogue
        super().__init__(np.asarray(embeddings)[self.catalogue], name=name, device=device)

    def search(self, query_vectors, k=10, exclude=None, batch_size=512):
        scores, local = super().search(query_vectors, k=k, exclude=None,
                                       batch_size=batch_size)
        return scores, self.catalogue[local]


# ----------------------------------------------------------- significance ----
def paired_bootstrap(scores_a, scores_b, n_resamples=2000, seed=0, alpha=0.05):
    """Bootstrap the mean difference between two models on the SAME queries.

    The previous gate used an unpaired binomial standard error,
    ``sqrt(p(1-p)/N)``, which assumes the two models were measured on
    independent samples. They are not - both are scored on identical queries, so
    per-query difficulty cancels and the paired interval is far tighter. The
    unpaired floor was therefore rejecting real improvements.

    Resamples queries, not predictions, mirroring ``prior_matched_metrics`` in
    ``src/evaluation/prior_shift.py``.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"paired comparison needs equal lengths, got {a.shape} and {b.shape}")

    difference = b - a
    observed = float(difference.mean())

    rng = np.random.default_rng(seed)
    n = len(difference)
    draws = rng.integers(0, n, size=(n_resamples, n))
    resampled = difference[draws].mean(axis=1)

    low, high = np.quantile(resampled, [alpha / 2, 1 - alpha / 2])
    # two-sided bootstrap p-value: how often the resampled mean crosses zero
    crossings = (resampled <= 0).mean() if observed > 0 else (resampled >= 0).mean()
    return {
        "delta": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(min(1.0, 2 * crossings)),
        "significant": bool(low > 0 or high < 0),
        "n_queries": int(n),
    }


def mcnemar(hits_a, hits_b):
    """Exact McNemar test over per-query binary outcomes.

    Use when the metric is hit/miss per query (P@1, or "did the top-10 contain a
    match"). For averaged rates such as P@10 use ``paired_bootstrap``.
    """
    from scipy.stats import binomtest

    a = np.asarray(hits_a, dtype=bool)
    b = np.asarray(hits_b, dtype=bool)
    b_only = int((~a & b).sum())
    a_only = int((a & ~b).sum())
    discordant = a_only + b_only
    if discordant == 0:
        return {"b_only": 0, "a_only": 0, "p_value": 1.0, "significant": False}
    result = binomtest(b_only, discordant, 0.5)
    return {
        "b_only": b_only,
        "a_only": a_only,
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < 0.05),
    }


def colour_families(values):
    """Map each ``baseColour`` to a family, by a lexical rule and nothing else.

    ``colour@10`` is an exact string match over 46 labels, so retrieving a Navy
    Blue shirt for a Blue query scores zero. 42% of the catalogue carries a
    colour that has a same-family sibling, which means a real share of the
    26-point gap between ``P@10`` (~80) and ``colour@10`` (~54) is vocabulary,
    not perception.

    The rule: a multi-word colour joins the family of whichever single-word
    colour in the vocabulary it contains. "Navy Blue" is a kind of Blue; "Grey
    Melange" is a kind of Grey. Everything else is its own family.

    This is deliberately lexical. Teal is not folded into Blue, Olive is not
    folded into Green, and Silver/Gold/Bronze/Copper are not folded into a
    metallics class - each of those would be a perceptual judgement about what
    counts as the same colour, and the point of this mapping is to remove a
    naming artefact, not to make the metric easier. Any number produced with it
    must be reported beside the exact-match one, never instead of it.
    """
    labels = [str(v) for v in pd.unique(pd.Series(list(values)).dropna())]
    singles = {label for label in labels if " " not in label}
    mapping = {}
    for label in labels:
        family = label
        if " " not in label:
            mapping[label] = label
            continue
        for word in label.split():
            if word in singles:
                family = word
                break
        mapping[label] = family
    return mapping


def bootstrap_mean(scores, n_resamples=2000, seed=0, alpha=0.05):
    """Percentile bootstrap interval for a single mean.

    ``paired_bootstrap`` answers "is A better than B"; this answers "what is this
    number, give or take". It exists for the real-photograph benchmark, where
    there is no second model to pair against and where the sample is small enough
    that quoting a bare mean would overstate what 31 photographs can establish.
    """
    values = np.asarray(scores, dtype=float)
    if len(values) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0}

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_resamples, len(values)))
    resampled = values[draws].mean(axis=1)
    low, high = np.quantile(resampled, [alpha / 2, 1 - alpha / 2])
    return {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": int(len(values)),
    }


def evaluate_real_photos(engine, labels, images_dir, k=10, mode="nobg",
                         confidence=None, seed=0, n_resamples=2000):
    """Score retrieval on hand-labelled photographs of real items.

    Every other Task 4 number is measured on a proxy. ``P@10 80.2`` is catalogue
    photographs retrieving catalogue photographs; ``P@10 60.6`` is catalogue
    items composited onto procedural backgrounds. Neither is a photograph someone
    took, and the gap between the two proxies - plus a mean top-1 similarity of
    0.664 on real uploads against 0.837 on catalogue images - is large enough
    that the synthetic benchmark cannot be assumed to stand in for the real one.

    Three groups are reported separately rather than pooled, because they fail
    for different reasons and averaging them hides all three:

    ``single``
        one garment, the case the encoder is built for. This is the headline.
    ``multi``
        several garments in one frame. The encoder emits one vector per image,
        so an outfit is averaged into a vector describing none of its parts;
        counting these as misses would blame the encoder for a known structural
        limit rather than measuring it.
    ``non_clothing``
        not a fashion item at all. There is no correct answer to retrieve, so
        the only meaningful question is whether the confidence gate declines.

    ``labels`` needs ``file`` and ``articleType``; ``baseColour``, ``n_garments``
    and ``notes`` are optional. Rows with no type are skipped and counted.

    Returns ``(summary, per_photo)``.
    """
    labels = pd.DataFrame(labels).copy()
    if "file" not in labels.columns or "articleType" not in labels.columns:
        raise ValueError("labels need at least 'file' and 'articleType' columns")

    labels["articleType"] = labels["articleType"].fillna("").astype(str).str.strip()
    if "baseColour" in labels.columns:
        labels["baseColour"] = labels["baseColour"].fillna("").astype(str).str.strip()
    else:
        labels["baseColour"] = ""
    if "n_garments" in labels.columns:
        labels["n_garments"] = pd.to_numeric(labels["n_garments"],
                                             errors="coerce").fillna(1).astype(int)
    else:
        labels["n_garments"] = 1

    unlabelled = int((labels["articleType"] == "").sum())
    scored = labels[labels["articleType"] != ""].reset_index(drop=True)
    if scored.empty:
        raise ValueError(
            "No labelled rows. Run scripts/build_label_sheet.py, label the "
            "photographs, and save the CSV it produces.")

    # A type that is not in the catalogue can never be retrieved, so scoring it
    # would report the vocabulary as an encoder failure.
    vocabulary = set(engine.metadata["articleType"].dropna().unique())
    unknown = sorted(set(scored["articleType"]) - vocabulary - {"none"})
    if unknown:
        raise ValueError(
            "These labels are not catalogue articleTypes: {}".format(unknown))

    images_dir = Path(images_dir)
    paths = [images_dir / name for name in scored["file"]]
    absent = [p.name for p in paths if not p.exists()]
    if absent:
        raise FileNotFoundError("Missing upload(s): {}".format(absent[:5]))

    results = engine.search(paths, k=k, mode=mode, with_diagnostics=True,
                            confidence=confidence)

    p_at_k = "P@{}".format(k)
    colour_at_k = "colour@{}".format(k)
    both_at_k = "both@{}".format(k)

    rows = []
    for position, record in scored.iterrows():
        got = results[results["query"] == Path(paths[position]).name]
        type_hit = (got["articleType"].to_numpy() == record["articleType"])
        colour_hit = (got["baseColour"].to_numpy() == record["baseColour"]
                      if record["baseColour"] else np.zeros(len(got), bool))
        group = ("non_clothing" if record["n_garments"] == 0
                 else "single" if record["n_garments"] == 1 else "multi")
        rows.append({
            "file": record["file"],
            "group": group,
            "n_garments": int(record["n_garments"]),
            "true_articleType": record["articleType"],
            "true_baseColour": record["baseColour"],
            "top1_articleType": got["articleType"].iloc[0] if len(got) else None,
            "P@1": float(type_hit[0]) if len(type_hit) else 0.0,
            p_at_k: float(type_hit.mean()) if len(type_hit) else 0.0,
            colour_at_k: float(colour_hit.mean()) if len(got) else 0.0,
            both_at_k: float((type_hit & colour_hit).mean()) if len(got) else 0.0,
            "top1_similarity": float(got["top1_similarity"].iloc[0]) if len(got) else np.nan,
            "coherence": float(got["coherence"].iloc[0]) if len(got) else np.nan,
            "confident": bool(got["confident"].iloc[0]) if len(got) else False,
            "ingest_method": got["ingest_method"].iloc[0] if len(got) else None,
            "ingest_fell_back": bool(got["ingest_fell_back"].iloc[0]) if len(got) else False,
            "notes": record.get("notes", ""),
        })
    per_photo = pd.DataFrame(rows)

    summary = {
        "n_photos": int(len(labels)),
        "n_unlabelled": unlabelled,
        "n_scored": int(len(per_photo)),
        "mode": mode,
        "k": int(k),
        "gate_pass_rate": float(per_photo["confident"].mean()),
        "ingest_fallback_rate": float(per_photo["ingest_fell_back"].mean()),
        "mean_top1_similarity": float(per_photo["top1_similarity"].mean()),
        "mean_coherence": float(per_photo["coherence"].mean()),
    }
    for group in ("single", "multi", "non_clothing"):
        subset = per_photo[per_photo["group"] == group]
        summary["n_" + group] = int(len(subset))
        if group == "non_clothing":
            # Nothing to retrieve, so the only score that means anything is
            # whether the system had the sense to say it did not know.
            summary["non_clothing_declined_rate"] = (
                float((~subset["confident"]).mean()) if len(subset) else float("nan"))
            continue
        for metric in ("P@1", p_at_k, colour_at_k, both_at_k):
            interval = bootstrap_mean(subset[metric], n_resamples=n_resamples,
                                      seed=seed)
            summary["{}_{}".format(group, metric)] = interval["mean"]
            summary["{}_{}_ci".format(group, metric)] = (
                interval["ci_low"], interval["ci_high"])

    return summary, per_photo


# --------------------------------------------------------------- protocol ----
@dataclass
class RetrievalProtocol:
    """Split, query set, relevance tables and metrics - built once, shared."""

    gallery: pd.DataFrame
    relevance: str = "articleType"
    control: str = "baseColour"
    holdout_fraction: float = 0.15
    n_queries: int = 2000
    stratified: bool = True
    max_queries_per_class: int = 40
    top_k: int = 10
    k_values: tuple = DEFAULT_K_VALUES
    seed: int = 42

    results: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        g = self.gallery
        self.article = g[self.relevance].to_numpy()
        self.subcat = g["subCategory"].to_numpy()
        self.mastercat = g["masterCategory"].to_numpy()
        self.colour = g[self.control].fillna("Unknown").to_numpy()
        # Same attribute, coarser vocabulary. Reported beside the exact match so
        # the naming artefact and the perceptual error can be told apart.
        self.colour_map = colour_families(self.colour)
        self.colour_group = np.array([self.colour_map.get(c, c) for c in self.colour])
        self.product_name = g["productDisplayName"].fillna("").to_numpy()

        # how many other items share each attribute - Recall@K and the nDCG ideal
        self.class_counts = g[self.relevance].value_counts()
        self.relevant_total = g[self.relevance].map(self.class_counts).to_numpy() - 1
        self.subcat_total = g["subCategory"].map(
            g["subCategory"].value_counts()).to_numpy() - 1
        self.master_total = g["masterCategory"].map(
            g["masterCategory"].value_counts()).to_numpy() - 1

        self._build_split()
        self._build_query_sets()

    # -- split ---------------------------------------------------------
    def _build_split(self):
        """Hold out whole PRODUCTS, so no sibling photo straddles the split."""
        g = self.gallery
        rng = np.random.default_rng(self.seed)
        key = g["productDisplayName"].fillna(g["id"].astype(str)).to_numpy()
        self.product_key = key

        unique_products = np.unique(key)
        rng.shuffle(unique_products)
        holdout = set(unique_products[:int(len(unique_products) * self.holdout_fraction)])
        is_holdout = np.array([p in holdout for p in key])

        self.catalogue_pos = np.where(~is_holdout)[0]
        heldout_pos = np.where(is_holdout)[0]

        if set(key[self.catalogue_pos]) & set(key[heldout_pos]):
            raise AssertionError("product leaked across the split")

        # a query is only answerable if its type still exists in the catalogue
        catalogue_types = set(pd.Series(self.article[self.catalogue_pos]).unique())
        usable = np.array([self.article[p] in catalogue_types for p in heldout_pos])
        self.heldout_pos = heldout_pos[usable]

        if len(self.heldout_pos) > self.n_queries:
            self.heldout_queries = np.sort(
                rng.choice(self.heldout_pos, self.n_queries, replace=False))
        else:
            self.heldout_queries = np.sort(self.heldout_pos)

    # -- query sets ----------------------------------------------------
    def _build_query_sets(self):
        """In-gallery queries, plus the exclusion lists that keep them honest."""
        rng = np.random.default_rng(self.seed)
        if self.stratified:
            # Random sampling is dominated by common classes - Watches alone
            # contributed 103 of 2000 queries once. Capping per class means rare
            # types are actually measured.
            picked = []
            for _, group in self.gallery.groupby(self.relevance).groups.items():
                members = np.asarray(group)
                take = min(len(members), self.max_queries_per_class)
                picked.extend(rng.choice(members, size=take, replace=False))
            picked = np.array(picked)
            if len(picked) > self.n_queries:
                picked = rng.choice(picked, size=self.n_queries, replace=False)
            self.query_positions = np.sort(picked)
        else:
            self.query_positions = np.sort(rng.choice(
                len(self.gallery), size=min(self.n_queries, len(self.gallery)),
                replace=False))

        self.exclude_standard = [np.array([p]) for p in self.query_positions]

        by_name = defaultdict(list)
        for position, name in enumerate(self.product_name):
            if name:
                by_name[name].append(position)
        self.exclude_strict = [
            np.unique(np.array([p] + list(by_name.get(self.product_name[p], []))))
            for p in self.query_positions
        ]

    # -- relevance -----------------------------------------------------
    def binary_relevance(self, query_position, result_positions):
        return (self.article[result_positions]
                == self.article[query_position]).astype(float)

    def graded_relevance(self, query_position, result_positions):
        """Partial credit for sharing a broader category."""
        grades = np.zeros(len(result_positions), dtype=float)
        grades[self.mastercat[result_positions] == self.mastercat[query_position]] = 0.25
        grades[self.subcat[result_positions] == self.subcat[query_position]] = 0.5
        grades[self.article[result_positions] == self.article[query_position]] = 1.0
        return grades

    @staticmethod
    def _dcg(gains):
        positions = np.arange(1, len(gains) + 1)
        return float((gains / np.log2(positions + 1)).sum())

    def _ideal_gains(self, query_position, k):
        """Best ranking actually achievable for this query - not ``ones(k)``.

        A type with only 3 other members cannot fill 10 slots with exact
        matches, so ``ones(k)`` understates nDCG for exactly the rare classes
        the metric is meant to protect.
        """
        n_exact = min(int(self.relevant_total[query_position]), k)
        remaining = k - n_exact

        n_sub = max(min(int(self.subcat_total[query_position])
                        - int(self.relevant_total[query_position]), remaining), 0)
        remaining -= n_sub
        n_master = max(min(int(self.master_total[query_position])
                           - int(self.subcat_total[query_position]), remaining), 0)
        remaining -= n_master

        return np.concatenate([
            np.ones(n_exact),
            np.full(n_sub, 0.5),
            np.full(n_master, 0.25),
            np.zeros(max(remaining, 0)),
        ])[:k]

    def evaluate_ranking(self, query_position, ranked_positions, k_values=None):
        """Metrics for one query, given positions ranked most-similar first."""
        k_values = k_values or self.k_values
        top = ranked_positions[:max(k_values)]

        hits = self.binary_relevance(query_position, top)
        grades = self.graded_relevance(query_position, top)
        n_relevant = max(int(self.relevant_total[query_position]), 1)

        out = {}
        for k in k_values:
            hits_k = hits[:k]
            out[f"P@{k}"] = float(hits_k.mean())
            out[f"R@{k}"] = float(hits_k.sum() / n_relevant)

            achievable = min(k, n_relevant)
            if hits_k.sum() > 0:
                precisions = np.cumsum(hits_k) / np.arange(1, k + 1)
                out[f"AP@{k}"] = float((precisions * hits_k).sum() / achievable)
            else:
                out[f"AP@{k}"] = 0.0

            ideal = self._dcg(self._ideal_gains(query_position, k))
            out[f"nDCG@{k}"] = self._dcg(grades[:k]) / ideal if ideal > 0 else 0.0

        # R-precision: precision at K = number of relevant items. Comparable
        # across classes of wildly different size, unlike Recall@10.
        r = min(n_relevant, len(ranked_positions))
        out["R-precision"] = float(
            self.binary_relevance(query_position, ranked_positions[:r]).mean()) if r else 0.0
        return out

    # -- harnesses -----------------------------------------------------
    def evaluate_index(self, index, embeddings, mode="standard", k_values=None,
                       store=True):
        """In-gallery protocol: a catalogue item queries its peers."""
        k_values = k_values or self.k_values
        exclude = self.exclude_standard if mode == "standard" else self.exclude_strict

        start = time.perf_counter()
        _, ranked = index.search(np.asarray(embeddings)[self.query_positions],
                                 k=max(k_values), exclude=exclude)
        elapsed = time.perf_counter() - start

        per_query = [self.evaluate_ranking(q, ranked[i], k_values)
                     for i, q in enumerate(self.query_positions)]
        summary = pd.DataFrame(per_query).mean().to_dict()
        summary["ms_per_query"] = elapsed / len(self.query_positions) * 1000
        summary["dim"] = index.dim

        if store:
            self.results.setdefault(index.name, {})[mode] = summary
        return summary, ranked

    def evaluate_deployment(self, index, embeddings, queries=None, k_values=None,
                            label=None, store=True):
        """Deployment protocol: unseen products query a catalogue-only index."""
        queries = self.heldout_queries if queries is None else queries
        k_values = k_values or self.k_values
        k = self.top_k

        start = time.perf_counter()
        _, ranked = index.search(np.asarray(embeddings)[queries], k=max(k_values))
        elapsed = (time.perf_counter() - start) / len(queries) * 1000

        per_query = [self.evaluate_ranking(q, ranked[i], k_values)
                     for i, q in enumerate(queries)]
        summary = pd.DataFrame(per_query).mean().to_dict()

        # A result is useful to a shopper when the TYPE and the COLOUR match.
        type_hit = self.article[ranked[:, :k]] == self.article[queries][:, None]
        colour_hit = self.colour[ranked[:, :k]] == self.colour[queries][:, None]
        family_hit = (self.colour_group[ranked[:, :k]]
                      == self.colour_group[queries][:, None])
        summary["type@10"] = float(type_hit.mean())
        summary["colour@10"] = float(colour_hit.mean())
        summary["both@10"] = float((type_hit & colour_hit).mean())
        # Colour scored on families rather than exact labels. Never a
        # replacement for colour@10 - the pair is the point, because the gap
        # between them is how much of the colour deficit is only naming.
        summary["colourfam@10"] = float(family_hit.mean())
        summary["bothfam@10"] = float((type_hit & family_hit).mean())
        summary["ms_per_query"] = elapsed
        summary["dim"] = index.dim
        # per-query means, so two models can be compared with a paired test
        summary["_per_query_both"] = (type_hit & colour_hit).mean(axis=1)
        summary["_per_query_type"] = type_hit.mean(axis=1)
        summary["_per_query_colour"] = colour_hit.mean(axis=1)
        summary["_per_query_colourfam"] = family_hit.mean(axis=1)
        summary["_per_query_bothfam"] = (type_hit & family_hit).mean(axis=1)

        if store:
            self.results.setdefault("deployment", {})[label or index.name] = summary
        return summary, ranked

    # -- reporting -----------------------------------------------------
    def deployment_table(self, results=None):
        k = self.top_k
        results = results if results is not None else self.results.get("deployment", {})
        columns = ["Method", "Dim", "P@1", f"P@{k}", f"mAP@{k}", f"nDCG@{k}",
                   "colour@10", "both@10", "ms/query"]
        if not results:
            # An empty dict used to raise KeyError('both@10') from sort_values,
            # which is what a caller hits when the table is built before any
            # index has been evaluated.
            return pd.DataFrame(columns=columns)
        rows = [{
            "Method": name,
            "Dim": int(s["dim"]),
            "P@1": round(s["P@1"] * 100, 2),
            f"P@{k}": round(s[f"P@{k}"] * 100, 2),
            f"mAP@{k}": round(s[f"AP@{k}"] * 100, 2),
            f"nDCG@{k}": round(s[f"nDCG@{k}"] * 100, 2),
            "colour@10": round(s["colour@10"] * 100, 2),
            "both@10": round(s["both@10"] * 100, 2),
            "ms/query": round(s["ms_per_query"], 2),
        } for name, s in results.items()]
        return pd.DataFrame(rows).sort_values("both@10", ascending=False)

    def compare(self, summary_a, summary_b, metric="both@10", **kwargs):
        """Paired comparison of two deployment summaries on the same queries."""
        key = {"both@10": "_per_query_both",
               "type@10": "_per_query_type",
               "colour@10": "_per_query_colour",
               "colourfam@10": "_per_query_colourfam",
               "bothfam@10": "_per_query_bothfam"}[metric]
        stats = paired_bootstrap(summary_a[key], summary_b[key], **kwargs)
        stats["metric"] = metric
        return stats
