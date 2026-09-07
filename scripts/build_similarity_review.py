#!/usr/bin/env python
"""Human judgement of visual similarity, which nothing else in the project measures.

Why this exists
---------------
Every retrieval number reported for Task 4 - P@10, mAP@10, nDCG@10 - scores a
result as relevant when it shares the query's ``articleType``. That is a metadata
proxy, and it is not the thing the task asks for. Two white t-shirts and a black
one are all `Tshirts`; a black t-shirt and a black polo shirt look far more alike
than either looks like a white tee. A metric built on class agreement cannot see
that difference, and a system tuned against it is being tuned toward
classification rather than similarity.

`05_task4_triplet_encoder.ipynb` names this as the first limitation of its
evaluation and suggests a small human-judged sample. This builds that sample.

    python scripts/build_similarity_review.py            # make the review sheet
    ...open outputs/similarity_review.html and score it...
    python scripts/build_similarity_review.py --score    # human P@10 vs the proxy

The sheet deliberately shows the query and its results without the retrieved
labels, so the judgement is made on appearance rather than on agreeing with the
metadata.
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "outputs"
REVIEW_CSV = OUTPUT_DIR / "similarity_review.csv"
REVIEW_HTML = OUTPUT_DIR / "similarity_review.html"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=25, help="queries to review")
    parser.add_argument("--k", type=int, default=10, help="results per query")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score", action="store_true",
                        help="score a filled-in review instead of building one")
    return parser.parse_args()


def thumbnail(path, size=(90, 120)):
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB").resize(size, Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build(args):
    from src.visual_search.search_engine import SearchEngine

    engine = SearchEngine.load()
    manifest = engine.manifest
    print("Engine: {} | {:,} catalogue items".format(
        manifest.get("best_method"), len(engine.index)))

    test_dir = (PROJECT_ROOT / "A2_FashionDataset" / "FashionDataset" / "test"
                / "images_test")
    upload_dir = PROJECT_ROOT / "A2_FashionDataset" / "input_images"

    rng = np.random.default_rng(args.seed)
    queries = []
    if test_dir.is_dir():
        pool = sorted(test_dir.glob("*.jpg"))
        take = min(len(pool), args.n * 2 // 3)
        queries += [(p, "catalogue") for p in rng.choice(pool, take, replace=False)]
    if upload_dir.is_dir():
        pool = [p for p in sorted(upload_dir.iterdir())
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        take = min(len(pool), args.n - len(queries))
        if take > 0:
            queries += [(p, "upload") for p in rng.choice(pool, take, replace=False)]
    if not queries:
        sys.exit("No query images found")
    print("Reviewing {} queries ({} catalogue, {} uploads)".format(
        len(queries),
        sum(1 for _, s in queries if s == "catalogue"),
        sum(1 for _, s in queries if s == "upload")))

    rows, blocks = [], []
    for query_path, source in queries:
        results = engine.search([query_path], k=args.k, mode="nobg",
                                with_diagnostics=True)
        query_b64 = thumbnail(query_path)

        cards = []
        for _, item in results.reset_index(drop=True).iterrows():
            image_path = None
            candidate = PROJECT_ROOT / str(item.get("image_path", ""))
            if candidate.exists():
                image_path = candidate
            if image_path is None:
                continue
            rows.append({
                "query": query_path.name,
                "source": source,
                "rank": int(item["rank"]),
                "result_id": item["id"],
                "similarity": item["similarity"],
                # the proxy's verdict, recorded for comparison but hidden in the sheet
                "same_articleType": bool(item["articleType"]
                                         == results.iloc[0]["articleType"]),
                "articleType": item["articleType"],
                "baseColour": item["baseColour"],
                "visually_similar": "",          # <- fill in with 1 or 0
            })
            cards.append(
                '<figure><img src="data:image/png;base64,{}" alt="">'
                '<figcaption>#{}</figcaption></figure>'.format(
                    thumbnail(image_path), int(item["rank"])))

        blocks.append(
            '<section><div class="q"><img src="data:image/png;base64,{}" alt="">'
            '<div><b>{}</b><br><span class="src">{}</span></div></div>'
            '<div class="grid">{}</div></section>'.format(
                query_b64, query_path.name, source, "".join(cards)))

    frame = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(exist_ok=True)
    frame.to_csv(REVIEW_CSV, index=False)

    html = """<!doctype html><meta charset="utf-8"><title>Visual similarity review</title>
<style>
 body{font:14px system-ui,sans-serif;margin:24px;background:#fafafa;color:#111}
 h1{font-size:20px} p{max-width:70ch;color:#444}
 section{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:14px;margin:14px 0}
 .q{display:flex;gap:12px;align-items:center;margin-bottom:10px}
 .q img{width:90px;height:120px;object-fit:contain;border:1px solid #ddd;background:#fff}
 .src{color:#777;font-size:12px}
 .grid{display:flex;flex-wrap:wrap;gap:10px}
 figure{margin:0;text-align:center}
 figure img{width:90px;height:120px;object-fit:contain;border:1px solid #ddd;background:#fff}
 figcaption{font-size:11px;color:#666;margin-top:2px}
</style>
<h1>Visual similarity review</h1>
<p>For each query on the left, decide which of the numbered results actually
<b>look like</b> it &mdash; same kind of garment, comparable colour and pattern.
Record a 1 or a 0 in the <code>visually_similar</code> column of
<code>outputs/similarity_review.csv</code>, matched on query and rank.</p>
<p>Labels are withheld on purpose: the point is to judge appearance, not to agree
with the metadata the automatic metric already uses.</p>
""" + "".join(blocks)
    REVIEW_HTML.write_text(html, encoding="utf-8")

    print("\nWrote {} ({} rows to judge)".format(REVIEW_CSV.name, len(frame)))
    print("Wrote {} - open it, then fill the CSV".format(REVIEW_HTML.name))


def score():
    if not REVIEW_CSV.exists():
        sys.exit("{} not found - build the review first".format(REVIEW_CSV))
    frame = pd.read_csv(REVIEW_CSV)
    judged = frame[frame["visually_similar"].notna()
                   & (frame["visually_similar"].astype(str).str.strip() != "")]
    if judged.empty:
        sys.exit("No judgements recorded yet - fill the visually_similar column "
                 "with 1 or 0 and re-run with --score")

    judged = judged.copy()
    judged["visually_similar"] = judged["visually_similar"].astype(float).astype(bool)
    print("Judged {} of {} rows across {} queries\n".format(
        len(judged), len(frame), judged["query"].nunique()))

    rows = []
    for source, group in judged.groupby("source"):
        per_query_human = group.groupby("query")["visually_similar"].mean()
        per_query_proxy = group.groupby("query")["same_articleType"].mean()
        rows.append({
            "source": source,
            "queries": int(group["query"].nunique()),
            "human P@k": round(float(per_query_human.mean()) * 100, 2),
            "metadata P@k": round(float(per_query_proxy.mean()) * 100, 2),
        })
    summary = pd.DataFrame(rows)
    summary["proxy overstates by"] = (summary["metadata P@k"]
                                      - summary["human P@k"]).round(2)
    print(summary.to_string(index=False))

    agreement = (judged["visually_similar"] == judged["same_articleType"]).mean()
    print("\nThe proxy and a human agree on {:.1%} of individual results.".format(
        agreement))
    print("Where they differ is where the metric is measuring the wrong thing:")
    both = judged.groupby(["same_articleType", "visually_similar"]).size()
    print(both.rename("results").to_string())

    summary.to_csv(OUTPUT_DIR / "similarity_review_summary.csv", index=False)
    print("\nWrote similarity_review_summary.csv")


if __name__ == "__main__":
    arguments = parse_args()
    score() if arguments.score else build(arguments)
