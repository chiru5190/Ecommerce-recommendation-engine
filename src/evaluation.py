"""Reproducible catalog diagnostics; no claims of labeled relevance quality."""

import argparse
import json
from pathlib import Path

from src.data_loader import DEFAULT_DATA_PATH, load_catalog
from src.recommender import ProductRecommender


def evaluate_catalog(engine: ProductRecommender, top_k: int = 5) -> dict:
    """Query every product and report coverage and structural sanity checks.

    Coverage = unique recommended IDs / catalog size. Category agreement is
    matching-category recommendations / all returned recommendations, a proxy
    influenced by category text being an input feature, not Precision@K.
    """
    if top_k < 1:
        raise ValueError("Evaluation top_k must be positive.")
    seen: set[str] = set()
    returned = agreement = queries_with_results = 0
    checks = dict(no_self_recommendations=True, unique_results=True,
                  descending_scores=True, valid_scores=True, at_most_k=True,
                  known_product_ids=True)
    known_ids = set(engine.catalog["product_id"])
    for product in engine.catalog.itertuples(index=False):
        result = engine.recommend(product.product_id, top_k=top_k)
        ids = result["product_id"].tolist()
        scores = result["similarity_score"]
        seen.update(ids)
        returned += len(result)
        queries_with_results += int(bool(len(result)))
        agreement += int(result["category"].eq(product.category).sum())
        checks["no_self_recommendations"] &= product.product_id not in ids
        checks["unique_results"] &= len(ids) == len(set(ids))
        checks["descending_scores"] &= bool(scores.is_monotonic_decreasing)
        checks["valid_scores"] &= bool(scores.between(0, 1).all())
        checks["at_most_k"] &= len(result) <= top_k
        checks["known_product_ids"] &= set(ids).issubset(known_ids)
    count = len(engine.catalog)
    return {
        "catalog_size": count, "top_k": top_k, "queries_evaluated": count,
        "queries_with_results": queries_with_results,
        "recommendations_returned": returned, "unique_recommended_products": len(seen),
        "catalog_coverage": len(seen) / count if count else 0.0,
        "category_agreement_proxy": agreement / returned if returned else None,
        "sanity_checks": checks,
        "note": "Category agreement is a metadata proxy, not ground-truth Precision@K. "
                "Empty-catalog sanity checks are vacuous. No relevance labels are available.",
    }


def main() -> None:
    """Print reproducible diagnostics for a local CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional JSON artifact path")
    args = parser.parse_args()
    report = evaluate_catalog(ProductRecommender(load_catalog(args.data)), args.top_k)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not all(report["sanity_checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
