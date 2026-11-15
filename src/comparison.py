"""Reproducible comparison on identical real catalog queries without relevance claims."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np

from src.data_loader import load_catalog
from src.flipkart import REAL_DATA_PATH, file_sha256
from src.hybrid import HybridRecommender
from src.recommender import ProductRecommender
from src.semantic import MODEL_ID, MODEL_REVISION, SemanticUnavailable, load_semantic_index


def compare(engine: HybridRecommender, query_ids: list[str], top_k: int = 5,
            weight: float = 0.65) -> dict:
    """Time warm per-query retrieval; exclude index/model loading and image I/O."""
    if top_k < 1 or len(query_ids) != len(set(query_ids)):
        raise ValueError("Use positive K and unique query IDs.")
    categories = engine.catalog.set_index("product_id").category
    if not set(query_ids).issubset(categories.index):
        raise ValueError("All comparison query IDs must exist in the catalog.")
    reports = {}
    for mode in ("TF-IDF", "Semantic", "Hybrid"):
        if mode != "TF-IDF" and engine.semantic is None:
            reports[mode] = {"status": "unavailable"}
            continue
        if query_ids:
            engine.recommend(query_ids[0], top_k=top_k, mode=mode, semantic_weight=weight)
        seen: set[str] = set()
        returned = agreement = 0
        latencies = []
        sanity = True
        for pid in query_ids:
            started = time.perf_counter()
            result = engine.recommend(pid, top_k=top_k, mode=mode, semantic_weight=weight)
            latencies.append((time.perf_counter() - started) * 1000)
            ids = result.product_id.tolist()
            seen.update(ids)
            returned += len(ids)
            agreement += int(result.category.eq(categories[pid]).sum())
            sanity &= (pid not in ids and len(ids) == len(set(ids)) and len(ids) <= top_k
                       and set(ids).issubset(categories.index)
                       and bool(result.similarity_score.between(-1, 1).all())
                       and bool(result.similarity_score.is_monotonic_decreasing))
        reports[mode] = {
            "status": "ok", "queries": len(query_ids), "recommendations_returned": returned,
            "unique_recommended_products": len(seen),
            "catalog_coverage": len(seen) / len(engine.catalog) if len(engine.catalog) else 0,
            "category_agreement_proxy": agreement / returned if returned else None,
            "latency_ms_mean": float(np.mean(latencies)) if latencies else None,
            "latency_ms_p50": float(np.percentile(latencies, 50)) if latencies else None,
            "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
            "sanity_checks_passed": bool(sanity),
        }
    return {"catalog_size": len(engine.catalog), "query_ids": query_ids, "top_k": top_k,
            "semantic_weight": weight, "methods": reports,
            "methodology": "Same seeded product queries, no filters, one warm-up per method. "
            "Coverage denominator is the entire catalog, even for sampled queries. "
            "Warm retrieval latency includes ranking/explanations, excludes model/index loading. "
            "Category agreement is a proxy, not Precision@K; category is an input feature. "
            "No genuine relevance labels are available; Precision/Recall/MRR/NDCG are not computed."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REAL_DATA_PATH)
    parser.add_argument("--queries", type=int, default=200, help="0 evaluates every product")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semantic-weight", type=float, default=0.65)
    parser.add_argument("--output", type=Path, default=Path("reports/model_comparison.json"))
    args = parser.parse_args()
    if args.queries < 0:
        parser.error("queries must be nonnegative")
    started = time.perf_counter()
    baseline = ProductRecommender(load_catalog(args.data))
    baseline_build_seconds = time.perf_counter() - started
    semantic = None
    semantic_error = None
    started = time.perf_counter()
    try:
        semantic = load_semantic_index(baseline.catalog)
    except SemanticUnavailable as exc:
        semantic_error = str(exc)
    semantic_load_seconds = time.perf_counter() - started
    ids = sorted(baseline.catalog.product_id)
    if args.queries and args.queries < len(ids):
        selected = np.random.default_rng(args.seed).choice(len(ids), args.queries, replace=False)
        ids = [ids[i] for i in sorted(selected)]
    report = compare(HybridRecommender(baseline, semantic), ids, args.top_k, args.semantic_weight)
    report.update(dataset_file=args.data.name, dataset_sha256=file_sha256(args.data),
                  query_seed=args.seed, model=MODEL_ID, model_revision=MODEL_REVISION,
                  embedding_fingerprint=semantic.fingerprint if semantic else None,
                  semantic_error=semantic_error, baseline_build_seconds=baseline_build_seconds,
                  semantic_load_seconds=semantic_load_seconds,
                  runtime={"python": platform.python_version(), "platform": platform.platform(),
                           "numpy": version("numpy"), "scikit-learn": version("scikit-learn")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "query_ids"}, indent=2))
    if any(not value.get("sanity_checks_passed", True) for value in report["methods"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
