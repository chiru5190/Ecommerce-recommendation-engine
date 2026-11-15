"""Exact single-query semantic/TF-IDF score blending with baseline delegation."""

import math
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.recommender import ProductRecommender
from src.semantic import SemanticIndex, SemanticUnavailable, catalog_fingerprint


def hybrid_scores(semantic: np.ndarray, tfidf: np.ndarray, weight: float = 0.65) -> np.ndarray:
    """Combine raw cosine values; no probability interpretation or calibration."""
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Semantic weight must be between 0 and 1.")
    if semantic.shape != tfidf.shape:
        raise ValueError("Score arrays must have identical ID alignment.")
    return weight * semantic + (1 - weight) * tfidf


class HybridRecommender:
    """Retain the original TF-IDF API and add optional semantic/hybrid modes."""

    def __init__(self, baseline: ProductRecommender, semantic: SemanticIndex | None = None) -> None:
        self.baseline = baseline
        self.catalog = baseline.catalog
        self.semantic = semantic
        if semantic is not None and set(semantic.product_ids) != set(self.catalog.product_id):
            raise ValueError("Semantic and TF-IDF product ID sets must match.")
        if (semantic is not None and semantic.catalog_signature is not None
                and semantic.catalog_signature != catalog_fingerprint(self.catalog)):
            raise ValueError("Semantic catalog metadata is stale or misaligned; rebuild the index.")

    def recommend(self, product_id: str, top_k: int = 5, category: str | None = None,
                  min_price: float | None = None, max_price: float | None = None,
                  mode: str = "TF-IDF", semantic_weight: float = 0.65) -> pd.DataFrame:
        """Rank filtered alternatives using exact, ID-aligned component scores."""
        if mode not in {"TF-IDF", "Semantic", "Hybrid"}:
            raise ValueError("Unknown recommendation mode.")
        if mode == "TF-IDF":
            return self.baseline.recommend(product_id, top_k, category, min_price, max_price)
        hybrid_scores(np.zeros(1), np.zeros(1), semantic_weight)
        # Reuse baseline control validation without constructing another similarity row.
        if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer.")
        self.baseline.recommend(product_id, 0, category, min_price, max_price)
        pid = str(product_id).strip()
        if pid not in self.baseline._positions or top_k == 0:
            return self.baseline._empty_result()
        if self.semantic is None:
            raise SemanticUnavailable("Semantic resources are unavailable. Select TF-IDF.")
        data = self.catalog
        semantic = self.semantic.cosine_scores(pid).reindex(data.product_id).to_numpy()
        selected = self.baseline._positions[pid]
        tfidf = np.zeros(len(data)) if self.baseline.matrix is None else cosine_similarity(
            self.baseline.matrix[selected], self.baseline.matrix).ravel()
        tfidf = np.clip(tfidf, 0, 1)
        scores = semantic if mode == "Semantic" else hybrid_scores(semantic, tfidf, semantic_weight)
        mask = data.product_id.ne(pid) & (scores > 0)
        if category is not None:
            mask &= data.category.eq(category)
        if min_price is not None:
            mask &= data.price.ge(min_price)
        if max_price is not None:
            mask &= data.price.le(max_price)
        result = data.loc[mask].copy()
        result["similarity_score"] = scores[mask]
        result["semantic_score"] = semantic[mask]
        result["tfidf_score"] = tfidf[mask]
        result = result.sort_values(["similarity_score", "product_id"], ascending=[False, True]).head(top_k)
        result["explanation"] = [
            f"Semantic cosine: {row.semantic_score:.3f}; TF-IDF cosine: {row.tfidf_score:.3f}. "
            + (f"Weighted {semantic_weight:.0%} semantic + {1-semantic_weight:.0%} TF-IDF."
               if mode == "Hybrid" else "Ranked by similarity of product-text embeddings.")
            for row in result.itertuples()
        ]
        return result.reset_index(drop=True)
