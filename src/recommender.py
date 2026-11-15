"""Sparse TF-IDF retrieval independent of the presentation layer."""

import math
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import validate_catalog
from src.preprocessing import build_product_text


class ProductRecommender:
    """Fit a catalog once and compute a single similarity row per request."""

    def __init__(self, catalog: pd.DataFrame) -> None:
        self.catalog = validate_catalog(catalog)
        self.vectorizer = TfidfVectorizer(
            lowercase=False, stop_words="english", ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w[\w+.\-]*", sublinear_tf=True,
            max_features=30_000,
        )
        self.matrix = None
        self._positions = {pid: i for i, pid in enumerate(self.catalog["product_id"])}
        if not self.catalog.empty:
            try:
                self.matrix = self.vectorizer.fit_transform(build_product_text(self.catalog))
            except ValueError as exc:
                if "empty vocabulary" not in str(exc):
                    raise

    def _empty_result(self) -> pd.DataFrame:
        return self.catalog.iloc[:0].assign(
            similarity_score=pd.Series(dtype=float), explanation=pd.Series(dtype=str)
        )

    def recommend(
        self, product_id: str, top_k: int = 5, category: str | None = None,
        min_price: float | None = None, max_price: float | None = None,
    ) -> pd.DataFrame:
        """Rank positive-overlap matches, with inclusive category/price filters.

        Unknown IDs, empty vocabularies and catalogs return an empty result.
        Invalid limits raise ValueError. Ties are resolved by product ID.
        Similarity scores are cosine values, not relevance probabilities.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer.")
        for price in (min_price, max_price):
            if price is not None and (not math.isfinite(price) or price < 0):
                raise ValueError("Price limits must be finite and nonnegative.")
        if min_price is not None and max_price is not None and min_price > max_price:
            raise ValueError("Minimum price must not exceed maximum price.")
        selected = self._positions.get(str(product_id).strip())
        if selected is None or self.matrix is None or top_k == 0:
            return self._empty_result()
        mask = self.catalog["product_id"].ne(self.catalog.iloc[selected]["product_id"])
        if category is not None:
            mask &= self.catalog["category"].eq(category)
        if min_price is not None:
            mask &= self.catalog["price"].ge(min_price)
        if max_price is not None:
            mask &= self.catalog["price"].le(max_price)
        positions = np.flatnonzero(mask.to_numpy())
        if not len(positions):
            return self._empty_result()
        scores = cosine_similarity(self.matrix[selected], self.matrix[positions]).ravel()
        result = self.catalog.iloc[positions].copy()
        result["similarity_score"] = np.clip(scores, 0, 1)
        result = result.loc[result["similarity_score"] > 0].sort_values(
            ["similarity_score", "product_id"], ascending=[False, True]
        ).head(top_k)
        terms = self.vectorizer.get_feature_names_out()
        explanations = []
        for row_index in result.index:
            overlap = self.matrix[selected].multiply(self.matrix[row_index]).tocoo()
            ordered = sorted(zip(overlap.col, overlap.data), key=lambda pair: (-pair[1], terms[pair[0]]))
            shared = ", ".join(terms[index] for index, _ in ordered[:3])
            explanations.append(f"Shared terms: {shared}.")
        result["explanation"] = explanations
        return result.reset_index(drop=True)
