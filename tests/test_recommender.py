"""Ranking, filters and edge cases using controlled product text."""

import pandas as pd
import pytest

from src.data_loader import REQUIRED_COLUMNS
from src.evaluation import evaluate_catalog
from src.preprocessing import clean_text
from src.recommender import ProductRecommender


@pytest.fixture
def catalog() -> pd.DataFrame:
    return pd.DataFrame([
        ["a", "Wireless Headphones", "Audio", "bluetooth noise cancelling headphones", 100],
        ["b", "Wireless Headphones", "Audio", "bluetooth noise cancelling headphones", 80],
        ["c", "Bluetooth Speaker", "Audio", "portable bluetooth speaker", 50],
        ["d", "Cotton Shirt", "Clothing", "soft cotton shirt", 20],
        ["e", "Headphones Case", "Accessories", "wireless headphones storage", 10],
    ], columns=REQUIRED_COLUMNS)


def test_ranking_and_exclusion(catalog) -> None:
    result = ProductRecommender(catalog).recommend("a")
    assert result.iloc[0]["product_id"] == "b"
    assert result.iloc[0]["similarity_score"] == pytest.approx(1)
    assert "a" not in set(result.product_id)
    assert "d" not in set(result.product_id)  # no overlap is not a meaningful match
    assert result.similarity_score.is_monotonic_decreasing
    assert result.similarity_score.between(0, 1).all()
    assert result.explanation.str.contains("headphones").any()


def test_filters(catalog) -> None:
    engine = ProductRecommender(catalog)
    assert engine.recommend("a", category="Audio", min_price=50, max_price=50).product_id.tolist() == ["c"]
    assert engine.recommend("a", category="Accessories").product_id.tolist() == ["e"]
    assert engine.recommend("a", category="Missing").empty
    assert engine.recommend("a", min_price=200).empty
    assert len(engine.recommend("a", top_k=1)) == 1


def test_unknown_id_and_zero_k(catalog) -> None:
    engine = ProductRecommender(catalog)
    for result in [engine.recommend("missing"), engine.recommend("a", top_k=0)]:
        assert result.empty
        assert {"similarity_score", "explanation"}.issubset(result.columns)


def test_duplicates(catalog) -> None:
    with pytest.warns(UserWarning, match="duplicate"):
        engine = ProductRecommender(pd.concat([catalog, catalog]))
    assert engine.recommend("a", top_k=100).product_id.is_unique
    assert len(engine.catalog) == len(catalog)


@pytest.mark.parametrize("size", [0, 1, 2])
def test_small_catalog(catalog, size) -> None:
    engine = ProductRecommender(catalog.head(size))
    assert len(engine.recommend("a", top_k=5)) == max(0, size - 1)


def test_empty_vocabulary() -> None:
    frame = pd.DataFrame([["1", "the", "and", "or", 1]], columns=REQUIRED_COLUMNS)
    engine = ProductRecommender(frame)
    assert engine.matrix is None
    assert engine.recommend("1").empty


@pytest.mark.parametrize("kwargs", [
    {"top_k": -1}, {"top_k": 1.5}, {"top_k": True},
    {"min_price": -1}, {"max_price": float("inf")},
    {"min_price": float("nan")}, {"min_price": 20, "max_price": 10},
])
def test_invalid_controls(catalog, kwargs) -> None:
    with pytest.raises(ValueError):
        ProductRecommender(catalog).recommend("a", **kwargs)


def test_deterministic_ties(catalog) -> None:
    duplicate = catalog.iloc[[1]].copy()
    duplicate["product_id"] = "z"
    engine = ProductRecommender(pd.concat([duplicate, catalog]))
    assert engine.recommend("a").product_id.tolist()[:2] == ["b", "z"]


def test_useful_product_terms() -> None:
    assert clean_text("  USB-C 65W / Wi-Fi 6 + C++ ") == "usb-c 65w wi-fi 6 + c++"


def test_evaluation_counts(catalog) -> None:
    engine = ProductRecommender(catalog)
    report = evaluate_catalog(engine, top_k=1)
    results = [engine.recommend(pid, top_k=1) for pid in catalog.product_id]
    ids = {pid for result in results for pid in result.product_id}
    assert report["catalog_coverage"] == len(ids) / len(catalog)
    assert report["queries_evaluated"] == 5
    assert all(report["sanity_checks"].values())


def test_empty_evaluation(catalog) -> None:
    report = evaluate_catalog(ProductRecommender(catalog.head(0)))
    assert report["catalog_coverage"] == 0
    assert report["category_agreement_proxy"] is None
