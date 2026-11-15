"""Known vector fixtures test FAISS and blending independently of model downloads."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from src.comparison import compare
from src.data_loader import REQUIRED_COLUMNS
from src.hybrid import HybridRecommender, hybrid_scores
from src.recommender import ProductRecommender
from src.semantic import SemanticIndex, SemanticUnavailable, catalog_fingerprint, load_semantic_index


@pytest.fixture
def base() -> ProductRecommender:
    return ProductRecommender(pd.DataFrame([
        ["b", "Wireless Headphones", "Audio", "bluetooth padded headphones", 1999],
        ["a", "Wireless Headphones", "Audio", "bluetooth padded headphones", 2499],
        ["c", "Speaker", "Audio", "portable audio", 999],
        ["d", "Cotton Shirt", "Clothing", "cotton shirt", 399],
    ], columns=REQUIRED_COLUMNS))


@pytest.fixture
def engine(base) -> HybridRecommender:
    pytest.importorskip("faiss")
    semantic = SemanticIndex(["a", "b", "c", "d"],
                             np.array([[1, 0], [0.8, 0.6], [1, 0], [-1, 0]]), "fixture")
    return HybridRecommender(base, semantic)


def test_faiss_normalization_retrieval_and_id_alignment(engine) -> None:
    assert np.allclose(np.linalg.norm(engine.semantic.embeddings, axis=1), 1)
    assert engine.semantic.index.ntotal == 4
    result = engine.recommend("a", mode="Semantic")
    assert result.product_id.tolist() == ["c", "b"]
    assert result.similarity_score.tolist() == pytest.approx([1, 0.8])
    assert result.product_id.is_unique
    assert "a" not in set(result.product_id)


def test_hybrid_weighting_exact_endpoints_and_filters(engine) -> None:
    semantic = engine.recommend("a", mode="Semantic")
    hybrid = engine.recommend("a", mode="Hybrid", semantic_weight=1)
    assert semantic.product_id.tolist() == hybrid.product_id.tolist()
    baseline = engine.recommend("a")
    zero = engine.recommend("a", mode="Hybrid", semantic_weight=0)
    assert baseline.product_id.tolist() == zero.product_id.tolist()
    result = engine.recommend("a", mode="Hybrid", category="Audio", min_price=1999, max_price=1999)
    assert result.product_id.tolist() == ["b"]
    expected = .65 * result.semantic_score + .35 * result.tfidf_score
    assert result.similarity_score.tolist() == pytest.approx(expected.tolist())
    assert engine.recommend("a", mode="Semantic", category="missing").empty
    assert engine.recommend("missing", mode="Hybrid").empty


def test_baseline_without_optional_semantics(base) -> None:
    engine = HybridRecommender(base)
    pd.testing.assert_frame_equal(engine.recommend("a"), base.recommend("a"))
    with pytest.raises(SemanticUnavailable):
        engine.recommend("a", mode="Hybrid")


def test_missing_faiss_does_not_break_tfidf(base, monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "faiss", None)
    with pytest.raises(SemanticUnavailable, match="FAISS unavailable"):
        load_semantic_index(base.catalog, tmp_path)
    assert not HybridRecommender(base).recommend("a").empty


@pytest.mark.parametrize("weight", [-1, 1.1, float("nan")])
def test_invalid_hybrid_weights(weight) -> None:
    with pytest.raises(ValueError):
        hybrid_scores(np.ones(2), np.ones(2), weight)


@pytest.mark.parametrize("kwargs", [{"top_k": True}, {"top_k": 1.5}, {"top_k": -1},
    {"min_price": 100, "max_price": 10}, {"max_price": float("inf")}])
def test_invalid_controls(engine, kwargs) -> None:
    with pytest.raises(ValueError):
        engine.recommend("a", mode="Semantic", **kwargs)


def test_fingerprint_content_order_and_revision(base) -> None:
    catalog = base.catalog
    original = catalog_fingerprint(catalog)
    assert original == catalog_fingerprint(catalog.iloc[::-1])
    changed = catalog.copy()
    changed.loc[0, "description"] = "new text"
    assert original != catalog_fingerprint(changed)
    changed = catalog.copy()
    changed.loc[0, "price"] += 1
    assert original != catalog_fingerprint(changed)
    assert original != catalog_fingerprint(catalog, revision="different")


def test_cache_hit_mapping_and_corruption_rebuild(base, tmp_path) -> None:
    pytest.importorskip("faiss")
    calls = []
    def encoder(texts):
        calls.append(len(texts))
        return np.tile(np.array([[1., 2., 3.]]), (len(texts), 1))
    first = load_semantic_index(base.catalog, tmp_path, encoder=encoder, model_id="test-fixture")
    second = load_semantic_index(base.catalog.iloc[::-1], tmp_path, encoder=encoder, model_id="test-fixture")
    assert calls == [4]
    assert first.product_ids == second.product_ids == ["a", "b", "c", "d"]
    assert first.cosine_scores("a").tolist() == pytest.approx(second.cosine_scores("a").tolist())
    (tmp_path / first.fingerprint / "index.faiss").write_bytes(b"corrupt")
    load_semantic_index(base.catalog, tmp_path, encoder=encoder, model_id="test-fixture")
    assert calls == [4, 4]
    changed = base.catalog.copy()
    changed.loc[0, "name"] = "changed"
    third = load_semantic_index(changed, tmp_path, encoder=encoder, model_id="test-fixture")
    assert third.fingerprint != first.fingerprint
    assert calls == [4, 4, 4]


def test_comparison_uses_identical_queries_and_labels_proxy(engine) -> None:
    report = compare(engine, ["a", "b"], top_k=2)
    assert set(report["methods"]) == {"TF-IDF", "Semantic", "Hybrid"}
    assert all(method["queries"] == 2 for method in report["methods"].values())
    assert all(method["sanity_checks_passed"] for method in report["methods"].values())
    assert "not Precision@K" in report["methodology"]
