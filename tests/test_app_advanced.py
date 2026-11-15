"""Graceful semantic fallback and real dataset integration when prepared locally."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.data_loader import DEFAULT_DATA_PATH
from src.flipkart import REAL_DATA_PATH
from src.semantic import SemanticUnavailable

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_semantic_error_shows_explicit_tfidf_fallback(monkeypatch) -> None:
    monkeypatch.setenv("CATALOG_PATH", str(DEFAULT_DATA_PATH))
    def unavailable(*args, **kwargs):
        raise SemanticUnavailable("Model deliberately unavailable in unit test.")
    monkeypatch.setattr("src.semantic.load_semantic_index", unavailable)
    app = AppTest.from_file(str(APP)).run(timeout=30)
    next(radio for radio in app.radio if radio.label == "Method").set_value("Hybrid").run(timeout=30)
    assert not app.exception
    assert any("Showing TF-IDF" in item.value for item in app.info)
    assert any(item.label == "Cosine similarity" for item in app.metric)


def test_prepared_real_catalog_and_all_modes(monkeypatch) -> None:
    if not REAL_DATA_PATH.is_file():
        pytest.skip("Optional real-data integration requires the downloaded/prepared Flipkart dataset.")
    monkeypatch.setenv("CATALOG_PATH", str(REAL_DATA_PATH))
    app = AppTest.from_file(str(APP)).run(timeout=60)
    assert not app.exception
    assert any("5,000 cleaned historical products" in item.value for item in app.info)
    assert not any("illustrative INR prices" in item.value for item in app.info)
    # Integration uses actual MiniLM caches only; unit vector fixtures are never substituted.
    for mode, label in [("Semantic", "Semantic cosine similarity"),
                        ("Hybrid", "Hybrid weighted similarity")]:
        next(radio for radio in app.radio if radio.label == "Method").set_value(mode).run(timeout=60)
        assert not app.exception
        if any("Showing TF-IDF" in item.value for item in app.info):
            pytest.skip("Optional real MiniLM cache is unavailable; fallback is tested separately.")
        assert any(item.label == label for item in app.metric)


def test_reported_product_metadata_stays_aligned_in_each_mode(monkeypatch) -> None:
    if not REAL_DATA_PATH.is_file():
        pytest.skip("Real prepared catalog not available.")
    monkeypatch.setenv("CATALOG_PATH", str(REAL_DATA_PATH))
    app = AppTest.from_file(str(APP)).run(timeout=60)
    app.selectbox[0].select("015b36dc0234a0a3ef68668e7cfc9c1c").run(timeout=60)
    for mode in ["TF-IDF", "Semantic", "Hybrid"]:
        next(radio for radio in app.radio if radio.label == "Method").set_value(mode).run(timeout=60)
        assert not app.exception
        assert any(item.value == "Tia by Ten on Ten Siana Women's T-Shirt Bra" for item in app.subheader)
        assert next(item.value for item in app.metric if item.label == "Price (INR)") == "₹399"
        assert any("Tia by Ten on Ten Siana" in item.value for item in app.markdown)
