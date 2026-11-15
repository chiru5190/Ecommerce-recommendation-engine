"""Execute the Streamlit script and exercise its main user controls."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def demo_catalog(monkeypatch):
    """Keep all original demo UI regressions deterministic as the real default evolves."""
    monkeypatch.setenv("CATALOG_PATH", str(APP.parent / "data" / "sample_products.csv"))


def test_app_selection_search_and_filters() -> None:
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert len(app.selectbox) == 2
    assert any(item.label == "Cosine similarity" for item in app.metric)
    assert next(item for item in app.metric if item.label == "Price (INR)").value == "₹2,499"
    assert any("Price (INR): ₹1,999" in item.value for item in app.caption)
    assert any("illustrative INR prices" in item.value for item in app.info)
    app.selectbox[0].select("DEMO002").run()
    assert not app.exception
    app.checkbox[0].check().run()
    app.number_input[0].set_value(app.number_input[1].value + 1).run()
    assert any("Minimum price" in item.value for item in app.warning)
    app.text_input[1].set_value("no-such-product-xyz").run()
    assert any("No products match" in item.value for item in app.info)
    assert not app.exception


def test_app_missing_catalog() -> None:
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app.text_input[0].set_value(str(APP.parent / "data" / "missing.csv")).run()
    assert app.error
    assert not app.exception
