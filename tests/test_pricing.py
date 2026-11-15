"""Synthetic INR plausibility, regeneration and formatting regressions."""

from decimal import Decimal

import pandas as pd
import pytest

from scripts.generate_demo import PRICE_BANDS, main as generate_demo
from src.data_loader import load_catalog
from src.formatting import format_inr
from src.recommender import ProductRecommender


def test_positive_numeric_prices_and_complete_type_policy() -> None:
    catalog = load_catalog().set_index("product_id")
    assert pd.api.types.is_numeric_dtype(catalog.price)
    assert catalog.price.gt(0).all()
    covered = []
    for product_type, numbers, low, high in PRICE_BANDS:
        for number in numbers:
            product_id = f"DEMO{number:03d}"
            covered.append(product_id)
            assert low <= catalog.loc[product_id, "price"] <= high, product_type
    assert len(covered) == len(set(covered)) == 72
    assert set(covered) == set(catalog.index)


@pytest.mark.parametrize("name,low,high", [
    ("Sport Wireless Earbuds", 799, 4999),
    ("Wireless Studio Headphones", 999, 7999),
    ("Classic Cotton T-Shirt", 299, 1499),
    ("Compact Hiking Backpack", 499, 2999),
    ("USB-C Travel Charger", 999, 3499),
    ("Ceramic Breakfast Mug", 149, 599),
    ("Pocket Travel Notebook", 49, 249),
    ("Rich Night Moisturizer", 349, 1499),
])
def test_representative_type_ranges(name, low, high) -> None:
    catalog = load_catalog()
    assert low <= catalog.loc[catalog.name.eq(name), "price"].item() <= high


def test_generator_matches_catalog_and_is_deterministic(tmp_path) -> None:
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    generate_demo(first)
    generate_demo(second)
    assert first.read_bytes() == second.read_bytes()
    pd.testing.assert_frame_equal(load_catalog(first), load_catalog())
    with pytest.raises(FileExistsError):
        generate_demo(first)


@pytest.mark.parametrize("amount,expected", [
    (1299, "₹1,299"), (1299.0, "₹1,299"), (0, "₹0"),
    (99, "₹99"), (29999, "₹29,999"), (149999, "₹1,49,999"),
    (12345678, "₹1,23,45,678"), (1299.5, "₹1,299.50"),
    (Decimal("1299.995"), "₹1,300"),
])
def test_indian_currency_formatting(amount, expected) -> None:
    assert format_inr(amount) == expected


@pytest.mark.parametrize("amount", [-1, float("nan"), float("inf")])
def test_invalid_currency_amount(amount) -> None:
    with pytest.raises(ValueError):
        format_inr(amount)


def test_inr_filters_are_inclusive_and_do_not_change_ranking() -> None:
    engine = ProductRecommender(load_catalog())
    unfiltered = engine.recommend("DEMO001", top_k=72)
    expected = unfiltered.loc[
        unfiltered.category.eq("Electronics") & unfiltered.price.between(1299, 1999)
    ].reset_index(drop=True)
    filtered = engine.recommend("DEMO001", top_k=72, category="Electronics",
                                min_price=1299, max_price=1999)
    pd.testing.assert_frame_equal(filtered, expected)
    assert engine.recommend("DEMO001", min_price=1999, max_price=1999).product_id.tolist() == ["DEMO002", "DEMO006"]
