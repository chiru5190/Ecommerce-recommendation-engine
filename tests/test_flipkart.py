"""Controlled raw-field fixtures; not substitute real catalog data."""

import pandas as pd
import pytest

from src.flipkart import clean_flipkart, diverse_sample, parse_category, parse_image, parse_price, prepare


@pytest.mark.parametrize("raw,expected", [
    ("1,299", 1299), ("₹1,29,999.50", 129999.5), ("Rs. 799", 799),
    ("INR 2499", 2499), ("0", 0), ("12,34", None), ("nan", None),
    ("inf", None), ("-1", None), ("", None), ("12 USD", None),
])
def test_price_parser(raw, expected) -> None:
    assert parse_price(raw) == expected


def test_category_and_image_parsing() -> None:
    assert parse_category('["Clothing >> Women >> T-Shirts"]') == (
        "Clothing", "Clothing > Women > T-Shirts")
    assert parse_category("['Home >> Kitchen']") == ("Home", "Home > Kitchen")
    assert parse_category("[malformed") == ("Uncategorized", "")
    assert parse_category("") == ("Uncategorized", "")
    assert parse_image('["javascript:bad", "https://img.example.com/a.jpg"]') == "https://img.example.com/a.jpg"
    assert parse_image("[broken") == ""
    assert parse_image("https://user:pass@example.com/a.jpg") == ""


def raw_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "uniq_id": ["a", "a", "", "c", "d", "e", "f", "g"],
        "pid": ["pid-a", "pid-a", "pid-b", "pid-c", "pid-d", "pid-e", "pid-a", "pid-g"],
        "product_name": ["Original name", "duplicate", "Fallback name", "Invalid price", "", "Other", "same pid", "Last"],
        "product_category_tree": ['["A >> Item"]'] * 6 + ['["B >> Item"]', '["B >> Other"]'],
        "description": [""] * 8,
        "discounted_price": ["1299", "999", "₹1,999", "oops", "55", "49", "99", "149"],
        "image": ["[broken"] * 8,
        "product_rating": ["4.5", "No rating available", "invalid", "8", "", "3", "", ""],
    })


def test_cleaning_counts_names_prices_ids_and_ratings() -> None:
    cleaned, report = clean_flipkart(raw_fixture())
    assert cleaned.product_id.tolist() == ["a", "e", "g", "pid-b"]
    assert report["raw_rows"] == 8 and report["cleaned_rows"] == 4
    assert report["filtering"] == {"missing_id": 0, "missing_name": 1,
        "invalid_or_missing_discounted_price": 1, "duplicate_product_id": 1, "duplicate_source_pid": 1}
    first = cleaned.iloc[0]
    assert first["name"] == "Original name" and first.price == 1299 and first.rating == 4.5
    assert cleaned.image_url.eq("").all()
    assert cleaned.description.eq("").all()
    assert cleaned.set_index("product_id").loc["pid-b", "price"] == 1999


def test_diverse_sample_is_order_independent_and_supports_all() -> None:
    cleaned, _ = clean_flipkart(raw_fixture())
    sample = diverse_sample(cleaned, 2)
    pd.testing.assert_frame_equal(sample, diverse_sample(cleaned.sample(frac=1, random_state=2), 2))
    assert set(sample.category) == {"A", "B"}
    assert len(diverse_sample(cleaned, 0)) == len(cleaned)


def test_preparation_report_and_schema(tmp_path) -> None:
    raw = tmp_path / "raw.csv"
    raw_fixture().to_csv(raw, index=False)
    report = prepare(raw, tmp_path / "processed", 2)
    assert report["sample_rows"] == 2
    assert report["license"] == "CC BY-SA 4.0"
    assert len(report["raw_sha256"]) == 64
    assert (tmp_path / "processed" / "cleaning_report.json").is_file()


def test_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="Not a Flipkart CSV"):
        clean_flipkart(pd.DataFrame({"name": ["fake"]}))
