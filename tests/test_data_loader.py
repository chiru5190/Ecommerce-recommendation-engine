"""Catalog schema and cleanup regression tests."""

import pandas as pd
import pytest

from src.data_loader import CatalogError, REQUIRED_COLUMNS, load_catalog, validate_catalog


def test_bundled_catalog() -> None:
    catalog = load_catalog()
    assert len(catalog) >= 60
    assert catalog.category.nunique() >= 5
    assert catalog.product_id.is_unique
    assert catalog.price.ge(0).all()


def test_cleanup_and_first_valid_duplicate() -> None:
    frame = pd.DataFrame([
        [" 001 ", "Bad", "A", None, "oops"],
        ["001", "First", None, None, "12.5"],
        ["001", "Second", "A", "description", "15"],
        ["002", "Negative", "A", "", "-1"],
        ["003", "Infinity", "A", "", "inf"],
        ["", "No ID", "A", "", "1"],
        ["004", " ", "A", "", "1"],
    ], columns=REQUIRED_COLUMNS)
    with pytest.warns(UserWarning, match="rejected 5 invalid rows; removed 1 duplicate"):
        result = validate_catalog(frame)
    assert result.product_id.tolist() == ["001"]
    assert result.iloc[0]["name"] == "First"
    assert result.iloc[0]["category"] == "Uncategorized"
    assert result.iloc[0]["description"] == ""
    assert result.iloc[0]["image_url"] == ""
    assert result.iloc[0]["price"] == 12.5


def test_csv_preserves_ids_and_missing_description(tmp_path) -> None:
    path = tmp_path / "catalog.csv"
    path.write_text("product_id,name,category,description,price\n001,Mouse,Tech,,0\n", encoding="utf-8")
    result = load_catalog(path)
    assert result.iloc[0]["product_id"] == "001"
    assert result.iloc[0]["description"] == ""


def test_missing_schema() -> None:
    with pytest.raises(CatalogError, match="Missing required columns"):
        validate_catalog(pd.DataFrame({"name": ["A"]}))


def test_header_only_catalog(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text(",".join(REQUIRED_COLUMNS) + "\n", encoding="utf-8")
    assert load_catalog(path).empty


def test_empty_file_and_missing_file(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    path.touch()
    for candidate in [path, tmp_path / "missing.csv"]:
        with pytest.raises(CatalogError, match="Cannot read catalog"):
            load_catalog(candidate)
