"""Load and validate local product catalogs without network access."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["product_id", "name", "category", "description", "price"]
CATALOG_COLUMNS = REQUIRED_COLUMNS + ["image_url"]
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_products.csv"


class CatalogError(ValueError):
    """The catalog is unreadable or does not follow the required schema."""


def validate_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy; warn about rejected rows and duplicate IDs.

    Blank IDs/names and nonfinite or negative prices are rejected. Missing
    descriptions/images become empty strings; missing categories become
    'Uncategorized'. Keep the first valid occurrence of each stripped ID.
    """
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise CatalogError(f"Missing required columns: {', '.join(sorted(missing))}")
    data = frame.copy()
    for column in ["product_id", "name", "category", "description", "image_url"]:
        if column not in data:
            data[column] = ""
        data[column] = data[column].fillna("").astype(str).str.strip()
    data["category"] = data["category"].replace("", "Uncategorized")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    valid = (data["product_id"].ne("") & data["name"].ne("")
             & np.isfinite(data["price"]) & data["price"].ge(0))
    invalid_count = int((~valid).sum())
    data = data.loc[valid].copy()
    duplicate_count = int(data.duplicated("product_id").sum())
    data = data.drop_duplicates("product_id", keep="first")
    if invalid_count or duplicate_count:
        warnings.warn(
            f"Catalog cleanup: rejected {invalid_count} invalid rows; removed "
            f"{duplicate_count} duplicate IDs (kept first valid occurrence).",
            UserWarning, stacklevel=2,
        )
    optional = [column for column in ("rating", "category_path", "source_pid", "quality_flags", "source_row_number") if column in data]
    if "quality_flags" in data:
        data["quality_flags"] = data["quality_flags"].fillna("").astype(str)
    if "rating" in data:
        ratings = pd.to_numeric(data["rating"], errors="coerce")
        data["rating"] = ratings.where(ratings.between(0, 5))
    return data[CATALOG_COLUMNS + optional].reset_index(drop=True)


def load_catalog(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read UTF-8 CSV, preserving string IDs (including leading zeroes)."""
    try:
        data = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise CatalogError(f"Cannot read catalog '{path}': {exc}") from exc
    return validate_catalog(data)
