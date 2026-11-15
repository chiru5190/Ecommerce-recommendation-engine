"""Conservative normalization that retains product model and unit tokens."""

import re
import unicodedata

import pandas as pd


def clean_text(text: str) -> str:
    """Normalize case/spacing; preserve letters, numbers, +, dots and hyphens."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w+.\-]+", " ", text)).strip()


def build_product_text(catalog: pd.DataFrame) -> pd.Series:
    """Combine product name, category and description without price or ID."""
    return catalog[["name", "category", "description"]].fillna("").agg(
        " ".join, axis=1
    ).map(clean_text)
