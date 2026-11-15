"""Auditable preparation of the real PromptCloud Flipkart historical catalog."""

import ast
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

import pandas as pd

SOURCE_URL = "https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products"
SOURCE_LICENSE = "CC BY-SA 4.0"
ROOT = Path(__file__).resolve().parents[1]
REAL_DATA_PATH = ROOT / "data" / "processed" / "flipkart_5000.csv"


def file_sha256(path: Path) -> str:
    """Hash files incrementally, including the original immutable raw CSV."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_price(value: object) -> float | None:
    """Parse a historical INR price; reject malformed grouping, negatives and NaN."""
    text = re.sub(r"^(?:₹|INR\s*|Rs\.?\s*)", "", str(value).strip(), flags=re.I).strip()
    numeric = r"(?:\d+|\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})*,\d{3})(?:\.\d+)?"
    if not re.fullmatch(numeric, text):
        return None
    price = float(text.replace(",", ""))
    return price if math.isfinite(price) and price >= 0 else None


def parse_list(value: str) -> list[str] | None:
    """Safely parse JSON/Python literal lists; never evaluate executable code."""
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError, RecursionError):
            return None
    return parsed if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed) else None


def parse_category(value: object) -> tuple[str, str]:
    """Return the first path's root and full path; malformed lists are uncategorized."""
    text = str(value).strip()
    if not text:
        return "Uncategorized", ""
    paths = parse_list(text)
    if paths is not None:
        text = paths[0] if paths else ""
    elif text.startswith(("[", "{")):
        return "Uncategorized", ""
    parts = [part.strip() for part in text.split(">>") if part.strip()]
    return (parts[0], " > ".join(parts)) if parts else ("Uncategorized", "")


def valid_image_url(value: str) -> bool:
    """Accept syntactically valid HTTP(S) URLs; availability is not assumed."""
    try:
        url = urlsplit(value)
        return bool(url.scheme in {"http", "https"} and url.hostname and "." in url.hostname
                    and not url.username and not url.password
                    and not re.search(r"[\s<>\"'\\]", value))
    except ValueError:
        return False


def parse_image(value: object) -> str:
    """Choose the first valid image URL; omit malformed or unavailable metadata."""
    text = str(value).strip()
    values = parse_list(text)
    if values is None:
        values = [text] if not text.startswith("[") else []
    return next((url.strip() for url in values if valid_image_url(url.strip())), "")


def source_quality_flags(row: pd.Series) -> list[str]:
    """Conservative review flags, never grounds for invented metadata or repricing.

    Embedded price differences may be stale listing text. Token/URL conflicts are
    review signals, not proof of corruption; no row is automatically removed.
    """
    flags = []
    description = str(row.get("description", ""))
    marker = re.search(r"\bPrice:\s*(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d+)?)", description, re.I)
    actual = parse_price(row.get("discounted_price", ""))
    if marker and actual is not None and parse_price(marker.group(1)) != actual:
        flags.append("description_price_differs_from_discounted_price")
    try:
        url_pid = parse_qs(urlsplit(str(row.get("product_url", ""))).query).get("pid", [""])[0]
        pid = str(row.get("pid", "")).strip()
        if url_pid and pid and url_pid != pid:
            flags.append("product_url_pid_mismatch")
    except ValueError:
        flags.append("malformed_product_url")
    stop = {"the", "for", "and", "with", "of", "in", "by", "on", "men", "women", "boys",
            "girls", "black", "white", "multicolor", "pack"}
    def tokens(value: str) -> set[str]:
        return {word for word in re.findall(r"[a-z0-9]+", value.casefold())
                if len(word) > 2 and word not in stop}
    title_terms, description_terms = tokens(str(row.get("product_name", ""))), tokens(description)
    if len(title_terms) >= 3 and len(description_terms) >= 8 and not title_terms & description_terms:
        flags.append("no_substantive_title_description_overlap")
    return flags


def clean_flipkart(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Map fields, sequentially filter invalid rows and retain genuine ratings.

    IDs use uniq_id, falling back to pid. The first valid product ID and source
    pid win; same-name variants with different IDs are retained. No price imputation,
    clipping, or invented names/ratings. Historical zero prices are retained.
    """
    required = {"product_name", "product_category_tree", "description", "discounted_price", "image"}
    missing = required - set(raw.columns)
    if missing or not ({"uniq_id", "pid"} & set(raw.columns)):
        raise ValueError(f"Not a Flipkart CSV: missing fields {sorted(missing)} or ID column.")
    raw = raw.fillna("").astype(str)
    data = pd.DataFrame(index=raw.index)
    uid = raw.get("uniq_id", pd.Series("", index=raw.index)).str.strip()
    pid = raw.get("pid", pd.Series("", index=raw.index)).str.strip()
    data["product_id"] = uid.where(uid.ne(""), pid)
    data["source_pid"] = pid
    data["name"] = raw.product_name.str.strip()
    parsed_categories = raw.product_category_tree.map(parse_category)
    data["category"] = parsed_categories.map(lambda value: value[0])
    data["category_path"] = parsed_categories.map(lambda value: value[1])
    data["description"] = raw.description.str.strip()
    data["price"] = raw.discounted_price.map(parse_price)
    data["image_url"] = raw.image.map(parse_image)
    data["quality_flags"] = raw.apply(lambda row: ";".join(source_quality_flags(row)), axis=1)
    # One-based data-record number, excluding header; not a physical text line
    # because descriptions in this CSV contain embedded newlines.
    data["source_row_number"] = range(1, len(raw) + 1)
    ratings = pd.to_numeric(raw.get("product_rating", pd.Series("", index=raw.index)), errors="coerce")
    data["rating"] = ratings.where(ratings.between(0, 5))
    report = {"raw_rows": len(raw), "id_fallback_to_pid": int((uid.eq("") & pid.ne("")).sum()),
              "filtering": {}, "currency": "INR", "price_policy": "preserve discounted_price; no retail_price fallback"}
    for reason, invalid in [
        ("missing_id", data.product_id.eq("")),
        ("missing_name", data.name.eq("")),
        ("invalid_or_missing_discounted_price", data.price.isna()),
    ]:
        reject = invalid.reindex(data.index)
        report["filtering"][reason] = int(reject.sum())
        data = data.loc[~reject].copy()
    duplicate = data.duplicated("product_id")
    report["filtering"]["duplicate_product_id"] = int(duplicate.sum())
    data = data.loc[~duplicate].copy()
    duplicate = data.source_pid.ne("") & data.duplicated("source_pid")
    report["filtering"]["duplicate_source_pid"] = int(duplicate.sum())
    data = data.loc[~duplicate].copy()
    data = data.sort_values("product_id").reset_index(drop=True)
    report.update(cleaned_rows=len(data), missing_descriptions_retained=int(data.description.eq("").sum()),
                  uncategorized_retained=int(data.category.eq("Uncategorized").sum()),
                  blank_or_invalid_images_retained=int(data.image_url.eq("").sum()),
                  missing_or_invalid_ratings_retained=int(data.rating.isna().sum()),
                  historical_zero_prices_retained=int(data.price.eq(0).sum()),
                  categories=int(data.category.nunique()))
    report["quality_policy"] = "Review flags retained; no semantic outlier exclusion, no metadata replacement."
    report["quality_flagged_rows_retained"] = int(data.quality_flags.ne("").sum())
    report["quality_flag_counts_retained"] = {
        code: int(data.quality_flags.str.split(";").map(lambda values: code in values).sum())
        for code in sorted({flag for value in data.quality_flags for flag in value.split(";") if flag})
    }
    assert len(raw) - sum(report["filtering"].values()) == len(data)
    return data, report


def diverse_sample(data: pd.DataFrame, size: int = 5000) -> pd.DataFrame:
    """Deterministic round-robin by root category and SHA256(product_id).

    Small categories are included early; this intentionally rebalances the source
    rather than claiming to preserve category prevalence. Zero requests all rows.
    """
    if size < 0:
        raise ValueError("Sample size must be nonnegative (0 means all).")
    ranked = data.copy()
    ranked["_hash"] = ranked.product_id.map(lambda pid: hashlib.sha256(pid.encode()).hexdigest())
    ranked = ranked.sort_values(["category", "_hash", "product_id"])
    ranked["_round"] = ranked.groupby("category").cumcount()
    ranked = ranked.sort_values(["_round", "category", "_hash", "product_id"])
    if size:
        ranked = ranked.head(size)
    return ranked.drop(columns=["_hash", "_round"]).sort_values("product_id").reset_index(drop=True)


def prepare(raw_path: Path, output_dir: Path, size: int = 5000) -> dict:
    """Write a full cleaned CSV, chosen sample and machine-readable audit."""
    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    data, report = clean_flipkart(raw)
    sample = diverse_sample(data, size)
    if sample.empty:
        raise ValueError("No valid Flipkart products remain after cleaning.")
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "flipkart_all.csv", index=False)
    sample_name = f"flipkart_{size}.csv" if size else "flipkart_all.csv"
    sample.to_csv(output_dir / sample_name, index=False)
    report.update(source=SOURCE_URL, attribution="PromptCloud; historical Flipkart product listings",
                  license=SOURCE_LICENSE, raw_file=raw_path.name, raw_sha256=file_sha256(raw_path),
                  sample_file=sample_name, sample_rows=len(sample), sample_sha256=file_sha256(output_dir / sample_name),
                  sampling="round-robin root categories, SHA256 product-ID order; output sorted by ID",
                  sample_category_counts={str(k): int(v) for k, v in sample.category.value_counts().sort_index().items()})
    (output_dir / "cleaning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
