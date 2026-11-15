"""Independent product-ID joins and source consistency audit; never reprice records."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_catalog
from src.flipkart import (ROOT, REAL_DATA_PATH, file_sha256, parse_category, parse_image,
                          parse_price, source_quality_flags)
from src.semantic import SemanticIndex, load_semantic_index

SUSPECT_ID = "015b36dc0234a0a3ef68668e7cfc9c1c"
RADIUS_ID = "4fcc6115656b061c0a97b358d230a0f9"
AUDIT_FIELDS = ["name", "description", "category", "category_path", "price", "image_url",
                "source_pid", "rating", "quality_flags", "source_row_number"]


def source_rows_by_id(raw: pd.DataFrame) -> dict[str, tuple[int, pd.Series]]:
    """Independently join first valid original rows by string ID, never position."""
    rows = {}
    for number, (_, row) in enumerate(raw.fillna("").astype(str).iterrows(), start=1):
        product_id = row.get("uniq_id", "").strip() or row.get("pid", "").strip()
        if (product_id and row.get("product_name", "").strip()
                and parse_price(row.get("discounted_price", "")) is not None):
            rows.setdefault(product_id, (number, row))
    return rows


def expected_metadata(number: int, row: pd.Series) -> dict:
    """Explicit raw-field mapping independent of clean_flipkart's DataFrame assignments."""
    category, category_path = parse_category(row.get("product_category_tree", ""))
    rating = pd.to_numeric(row.get("product_rating", ""), errors="coerce")
    return {"name": row.product_name.strip(), "description": row.get("description", "").strip(),
            "category": category, "category_path": category_path,
            "price": parse_price(row.discounted_price), "image_url": parse_image(row.get("image", "")),
            "source_pid": row.get("pid", "").strip(),
            "rating": float(rating) if pd.notna(rating) and 0 <= rating <= 5 else "",
            "quality_flags": ";".join(source_quality_flags(row)), "source_row_number": str(number)}


def audit_alignment(raw: pd.DataFrame, catalog: pd.DataFrame) -> dict:
    """Check every retained product's fields against its original source row.

    Optional fields are checked when present. Numeric comparisons use numeric
    values rather than CSV formatting; all text must match documented parsing.
    """
    source = source_rows_by_id(raw)
    mismatches = []
    unknown_ids = []
    checked_fields = [field for field in AUDIT_FIELDS if field in catalog]
    for _, actual in catalog.iterrows():
        product_id = str(actual.product_id).strip()
        if product_id not in source:
            unknown_ids.append(product_id)
            continue
        number, original = source[product_id]
        expected = expected_metadata(number, original)
        for field in checked_fields:
            value = actual[field]
            value = "" if pd.isna(value) else value
            if field in {"price", "rating"}:
                normalized = pd.to_numeric(value, errors="coerce")
                equal = (pd.isna(normalized) and expected[field] == "") or normalized == expected[field]
            else:
                equal = str(value) == str(expected[field])
            if not equal:
                mismatches.append({"product_id": product_id, "source_row_number": number,
                                   "field": field, "expected": expected[field], "actual": str(value)})
    duplicate_ids = int(catalog.product_id.duplicated().sum())
    counts = Counter(mismatch["field"] for mismatch in mismatches)
    return {"catalog_rows": len(catalog), "checked_fields": checked_fields,
            "field_mismatch_counts": {field: counts[field] for field in checked_fields},
            "mismatched_product_count": len({row["product_id"] for row in mismatches}),
            "unknown_ids": unknown_ids, "duplicate_ids": duplicate_ids, "mismatches": mismatches,
            "passed": not mismatches and not unknown_ids and duplicate_ids == 0}


def audit_semantic_alignment(catalog: pd.DataFrame, semantic: SemanticIndex) -> dict:
    """Verify all FAISS positions, vectors, catalog IDs and public ID-position map."""
    expected_ids = sorted(catalog.product_id.astype(str))
    ids_match = semantic.product_ids == expected_ids
    positions_match = semantic.positions == {pid: index for index, pid in enumerate(expected_ids)}
    vectors_match = bool(np.allclose(
        semantic.index.reconstruct_n(0, semantic.index.ntotal), semantic.embeddings, atol=1e-6))
    return {"positions_checked": len(expected_ids), "canonical_ids_match": ids_match,
            "id_position_mapping_matches": positions_match, "index_vectors_match_embeddings": vectors_match,
            "fingerprint": semantic.fingerprint,
            "passed": ids_match and positions_match and vectors_match}


def audit_source(raw: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Flag review candidates across ALL original records without deleting them."""
    findings = []
    for number, (_, row) in enumerate(raw.fillna("").astype(str).iterrows(), start=1):
        flags = source_quality_flags(row)
        if flags:
            findings.append({"source_row_number": number, "product_id": row.get("uniq_id", "") or row.get("pid", ""),
                             "name": row.get("product_name", ""), "flags": ";".join(flags),
                             "discounted_price": row.get("discounted_price", ""),
                             "description": row.get("description", "")})
    counts = Counter(code for finding in findings for code in finding["flags"].split(";"))
    rules = ["description_price_differs_from_discounted_price", "product_url_pid_mismatch",
             "malformed_product_url", "no_substantive_title_description_overlap"]
    return ({"raw_rows_checked": len(raw), "flagged_rows": len(findings),
             "flag_counts": {code: counts[code] for code in rules},
             "policy": "Review flags only. No replacement descriptions/prices; no automatic semantic exclusions.",
             "limitations": "Rule-based screening is not proof of semantic correctness or exhaustive corruption detection."},
            pd.DataFrame(findings, columns=["source_row_number", "product_id", "name", "flags", "discounted_price", "description"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--sample", type=Path, default=REAL_DATA_PATH)
    parser.add_argument("--all", type=Path, default=ROOT / "data" / "processed" / "flipkart_all.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "data_quality_audit.json")
    args = parser.parse_args()
    raw_path = args.raw or next((path for path in [ROOT / "data/raw/flipkart_com-ecommerce_sample.csv",
                                                  ROOT / "flipkart_com-ecommerce_sample.csv"] if path.is_file()), None)
    if raw_path is None:
        parser.error("Real raw Flipkart CSV not found; pass --raw.")
    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    sample = load_catalog(args.sample)
    full = load_catalog(args.all)
    semantic = load_semantic_index(sample)
    report, findings = audit_source(raw)
    report.update(raw_sha256=file_sha256(raw_path), sample_sha256=file_sha256(args.sample),
                  catalogs={"sample": audit_alignment(raw, sample), "all": audit_alignment(raw, full)},
                  semantic=audit_semantic_alignment(sample, semantic))
    source = source_rows_by_id(raw)
    report["traced_products"] = {}
    for product_id in (SUSPECT_ID, RADIUS_ID):
        if product_id not in source:
            continue
        number, original = source[product_id]
        report["traced_products"][product_id] = {
            "raw_index_zero_based": number - 1, "source_data_record_number_one_based": number,
            "raw_fields": {field: original.get(field, "") for field in ["uniq_id", "pid", "product_name", "description",
                            "product_category_tree", "discounted_price", "image", "product_url"]},
            "sample_faiss_position": semantic.positions.get(product_id),
            "sample_metadata": sample.loc[sample.product_id.eq(product_id)].fillna("").to_dict(orient="records"),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    findings.to_csv(args.output.with_name("source_quality_flags.csv"), index=False)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"raw_rows_checked": len(raw), "review_flagged_rows": report["flagged_rows"],
                      "source_flag_counts": report["flag_counts"],
                      "catalog_alignment_passed": {k: v["passed"] for k, v in report["catalogs"].items()},
                      "semantic": report["semantic"]}, indent=2))
    if not report["semantic"]["passed"] or not all(value["passed"] for value in report["catalogs"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
