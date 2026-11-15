"""Regression guards for independent source joins and deliberate metadata corruption."""

import numpy as np
import pandas as pd
import pytest

from src.audit import SUSPECT_ID, audit_alignment, audit_semantic_alignment, audit_source
from src.data_loader import load_catalog
from src.flipkart import ROOT, clean_flipkart, source_quality_flags
from src.hybrid import HybridRecommender
from src.recommender import ProductRecommender
from src.semantic import SemanticIndex, catalog_fingerprint


def fixture_raw() -> pd.DataFrame:
    """Tiny controlled fixture, not a substitute for the real dataset audit."""
    return pd.DataFrame({
        "uniq_id": ["bra-id", "audio-id"], "pid": ["BRA123", "AUD123"],
        "product_name": ["Cotton Everyday Bra", "Portable Audio Headphones"],
        "product_category_tree": ['["Clothing >> Bras"]', '["Electronics >> Headphones"]'],
        "description": ["Cotton Everyday Bra Price: Rs. 399", "Portable Audio Headphones Price: Rs. 22630"],
        "discounted_price": ["399", "22630"],
        "image": ['["https://images.example.com/bra.jpg"]', '["https://images.example.com/headphones.jpg"]'],
        "product_url": ["https://shop.example.com/bra?pid=BRA123", "https://shop.example.com/audio?pid=AUD123"],
        "product_rating": ["4", "3.5"],
    })


def test_source_join_is_by_id_not_row_position() -> None:
    raw = fixture_raw()
    catalog, _ = clean_flipkart(raw)
    report = audit_alignment(raw, catalog.iloc[::-1])
    assert report["passed"]
    assert set(report["field_mismatch_counts"].values()) == {0}


@pytest.mark.parametrize("field", ["name", "description", "category", "category_path", "price",
                                   "image_url", "source_pid", "rating", "source_row_number"])
def test_detects_cross_product_metadata_swap(field) -> None:
    raw = fixture_raw()
    catalog, _ = clean_flipkart(raw)
    catalog[field] = catalog[field].iloc[::-1].to_numpy()
    report = audit_alignment(raw, catalog)
    assert not report["passed"]
    assert report["mismatched_product_count"] == 2
    assert report["field_mismatch_counts"][field] == 2


def test_unknown_or_duplicate_ids_are_reported() -> None:
    raw = fixture_raw()
    catalog, _ = clean_flipkart(raw)
    catalog.loc[0, "product_id"] = "unknown"
    assert audit_alignment(raw, catalog)["unknown_ids"] == ["unknown"]
    catalog = pd.concat([catalog, catalog.iloc[[1]]])
    assert audit_alignment(raw, catalog)["duplicate_ids"] == 1


def test_price_conflict_is_flagged_without_replacement_or_exclusion() -> None:
    raw = fixture_raw()
    raw.loc[0, "description"] = "Cotton Everyday Bra Price: Rs. 499"
    catalog, report = clean_flipkart(raw)
    bra = catalog.set_index("product_id").loc["bra-id"]
    assert bra.price == 399
    assert "Rs. 499" in bra.description
    assert bra.quality_flags == "description_price_differs_from_discounted_price"
    assert report["cleaned_rows"] == 2
    assert report["quality_flagged_rows_retained"] == 1
    source_report, findings = audit_source(raw)
    assert source_report["flagged_rows"] == 1
    assert findings.iloc[0].product_id == "bra-id"


def test_cross_type_description_and_url_pid_are_flagged() -> None:
    row = fixture_raw().iloc[0].copy()
    row["description"] = "Radius earbuds headphones wireless bluetooth sound audio drivers bass stereo microphone"
    row["product_url"] = "https://shop.example.com/audio?pid=AUD123"
    flags = source_quality_flags(row)
    assert "no_substantive_title_description_overlap" in flags
    assert "product_url_pid_mismatch" in flags


def test_faiss_rejects_same_shape_but_reordered_vector_rows() -> None:
    faiss = pytest.importorskip("faiss")
    vectors = np.array([[1, 0], [0, 1]], dtype=np.float32)
    index = faiss.IndexFlatIP(2)
    index.add(vectors[::-1].copy())
    with pytest.raises(ValueError, match="FAISS positions"):
        SemanticIndex(["a", "b"], vectors, "test", index=index)


def test_stale_catalog_metadata_rejected_even_with_same_product_ids() -> None:
    pytest.importorskip("faiss")
    catalog, _ = clean_flipkart(fixture_raw())
    base = ProductRecommender(catalog)
    signature = catalog_fingerprint(base.catalog)
    semantic = SemanticIndex(base.catalog.product_id.tolist(), np.eye(2), signature,
                             catalog_signature=signature)
    HybridRecommender(base, semantic)
    catalog["description"] = catalog.description.iloc[::-1].to_numpy()
    with pytest.raises(ValueError, match="stale or misaligned"):
        HybridRecommender(ProductRecommender(catalog), semantic)


def test_semantic_audit_checks_public_mapping() -> None:
    pytest.importorskip("faiss")
    catalog, _ = clean_flipkart(fixture_raw())
    ids = sorted(catalog.product_id)
    semantic = SemanticIndex(ids, np.eye(2), "test")
    assert audit_semantic_alignment(catalog, semantic)["passed"]
    semantic.positions[ids[0]] = 1
    assert not audit_semantic_alignment(catalog, semantic)["passed"]


def test_genuine_reported_product_has_matching_raw_fields() -> None:
    path = ROOT / "flipkart_com-ecommerce_sample.csv"
    if not path.is_file():
        pytest.skip("Real raw CSV not downloaded.")
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    row = raw.loc[raw.uniq_id.eq(SUSPECT_ID)].iloc[0]
    assert row["product_name"] == "Tia by Ten on Ten Siana Women's T-Shirt Bra"
    assert row["discounted_price"] == "399"
    assert "Radius" not in row["description"]
    assert "Tia by Ten on Ten Siana" in row["description"]
    assert not source_quality_flags(row)
