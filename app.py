"""Local Streamlit interface for content-based product discovery."""

from pathlib import Path
from html import escape
import os
import warnings

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.data_loader import CatalogError, DEFAULT_DATA_PATH, load_catalog
from src.formatting import format_inr
from src.flipkart import REAL_DATA_PATH, ROOT, file_sha256, valid_image_url
from src.hybrid import HybridRecommender
from src.recommender import ProductRecommender
from src.semantic import SemanticUnavailable, catalog_fingerprint, load_semantic_index

APP_TITLE = "AI-Powered E-Commerce Product Recommendation Engine"

st.set_page_config(page_title=APP_TITLE, page_icon="🛍️", layout="wide")

PAGE_STYLE = """
<style>
    .block-container { max-width: 1180px; padding-top: 2.5rem; padding-bottom: 3rem; }
    h1 { font-size: clamp(1.8rem, 3vw, 2.65rem) !important;
         line-height: 1.18 !important; letter-spacing: -0.035em; }
    h2 { letter-spacing: -0.02em; }
    h3 { line-height: 1.35 !important; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
    }
    [data-testid="stMetricValue"] { font-size: 1.65rem; font-weight: 650; }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; }
    @media (max-width: 640px) {
        .block-container { padding-top: 1.5rem; padding-left: 1rem; padding-right: 1rem; }
    }
</style>
"""


@st.cache_resource(show_spinner="Indexing product catalog…")
def get_engine(path: str, content_hash: str) -> tuple[ProductRecommender, list[str]]:
    """Cache the fitted sparse model, invalidating when CSV content changes."""
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always", UserWarning)
        engine = ProductRecommender(load_catalog(path))
    return engine, [str(notice.message) for notice in notices]


@st.cache_resource(show_spinner="Loading semantic index…")
def get_semantic(path: str, content_hash: str, model_fingerprint: str):
    """Reuse a fingerprinted index; model downloads are an explicit CLI step."""
    return load_semantic_index(load_catalog(path))


def show_metadata(product, show_images: bool) -> None:
    """Show genuine available ratings and a browser-side broken-image fallback."""
    rating = product.get("rating") if isinstance(product, pd.Series) else getattr(product, "rating", None)
    if rating is not None and pd.notna(rating):
        st.caption(f"Historical product rating: {float(rating):g}/5 (source data)")
    flags = product.get("quality_flags", "") if isinstance(product, pd.Series) else getattr(product, "quality_flags", "")
    if flags:
        st.caption("Source metadata review flag: " + str(flags).replace("_", " ").replace(";", "; "))
    image = product.get("image_url", "") if isinstance(product, pd.Series) else product.image_url
    if show_images and valid_image_url(image):
        # Escape URL attributes; no server-side remote requests or fabricated replacement images.
        components.html(
            '<div style="font:13px sans-serif;color:#aaa;height:150px">'
            '<span id="fallback" hidden>Historical product image unavailable.</span>'
            f'<img src="{escape(image, quote=True)}" alt="Product image" '
            'style="max-width:100%;height:140px;object-fit:contain" referrerpolicy="no-referrer" '
            'onerror="this.hidden=true;document.getElementById(\'fallback\').hidden=false">'
            '</div><script>setTimeout(()=>{const i=document.querySelector("img");'
            'if(!i.complete||!i.naturalWidth){i.hidden=true;'
            'document.getElementById("fallback").hidden=false;}},5000);</script>', height=155,
        )


def main() -> None:
    """Render selection, filters, product details and ranked recommendations."""
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)
    st.caption("CONTENT-BASED PRODUCT DISCOVERY")
    st.title(APP_TITLE)
    st.markdown(
        "Discover related products with **TF-IDF**, **semantic embeddings**, and "
        "**hybrid cosine similarity**, using product names, categories, and descriptions."
    )
    with st.sidebar:
        st.header("Explore the catalog")
        st.caption("Choose a product, then refine its recommendations.")
        with st.expander("Advanced Settings"):
            default = Path(os.environ.get("CATALOG_PATH", str(REAL_DATA_PATH if REAL_DATA_PATH.exists() else DEFAULT_DATA_PATH)))
            sources = {"Flipkart · 5,000-product sample": REAL_DATA_PATH,
                       "Flipkart · all valid products": ROOT / "data" / "processed" / "flipkart_all.csv",
                       "Synthetic demo · 72 fictional products": DEFAULT_DATA_PATH,
                       "Custom CSV": default}
            default_source = next((name for name, source_path in sources.items() if source_path == default), "Custom CSV")
            dataset_name = st.radio("Dataset", list(sources), index=list(sources).index(default_source))
            dataset = st.text_input("Local catalog CSV", value=str(sources[dataset_name]),
                                    help="Load a local CSV that follows the schema in README.md.")
            st.caption("Prepare real data with: python -m scripts.prepare_flipkart. The synthetic demo remains available.")
            show_images = st.toggle("Show available product images", value=default != DEFAULT_DATA_PATH,
                                    help="Historical external URLs may be broken. Images are optional.")
    try:
        path = Path(dataset).expanduser().resolve()
        content_hash = file_sha256(path)
        engine, notices = get_engine(str(path), content_hash)
    except (OSError, CatalogError, ValueError) as exc:
        st.error(f"Unable to load this catalog. {exc}")
        st.info("Check the file path and required columns in README.md. Prepare the downloaded raw file "
                "with python -m scripts.prepare_flipkart, or choose the explicitly labeled synthetic demo.")
        return
    for notice in notices:
        st.warning(notice)
    catalog = engine.catalog
    if catalog.empty:
        st.info("No valid products are available. Add products with IDs, names and valid prices.")
        return
    if path == DEFAULT_DATA_PATH.resolve():
        st.info("Synthetic demo catalog • Fictional products and illustrative INR prices for functional testing only.")
    elif path in {REAL_DATA_PATH.resolve(), (ROOT / "data" / "processed" / "flipkart_all.csv").resolve()}:
        st.info(f"Flipkart Products · {len(catalog):,} cleaned historical products · PromptCloud / Kaggle · "
                "CC BY-SA 4.0 · Historical INR prices and ratings; availability is not verified.")
    else:
        st.caption("Custom local catalog • Prices are interpreted as INR; provenance is supplied by the catalog owner.")
    st.caption(f"Dataset: {path.name} · Actual loaded product count: {len(catalog):,}")
    with st.sidebar:
        st.caption(f"{len(catalog)} products · {catalog['category'].nunique()} categories")
        st.subheader("1. Choose a product")
        query = st.text_input("Search name, category or ID", placeholder="e.g. headphones",
                              help="Search narrows the product selector below.")
        matches = catalog.loc[catalog[["product_id", "name", "category"]].agg(" ".join, axis=1)
                              .str.contains(query.strip(), case=False, regex=False)]
        if matches.empty:
            st.info("No products match your search. Try a broader term.")
            return
        labels = {row.product_id: f"{row.name} ({row.product_id})" for row in matches.itertuples()}
        product_id = st.selectbox("Select a product", matches["product_id"].tolist(),
                                  format_func=labels.get)
        st.divider()
        st.subheader("2. Refine recommendations")
        st.caption("These filters apply to recommended products.")
        category = st.selectbox("Recommendation category", ["All categories"] + sorted(catalog["category"].unique()),
                                help="Choose a category or search across the full catalog.")
        limit_prices = st.checkbox("Filter by price", help="Use inclusive minimum and maximum prices.")
        min_price = max_price = None
        if limit_prices:
            min_price = st.number_input("Minimum price (₹)", min_value=0.0, value=0.0, step=1.0)
            max_price = st.number_input("Maximum price (₹)", min_value=0.0,
                                        value=float(catalog["price"].max()), step=1.0)
            st.caption(f"Price range: {format_inr(min_price)} – {format_inr(max_price)}")
        st.divider()
        st.caption("All prices are INR (₹). Custom CSV prices must also be numeric INR values; no currency conversion is performed.")
        st.subheader("3. Recommendation method")
        mode = st.radio("Method", ["TF-IDF", "Semantic", "Hybrid"],
                        help="TF-IDF works without the optional model. Semantic and Hybrid require the local MiniLM index.")
        top_k = st.slider("Number of recommendations", min_value=1, max_value=20, value=5)
        semantic_weight = 0.65
        if mode == "Hybrid":
            semantic_weight = st.slider("Semantic weight", 0.0, 1.0, 0.65, 0.05)
            st.caption(f"Semantic {semantic_weight:.0%} + TF-IDF {1-semantic_weight:.0%}")
    advanced = HybridRecommender(engine)
    if mode != "TF-IDF":
        try:
            semantic = get_semantic(str(path), content_hash, catalog_fingerprint(catalog))
            advanced = HybridRecommender(engine, semantic)
        except (SemanticUnavailable, ValueError) as exc:
            st.warning(str(exc))
            st.info(f"Requested {mode} is unavailable. Showing TF-IDF results instead.")
            mode = "TF-IDF"
    selected = catalog.loc[catalog["product_id"].eq(product_id)].iloc[0]
    price_label = "Price (INR)"
    st.divider()
    st.subheader("Selected product")
    with st.container(border=True):
        details, price = st.columns([3, 1], gap="large")
        with details:
            st.caption(f"{selected['category']} · ID: {product_id}")
            st.subheader(selected["name"])
            st.write(selected["description"] or "No description provided.")
            show_metadata(selected, show_images)
        with price:
            st.metric(price_label, format_inr(selected["price"]))
    st.subheader("Recommended for this product")
    st.caption(
        f"Up to {top_k} matches · Active method: {mode}. "
        "TF-IDF cosine ranges from 0 to 1; semantic cosine and its hybrid weighted sum can range "
        "from −1 to 1. Only positive matches are shown. Scores are not confidence or probability."
    )
    try:
        results = advanced.recommend(product_id, top_k=top_k, category=None if category == "All categories" else category,
                                     min_price=min_price, max_price=max_price, mode=mode,
                                     semantic_weight=semantic_weight)
    except ValueError as exc:
        st.warning(str(exc))
        return
    if results.empty:
        st.info("No similar products match these controls. Broaden the filters or select another product.")
    else:
        st.caption(f"Showing {len(results)} matching product{'s' if len(results) != 1 else ''}")
    for rank, product in enumerate(results.itertuples(index=False), start=1):
        with st.container(border=True):
            details, score = st.columns([3, 1], gap="large")
            with details:
                st.caption(f"MATCH {rank:02d} · {product.category}")
                st.subheader(product.name)
                description = product.description or "No description provided."
                st.write(description[:320] + ("…" if len(description) > 320 else ""))
                show_metadata(product, show_images)
                st.caption(f"{price_label}: {format_inr(product.price)}")
                st.markdown("**Why this appears**")
                if mode == "TF-IDF":
                    shared_terms = product.explanation.removeprefix("Shared terms: ").removesuffix(".")
                    st.write(f"Shares these weighted terms with your selected product: {shared_terms}.")
                else:
                    st.write(product.explanation)
            with score:
                score_label = {"TF-IDF": "Cosine similarity", "Semantic": "Semantic cosine similarity",
                               "Hybrid": "Hybrid weighted similarity"}[mode]
                st.metric(score_label, f"{product.similarity_score:.3f}")
    st.divider()
    with st.expander("How recommendations work & dataset details"):
        st.write("Name, category and description are normalized and indexed with TF-IDF words and "
                 "two-word phrases. Cosine similarity ranks products by shared weighted terms. "
                 "Scores measure text overlap, not purchase likelihood. Zero-overlap products are omitted.")
        st.write("Each explanation lists up to three shared terms contributing most to the cosine "
                 "score. A shared category or phrase can contribute to a match; explanations do not "
                 "imply verified compatibility, personal preferences, or purchase behavior.")
        st.write("Semantic mode uses all-MiniLM-L6-v2 sentence embeddings normalized to unit length "
                 "and FAISS IndexFlatIP on CPU. Hybrid initially weights semantic cosine 65% and "
                 "TF-IDF cosine 35%; these weights are not tuned or evidence of superior quality.")
        st.write("TF-IDF requires no model download. The optional public semantic model requires a "
                 "one-time download. No customer history, paid APIs, API keys, GPU, or product images "
                 "are required. Historical catalog ratings are not recommendation relevance labels.")
        st.caption(f"Source file: {path.name} · {len(catalog)} valid products")


if __name__ == "__main__":
    main()
