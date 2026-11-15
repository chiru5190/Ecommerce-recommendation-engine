# AI-Powered E-Commerce Product Recommendation Engine

A local, CPU-capable Streamlit app for historical Flipkart product discovery.
It retains the original TF-IDF + cosine baseline and adds optional MiniLM semantic
retrieval and configurable hybrid scoring. Prices are numeric INR; cards use
Indian currency formatting. No paid API, API key, or GPU is needed.

The prepared default is **5,000 real products** from PromptCloud's Flipkart dataset.
The original **72-product synthetic demo** remains a separately labeled fallback.
The UI never silently substitutes synthetic data for a failed real-data load.

## Quick start — Windows PowerShell

Run from this project folder. Python 3.11-compatible code and direct dependencies
are used; actual verification here used **Python 3.12.7** because the registered
Python 3.11 executable was missing.

```powershell
# For a fresh environment (skip if .venv already exists):
py -3.11 -m venv .venv
# Or use a working Python 3.11/3.12 installation: python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.prepare_flipkart
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the localhost URL printed by Streamlit, normally http://localhost:8501.
Stop with Ctrl+C. Activation is optional; these commands avoid PowerShell execution
policy changes. Select `.venv\Scripts\python.exe` as the VS Code interpreter.
For macOS/Linux, use `.venv/bin/python` after creating a virtual environment.

TF-IDF is immediately usable with the baseline requirements. To enable real
semantic/hybrid modes on CPU:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-semantic.txt
.\.venv\Scripts\python.exe -m scripts.build_semantic --download
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The first semantic build downloads the pinned public model (~91 MB of model weights)
and encodes the catalog. This workspace already has the real-data cache built.
After setup, the app loads local models/caches without requesting downloads.
Unavailable optional dependencies or models produce an explicit TF-IDF fallback.
The baseline requirements file does not import or require PyTorch/FAISS.

## Dataset, provenance, and license

Source: [Flipkart Products on Kaggle](https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products).
Creator: **PromptCloud**, using its web-crawling service on Flipkart listings.
The [source data card](https://www.kaggle.com/PromptCloudHQ/flipkart-products/tasks)
lists **CC BY-SA 4.0**. Attribution and adaptations are recorded in
[data/DATASET_ATTRIBUTION.md](data/DATASET_ATTRIBUTION.md). Processed catalogs retain
that dataset license. This is historical product metadata, not customer interactions
or recommendation relevance judgments. Prices, ratings, and image links may be stale.
No contemporary retail-price corrections are applied to real records.

The downloaded `flipkart_com-ecommerce_sample.csv` was found in the workspace root.
If it is missing on another machine, sign in to Kaggle, open the source page, choose
Download, extract the ZIP, and place the original CSV under `data/raw/` (or the root).
No Kaggle client or credentials are required by the app itself. Download/access
requirements are determined by Kaggle. No alternate or fake dataset is substituted.
The raw file is never overwritten by preparation; it and generated CSVs are gitignored.

### Observed cleaning audit

| Stage/decision | Rows |
| --- | ---: |
| Original downloaded records | 20,000 |
| Missing ID rejected | 0 |
| Missing name rejected | 0 |
| Invalid/missing discounted price rejected | 78 |
| Duplicate selected product ID removed | 0 |
| Duplicate source pid removed | 2 |
| Valid cleaned products | 19,920 |
| Deterministic initial sample | 5,000 |

Among the cleaned rows, 2 missing descriptions, 3 blank/invalid image fields, and
18,081 unavailable/invalid ratings were retained as missing. There are 262 parsed
root categories; all occur in the 5,000-item sample. Some source paths contain only
a product title, which is retained as its root instead of inventing a category.

The final consistency audit flags **280** source records where an explicit
description price differs from `discounted_price`. Both original values are
preserved; these review flags do not trigger repricing or automatic exclusion.
Cards disclose flags when present. See [the final audit](reports/DATA_QUALITY_AUDIT.md)
for the Tia/Radius investigation and exact verification results.

[data/processed/cleaning_report.json](data/processed/cleaning_report.json) records
filtering counts, raw/sample SHA256, license, sampling policy, and sample category
counts. Its raw SHA256 is
`56f8f699c9e847356666c2eab3c3ab1244340f6a98ad08e39ea2199ebe993ad1`.

### Field mapping and decisions

| Raw field | Prepared field | Rule |
| --- | --- | --- |
| `uniq_id`, else `pid` | `product_id` | Preserve string ID; reject if both blank |
| `product_name` | `name` | Strip outer whitespace; no fabricated replacements |
| `product_category_tree` | `category` | Safely parse JSON/Python string list; first path's root |
| `product_category_tree` | `category_path` | Preserve first hierarchy as readable path |
| `description` | `description` | Keep text; missing becomes empty |
| `discounted_price` | `price` | Preserve historical INR number; no imputation/clipping |
| `image` | `image_url` | First syntactically valid HTTP(S) URL; missing is allowed |
| `product_rating` | `rating` | Genuine numeric 0–5 rating, otherwise missing |
| `pid` | `source_pid` | Preserve for source-product deduplication/audit |
| Original record position | `source_row_number` | One-based data record excluding header, not physical line |
| Source consistency rules | `quality_flags` | Review codes; no replacement metadata |

Price parsing accepts numeric INR and well-formed Indian/Western separators; rejects
negatives, infinities, NaN, malformed grouping, and missing values. Historical zero
prices would be retained, though none remain in this file. `retail_price` is not used
as an undocumented fallback. Filters are sequential, so rejection counts sum exactly.
The first valid ID/source-pid occurrence wins; same-name variants with distinct IDs
are retained. Missing/invalid categories become `Uncategorized`; malformed optional
images never remove an otherwise valid product. URLs are syntax checked, not claimed
to be reachable. No image downloads occur on the server.

### Reproduce preparation and scale up

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_flipkart
.\.venv\Scripts\python.exe -m scripts.prepare_flipkart --raw data/raw/flipkart_com-ecommerce_sample.csv --sample-size 5000
# 0 selects all valid rows; a full cleaned CSV is also always written by preparation:
.\.venv\Scripts\python.exe -m scripts.prepare_flipkart --sample-size 0
.\.venv\Scripts\python.exe -m scripts.build_semantic --data data/processed/flipkart_all.csv --download
```

The sample uses round-robin root categories with SHA256(product_id) ordering within
each category, then sorts output by ID. This is reproducible and category-diverse,
but deliberately rebalances source prevalence. It is not a random training/test split.
Select the sample, full cleaned catalog, synthetic demo, or custom schema-compatible
CSV in **Advanced Settings**. The UI shows the actual loaded count and dataset file.
You can also set `$env:CATALOG_PATH = 'data/processed/flipkart_all.csv'` before launch.
Full-catalog TF-IDF is supported; a separate embedding cache is required for its
semantic modes. The 19,920-item semantic build has not been timed here.

## Architecture and algorithms

```text
app.py                      Dark Streamlit UI, datasets, methods, filters, caching
src/data_loader.py          Schema validation; preserves optional source metadata
src/flipkart.py             Safe parsing, sequential cleaning, deterministic sampling
src/preprocessing.py        Shared normalized name/category/description text
src/recommender.py          Preserved original sparse TF-IDF/cosine baseline
src/semantic.py             Stable ID mapping, normalized FAISS index, disk caching
src/embedding_worker.py     CPU MiniLM encoding in a separate short-lived process
src/hybrid.py               Optional methods and exact weighted score blending
src/formatting.py           Indian INR grouping and paise handling
src/evaluation.py           Original full-query catalog sanity/coverage diagnostics
src/comparison.py           Identical-query method comparison and measured latency
scripts/prepare_flipkart.py Reproducible real-data preparation CLI
scripts/build_semantic.py   Explicit model download/index-build CLI
scripts/generate_demo.py    Original deterministic synthetic INR fallback generator
scripts/smoke_test.py       Local server health/root check and subprocess shutdown
reports/                    Actual comparison and catalog-evaluation JSON artifacts
```

**TF-IDF:** normalized product name, category and description; word unigrams/bigrams,
English stop words, sublinear TF, L2 normalization, and 30,000-feature cap. Cosine
scores are computed only for the selected item. Matching results are sorted by score
then ID; self/duplicate results and nonpositive scores are omitted. The original API
and shared-term explanations remain available:

```python
from src.data_loader import load_catalog
from src.recommender import ProductRecommender

baseline = ProductRecommender(load_catalog('data/processed/flipkart_5000.csv'))
product_id = baseline.catalog.iloc[0]['product_id']
results = baseline.recommend(product_id, top_k=5, min_price=500, max_price=5000)
```

**Semantic:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
Apache-2.0, pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
It encodes the same normalized text into 384-dimensional CPU float32 embeddings.
The model truncates at 256 tokens, so long descriptions are not completely represented.
L2-normalized vectors feed FAISS **IndexFlatIP**; inner product equals cosine.
PyTorch encoding is isolated from the FAISS process to avoid an observed Windows
OpenMP DLL conflict, without unsafe duplicate-runtime environment overrides.

**Hybrid:** `score = semantic_weight * semantic_cosine + (1 - semantic_weight) * tfidf_cosine`.
The initial configurable semantic weight is **0.65**, with **0.35** TF-IDF. These are
starting weights, not optimized or calibrated values. Raw component cosines are
blended without treating them as probabilities. TF-IDF lies in [0,1]; semantic
and hybrid scores can lie in [-1,1]. Only positive-scoring candidates are shown.
Explanations disclose component values and weights, not invented compatibility or
personal preference claims. Category/INR price filters are inclusive and exact.
Top-K is configurable (1–20 in the UI; the API supports larger K).

```python
from src.hybrid import HybridRecommender
from src.semantic import load_semantic_index

engine = HybridRecommender(baseline, load_semantic_index(baseline.catalog))
results = engine.recommend(product_id, top_k=5, mode='Hybrid', semantic_weight=0.65)
```

An exact FAISS search for a selected product returns one O(N) score row so filters
and hybrid ranking consider every eligible candidate; no approximate candidate pool
or dense all-products similarity matrix is used. Full evaluation has quadratic total
pair comparisons, but each query keeps only its own scores.

Caches use canonical sorted IDs, all cleaned catalog content, normalized text,
preprocessing schema, immutable model revision, and dependency versions as a SHA256
fingerprint. ID lists, normalized vectors, dimensions, and file checksums are checked
on load. Changed content or revision invalidates the cache; corruption triggers a
rebuild. Embeddings, FAISS index, and manifest are saved with atomic file replacements
under `artifacts/semantic/`. Streamlit separately caches resources using CSV content
hashes; it never reuses indexes by filename alone. Runtime caches/models are local
and gitignored. Changed model preprocessing requires a cache-schema version bump.

## UI and offline behavior

Search by product name/category/ID, choose a product, then set category, INR price,
Top-K and method controls. Hybrid exposes a semantic-weight slider. Cards show real
names/prices and source ratings where available. Historical images are optional;
syntactically valid URLs render in the browser with a broken/slow-image placeholder.
Images may fail or be blocked by browser policy; recommendations work without them.
No ratings, discounts, brands, purchases, or popularity statistics are synthesized.

If real preparation is absent, the default is explicitly disclosed as the synthetic
demo. Failed explicit real/custom loads show a useful error and offer preparation or
manual demo selection. Unknown IDs/empty catalogs/vocabularies/filter results are
handled by the engine; invalid bounds produce a friendly warning. Semantic failures
are explicitly disclosed before displaying TF-IDF results with TF-IDF score labels.

## Testing and evaluation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/smoke_test.py
.\.venv\Scripts\python.exe -m pip check
# Original all-query diagnostic (defaults to synthetic if --data is omitted):
.\.venv\Scripts\python.exe -m src.evaluation --data data/processed/flipkart_5000.csv --output reports/tfidf_catalog_evaluation.json
# Same 200 queries for all three methods; use --queries 0 for all catalog items:
.\.venv\Scripts\python.exe -m src.comparison --queries 200 --top-k 5 --seed 42 --output reports/model_comparison.json
```

All original baseline tests are retained. Original UI regressions explicitly select
the synthetic catalog to keep their assertions deterministic. New tests cover price
and category parsing, filtering decisions, preserved IDs, category-diverse sampling,
normalized FAISS retrieval, ID-aligned blending, filters, cache invalidation/corruption,
explicit semantic fallback, and real-data UI modes. Controlled vector fixtures test
math; they are never passed off as actual model evaluation. Optional FAISS/model/data
integration tests may skip on baseline-only machines and do not claim semantic success.

**Metrics:** catalog coverage = unique recommended IDs / full catalog size. Category
agreement = returned items sharing the query's root category / all returned items.
It is a metadata proxy, partly circular because category is an input, **not Precision@K**.
Neither metric establishes relevance, accuracy, or superiority. Source product ratings
are not query-product relevance judgments. Precision@K, Recall@K, MRR and NDCG@K are
therefore not computed. Add genuine/documented judgments before implementing/reporting
those metrics; do not derive them from categories and call them ground truth.

The comparison fixes catalog hash, seed, query IDs, K, model revision, and weight.
It warms each method once and records per-query mean/p50/p95 latency, excluding model
load, index build and images but including result assembly/explanations. Timing varies
by CPU, concurrent workload and environment. Sample-query coverage uses the full
5,000-product denominator; it must not be compared directly with all-query coverage.
See [reports/RESULTS.md](reports/RESULTS.md) for observed local outcomes and limitations.

## Reproduce the final data-quality audit

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_flipkart
.\.venv\Scripts\python.exe -m scripts.build_semantic
.\.venv\Scripts\python.exe -m src.audit
```

The audit independently joins each retained ID to its first valid raw source row
and checks names, descriptions, categories/paths, numeric prices, image URLs,
source pids, ratings, review flags and source record numbers across all 19,920
cleaned rows and the 5,000-row sample. It reconstructs all FAISS vectors and checks
their positions against the explicit ID/embedding mapping, not just array sizes.
Hybrid engines reject production caches with stale catalog metadata even when
the ID sets match. The audit changes bumped the cache schema and forced a rebuild.

Raw screening flags explicit description-price differences, URL query-pid conflicts,
malformed product URLs, and substantive title/description token disjointness.
These are review signals, not exhaustive semantic validation. Candidates are in
`reports/source_quality_flags.csv`; independent checks and source traces are in
`reports/data_quality_audit.json`. No source fields were replaced or repriced.

The reported bra currently shows ₹399 and its matching description in raw/prepared
files and all-mode UI tests. The original mixed display could not be reproduced,
so its past runtime cause is not established. See the audit report for details.

## Baseline synthetic catalog and schema

`data/sample_products.csv` and `scripts/generate_demo.py` retain the 72 fictional
products with fixed illustrative INR prices. They are functional fallback data, not
real training/evaluation data. Product-type assumptions and earlier checks remain in
[docs/synthetic_pricing.md](docs/synthetic_pricing.md). The generator refuses to
overwrite an existing catalog.

Custom CSVs require `product_id,name,category,description,price`; `image_url` is
optional. IDs/names must be nonblank and prices finite nonnegative numeric INR.
Missing descriptions/images are allowed; missing categories become `Uncategorized`.
The general loader warns about invalid rows and duplicate IDs. Optional `rating`,
`category_path`, and `source_pid` are retained; extra unrelated columns are ignored.
`quality_flags` and `source_row_number` are also retained when present.

## Screenshot placeholder

TODO: Capture the real-data app with its dataset disclosure, selected product,
method controls and recommendation cards. Automated AppTest and HTTP startup checks
were performed; no browser visual screenshot review is claimed.

## Limitations and next steps

Historical metadata and image URLs are stale; displayed prices are not live offers.
Missing ratings are common. Some category roots are product titles or tiny categories;
category agreement reflects these source inconsistencies. Balanced sampling changes
category prevalence. Neither ratings nor row frequency is used as popularity weighting,
but catalog composition, repeated seller templates and common features can concentrate
exposure. No claim of popularity neutrality, fairness or diversity is made.

This is item-to-item content retrieval without user personalization. New users can
browse but preferences are unknown; new/sparse products need useful metadata and a
rebuilt index. MiniLM may miss domain terminology and truncates long descriptions;
TF-IDF misses synonyms and depends on exact wording. English normalization is not
validated for multilingual products. Hybrid weights and uncalibrated score scales
need evaluation with genuine relevance labels before tuning or quality claims.

The full semantic catalog is supported but requires its own CPU build and cache.
Future work: document expert relevance judgments, measure labeled retrieval quality
and exposure, improve source-category taxonomy with an auditable mapping, test diverse
reranking, and consider approximate indexing only when exact search becomes insufficient.
Pin transitive dependencies for longer-term environment reproduction. No code has
been committed or pushed by the implementation workflow.
