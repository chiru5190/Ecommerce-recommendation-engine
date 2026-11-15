# Observed local verification

These results were produced with the actual downloaded Flipkart CSV, the pinned
MiniLM model, and CPU FAISS. They are not simulated results or recommendation
accuracy claims. Platform: Windows 11, Python 3.12.7, NumPy 2.1.3,
Scikit-learn 1.5.2, torch 2.5.1+cpu, Sentence Transformers 3.3.1, FAISS CPU 1.10.0.
Python 3.11 execution was not tested on this machine.

These are the post-audit results. See [DATA_QUALITY_AUDIT.md](DATA_QUALITY_AUDIT.md)
for the unreproduced Tia/Radius display investigation and independent metadata checks.

## Data and index

- Original rows: **20,000**; cleaned: **19,920**; initial sample: **5,000**.
- Removed: 78 missing/invalid discounted prices, 2 duplicate source pids.
- The sample represents all 262 parsed root categories; source taxonomy anomalies
  are preserved and documented instead of silently replaced.
- All 19,920 cleaned historical names and prices were checked against the original
  source by product ID. Full-catalog TF-IDF indexing/querying returned five results.
- Real sample embedding/index rebuild: **5,000 × 384** normalized vectors,
  **164.73 seconds** using the previously downloaded model and new alignment safeguards.
- Source review flags: 280 full-catalog records (81 in the sample) have embedded
  description prices differing from `discounted_price`; values are preserved.
- Model: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Dataset SHA256: `397b8999c9bf9729f6e400b4212777f9b99865cadac4ec351c344d5037f2ff3b`.
- Embedding fingerprint: `ded107b99dd0855974bad9eab5c9ee8fbf9e862df1238f251c21264f102c9d03`.

## Same-query method comparison

Command:

```powershell
.\.venv\Scripts\python.exe -m src.comparison --queries 200 --top-k 5 --seed 42 --output reports/model_comparison.json
```

Each method queried the same 200 sampled IDs, returned 1,000 total recommendations,
and passed sanity checks. Hybrid used 65% semantic and 35% TF-IDF cosine. No category
or price filters were active. One warm-up preceded each method's measurements.

| Method | Distinct recommended IDs | Coverage of 5,000 | Category agreement proxy | Mean query ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | 859 | 17.18% | 87.40% | 44.81 | 43.56 | 57.53 |
| Semantic | 854 | 17.08% | 86.90% | 20.66 | 20.39 | 25.42 |
| Hybrid | 861 | 17.22% | 88.00% | 21.04 | 20.79 | 25.19 |

The JSON artifact records exact query IDs, hashes, revision, runtime information and
unrounded measurements. Catalog TF-IDF build took 1.25 seconds; semantic cache load
took 0.55 seconds in this run. These startup costs are excluded from per-query timing.

This was one local run, with other verification work running during part of it;
timing is workload/hardware dependent and is not an isolated performance benchmark.
TF-IDF includes shared-term explanation generation; the other modes report component
scores. Latencies therefore include differing result/explanation work. Results do
not establish semantic/hybrid quality or speed superiority in other environments.

Coverage uses the whole 5,000-product denominator despite querying only 200 products.
Category agreement is partly circular because category is an input and source root
taxonomy is inconsistent. It is **not ground-truth Precision@K**. Genuine relevance
judgments are absent, so Precision@K, Recall@K, MRR and NDCG@K were not computed.

## Original full-query TF-IDF evaluation

```powershell
.\.venv\Scripts\python.exe -m src.evaluation --data data/processed/flipkart_5000.csv --output reports/tfidf_catalog_evaluation.json
```

5,000 queries yielded 25,000 recommendations, with 4,562 distinct recommended products:
**91.24%** all-query catalog coverage and **86.368%** category agreement proxy.
All six structural checks passed: self exclusion, unique results, descending scores,
bounded scores, at most K results, and known IDs. This all-query coverage cannot be
directly compared to the 200-query coverage in the table.

## Tests and startup

- `python -m pytest -q`: **101 passed, 1 warning in 13.61s**. No tests skipped locally.
- Tests include original demo regressions, controlled-vector math, missing optional
  dependency/model fallback, and actual 5,000-product UI execution in all three modes.
- `python scripts/smoke_test.py`: actual Streamlit health and root responses were
  **HTTP 200**; the spawned test server was terminated afterward.
- `python -m pip check`: **No broken requirements found.**
- One third-party warning: FAISS 1.10.0 imports a deprecated private NumPy namespace.
  It did not cause test failures; no global warning suppression was introduced.

## Remaining practical limits

The full 19,920-row semantic cache has not been built/timed; the CLI supports it with
the same model and a separate fingerprint. External historical images may fail and
have a browser-side placeholder. Browser visual screenshot review was not performed.
The initial Windows PyTorch/FAISS OpenMP conflict was resolved by isolating encoding
in a subprocess, and the successful real cache build verifies that path. Model and
index artifacts are gitignored and must be built/copied for another environment.
