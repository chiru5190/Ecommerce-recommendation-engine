# Final data-quality audit

## Reported Tia/Radius mismatch

The reported mixed display **could not be reproduced in the current workspace**.
The bra's original record and both prepared catalogs have matching bra metadata
and a price of **₹399**, not ₹22,630. No replacement description or price was made.

| Field | Bra | Separate Radius headphones |
| --- | --- | --- |
| Raw zero-based record index | 2249 | 7958 |
| Sample FAISS position | 27 | 1537 |
| One-based data record, excluding header | 2250 | 7959 |
| `uniq_id` / `product_id` | `015b36dc0234a0a3ef68668e7cfc9c1c` | `4fcc6115656b061c0a97b358d230a0f9` |
| Source `pid` | `BRAEB2S8ZUHV9YEA` | `ACCEEV5V5VD5DR9C` |
| Product name | Tia by Ten on Ten Siana Women's T-Shirt Bra | Radius Earbuds Radheadphones Hp-Rhf41P Headphones |
| Historical `discounted_price` | 399 | 22630 |

Record numbers are CSV **records**, not physical text lines: descriptions contain
embedded newlines. The exact raw category, description, price, image URLs, product
URL, prepared metadata and sample FAISS position for both IDs are preserved in
`reports/data_quality_audit.json` under `traced_products`.

For the bra, the source category is Clothing > Women's Clothing > Lingerie, Sleep &
Swimwear > Bras > Tia by Ten on Ten Bras. Its description begins with the same bra
title and `Price: Rs. 399`; its image path is `/image/bra/...`. The Radius record
contains the Radius headphones description and a ₹22,630 historical price.

The bra is FAISS position **27** in the canonical 5,000-ID mapping. The raw product
ID is preserved through cleaning, sampling, TF-IDF and semantic retrieval. All three
UI modes were exercised with this exact selected ID; the selected heading remained
the bra and its displayed price remained ₹399 with the matching description.

**Root-cause conclusion:** the specific source row is not corrupted, and the current
preparation, field mapping, stored catalog, FAISS positions, and tested UI do not
show the claimed mix-up. The historical runtime origin of the reported display is
**not established**. A stale/alternate running session is a possibility, not a proven
cause. Do not attribute this incident to source corruption or claim an identified
mapping fix without reproduction evidence. The dataset/path and mode active during
the original observation were requested for further diagnosis.

## Whole-catalog metadata alignment

`python -m src.audit` independently joins original rows by string product ID, not
by matching DataFrame positions. It checks names, descriptions, categories/paths,
numeric prices, image URLs, source pids, ratings, review flags and source record
numbers for every retained row:

- **19,920 full-catalog rows:** zero mismatched fields, unknown IDs or duplicate IDs.
- **5,000 sample rows:** zero mismatched fields, unknown IDs or duplicate IDs.
- **5,000 FAISS positions:** canonical IDs, ID-to-position mapping, and reconstructed
  index vectors all match their stored embedding rows.
- Original raw CSV SHA256 remains
  `56f8f699c9e847356666c2eab3c3ab1244340f6a98ad08e39ea2199ebe993ad1`.

This verifies provenance/alignment and documented parsing; it does not prove the
semantic truth of every historical listing or that external images are reachable.

## Broader raw-source consistency screen

All **20,000** raw records were screened using deterministic rules:

| Review rule | Raw records flagged |
| --- | ---: |
| Explicit `Price: Rs./INR/₹` in description differs from numeric `discounted_price` | 280 |
| Query-string `pid` in product URL conflicts with source `pid` | 0 |
| Product URL cannot be parsed | 0 |
| At least three substantive title terms and eight description terms, with no overlap | 0 |

The token rule uses lowercase alphanumeric tokens longer than two characters and
a documented small stop-word list in `source_quality_flags`. It is a limited review
heuristic: some real corruptions may still share words; low-overlap text may be valid.

**Affected row count:** 280 source records have the embedded-price review flag;
all 280 remain in the valid full catalog. **No additional rows were excluded**, and
no product names, descriptions, prices, categories or image values were replaced.
The prior 78 invalid-price and two duplicate-source-pid removals remain unchanged.
The reported bra is not flagged. Embedded price differences may reflect stale listing
text rather than incorrect products; no retail-price assumptions or fabricated
replacements were used. `reports/source_quality_flags.csv` records all candidates
with original descriptions and source record numbers.

Of these flagged records, **81** occur in the 5,000-product sample.

Prepared records now include `quality_flags` and `source_row_number`; the loader
preserves them and the existing cards disclose review flags without a redesign.
Product text and the recommendation scoring formula are unchanged.

## Defensive mapping improvements and rebuild

These safeguards address potential validation gaps; they are not claimed as the
proven cause of the original screenshot/report:

- FAISS cache construction/load verifies exact IndexFlatIP type and reconstructs
  every vector to detect same-shaped but reordered index rows.
- Production semantic indexes carry a catalog signature. Hybrid construction rejects
  stale metadata even when the old/new product-ID sets happen to match.
- Streamlit retains its explicit TF-IDF fallback for failed optional index validation.
- New tests deliberately swap individual metadata columns, reorder FAISS rows,
  corrupt ID-position maps, and reuse stale catalogs, as well as exercise the
  reported real product in each UI mode.

The cache schema was bumped. The **5,000 × 384** real CPU embeddings and FAISS index
were rebuilt successfully in **164.73 seconds**, with fingerprint
`ded107b99dd0855974bad9eab5c9ee8fbf9e862df1238f251c21264f102c9d03`.
Old fingerprint directories are retained but are not selected by the current code.
The full 19,920-item semantic index had not previously been built; its build command
remains supported and will use its own new fingerprint.

## Verification

- Complete suite: **101 passed, 1 warning in 13.61s** on Python 3.12.7; no skips.
- The warning is the existing FAISS/NumPy private-namespace deprecation warning.
- Actual Streamlit health and root endpoints returned **HTTP 200**; the smoke-test
  subprocess was terminated afterward. Existing user-launched servers were not stopped.
- `python -m pip check`: **No broken requirements found.**
- Independent audit passed for both catalogs and all 5,000 index positions.
- Three-method comparison was rerun using identical 200 queries, K=5, seed 42 and
  65% semantic weight. All sanity checks passed; exact current measurements are in
  `reports/model_comparison.json` and summarized in `reports/RESULTS.md`.
- Full-query baseline results are in `reports/tfidf_catalog_evaluation.json`.
  Category agreement remains a proxy, not recommendation relevance/accuracy.

Previous evaluation JSON artifacts were preserved under `reports/pre_audit/`.
No pushes, publication or commits were performed. Restart any long-lived Streamlit
session before rechecking the demo so it executes the current code and resource keys;
this is an operational step, not proof that a stale session caused the original issue.
