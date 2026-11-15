# Dataset attribution and adaptations

**Flipkart Products** was created by **PromptCloud** using its web-crawling service
to extract historical Flipkart product listings.

- Official source: https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products
- Source data card/license: https://www.kaggle.com/PromptCloudHQ/flipkart-products/tasks
- Listed dataset license: **Creative Commons Attribution-ShareAlike 4.0 International**
- License text: https://creativecommons.org/licenses/by-sa/4.0/
- Original local file: `flipkart_com-ecommerce_sample.csv`, 20,000 rows.
- Original file SHA256: `56f8f699c9e847356666c2eab3c3ab1244340f6a98ad08e39ea2199ebe993ad1`

Adaptations in this project: documented field mapping, safe category/image parsing,
invalid-record removal, ID/source-pid deduplication, and a deterministic 5,000-product
category-diverse sample. Product names, descriptions and historical discounted INR
prices are not fabricated or repriced. The original raw file remains unmodified.
The final audit adds source-record provenance and reproducible review flags for
metadata inconsistencies. Flagged rows retain their original descriptions and prices;
these annotations do not certify semantic correctness of every source record.
Processed datasets retain CC BY-SA 4.0 with this attribution and change notice.
See `data/processed/cleaning_report.json` for exact filtering counts and fingerprints.

Optional product images remain external historical URLs; image binaries are not
copied into or redistributed with the catalog artifacts. Product ratings are source
metadata, not recommendation relevance labels or verified current reviews.

The separate `data/sample_products.csv` is original **synthetic demo data** and is
not part of the PromptCloud dataset. Its fictional INR prices remain labeled as
illustrative. Do not attribute the synthetic catalog to PromptCloud or Flipkart.

Semantic model: `sentence-transformers/all-MiniLM-L6-v2`, Apache-2.0 per its model
card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 . Its model license
is separate from the dataset license. No affiliation or endorsement is implied.
