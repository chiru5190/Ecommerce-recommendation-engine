"""Reproduce the real catalog preparation: python -m scripts.prepare_flipkart."""

import argparse
import json
from pathlib import Path

from src.flipkart import ROOT, prepare


def main() -> None:
    """Locate the downloaded original; never substitute generated data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--sample-size", type=int, default=5000, help="0 uses all valid products")
    args = parser.parse_args()
    candidates = [ROOT / "data" / "raw" / "flipkart_com-ecommerce_sample.csv",
                  ROOT / "flipkart_com-ecommerce_sample.csv"]
    raw = args.raw or next((path for path in candidates if path.is_file()), None)
    if raw is None:
        parser.error("Download flipkart_com-ecommerce_sample.csv from "
                     "https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products "
                     "and extract it into data/raw/. No real dataset was substituted.")
    report = prepare(raw, args.output, args.sample_size)
    print(json.dumps({k: v for k, v in report.items() if k != "sample_category_counts"}, indent=2))


if __name__ == "__main__":
    main()
