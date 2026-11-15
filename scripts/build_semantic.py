"""Build the pinned CPU model/index cache without starting Streamlit."""

import argparse
import json
from pathlib import Path
import time

from src.data_loader import load_catalog
from src.flipkart import REAL_DATA_PATH
from src.semantic import load_semantic_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REAL_DATA_PATH)
    parser.add_argument("--download", action="store_true", help="Allow first-time public model download")
    args = parser.parse_args()
    started = time.perf_counter()
    index = load_semantic_index(load_catalog(args.data), allow_download=args.download)
    print(json.dumps({"products": len(index.product_ids), "dimension": index.index.d,
                      "fingerprint": index.fingerprint, "seconds": time.perf_counter() - started}, indent=2))


if __name__ == "__main__":
    main()
