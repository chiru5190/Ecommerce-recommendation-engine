"""Isolate PyTorch from FAISS's Windows OpenMP runtime in a short-lived process."""

import json
import sys

import numpy as np


def main() -> None:
    """Read a local encoding request from stdin and write float32 NumPy vectors."""
    from sentence_transformers import SentenceTransformer
    import torch

    torch.set_num_threads(2)
    request = json.load(sys.stdin)
    model = SentenceTransformer(request["model_id"], revision=request["revision"], device="cpu",
                                cache_folder=request["model_cache"],
                                local_files_only=not request["allow_download"], trust_remote_code=False)
    vectors = model.encode(request["texts"], batch_size=32, show_progress_bar=True,
                           convert_to_numpy=True, normalize_embeddings=True)
    np.save(request["output"], vectors.astype(np.float32), allow_pickle=False)


if __name__ == "__main__":
    main()
