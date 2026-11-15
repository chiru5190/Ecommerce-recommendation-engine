"""Optional CPU sentence embeddings with validated, fingerprinted FAISS caches."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable
from uuid import uuid4

import numpy as np
import pandas as pd

from src.data_loader import validate_catalog
from src.preprocessing import build_product_text

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "artifacts" / "semantic"
CACHE_SCHEMA = "minilm-normalized-product-text-v2-verified-positions"


class SemanticUnavailable(RuntimeError):
    """Optional retrieval cannot be used; TF-IDF remains available."""


def _faiss():
    try:
        import faiss
        # Small exact searches suffer from excessive thread startup overhead.
        faiss.omp_set_num_threads(2)
        return faiss
    except Exception as exc:
        raise SemanticUnavailable("FAISS unavailable. Install requirements-semantic.txt; TF-IDF still works.") from exc


def catalog_fingerprint(catalog: pd.DataFrame, model_id: str = MODEL_ID,
                        revision: str = MODEL_REVISION) -> str:
    """Fingerprint content, sorted ID mapping, preprocessing, model and libraries."""
    ordered = validate_catalog(catalog).sort_values("product_id").reset_index(drop=True)
    packages = {}
    for package in ("sentence-transformers", "transformers", "torch", "faiss-cpu", "numpy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "unavailable"
    payload = {"schema": CACHE_SCHEMA, "model": model_id, "revision": revision,
               "packages": packages, "catalog": ordered.to_json(orient="records"),
               "texts": build_product_text(ordered).tolist() if len(ordered) else []}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class SemanticIndex:
    """IndexFlatIP over L2-normalized float32 rows with explicit product IDs."""

    def __init__(self, product_ids: list[str], embeddings: np.ndarray, fingerprint: str,
                 index=None, catalog_signature: str | None = None) -> None:
        faiss = _faiss()
        vectors = np.array(embeddings, dtype=np.float32, order="C", copy=True)
        if vectors.ndim != 2 or len(vectors) != len(product_ids) or not vectors.shape[1]:
            raise ValueError("Embedding dimensions do not match product-ID mapping.")
        if len(set(product_ids)) != len(product_ids):
            raise ValueError("Semantic product IDs must be unique.")
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("Embeddings must be finite, nonzero vectors.")
        faiss.normalize_L2(vectors)
        self.product_ids = list(product_ids)
        self.embeddings = vectors
        self.fingerprint = fingerprint
        self.catalog_signature = catalog_signature
        self.positions = {pid: i for i, pid in enumerate(product_ids)}
        self.index = index if index is not None else faiss.IndexFlatIP(vectors.shape[1])
        if index is None:
            self.index.add(vectors)
        if self.index.ntotal != len(vectors) or self.index.d != vectors.shape[1]:
            raise ValueError("Cached FAISS index does not match embeddings.")
        if not isinstance(self.index, faiss.IndexFlatIP):
            raise ValueError("Expected an exact FAISS IndexFlatIP index.")
        if not np.allclose(self.index.reconstruct_n(0, len(vectors)), vectors, atol=1e-6):
            raise ValueError("FAISS positions do not match the product-ID embedding rows.")

    def cosine_scores(self, product_id: str) -> pd.Series:
        """Search one query against all rows for exact filtering/hybrid scoring.

        This allocates O(N) query scores, never an N×N similarity matrix.
        """
        if product_id not in self.positions:
            return pd.Series(dtype=float)
        query = self.embeddings[self.positions[product_id]:self.positions[product_id] + 1]
        scores, positions = self.index.search(query, len(self.product_ids))
        aligned = np.empty(len(self.product_ids), dtype=np.float32)
        aligned[positions[0]] = np.clip(scores[0], -1, 1)
        return pd.Series(aligned, index=self.product_ids)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_semantic_index(catalog: pd.DataFrame, cache_dir: Path = DEFAULT_CACHE,
                        allow_download: bool = False,
                        encoder: Callable[[list[str]], np.ndarray] | None = None,
                        model_id: str = MODEL_ID, revision: str = MODEL_REVISION) -> SemanticIndex:
    """Load a verified cache or encode on CPU; optional imports stay isolated.

    The injectable encoder is only for deterministic unit fixtures. Production
    always uses the pinned MiniLM model. Network downloads require allow_download.
    """
    faiss = _faiss()
    ordered = validate_catalog(catalog).sort_values("product_id").reset_index(drop=True)
    if ordered.empty:
        raise SemanticUnavailable("No products are available to embed.")
    ids = ordered.product_id.tolist()
    fingerprint = catalog_fingerprint(ordered, model_id, revision)
    directory = Path(cache_dir) / fingerprint
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["fingerprint"] != fingerprint or manifest["product_ids"] != ids:
            raise ValueError("Cache ID mapping changed.")
        for name in ("embeddings.npy", "index.faiss"):
            if manifest["sha256"][name] != _sha(directory / name):
                raise ValueError("Cache checksum mismatch.")
        vectors = np.load(directory / "embeddings.npy", allow_pickle=False)
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
            raise ValueError("Cached vectors are not normalized.")
        index = faiss.read_index(str(directory / "index.faiss"))
        return SemanticIndex(ids, vectors, fingerprint, index=index, catalog_signature=fingerprint)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        pass  # A missing/corrupt cache is rebuilt, never silently reused.
    texts = build_product_text(ordered).tolist()
    try:
        if encoder is None:
            # PyTorch and FAISS wheels can ship incompatible Windows OpenMP DLLs.
            # Isolate encoding instead of enabling unsafe duplicate-runtime workarounds.
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=cache_dir) as temporary:
                output = Path(temporary) / "vectors.npy"
                request = {"model_id": model_id, "revision": revision, "model_cache": str(ROOT / "models"),
                           "allow_download": allow_download, "texts": texts, "output": str(output)}
                worker = subprocess.run(
                    [sys.executable, "-m", "src.embedding_worker"], cwd=ROOT,
                    input=json.dumps(request), text=True, encoding="utf-8", capture_output=True,
                    timeout=1800, env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                                       "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"},
                )
                if worker.returncode:
                    raise RuntimeError(worker.stderr[-2000:] or worker.stdout[-2000:])
                vectors = np.load(output, allow_pickle=False)
        else:
            vectors = encoder(texts)
        result = SemanticIndex(ids, vectors, fingerprint, catalog_signature=fingerprint)
        directory.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        vector_tmp = directory / f"embeddings.{token}.tmp"
        with vector_tmp.open("wb") as file:
            np.save(file, result.embeddings, allow_pickle=False)
        os.replace(vector_tmp, directory / "embeddings.npy")
        index_tmp = directory / f"index.{token}.tmp"
        faiss.write_index(result.index, str(index_tmp))
        os.replace(index_tmp, directory / "index.faiss")
        manifest = {"fingerprint": fingerprint, "model": model_id, "revision": revision,
                    "schema": CACHE_SCHEMA, "product_ids": ids,
                    "dimension": result.embeddings.shape[1],
                    "sha256": {name: _sha(directory / name) for name in ("embeddings.npy", "index.faiss")}}
        tmp = directory / f"manifest.{token}.tmp"
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp, manifest_path)
        return result
    except Exception as exc:
        raise SemanticUnavailable(
            "Semantic model/index unavailable. Run the optional install and "
            "python -m scripts.build_semantic --download. TF-IDF remains available. "
            f"Details: {type(exc).__name__}: {exc}"
        ) from exc
