"""Embedding backend for SOP-name similarity search (match_sop).

Uses sentence-transformers ``all-MiniLM-L6-v2`` only. There is NO lexical /
difflib fallback: if the model cannot be loaded, we raise loudly so the failure
is surfaced rather than silently degraded (per project policy).
"""
from __future__ import annotations

from typing import List

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_MODEL = None


def _pick_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_embedder():
    """Lazily load and cache the MiniLM model (one copy per process)."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(MODEL_NAME, device=_pick_device())
    return _MODEL


def embed(texts: List[str]) -> np.ndarray:
    """Return L2-normalized embeddings, shape (len(texts), dim)."""
    model = get_embedder()
    vecs = model.encode(
        list(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype=np.float32)


def cosine_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarities between one query vector and rows of ``matrix``.

    Both are assumed L2-normalized, so this is a dot product.
    """
    return matrix @ query_vec
