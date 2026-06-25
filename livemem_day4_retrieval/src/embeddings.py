"""
Wraps a local sentence-transformers model for embeddings, so retrieval
similarity scoring never touches the Cerebras API or its rate limits.

Loading the model is slow (a few seconds) so we cache it as a module-level
singleton — every system that needs embeddings imports embed_texts() from
here rather than loading its own copy of the model.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from . import config
except ImportError:
    # allows `python3 src/embeddings.py` to work directly for quick testing
    import config

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME} (one-time, a few seconds)...")
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list) -> np.ndarray:
    """Returns an (N, dim) array of embeddings for a list of strings."""
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Returns a 1D array of cosine similarities between one query vector and many doc vectors."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
    return doc_norms @ query_norm


def top_k_indices(query_text: str, candidate_texts: list, k: int) -> list:
    """
    The core retrieval primitive every chunk-based RAG baseline will call:
    given a query and a list of candidate texts, return the indices of the
    top-k most similar candidates, ranked best-first.
    """
    if not candidate_texts:
        return []
    query_vec = embed_texts([query_text])[0]
    doc_vecs = embed_texts(candidate_texts)
    sims = cosine_similarity(query_vec, doc_vecs)
    ranked = np.argsort(-sims)[:k]
    return ranked.tolist()


if __name__ == "__main__":
    # Quick self-test
    candidates = [
        "The backend framework was initially Flask.",
        "The backend framework was changed to FastAPI for async support.",
        "Deployment is on Railway.",
    ]
    idxs = top_k_indices("What backend framework is currently used?", candidates, k=2)
    print("Top matches:")
    for i in idxs:
        print(f"  - {candidates[i]}")
