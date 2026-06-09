from functools import lru_cache
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    """Load cross-encoder once and cache it."""
    return CrossEncoder(RERANKER_MODEL)


def rerank(
    query:     str,
    chunks:    list[str],
    metadatas: list[dict],
    top_n:     int = 5,
) -> tuple[list[str], list[dict]]:
    """
    Rerank retrieved chunks using a cross-encoder model.
    Returns top_n chunks sorted by relevance score descending.

    Parameters
    ----------
    query     : user query string
    chunks    : list of retrieved chunk texts
    metadatas : list of corresponding metadata dicts
    top_n     : number of chunks to return after reranking

    Returns
    -------
    reranked_chunks    : top_n most relevant chunk texts
    reranked_metadatas : corresponding metadata dicts
    """
    if not chunks:
        return [], []

    reranker = _get_reranker()

    # Cross-encoder scores each (query, chunk) pair
    pairs  = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)

    # Sort by score descending
    ranked = sorted(
        zip(scores, chunks, metadatas),
        key=lambda x: x[0],
        reverse=True,
    )

    top = ranked[:top_n]

    reranked_chunks    = [item[1] for item in top]
    reranked_metadatas = [item[2] for item in top]

    return reranked_chunks, reranked_metadatas
