from functools import lru_cache
from pathlib import Path
import torch
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH      = Path(__file__).parent.parent / "data" / "chroma"
COLLECTION_NAME  = "omnijuris"
EMBEDDING_MODEL  = "intfloat/multilingual-e5-base"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    """Load multilingual-e5-base once and cache it."""
    return SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)


@lru_cache(maxsize=1)
def _get_collection() -> chromadb.Collection:
    """Load ChromaDB collection once and cache it."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_collection(name=COLLECTION_NAME)


def retrieve(query: str, top_k: int = 10) -> tuple[list[str], list[dict]]:
    """
    Embed query and retrieve top-K most similar chunks from ChromaDB.

    Returns
    -------
    chunks    : list of chunk text strings
    metadatas : list of metadata dicts (source, type, etc.)
    """
    embedder   = _get_embedder()
    collection = _get_collection()

    # multilingual-e5 requires "query: " prefix at inference time
    query_vector = embedder.encode(
        f"query: {query}",
        normalize_embeddings=True,
        device=DEVICE,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]

    return chunks, metadatas