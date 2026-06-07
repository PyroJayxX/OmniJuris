import logging
from pathlib import Path
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import torch

INPUT_PATH      = Path(__file__).parent.parent / "data" / "raw" / "chunks.parquet"
CHROMA_PATH     = Path(__file__).parent.parent / "data" / "chroma"
COLLECTION_NAME = "omnijuris"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

ENCODE_BATCH_SIZE = 512
CHROMA_ADD_SIZE   = 5000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("omnijuris.embedder")

log.info("Loading embedding model '%s' on device: %s", EMBEDDING_MODEL, DEVICE)
embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)


def build_metadatas(batch_df: pd.DataFrame) -> list[dict]:
    meta_cols = [c for c in batch_df.columns if c not in ("chunk_id", "chunk_text", "chunk_index")]
    sub = batch_df[meta_cols].astype(str).replace({"nan": None, "": None, "None": None})
    records = sub.to_dict(orient="records")
    out = []
    for rec in records:
        meta = {k: v for k, v in rec.items() if v is not None}
        if not meta:
            meta = {"source": "philippine-omnicorpus"}
        out.append(meta)
    return out


def get_existing_ids(collection) -> set[str]:
    log.info("Fetching existing IDs for resume support...")
    result = collection.get(include=[])
    existing = set(result["ids"])
    log.info("Found %d existing vectors", len(existing))
    return existing


def run() -> None:
    log.info("Loading chunks from %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    log.info("Loaded %d chunks", len(df))

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client     = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = get_existing_ids(collection)
    if existing_ids:
        before = len(df)
        df = df[~df["chunk_id"].isin(existing_ids)].reset_index(drop=True)
        log.info("Skipped %d already-embedded chunks, %d remaining", before - len(df), len(df))

    if df.empty:
        log.info("Nothing to embed.")
        return

    total = len(df)
    log.info("Embedding %d chunks (encode_batch=%d, chroma_batch=%d)",
             total, ENCODE_BATCH_SIZE, CHROMA_ADD_SIZE)

    for chroma_start in range(0, total, CHROMA_ADD_SIZE):
        chunk_df = df.iloc[chroma_start : chroma_start + CHROMA_ADD_SIZE]

        texts     = chunk_df["chunk_text"].tolist()
        ids       = chunk_df["chunk_id"].tolist()
        metadatas = build_metadatas(chunk_df)

        # Prefix only for the encoder, not stored in ChromaDB
        encoded_texts = ["passage: " + t for t in texts]

        embeddings = embedder.encode(
            encoded_texts,
            batch_size=ENCODE_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,        # original text, no prefix
            metadatas=metadatas,
        )

        done = min(chroma_start + CHROMA_ADD_SIZE, total)
        log.info("Committed %d/%d (%.1f%%)", done, total, done / total * 100)

    log.info("Done. Collection now has %d vectors", collection.count())


if __name__ == "__main__":
    run()