import logging
from pathlib import Path
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

INPUT_PATH  = Path(__file__).parent.parent / "data" / "raw" / "legal_corpus.parquet"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "chunks.parquet"

CHUNK_SIZE    = 512    # tokens per chunk
CHUNK_OVERLAP = 64     # overlap between chunks to avoid cutting context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("omnijuris.chunker")


def chunk_documents(df: pd.DataFrame, text_col: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = []
    for _, row in df.iterrows():
        text = str(row[text_col]).strip()
        if not text:
            continue

        doc_chunks = splitter.split_text(text)

        # Parse citation_information JSON
        metadata = {"label": str(row.get("label", ""))}
        try:
            citation = json.loads(row.get("citation_information", "{}"))
            if citation.get("type"):
                metadata["type"] = citation["type"]
            if citation.get("court"):
                metadata["court"] = citation["court"]
            if citation.get("citation"):
                metadata["source"] = citation["citation"]
            if citation.get("date") or citation.get("date_of_issuance"):
                metadata["date"] = citation.get("date") or citation.get("date_of_issuance")
            if citation.get("case_name") or citation.get("title"):
                metadata["title"] = citation.get("case_name") or citation.get("title")
        except (json.JSONDecodeError, TypeError):
            pass

        if row.get("url"):
            metadata["url"] = str(row["url"])

        # THIS WAS MISSING — append each chunk
        for i, chunk_text in enumerate(doc_chunks):
            chunks.append({
                "chunk_id":    f"{row.name}_{i}",
                "chunk_text":  chunk_text,
                "chunk_index": i,
                **metadata,
            })

    return chunks


def run() -> None:
    log.info("Loading legal corpus from %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    log.info("Loaded %d documents", len(df))

    # Detect text column
    text_col = None
    for col in ["text", "content", "body", "document"]:
        if col in df.columns:
            text_col = col
            break

    if not text_col:
        raise ValueError(f"No text column found. Available columns: {df.columns.tolist()}")

    log.info("Using text column: '%s'", text_col)

    chunks = chunk_documents(df, text_col)
    log.info("Generated %d chunks from %d documents", len(chunks), len(df))

    chunks_df = pd.DataFrame(chunks)
    chunks_df.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")

    log.info("Saved %d chunks → %s", len(chunks_df), OUTPUT_PATH.resolve())
    log.info("Average chunk length: %.0f chars", chunks_df["chunk_text"].str.len().mean())


if __name__ == "__main__":
    run()
