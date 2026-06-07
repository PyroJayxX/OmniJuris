import logging
import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset

OUTPUT_DIR  = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_PATH = OUTPUT_DIR / "legal_corpus.parquet"

# Legal document types to keep from the OmniCorpus
LEGAL_CATEGORIES = [
    "legal",
    "law",
    "jurisprudence",
    "supreme_court",
    "statute",
    "executive",
    "administrative",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("omnijuris.loader")


def run() -> None:
    """
    Download Philippine OmniCorpus, filter to legal documents,
    and save as Parquet to data/raw/legal_corpus.parquet
    """
    log.info("Loading Philippine OmniCorpus from HuggingFace...")

    import os
    from dotenv import load_dotenv
    load_dotenv()

    dataset = load_dataset(
        "mongramosjr/philippine-omnicorpus",
        split="train",
        token=os.getenv("HF_TOKEN"),
    )

    log.info("Dataset loaded: %d total records", len(dataset))
    log.info("Columns: %s", dataset.column_names)

    # Convert to DataFrame for filtering
    df = dataset.to_pandas()

    log.info("Sample row:\n%s", df.iloc[0].to_dict())

    # Filter to legal documents
    # Column name may vary — check actual column names from sample row above
    # Common column names: 'category', 'type', 'source', 'domain'
    category_col = "label"
    for col in ["category", "type", "domain", "source_type"]:
        if col in df.columns:
            category_col = col
            break

    if category_col:
        legal_mask = df[category_col].str.lower().str.contains(
            "|".join(LEGAL_CATEGORIES),
            na=False,
        )
        df_legal = df[legal_mask].copy()
        log.info(
            "Filtered to %d legal documents (from %d total) using column '%s'",
            len(df_legal), len(df), category_col
        )
    else:
        log.warning("No category column found — keeping all documents")
        df_legal = df.copy()

    # Drop rows with empty text
    text_col = None
    for col in ["text", "content", "body", "document"]:
        if col in df.columns:
            text_col = col
            break

    if text_col:
        df_legal = df_legal[df_legal[text_col].notna()]
        df_legal = df_legal[df_legal[text_col].str.strip() != ""]
        log.info("After dropping empty text: %d documents", len(df_legal))

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_legal.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")

    log.info("Saved %d legal documents → %s", len(df_legal), OUTPUT_PATH.resolve())


if __name__ == "__main__":
    run()
