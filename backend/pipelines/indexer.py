import logging
from pipelines.loader import run as run_loader
from pipelines.chunker import run as run_chunker
from pipelines.embedder import run as run_embedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("omnijuris.indexer")


def run() -> None:
    log.info("OmniJuris Indexer: Starting...")

    log.info("Stage 1: Loading Philippine OmniCorpus")
    run_loader()

    log.info("Stage 2: Chunking Documents")
    run_chunker()

    log.info("Stage 3: Embedding and Indexing")
    run_embedder()

    log.info("OmniJuris Indexer: Complete.")

if __name__ == "__main__":
    run()
