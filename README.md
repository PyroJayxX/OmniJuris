# OmniJuris - Philippine Legal Intelligence

> Note: This project is a work in progress.

OmniJuris is a full-stack retrieval-augmented generation system over Philippine jurisprudence, built on the **Philippine OmniCorpus** dataset (Ramos, 2024). Powered by BGE-M3 embeddings, ChromaDB vector search, cross-encoder reranking, and a locally-hosted Gemma2 27B LLM via ollama. 

## Stack

| Component | Technology |
|---|---|
| Dataset | Philippine OmniCorpus (HuggingFace) |
| Embedding Model | BGE-M3 (sentence-transformers) |
| Vector Store | ChromaDB |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Gemma2 27B via Ollama |
| Memory | SQLite + sliding window (last 10 exchanges) |
| Backend | FastAPI |
| Frontend | React + TypeScript |

## Dataset

This project uses the [**Philippine OmniCorpus**](https://huggingface.co/datasets/mongramosjr/philippine-omnicorpus) dataset by Dominador B. Ramos Jr., licensed under the Open Data Commons Attribution License (ODC-By) v1.0.