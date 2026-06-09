"""
routers/query.py
OmniJuris — POST /query endpoint
Handles retrieval, reranking, and LLM generation for both local and cloud modes.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.retriever import retrieve
from core.reranker import rerank
from core.prompt import build_prompt
from engines.local import generate_local
from engines.cloud import generate_cloud

router = APIRouter(tags=["query"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query:         str
    override_mode: str = "local"   # "local" or "cloud"
    thinking_mode: bool = False    # Qwen3 extended reasoning (local only)
    top_k:         int = 10        # chunks to retrieve before reranking
    top_n:         int = 5         # chunks to keep after reranking


class QueryResponse(BaseModel):
    answer:      str
    engine_mode: str
    citations:   list[str]
    chunks_used: int


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Full RAG pipeline:
    1. Retrieve top_k chunks from ChromaDB
    2. Rerank to top_n using cross-encoder
    3. Build prompt from reranked chunks
    4. Generate answer via local (Ollama) or cloud (Gemini) engine
    5. Return answer + citations
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    mode = request.override_mode.lower()
    if mode not in ("local", "cloud"):
        raise HTTPException(status_code=400, detail="override_mode must be 'local' or 'cloud'.")

    # Step 1 — Retrieve
    chunks, metadatas = retrieve(request.query, top_k=request.top_k)

    if not chunks:
        return QueryResponse(
            answer="I could not find relevant information in the legal corpus for your query.",
            engine_mode=mode,
            citations=[],
            chunks_used=0,
        )

    # Step 2 — Rerank
    reranked_chunks, reranked_metadatas = rerank(
        query=request.query,
        chunks=chunks,
        metadatas=metadatas,
        top_n=request.top_n,
    )

    # Step 3 — Build prompt
    prompt = build_prompt(
        query=request.query,
        chunks=reranked_chunks,
        thinking_mode=request.thinking_mode,
    )

    # Step 4 — Generate
    if mode == "local":
        answer = generate_local(prompt, thinking_mode=request.thinking_mode)
    else:
        answer = generate_cloud(prompt)

    # Step 5 — Extract citations from metadata
    citations = []
    for meta in reranked_metadatas:
        source = meta.get("source", "")
        if source and source not in citations:
            citations.append(source)

    return QueryResponse(
        answer=answer,
        engine_mode=mode,
        citations=citations,
        chunks_used=len(reranked_chunks),
    )
