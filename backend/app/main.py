"""
main.py
OmniJuris — FastAPI entrypoint
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.query import router as query_router

app = FastAPI(
    title="OmniJuris API",
    description="Philippine Legal Intelligence — RAG over Philippine jurisprudence",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "OmniJuris API"}
