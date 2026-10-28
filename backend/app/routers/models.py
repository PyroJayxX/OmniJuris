"""
routers/models.py
OmniJuris — GET /models/local endpoint
Exposes installed Ollama models for the frontend model picker.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from engines.local import list_local_models

router = APIRouter(tags=["models"])


class LocalModelsResponse(BaseModel):
    available:     bool
    models:        list[str]
    default_model: str | None
    message:       str | None = None


@router.get("/models/local", response_model=LocalModelsResponse)
async def get_local_models() -> LocalModelsResponse:
    result = list_local_models()
    default_model = result["models"][0] if result["models"] else None
    return LocalModelsResponse(
        available=result["available"],
        models=result["models"],
        default_model=default_model,
        message=result["message"],
    )
