"""GET /health — probe Ollama and report which models are loaded."""
from __future__ import annotations

from fastapi import APIRouter

from ..deps import DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_MODEL, get_embedder, get_ollama_client
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ollama_ok = get_ollama_client(DEFAULT_OLLAMA_MODEL).health()
    return HealthResponse(
        ollama_ok=ollama_ok,
        default_model=DEFAULT_OLLAMA_MODEL,
        embedder_loaded=get_embedder.cache_info().currsize > 0,
    )


@router.get("/health/warmup")
def warmup() -> dict:
    """Force-load both the default Ollama client and the default embedder."""
    get_ollama_client(DEFAULT_OLLAMA_MODEL)
    get_embedder(DEFAULT_EMBED_MODEL)
    return {"ok": True, "ollama_model": DEFAULT_OLLAMA_MODEL, "embed_model": DEFAULT_EMBED_MODEL}
