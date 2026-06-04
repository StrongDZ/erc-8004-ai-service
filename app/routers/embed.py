"""POST /embed — batch text embedding via sentence-transformers."""
from __future__ import annotations

from fastapi import APIRouter

from ..deps import DEFAULT_EMBED_MODEL, get_embedder
from ..schemas import EmbedRequest, EmbedResponse

router = APIRouter()


@router.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    model = req.model or DEFAULT_EMBED_MODEL
    encoder = get_embedder(model)
    vectors = encoder.encode(req.texts, normalize_embeddings=True)
    embeddings = [vec.tolist() for vec in vectors]
    dim = len(embeddings[0]) if embeddings else int(encoder.get_sentence_embedding_dimension() or 0)
    return EmbedResponse(embeddings=embeddings, model=model, dim=dim)
