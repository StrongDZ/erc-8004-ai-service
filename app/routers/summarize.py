"""POST /summarize — produce a one-sentence business/domain summary of an agent description.

Consumed by the Go desc-summarizer worker. Reuses the existing
AGENT_SUMMARY_SYSTEM prompt + OllamaClient.summarize().
"""
from __future__ import annotations

import time

from fastapi import APIRouter
from pydantic import BaseModel

from shared.prompts import AGENT_SUMMARY_SYSTEM, agent_summary_user_msg

from ..deps import DEFAULT_OLLAMA_MODEL, get_ollama_client

router = APIRouter()


class SummarizeRequest(BaseModel):
    agent_id: str = ""
    description: str
    model: str | None = None


class SummarizeResponse(BaseModel):
    summary: str
    model_ver: str
    latency_ms: int = 0


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest) -> SummarizeResponse:
    model = req.model or DEFAULT_OLLAMA_MODEL
    client = get_ollama_client(model)
    user_msg = agent_summary_user_msg(name="", description=req.description, services_flat="")
    t0 = time.monotonic()
    summary = client.summarize(AGENT_SUMMARY_SYSTEM, user_msg)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return SummarizeResponse(summary=summary, model_ver=model, latency_ms=elapsed_ms)
