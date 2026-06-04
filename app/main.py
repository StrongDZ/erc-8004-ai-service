"""FastAPI entrypoint for the ERC-8004 AI service.

Wraps the Python research modules (shared/) behind a small HTTP API:

  POST /classify  — classify one feedback record via Ollama
  POST /summarize — one-sentence agent business/domain summary via Ollama
  POST /embed     — batch text embeddings via sentence-transformers
  GET  /health    — Ollama reachability + loaded models

Run with:
    uv run uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from .deps import DEFAULT_OLLAMA_MODEL, get_ollama_client  # noqa: E402
from .routers import classify, embed, health, summarize  # noqa: E402


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_ollama_client(DEFAULT_OLLAMA_MODEL)
    yield


app = FastAPI(title="erc-8004-ai-service", version="0.1.0", lifespan=lifespan)
app.include_router(classify.router, tags=["classify"])
app.include_router(summarize.router, tags=["summarize"])
app.include_router(embed.router, tags=["embed"])
app.include_router(health.router, tags=["health"])
