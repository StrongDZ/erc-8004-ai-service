"""Singleton dependency providers for the FastAPI app.

OllamaClient and SentenceTransformer are expensive to construct (the embedder
loads model weights to RAM). They're built once at app startup and reused for
every request.
"""
from __future__ import annotations

import os
from functools import lru_cache

from shared.ollama_client import OllamaClient
from shared.prompts import system_prompt_v5


DEFAULT_OLLAMA_MODEL = os.getenv("AI_SERVICE_DEFAULT_MODEL", "qwen2.5:3b")
DEFAULT_EMBED_MODEL = os.getenv("AI_SERVICE_DEFAULT_EMBED_MODEL", "BAAI/bge-base-en-v1.5")


@lru_cache(maxsize=8)
def get_ollama_client(model: str) -> OllamaClient:
    """One OllamaClient per model (Ollama keeps weights warm per model).

    Default prompt is V5 (4-category spec + agent_domain + endpoint_matched + tier rule
    + first-match ordering). Few-shot block included by default for smaller models.
    """
    return OllamaClient(model=model, system_prompt=system_prompt_v5(include_few_shot=True))


@lru_cache(maxsize=4)
def get_embedder(model: str):
    # Lazy import: /summarize and /classify do not need torch or sentence-transformers.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model)


@lru_cache(maxsize=1)
def get_knn_classifier():
    """Lazy singleton KNNCorpus — built on first call, cached for the process lifetime.

    Building the corpus (sampling ~4K records from Mongo + encoding) takes
    ~15-30 s on CPU. Subsequent calls return the cached instance immediately.
    """
    from shared.knn_classifier import KNNCorpus

    encoder = get_embedder(DEFAULT_EMBED_MODEL)
    corpus = KNNCorpus(encoder)
    corpus.build()
    return corpus


@lru_cache(maxsize=1)
def get_linear_classifier():
    """Lazy singleton logistic-regression head.

    Reuses the kNN corpus (identical vectors + labels) and fits a
    LogisticRegression(class_weight='balanced') on top, so the comparison vs kNN
    isolates the decision rule (learned boundary vs cosine majority vote).
    """
    from shared.linear_classifier import EmbeddingLinearClassifier

    corpus = get_knn_classifier()
    clf = EmbeddingLinearClassifier(corpus.embedder, corpus.vectors, corpus.labels)
    clf.build()
    return clf
