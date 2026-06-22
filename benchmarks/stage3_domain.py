#!/usr/bin/env python3
"""Stage 3: FAISS agent domain check + value_scale heuristic fallback.

classify(tag1, tag2, value_scale, value_decimals, agent_key):
  1. If agent not in index → apply scale_heuristic → quality/quantity/None(LLM)
  2. Encode max(cos(tag1_vec, agent_vec), cos(tag2_vec, agent_vec))
  3. cosine > 0.55 → in domain → scale determines quality/quantity
  4. cosine < 0.35 → not in domain → junk
  5. 0.35-0.55 → borderline → None (LLM)
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

# sentence_transformers (torch) must import before faiss: faiss-cpu and torch
# bundle conflicting native OpenMP/BLAS runtimes on Apple Silicon, and loading
# faiss first segfaults (SIGSEGV) as soon as the transformer model is constructed.
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
FAISS_INDEX_PATH = ROOT / "data/faiss/agent_index.faiss"
FAISS_KEYS_PATH = ROOT / "data/faiss/agent_keys.json"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Cosine thresholds — empirically tuned (bge-small baseline ~0.30-0.45)
THRESH_IN_DOMAIN = 0.55


@lru_cache(maxsize=1)
def _load_index() -> tuple[faiss.Index, dict[str, int]]:
    """Load FAISS index and build agent_key -> position map."""
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    keys = json.loads(FAISS_KEYS_PATH.read_text())
    key_to_pos = {k: i for i, k in enumerate(keys)}
    return index, key_to_pos


@lru_cache(maxsize=1)
def _load_model():
    # CPU device + capped seq length: see build_agent_index.py for why MPS +
    # large batches exhausted this machine's 24GB unified memory. Single-text
    # encode calls here are tiny, but keep config consistent with the index build.
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    model.max_seq_length = 256
    return model


def _encode(text: str) -> np.ndarray:
    model = _load_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.astype("float32")


def _cosine_to_agent(tag: str, agent_vec: np.ndarray) -> float:
    """Cosine similarity between tag embedding and agent vector (both L2-normalized)."""
    tag_vec = _encode(tag)
    return float(np.dot(tag_vec, agent_vec))


def scale_heuristic(value_scale: str, value_decimals: int) -> str | None:
    """Fallback when agent has no FAISS vector. Returns quality/quantity/None."""
    s = (value_scale or "").strip().lower()
    if s == "unbounded" or value_decimals > 0:
        return "quantity"
    if s in ("star5", "star10", "binary"):
        return "quality"
    return None


def _scale_to_label(value_scale: str) -> str:
    """When in domain, use scale to pick quality vs quantity."""
    s = (value_scale or "").strip().lower()
    if s == "unbounded":
        return "quantity"
    return "quality"


class DomainClassifier:
    """Stage 3 FAISS domain classifier. Thread-safe (reads only)."""

    def check_in_domain(
        self,
        tag1: str,
        tag2: str,
        agent_key: str,
    ) -> tuple[bool | None, float]:
        """Check if tags are in the agent's domain.

        Returns (in_domain, best_cos):
          True  = cosine > THRESH_IN_DOMAIN → in domain
          False = cosine <= THRESH_IN_DOMAIN → not in domain
          None  = agent not indexed (caller should use scale_heuristic)
        """
        index, key_to_pos = _load_index()
        pos = key_to_pos.get(agent_key)
        if pos is None:
            return None, 0.0

        agent_vec = index.reconstruct(pos)
        tags = [t for t in (tag1.strip(), tag2.strip()) if t]
        if not tags:
            return None, 0.0

        cos_scores = [_cosine_to_agent(t, agent_vec) for t in tags]
        best_cos = max(cos_scores)
        return best_cos > THRESH_IN_DOMAIN, best_cos

    def classify(
        self,
        tag1: str,
        tag2: str,
        value_scale: str,
        value_decimals: int,
        agent_key: str,
    ) -> tuple[str | None, str]:
        """
        Returns (label, reason).
        label: 'quality' | 'quantity' | 'junk' | None (None = escalate to LLM)
        """
        index, key_to_pos = _load_index()
        pos = key_to_pos.get(agent_key)

        if pos is None:
            # Agent not indexed — use value_scale heuristic
            label = scale_heuristic(value_scale, value_decimals)
            reason = f"no_metadata_heuristic:scale={value_scale},decimals={value_decimals}"
            return label, reason

        # Fetch agent vector from FAISS
        agent_vec = index.reconstruct(pos)

        # Max pooling over tag1 and tag2
        tags = [t for t in (tag1.strip(), tag2.strip()) if t]
        if not tags:
            # Empty tags → use heuristic
            label = scale_heuristic(value_scale, value_decimals)
            return label, "no_tags_heuristic"

        cos_scores = [_cosine_to_agent(t, agent_vec) for t in tags]
        best_cos = max(cos_scores)

        if best_cos > THRESH_IN_DOMAIN:
            label = _scale_to_label(value_scale)
            return label, f"in_domain:cos={best_cos:.3f}"

        # Borderline → LLM
        return None, f"borderline:cos={best_cos:.3f}"
