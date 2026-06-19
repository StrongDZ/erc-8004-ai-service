#!/usr/bin/env python3
"""Build FAISS agent domain index from MongoDB agent metadata.

Encodes: f"{summarizedDescription or description} {' '.join(service_names)}"
using BAAI/bge-small-en-v1.5 (384-dim). Only agents with description OR services.

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m scripts.build_agent_index [--batch-size 512]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# sentence_transformers (torch) must import before faiss: faiss-cpu and torch
# bundle conflicting native OpenMP/BLAS runtimes on Apple Silicon, and loading
# faiss first segfaults (SIGSEGV) as soon as the transformer model is constructed.
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.mongo_client import agents_coll

OUT_DIR = Path(__file__).resolve().parent.parent / "data/faiss"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384


def _agent_text(ag: dict) -> str:
    desc = (ag.get("summarizedDescription") or ag.get("description") or "").strip()
    svcs = " ".join(
        s.get("name", "") for s in (ag.get("services") or []) if s.get("name")
    )
    return f"{desc} {svcs}".strip()


def build_index(batch_size: int = 64) -> None:
    print(f"Loading {EMBED_MODEL} on CPU...")
    # Force CPU: MPS (Apple GPU) shares unified memory with the OS on this
    # machine (24GB total). Attention memory scales batch x seq^2, so a
    # long-tailed agent description in a batch_size=512 batch under MPS's
    # caching allocator exhausted RAM and swapped. CPU keeps memory flat.
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    model.max_seq_length = 256  # descriptions/services text never needs 512 tokens

    coll = agents_coll()
    # Only agents with description OR services
    query = {
        "$or": [
            {"description": {"$exists": True, "$ne": ""}},
            {"summarizedDescription": {"$exists": True, "$ne": ""}},
            {"services": {"$exists": True, "$not": {"$size": 0}}},
        ]
    }
    total = coll.count_documents(query)
    print(f"Found {total} agents with metadata to index.")

    index = faiss.IndexFlatIP(DIM)
    keys: list[str] = []

    batch_texts: list[str] = []
    batch_keys: list[str] = []

    def flush():
        if not batch_texts:
            return
        vecs = model.encode(batch_texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        index.add(np.array(vecs, dtype="float32"))
        keys.extend(batch_keys)
        batch_texts.clear()
        batch_keys.clear()

    cursor = coll.find(query, {"_id": 1, "description": 1, "summarizedDescription": 1, "services": 1})
    for ag in tqdm(cursor, total=total, desc="Encoding agents"):
        text = _agent_text(ag)
        if not text:
            continue
        batch_texts.append(text)
        batch_keys.append(str(ag["_id"]))
        if len(batch_texts) >= batch_size:
            flush()
    flush()

    faiss.write_index(index, str(OUT_DIR / "agent_index.faiss"))
    (OUT_DIR / "agent_keys.json").write_text(json.dumps(keys))
    print(f"Index built: {index.ntotal} vectors → {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    build_index(args.batch_size)
