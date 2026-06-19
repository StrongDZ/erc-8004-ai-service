#!/usr/bin/env python3
"""Build agent-enriched feedback dataset for per-tag pipeline benchmark.

Group A: agents with description or summarizedDescription (≤5 per (tag1,tag2) pair)
Group B: agents with no metadata at all (≤5 per (tag1,tag2) pair)

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m scripts.build_agent_enriched_dataset
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.mongo_client import agents_coll, feedback_coll, fetch_agents_by_keys
from shared.types import LLM_OUTPUT_CATEGORIES, MONGO_CATEGORY_ALIASES, RULE_TO_CAT

SEED = 42
MAX_PER_PAIR = 5
OUT_DIR = Path(__file__).resolve().parent.parent / "data/splits/agent_enriched"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["quality", "quantity", "junk"]


def _agent_meta(ag: dict) -> tuple[str, list[str]]:
    """Return (description, service_names) from agent doc."""
    desc = (ag.get("summarizedDescription") or ag.get("description") or "").strip()
    svcs = [s.get("name", "") for s in (ag.get("services") or []) if s.get("name")]
    return desc, svcs


def _has_metadata(desc: str, svcs: list[str]) -> bool:
    return bool(desc) or bool(svcs)


def _fetch_feedback_docs(coll_fb) -> list[dict]:
    """Sample up to 20k feedback docs per rule category across quality/quantity/junk."""
    docs: list[dict] = []
    for cat in CATEGORIES:
        aliases = MONGO_CATEGORY_ALIASES.get(cat, [cat])
        cursor = coll_fb.aggregate([
            {"$match": {"classification.rule.category": {"$in": aliases}}},
            {"$sample": {"size": 20000}},
            {"$project": {
                "_id": 1, "agentId": 1, "chainId": 1,
                "tag1": 1, "tag2": 1, "valueScale": 1, "valueDecimals": 1, "value": 1,
                "feedbackParsed": 1, "classification.rule.category": 1,
            }},
        ], allowDiskUse=True)
        docs.extend(cursor)
    return docs


def build_group(
    docs: list[dict],
    agent_meta_by_key: dict[str, tuple[str, list[str]]],
    want_metadata: bool,
    rng: random.Random,
) -> pd.DataFrame:
    """Sample up to MAX_PER_PAIR records per (tag1, tag2) pair for one metadata group."""
    pair_buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for doc in docs:
        agent_key = f"{doc.get('chainId', 0)}:{doc.get('agentId', '')}"
        desc, svcs = agent_meta_by_key.get(agent_key, ("", []))
        if _has_metadata(desc, svcs) != want_metadata:
            continue

        t1 = str(doc.get("tag1") or "").strip()
        t2 = str(doc.get("tag2") or "").strip()
        pair_buckets[(t1, t2)].append({
            "id": str(doc["_id"]),
            "agent_id": str(doc.get("agentId", "")),
            "chain_id": int(doc.get("chainId", 0) or 0),
            "tag1": t1,
            "tag2": t2,
            "value_scale": str(doc.get("valueScale") or "").strip(),
            "value_decimals": int(doc.get("valueDecimals") or 0),
            "value": str(doc.get("value") or ""),
            "label": RULE_TO_CAT.get(
                (doc.get("classification") or {}).get("rule", {}).get("category", "others"),
                "others",
            ),
            "agent_description": desc,
            "agent_services": json.dumps(svcs),
            "has_agent_metadata": want_metadata,
        })

    rows = []
    for (t1, t2), pdocs in pair_buckets.items():
        pdocs = [d for d in pdocs if d["label"] in LLM_OUTPUT_CATEGORIES]
        rng.shuffle(pdocs)
        rows.extend(pdocs[:MAX_PER_PAIR])

    rng.shuffle(rows)
    return pd.DataFrame(rows)


def main() -> None:
    rng = random.Random(SEED)
    coll_fb = feedback_coll()
    agents_coll()  # ensure collection/env wiring is valid before the heavy fetch

    print("Sampling feedback documents (quality/quantity/junk)...")
    docs = _fetch_feedback_docs(coll_fb)
    print(f"  Fetched {len(docs)} feedback docs")

    print("Bulk-fetching agent metadata for all referenced agents...")
    keys = {(int(d.get("chainId", 0) or 0), str(d.get("agentId", ""))) for d in docs}
    agents_by_id = fetch_agents_by_keys(keys)
    agent_meta_by_key = {
        agent_id: _agent_meta(ag) for agent_id, ag in agents_by_id.items()
    }
    print(f"  Resolved metadata for {len(agent_meta_by_key)} / {len(keys)} agent keys")

    print("\nBuilding Group A (agents WITH metadata)...")
    group_a = build_group(docs, agent_meta_by_key, want_metadata=True, rng=rng)
    print(f"  Group A: {len(group_a)} records")
    print(f"  Label dist:\n{group_a['label'].value_counts().to_string()}")
    group_a.to_parquet(OUT_DIR / "group_a.parquet", index=False)

    print("\nBuilding Group B (agents WITHOUT metadata)...")
    group_b = build_group(docs, agent_meta_by_key, want_metadata=False, rng=rng)
    print(f"  Group B: {len(group_b)} records")
    print(f"  Label dist:\n{group_b['label'].value_counts().to_string()}")
    group_b.to_parquet(OUT_DIR / "group_b.parquet", index=False)

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
