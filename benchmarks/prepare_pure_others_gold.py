#!/usr/bin/env python3
"""Build a clean, non-circular "others" labeling pool.

Problem: gold_combined_clean_v2.csv mixes two kinds of "AI-service" records
(those that escaped the Go rule engine, i.e. classification.rule.category ==
"others"):
  - 249  human-labeled (source in {gold_v1, round2})
  - 1038 silver-labeled via convention_classify() — the SAME rule logic as
    Stage 1, just run again in Python. These are NOT independent of the rule
    engine and inflate the benchmark's apparent accuracy.

This script assembles a single CSV of *only* others-pool records (rule could
not decide) for full manual re-labeling, sourced from three places:
  1. The 1,287 AI-service records already in gold_combined_clean_v2.csv
     (existing silver/human label kept as `existing_label` for reference only
     — it must not be treated as ground truth).
  2. Agent context backfilled from the two existing local labeling sheets
     (silver_candidates.csv, others_to_label.csv) where the feedback_id
     matches, to avoid re-querying MongoDB for content already on disk.
  3. `--new-sample` additional records freshly drawn from MongoDB
     (classification.rule.category == "others") that are not in gold or
     either local sheet, enriched with the same agent context, to grow the
     pool.

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.prepare_pure_others_gold --new-sample 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.mongo_client import agents_coll, feedback_coll
from shared.oasf_enrich import expand_oasf, oasf_lookup
from shared.types import AgentMeta

ROOT = Path(__file__).resolve().parent.parent
GOLD_CSV = ROOT / "data/labelled/gold_combined_clean_v2.csv"
CACHE_PARQUET = ROOT / "data/benchmark_results/tune_cache.parquet"
SILVER_CANDIDATES_CSV = ROOT / "data/labelled/silver_candidates.csv"
OTHERS_TO_LABEL_CSV = ROOT / "data/labelled/others_to_label.csv"
DEFAULT_OUT = ROOT / "data/labelled/pure_others_to_label.csv"

MAX_DESC_CHARS = 500
MAX_OFFCHAIN_CHARS = 400
MAX_PER_AGENT = 15

CONTEXT_COLS = [
    "endpoint", "offchain_note",
    "agent_key", "agent_name", "agent_description", "agent_services",
    "agent_oasf_domains_raw", "agent_oasf_domains_text",
    "agent_oasf_skills_raw", "agent_oasf_skills_text",
    "agent_tags", "agent_domain_text",
]

OUTPUT_COLS = [
    "feedback_id", "tag1", "tag2", "value", "value_decimals", "scale",
    *CONTEXT_COLS,
    "existing_label", "pool_origin", "human_label",
]


def _fetch_agent_meta(ag_coll, agent_key: str) -> AgentMeta:
    chain_id_str, agent_id = agent_key.split(":", 1) if ":" in agent_key else ("0", agent_key)
    doc = ag_coll.find_one({"_id": agent_key}, {
        "name": 1, "description": 1, "summarizedDescription": 1,
        "services": 1, "oasfDomains": 1, "oasfSkills": 1, "tags": 1,
    }) or {}
    return AgentMeta(
        chain_id=int(chain_id_str) if chain_id_str.lstrip("-").isdigit() else 0,
        agent_id=agent_id,
        name=(doc.get("name") or "").strip(),
        description=(doc.get("summarizedDescription") or doc.get("description") or "").strip(),
        services=doc.get("services") or [],
        oasf_domains=doc.get("oasfDomains") or [],
        oasf_skills=doc.get("oasfSkills") or [],
        tags=[t for t in (doc.get("tags") or []) if t],
    )


def _offchain_extract(fp) -> str:
    if fp is None:
        return ""
    if isinstance(fp, str):
        return "" if fp.strip() in ("", "null", "None") else fp[:MAX_OFFCHAIN_CHARS]
    if isinstance(fp, dict):
        note = (fp.get("note") or fp.get("text") or fp.get("summary") or fp.get("content") or "").strip()
        return note[:MAX_OFFCHAIN_CHARS]
    return ""


def _format_services(services: list[dict]) -> str:
    out = []
    for svc in (services or [])[:6]:
        name = (svc.get("name") or "").strip()
        ep = (svc.get("endpoint") or "").strip()
        if name and ep:
            out.append(f"{name}:{ep[:60]}")
        elif name:
            out.append(name)
    return " | ".join(out)


def _enrich_one(ag_coll, fid: str, tag1: str, tag2: str, value: str, value_decimals: int,
                 scale: str, endpoint: str, feedback_parsed, agent_key: str) -> dict:
    ag = _fetch_agent_meta(ag_coll, agent_key)
    oasf_dom_text = expand_oasf(ag.oasf_domains)
    oasf_sk_text = expand_oasf(ag.oasf_skills)
    desc_trimmed = ag.description[:MAX_DESC_CHARS]
    domain_parts = [p for p in [desc_trimmed, oasf_dom_text, oasf_sk_text] if p]
    if ag.tags:
        domain_parts.append(", ".join(ag.tags[:8]))
    agent_domain_text = " | ".join(domain_parts)

    return {
        "feedback_id": fid,
        "tag1": tag1, "tag2": tag2, "value": value,
        "value_decimals": value_decimals, "scale": scale,
        "endpoint": endpoint,
        "offchain_note": _offchain_extract(feedback_parsed),
        "agent_key": agent_key,
        "agent_name": ag.name,
        "agent_description": desc_trimmed,
        "agent_services": _format_services(ag.services),
        "agent_oasf_domains_raw": ", ".join(ag.oasf_domains[:5]),
        "agent_oasf_domains_text": oasf_dom_text[:300],
        "agent_oasf_skills_raw": ", ".join(ag.oasf_skills[:8]),
        "agent_oasf_skills_text": oasf_sk_text[:400],
        "agent_tags": ", ".join(ag.tags[:10]),
        "agent_domain_text": agent_domain_text[:600],
    }


def _load_existing_ai_service_pool() -> tuple[pd.DataFrame, set[str]]:
    """The records already in gold that escaped the Stage-1 rule engine.

    `cache["stage_early"]` comes from `benchmarks.pipeline_3tier.rule_classify()`,
    a *local* re-implementation of Stage 1 that does not run the Stage 0
    self-feedback gate (clientAddress == owner/agentWallet). Production's Go
    self-feedback gate runs upstream of Stage 1 and writes
    `classification.rule.category = "junk"` for those rows, so they are NOT
    actually unresolved "others" records — they are rule-resolved junk that
    leaked into the local "stage_early == ''" bucket. Cross-check against
    MongoDB's real `classification.rule.category` and drop anything that
    isn't genuinely "others" before treating it as part of the labeling pool.
    """
    gold = pd.read_csv(GOLD_CSV)
    cache = pd.read_parquet(CACHE_PARQUET)
    assert len(gold) == len(cache), "gold/cache row-count mismatch — re-run tune_thresholds precompute"
    gold["stage_early"] = cache["stage_early"].values
    ai_gold = gold[gold["stage_early"] == ""].copy()
    ai_gold = ai_gold.rename(columns={"category": "existing_label"})
    ai_gold = ai_gold[["feedback_id", "tag1", "tag2", "value", "value_decimals", "scale", "existing_label"]]

    fb_coll = feedback_coll()
    mongo_rule_cat = {
        d["_id"]: d.get("classification", {}).get("rule", {}).get("category", "")
        for d in fb_coll.find(
            {"_id": {"$in": ai_gold["feedback_id"].tolist()}},
            {"classification.rule.category": 1},
        )
    }
    real_others_mask = ai_gold["feedback_id"].map(mongo_rule_cat).fillna("") == "others"
    n_leaked = (~real_others_mask).sum()
    if n_leaked:
        print(f"  [filter] dropping {n_leaked} records that local rule_classify() missed but "
              f"production's Go rule engine already resolved (e.g. Stage-0 self-feedback)")
    ai_gold = ai_gold[real_others_mask].reset_index(drop=True)

    return ai_gold, set(gold["feedback_id"])


def _backfill_context_from_local_sheets(rows: pd.DataFrame) -> pd.DataFrame:
    """Fill CONTEXT_COLS for `rows` from silver_candidates.csv / others_to_label.csv where possible."""
    sheets = []
    if SILVER_CANDIDATES_CSV.exists():
        sheets.append(pd.read_csv(SILVER_CANDIDATES_CSV))
    if OTHERS_TO_LABEL_CSV.exists():
        sheets.append(pd.read_csv(OTHERS_TO_LABEL_CSV))

    context_lookup: dict[str, dict] = {}
    for sheet in sheets:
        for _, r in sheet.iterrows():
            fid = r["feedback_id"]
            if fid in context_lookup:
                continue
            context_lookup[fid] = {c: r.get(c, "") for c in CONTEXT_COLS}

    for c in CONTEXT_COLS:
        rows[c] = ""
    found = 0
    for i, fid in enumerate(rows["feedback_id"]):
        ctx = context_lookup.get(fid)
        if ctx is not None:
            for c in CONTEXT_COLS:
                rows.iat[i, rows.columns.get_loc(c)] = ctx[c]
            found += 1
    print(f"  Backfilled context for {found}/{len(rows)} records from local sheets")
    return rows


def _fetch_missing_context(rows: pd.DataFrame) -> pd.DataFrame:
    """For rows still missing agent_key (no local-sheet match), query MongoDB live."""
    fb_coll = feedback_coll()
    ag_coll = agents_coll()
    missing_mask = rows["agent_key"] == ""
    missing_ids = rows.loc[missing_mask, "feedback_id"].tolist()
    print(f"  Querying MongoDB for {len(missing_ids)} records with no local context...")

    docs = {
        d["_id"]: d
        for d in fb_coll.find(
            {"_id": {"$in": missing_ids}},
            {"tag1": 1, "tag2": 1, "value": 1, "valueDecimals": 1, "valueScale": 1,
             "endpoint": 1, "feedbackParsed": 1, "agentId": 1, "chainId": 1},
        )
    }
    n_not_found = 0
    for i in rows.index[missing_mask]:
        fid = rows.at[i, "feedback_id"]
        doc = docs.get(fid)
        if doc is None:
            n_not_found += 1
            continue
        chain_id = doc.get("chainId", 0)
        agent_id = str(doc.get("agentId", ""))
        agent_key = f"{chain_id}:{agent_id}"
        enriched = _enrich_one(
            ag_coll, fid,
            str(doc.get("tag1", "") or "").strip(), str(doc.get("tag2", "") or "").strip(),
            str(doc.get("value", "") or ""), int(doc.get("valueDecimals", 0) or 0),
            str(doc.get("valueScale", "") or "").strip(), str(doc.get("endpoint", "") or "").strip(),
            doc.get("feedbackParsed"), agent_key,
        )
        for c in CONTEXT_COLS:
            rows.at[i, c] = enriched[c]
    if n_not_found:
        print(f"  [warn] {n_not_found} feedback_ids not found in MongoDB (stale gold entries)")
    return rows


def _sample_new_others(n: int, exclude_ids: set[str]) -> pd.DataFrame:
    """Draw `n` fresh others-pool records from MongoDB not already covered.

    Uses $sample for a randomized draw across the *whole* others-pool collection
    (6,127 unique agents; one agent alone has 1,570 records) rather than an
    `_id`-sorted cursor — sorting by `_id` clusters consecutive documents under
    the same agent, so a flat `limit` mostly exhausts itself on a handful of
    agents before the per-agent cap can diversify the sample.
    """
    fb_coll = feedback_coll()
    ag_coll = agents_coll()

    # Oversample generously: many draws get dropped by exclude_ids / per-agent cap.
    sample_size = max(n * 15, 5000)
    pipeline = [
        {"$match": {"classification.rule.category": "others"}},
        {"$sample": {"size": sample_size}},
        {"$project": {
            "tag1": 1, "tag2": 1, "value": 1, "valueDecimals": 1, "valueScale": 1,
            "endpoint": 1, "feedbackParsed": 1, "agentId": 1, "chainId": 1,
        }},
    ]
    candidates = list(fb_coll.aggregate(pipeline))
    print(f"  Drew {len(candidates)} randomized candidates from the others pool")

    rows = []
    seen_agents: dict[str, int] = {}
    for doc in candidates:
        fid = str(doc["_id"])
        if fid in exclude_ids:
            continue
        chain_id = doc.get("chainId", 0)
        agent_id = str(doc.get("agentId", ""))
        agent_key = f"{chain_id}:{agent_id}"
        if seen_agents.get(agent_key, 0) >= MAX_PER_AGENT:
            continue

        enriched = _enrich_one(
            ag_coll, fid,
            str(doc.get("tag1", "") or "").strip(), str(doc.get("tag2", "") or "").strip(),
            str(doc.get("value", "") or ""), int(doc.get("valueDecimals", 0) or 0),
            str(doc.get("valueScale", "") or "").strip(), str(doc.get("endpoint", "") or "").strip(),
            doc.get("feedbackParsed"), agent_key,
        )
        enriched["existing_label"] = ""
        rows.append(enriched)
        seen_agents[agent_key] = seen_agents.get(agent_key, 0) + 1
        exclude_ids.add(fid)
        if len(rows) >= n:
            break

    print(f"  Sampled {len(rows)} new records from {len(seen_agents)} unique agents")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-sample", type=int, default=1000,
                         help="Number of NEW others-pool records to add beyond the existing gold pool")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    _ = oasf_lookup()  # warm cache

    print("Step 1: loading existing AI-service ('others') records from gold...")
    existing, gold_ids = _load_existing_ai_service_pool()
    print(f"  {len(existing)} records (rule could not decide under current production logic)")

    print("\nStep 2: backfilling agent context from local labeling sheets...")
    existing = _backfill_context_from_local_sheets(existing)

    print("\nStep 3: fetching context for records with no local-sheet match...")
    existing = _fetch_missing_context(existing)
    existing["pool_origin"] = "existing_gold_ai_service"
    existing["human_label"] = ""

    new_rows = pd.DataFrame()
    if args.new_sample > 0:
        print(f"\nStep 4: sampling {args.new_sample} NEW others-pool records from MongoDB...")
        sc_ids = set(pd.read_csv(SILVER_CANDIDATES_CSV, usecols=["feedback_id"])["feedback_id"]) if SILVER_CANDIDATES_CSV.exists() else set()
        ol_ids = set(pd.read_csv(OTHERS_TO_LABEL_CSV, usecols=["feedback_id"])["feedback_id"]) if OTHERS_TO_LABEL_CSV.exists() else set()
        exclude = gold_ids | sc_ids | ol_ids
        new_rows = _sample_new_others(args.new_sample, exclude)
        new_rows["pool_origin"] = "new_mongo_sample"
        new_rows["human_label"] = ""

    merged = pd.concat([existing, new_rows], ignore_index=True)
    merged = merged[OUTPUT_COLS]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    print(f"\n{'='*60}")
    print(f"Saved {len(merged)} records -> {args.out}")
    print(f"  existing_gold_ai_service: {(merged['pool_origin']=='existing_gold_ai_service').sum()}")
    print(f"  new_mongo_sample:         {(merged['pool_origin']=='new_mongo_sample').sum()}")
    has_ctx = (merged["agent_description"].fillna("").str.len() > 0) | (merged["agent_key"].fillna("") != "")
    print(f"  has any agent context:   {has_ctx.sum()} ({has_ctx.mean()*100:.1f}%)")
    print(f"\nFill in the 'human_label' column with one of: quality, quantity, junk")
    print("`existing_label` is shown for reference ONLY — it may be rule-derived silver, not ground truth.")


if __name__ == "__main__":
    main()
