#!/usr/bin/env python3
"""Build a diverse, class-balanced silver candidate set for Opus labeling.

Problem this solves: the "others" pool is quality-by-nature (the rule engine
already routes named-metric->quantity and gibberish->junk upstream), and it is
dominated by near-duplicate epoch-fitness records.  Sampling only "others"
yields a 98% quality, low-diversity set that cannot evaluate quantity/junk.

Strategy: stratified sampling across ALL FOUR runtime rule-categories
(quality / quantity / junk / others), with a hard cap of K records per
(tag1, tag2, value_scale) signature for diversity and M records per agent.
The rule-category is used ONLY to stratify the draw so every class is well
represented; the TRUE label is assigned afterwards by the convention labeler
(Opus), and rule<->Opus disagreement is itself a useful signal.

Each row is enriched with full agent context (description, services, OASF
domains+skills text, tags).

Usage:
    .venv/bin/python3 -m benchmarks.prepare_balanced_silver \\
        --out data/labelled/silver_candidates.csv

Then:
    .venv/bin/python3 -m benchmarks.convention_label --records data/labelled/silver_candidates.csv
    # Opus reviews quantity/junk picks; finally merge_silver_gold.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.prepare_others_for_labeling import (
    _fetch_agent_meta, _format_services, _offchain_extract, MAX_DESC_CHARS,
)
from shared.mongo_client import agents_coll, feedback_coll
from shared.oasf_enrich import expand_oasf, oasf_lookup
from shared.types import MONGO_CATEGORY_ALIASES

ROOT = Path(__file__).resolve().parent.parent
GOLD_CSV = ROOT.parent / "erc-8004-benchmarking-be/scripts/labelled/gold_final.csv"
DEFAULT_OUT = ROOT / "data/labelled/silver_candidates.csv"

# Per-category sampling plan: (max_total, max_per_signature, max_per_agent).
# Tuned to the diversity ceiling of each bucket (junk has only 61 signatures).
PLAN = {
    "quantity": (900, 3, 12),
    "quality":  (700, 2, 10),
    "others":   (700, 5, 12),
    "junk":     (300, 6, 20),
}


def _load_gold_ids() -> set[str]:
    if not GOLD_CSV.exists():
        return set()
    return set(pd.read_csv(GOLD_CSV, usecols=["feedback_id"])["feedback_id"].astype(str))


def _sample_bucket(fb_coll, ag_coll, aliases, max_total, cap_sig, cap_agent,
                   gold_ids, taken_ids) -> list[dict]:
    cursor = fb_coll.find(
        {"category": {"$in": aliases}},
        {"_id": 1, "tag1": 1, "tag2": 1, "value": 1, "valueDecimals": 1,
         "valueScale": 1, "endpoint": 1, "feedbackParsed": 1,
         "agentId": 1, "chainId": 1, "category": 1},
    ).sort("_id", 1)

    rows = []
    sig_count: dict[tuple, int] = {}
    agent_count: dict[str, int] = {}

    for doc in cursor:
        fid = str(doc["_id"])
        if fid in gold_ids or fid in taken_ids:
            continue
        tag1 = str(doc.get("tag1", "") or "").strip()
        tag2 = str(doc.get("tag2", "") or "").strip()
        scale = str(doc.get("valueScale", "") or "").strip()
        sig = (tag1.lower(), tag2.lower(), scale.lower())
        if sig_count.get(sig, 0) >= cap_sig:
            continue
        chain_id = doc.get("chainId", 0)
        agent_id = str(doc.get("agentId", ""))
        agent_key = f"{chain_id}:{agent_id}"
        if agent_count.get(agent_key, 0) >= cap_agent:
            continue

        ag = _fetch_agent_meta(ag_coll, agent_key)
        oasf_dom_text = expand_oasf(ag.oasf_domains)
        oasf_sk_text = expand_oasf(ag.oasf_skills)
        desc = ag.description[:MAX_DESC_CHARS]
        dom_parts = [p for p in [desc, oasf_dom_text, oasf_sk_text] if p]
        if ag.tags:
            dom_parts.append(", ".join(ag.tags[:8]))
        offchain_note, offchain_json = _offchain_extract(doc.get("feedbackParsed"))

        rows.append({
            "feedback_id": fid,
            "tag1": tag1, "tag2": tag2,
            "value": str(doc.get("value", "") or ""),
            "value_decimals": int(doc.get("valueDecimals", 0) or 0),
            "scale": scale,
            "endpoint": str(doc.get("endpoint", "") or "").strip(),
            "offchain_note": offchain_note, "offchain_json": offchain_json,
            "agent_key": agent_key, "agent_name": ag.name,
            "agent_description": desc,
            "agent_services": _format_services(ag.services),
            "agent_oasf_domains_raw": ", ".join(ag.oasf_domains[:5]),
            "agent_oasf_domains_text": oasf_dom_text[:300],
            "agent_oasf_skills_raw": ", ".join(ag.oasf_skills[:8]),
            "agent_oasf_skills_text": oasf_sk_text[:400],
            "agent_tags": ", ".join(ag.tags[:10]),
            "agent_domain_text": " | ".join(dom_parts)[:600],
            "rule_category": str(doc.get("category", "")),
            "opus_label": "", "opus_reason": "",
            "category": "", "feature": "",
            "agent_domains": ", ".join(ag.oasf_domains[:3]),
            "source": "silver_opus",
        })
        sig_count[sig] = sig_count.get(sig, 0) + 1
        agent_count[agent_key] = agent_count.get(agent_key, 0) + 1
        taken_ids.add(fid)
        if len(rows) >= max_total:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    _ = oasf_lookup()
    gold_ids = _load_gold_ids()
    taken_ids: set[str] = set()
    fb_coll, ag_coll = feedback_coll(), agents_coll()

    all_rows = []
    for cat, (mt, cs, ca) in PLAN.items():
        aliases = MONGO_CATEGORY_ALIASES[cat]
        rows = _sample_bucket(fb_coll, ag_coll, aliases, mt, cs, ca, gold_ids, taken_ids)
        print(f"  rule={cat:9s}: sampled {len(rows):4d} (cap {cs}/sig, {ca}/agent, target {mt})")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nTotal silver candidates: {len(df)} -> {args.out}")
    print(f"Unique (tag1,tag2,scale) signatures: {df.groupby(['tag1','tag2','scale']).ngroups}")
    print(f"Unique agents: {df['agent_key'].nunique()}")
    print("\nBy rule_category (sampling strata):")
    print(df["rule_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
