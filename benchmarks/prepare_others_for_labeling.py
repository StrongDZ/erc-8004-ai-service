#!/usr/bin/env python3
"""Prepare "others" pool records for silver labeling by Claude Opus.

Queries MongoDB for feedback records the rule cascade could not classify
(category="others"), enriches each with full agent context (description,
summarizedDescription, services, oasfDomains + expanded text, oasfSkills +
expanded text, agent tags), and exports a CSV ready for LLM labeling.

Deduplicates against the existing gold set so nothing already labeled ends
up in the output.  Applies a per-agent cap to keep diversity.

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.prepare_others_for_labeling \\
        --sample 1500 \\
        --out data/labelled/others_to_label.csv

Output columns:
  feedback_id, tag1, tag2, value, value_decimals, scale, endpoint,
  offchain_note, offchain_json,
  agent_key, agent_name, agent_description, agent_services,
  agent_oasf_domains_raw, agent_oasf_domains_text,
  agent_oasf_skills_raw, agent_oasf_skills_text,
  agent_tags, agent_domain_text,
  opus_label, opus_reason,        <- to be filled by labeler
  category, feature, agent_domains, source   <- gold-compatible merge columns
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
GOLD_CSV = ROOT.parent / "erc-8004-benchmarking-be/scripts/labelled/gold_final.csv"
DEFAULT_OUT = ROOT / "data/labelled/others_to_label.csv"

MAX_DESC_CHARS = 500
MAX_OFFCHAIN_CHARS = 400
MAX_PER_AGENT = 15  # cap per agent for diversity


def _load_gold_ids() -> set[str]:
    if not GOLD_CSV.exists():
        print(f"[warn] gold CSV not found at {GOLD_CSV}, skipping dedup")
        return set()
    df = pd.read_csv(GOLD_CSV, usecols=["feedback_id"])
    return set(df["feedback_id"].astype(str).tolist())


def _fetch_agent_meta(ag_coll, agent_key: str) -> AgentMeta:
    chain_id_str, agent_id = agent_key.split(":", 1) if ":" in agent_key else ("0", agent_key)
    doc = ag_coll.find_one({"_id": agent_key}, {
        "name": 1, "description": 1, "summarizedDescription": 1,
        "services": 1, "oasfDomains": 1, "oasfSkills": 1, "tags": 1,
    }) or {}
    return AgentMeta(
        chain_id=int(chain_id_str),
        agent_id=agent_id,
        name=(doc.get("name") or "").strip(),
        description=(doc.get("summarizedDescription") or doc.get("description") or "").strip(),
        services=doc.get("services") or [],
        oasf_domains=doc.get("oasfDomains") or [],
        oasf_skills=doc.get("oasfSkills") or [],
        tags=[t for t in (doc.get("tags") or []) if t],
    )


def _offchain_extract(fp) -> tuple[str, str]:
    """Return (note_text, full_json_snippet) from feedbackParsed."""
    if fp is None:
        return "", ""
    if isinstance(fp, str):
        if fp.strip() in ("", "null", "None"):
            return "", ""
        return fp[:MAX_OFFCHAIN_CHARS], fp[:MAX_OFFCHAIN_CHARS]
    if isinstance(fp, dict):
        note = (fp.get("note") or fp.get("text") or fp.get("summary") or fp.get("content") or "").strip()
        full = json.dumps(fp, ensure_ascii=False)
        return note[:MAX_OFFCHAIN_CHARS], full[:MAX_OFFCHAIN_CHARS]
    return "", ""


def _format_services(services: list[dict]) -> str:
    """'name:endpoint | name:endpoint | ...' (up to 6 services)."""
    out = []
    for svc in (services or [])[:6]:
        name = (svc.get("name") or "").strip()
        ep = (svc.get("endpoint") or "").strip()
        if name and ep:
            out.append(f"{name}:{ep[:60]}")
        elif name:
            out.append(name)
    return " | ".join(out)


def prepare(sample: int, out_path: Path) -> None:
    # Preload OASF lookup so we expand paths to human text
    _ = oasf_lookup()  # warms cache

    gold_ids = _load_gold_ids()
    print(f"Gold set: {len(gold_ids)} records (excluded from output)")

    fb_coll = feedback_coll()
    ag_coll = agents_coll()

    cursor = fb_coll.find(
        {"category": "others"},
        {
            "_id": 1,
            "tag1": 1, "tag2": 1,
            "value": 1, "valueDecimals": 1, "valueScale": 1,
            "endpoint": 1, "feedbackParsed": 1,
            "agentId": 1, "chainId": 1,
        },
        limit=sample * 5,  # over-fetch to absorb dedup + per-agent cap
    ).sort("_id", 1)

    rows = []
    seen_agents: dict[str, int] = {}

    for doc in cursor:
        fid = str(doc["_id"])
        if fid in gold_ids:
            continue

        chain_id = doc.get("chainId", 0)
        agent_id = str(doc.get("agentId", ""))
        agent_key = f"{chain_id}:{agent_id}"

        if seen_agents.get(agent_key, 0) >= MAX_PER_AGENT:
            continue

        ag = _fetch_agent_meta(ag_coll, agent_key)

        # Expand OASF paths → human-readable text
        oasf_dom_text = expand_oasf(ag.oasf_domains)
        oasf_sk_text = expand_oasf(ag.oasf_skills)

        # Combined domain signal for Opus prompt
        desc_trimmed = ag.description[:MAX_DESC_CHARS]
        domain_parts = [p for p in [desc_trimmed, oasf_dom_text, oasf_sk_text] if p]
        if ag.tags:
            domain_parts.append(", ".join(ag.tags[:8]))
        agent_domain_text = " | ".join(domain_parts)

        # Offchain content
        offchain_note, offchain_json = _offchain_extract(doc.get("feedbackParsed"))

        rows.append({
            # Feedback
            "feedback_id": fid,
            "tag1": str(doc.get("tag1", "") or "").strip(),
            "tag2": str(doc.get("tag2", "") or "").strip(),
            "value": str(doc.get("value", "") or ""),
            "value_decimals": int(doc.get("valueDecimals", 0) or 0),
            "scale": str(doc.get("valueScale", "") or "").strip(),
            "endpoint": str(doc.get("endpoint", "") or "").strip(),
            "offchain_note": offchain_note,
            "offchain_json": offchain_json,
            # Agent context
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
            # To be filled by labeler
            "opus_label": "",
            "opus_reason": "",
            # Gold-compatible merge columns (empty until merge step)
            "category": "",
            "feature": "",
            "agent_domains": ", ".join(ag.oasf_domains[:3]),
            "source": "silver_opus",
        })

        seen_agents[agent_key] = seen_agents.get(agent_key, 0) + 1
        if len(rows) >= sample:
            break

    print(f"Collected {len(rows)} records from {len(seen_agents)} unique agents")

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

    # Coverage stats
    has_desc = df["agent_description"].str.len() > 0
    has_svcs = df["agent_services"].str.len() > 0
    has_oasf_dom = df["agent_oasf_domains_raw"].str.len() > 0
    has_oasf_sk = df["agent_oasf_skills_raw"].str.len() > 0
    has_offchain = df["offchain_note"].str.len() > 0
    any_context = has_desc | has_svcs | has_oasf_dom | has_oasf_sk | has_offchain

    n = len(df)
    print(f"\nAgent context coverage (N={n}):")
    print(f"  description          : {has_desc.sum():4d} ({has_desc.mean()*100:.1f}%)")
    print(f"  services             : {has_svcs.sum():4d} ({has_svcs.mean()*100:.1f}%)")
    print(f"  oasf domains         : {has_oasf_dom.sum():4d} ({has_oasf_dom.mean()*100:.1f}%)")
    print(f"  oasf skills          : {has_oasf_sk.sum():4d} ({has_oasf_sk.mean()*100:.1f}%)")
    print(f"  offchain note        : {has_offchain.sum():4d} ({has_offchain.mean()*100:.1f}%)")
    print(f"  ANY context          : {any_context.sum():4d} ({any_context.mean()*100:.1f}%)")
    print(f"  NO context at all    : {(~any_context).sum():4d} ({(~any_context).mean()*100:.1f}%)")

    tag1_empty = (df["tag1"] == "").sum()
    both_empty = ((df["tag1"] == "") & (df["tag2"] == "")).sum()
    print(f"\nTag coverage:")
    print(f"  tag1 empty           : {tag1_empty:4d} ({tag1_empty/n*100:.1f}%)")
    print(f"  both tags empty      : {both_empty:4d} ({both_empty/n*100:.1f}%)")

    print(f"\nTop 10 scales:\n{df['scale'].value_counts().head(10).to_string()}")
    print(f"\nTop 10 tag1 values:\n{df['tag1'].replace('', '(empty)').value_counts().head(10).to_string()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=1500)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    prepare(args.sample, args.out)


if __name__ == "__main__":
    main()
