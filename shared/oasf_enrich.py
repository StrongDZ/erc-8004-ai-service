"""Resolve agent OASF paths -> descriptive text and build the agent-domain tower.

OASF schema (caption + description per taxonomy path) lives in the main backend
DB (erc8004), keyed by the hierarchical path stored on agents in
oasfDomains / oasfSkills. The whole table is tiny (~340 rows) so we load it once
into a dict and expand each agent's paths into natural-language text for embedding.

This is the enrichment signal validated for the 'others' pool: agent.description
covers 100% of feedback (feedback-weighted) and OASF paths cover ~57% of the
others pool, both targeting the app_specific vs service_feedback boundary.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from .mongo_client import oasf_domains_coll, oasf_skills_coll

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def oasf_lookup() -> dict[str, str]:
    """path -> 'Caption: description' for every OASF domain + skill entry."""
    out: dict[str, str] = {}
    for coll in (oasf_domains_coll(), oasf_skills_coll()):
        for e in coll.find({}, {"_id": 1, "caption": 1, "description": 1}):
            cap = (e.get("caption") or e["_id"]).strip()
            desc = (e.get("description") or "").strip()
            out[e["_id"]] = (f"{cap}: {desc}" if desc else cap).strip()
    log.info("OASF lookup loaded: %d entries", len(out))
    return out


def expand_oasf(paths) -> str:
    """Join the descriptive text of each OASF path (unknown paths pass through)."""
    lut = oasf_lookup()
    return " ".join(lut.get(p, p) for p in (paths or []) if p)


def agent_domain_text(
    description: str = "",
    oasf_domains=None,
    oasf_skills=None,
    max_chars: int = 1000,
) -> str:
    """Agent-tower text = description + expanded OASF domain/skill descriptions.

    Returns '' when no signal is available, so the caller can use a zero vector
    (missing agent context contributes nothing to the late-fusion concatenation).
    """
    parts: list[str] = []
    if (description or "").strip():
        parts.append(description.strip())
    d = expand_oasf(oasf_domains)
    if d:
        parts.append(d)
    s = expand_oasf(oasf_skills)
    if s:
        parts.append(s)
    return " | ".join(parts)[:max_chars]
