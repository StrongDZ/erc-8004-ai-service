"""Build compact agent + feedback context for LLM / embedding consumption.

The Go backend (`internal/domain/classifier/humanize.go`) does this in
ad-hoc form per prompt version. The Python port reorganises into a single
context dict consumed by every approach, with three design changes:

1. agent_domain → service names only (generic web/OASF/A2A/email stripped) + OASF paths + tags
2. oasfDomains/Skills → top-K shallow hierarchical paths (avoids BSCAI-style flood)
3. description → use `agentSummary` field if present (filled by 02_agent_summary
   notebook); fall back to truncated description.
"""
from __future__ import annotations

import json
from collections.abc import Iterable

from .types import AgentMeta, FeedbackRecord


# Service names treated as protocol/infra plumbing rather than domain signal.
# These are filtered out when extracting "what business does this agent do".
_GENERIC_SERVICE_NAMES = frozenset({"web", "oasf", "a2a", "email"})


def is_generic_service(name: str) -> bool:
    """True when a service name is plumbing (web/OASF/A2A/email) rather than business."""
    return (name or "").strip().lower() in _GENERIC_SERVICE_NAMES


def _service_line(svc: dict) -> str | None:
    name = (svc.get("name") or "").strip()
    endpoint = (svc.get("endpoint") or "").strip()
    if not name or not endpoint:
        return None
    if len(endpoint) > 80:
        endpoint = endpoint[:77] + "…"
    return f"{name}:{endpoint}"


def format_services(services: Iterable[dict], max_services: int = 5) -> str:
    """Format services list as 'name:endpoint | name:endpoint | ...'."""
    out: list[str] = []
    for svc in services:
        line = _service_line(svc)
        if not line:
            continue
        out.append(line)
        if len(out) >= max_services:
            break
    return " | ".join(out)


def format_service_names(services: Iterable[dict], max_services: int = 5) -> str:
    """Format services list as 'name1, name2, ...' (names only, no endpoints)."""
    out: list[str] = []
    for svc in services:
        name = (svc.get("name") or "").strip() if isinstance(svc, dict) else (getattr(svc, "name", "") or "").strip()
        if not name:
            continue
        out.append(name)
        if len(out) >= max_services:
            break
    return ", ".join(out)


def _normalize_url(u: str) -> str:
    """Lowercase + strip trailing slash for endpoint comparison."""
    return (u or "").strip().lower().rstrip("/")


def endpoint_matches_services(feedback_endpoint: str, services: Iterable[dict]) -> bool:
    """True when the feedback endpoint matches any agent service endpoint.

    Matching is case-insensitive, ignores trailing slashes, and accepts
    bidirectional substring matches so a feedback against `https://x.com/v1` will
    match a service registered at `https://x.com`. Returns False when either
    side is empty.
    """
    fe = _normalize_url(feedback_endpoint)
    if not fe:
        return False
    for svc in services:
        se = _normalize_url(svc.get("endpoint", "") if isinstance(svc, dict) else getattr(svc, "endpoint", ""))
        if not se:
            continue
        if fe == se or fe in se or se in fe:
            return True
    return False


def find_matched_service(feedback_endpoint: str, services: Iterable[dict]) -> dict | None:
    """Return the first service whose endpoint matches the feedback endpoint, or None."""
    fe = _normalize_url(feedback_endpoint)
    if not fe:
        return None
    for svc in services:
        se = _normalize_url(svc.get("endpoint", "") if isinstance(svc, dict) else getattr(svc, "endpoint", ""))
        if not se:
            continue
        if fe == se or fe in se or se in fe:
            return svc if isinstance(svc, dict) else svc.__dict__
    return None


def domain_service_names(services: Iterable[dict], max_names: int = 5) -> list[str]:
    """Names of services that signal business/domain (i.e. not generic plumbing)."""
    out: list[str] = []
    seen: set[str] = set()
    for svc in services:
        name = (svc.get("name") or "").strip() if isinstance(svc, dict) else (getattr(svc, "name", "") or "").strip()
        if not name or is_generic_service(name):
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
        if len(out) >= max_names:
            break
    return out


def agent_domain_block(
    services: Iterable[dict],
    oasf_domains: list[str],
    oasf_skills: list[str],
    tags: list[str],
    domains_k: int = 3,
    skills_k: int = 5,
) -> dict:
    """Compact 'what is this agent about' bundle, generic services stripped.

    Combines: special service names (sans generic plumbing) + top-K OASF domain
    paths + top-K OASF skill paths + agent tags. This is the canonical
    "agent domain / lĩnh vực" signal the classifier consumes when deciding
    between app_specific and service_feedback for ambiguous tags.
    """
    return {
        "service_names": domain_service_names(services or []),
        "oasf_domains": top_k_hierarchical(list(oasf_domains or []), domains_k),
        "oasf_skills": top_k_hierarchical(list(oasf_skills or []), skills_k),
        "tags": [t.strip() for t in (tags or [])[:10] if (t or "").strip()],
    }


def top_k_hierarchical(paths: list[str], k: int) -> list[str]:
    """Return the K shallowest unique paths from an OASF list.

    Shallow paths (`technology/blockchain`) are more informative for
    industry-level discrimination than deep leaves (`technology/blockchain/defi`).
    Outliers like BSCAI list 200+ domains; we just take the first K after sorting
    by depth ascending (then by lexicographic for determinism).
    """
    if not paths:
        return []
    seen: set[str] = set()
    deduped: list[str] = []
    for p in paths:
        p = p.strip()
        if not p or p in seen:
            continue
        seen.add(p)
        deduped.append(p)
    deduped.sort(key=lambda s: (s.count("/"), s))
    return deduped[:k]


def agent_block(agent: AgentMeta, domains_k: int = 3, skills_k: int = 5) -> dict:
    """Compact agent context dict consumed by both prompt builders and embedders.

    Adds an `agent_domain` flat string that fuses the filtered service names
    (generic web/OASF/A2A/email stripped) + OASF domain paths + OASF skill paths +
    tags. This is the canonical "lĩnh vực" signal used by the classifier to
    decide whether an ambiguous tag belongs to the agent's business or not.
    """
    summary = agent.summary or agent.description or ""
    if len(summary) > 400:
        summary = summary[:397] + "…"

    dom = agent_domain_block(agent.services, agent.oasf_domains, agent.oasf_skills, agent.tags,
                             domains_k=domains_k, skills_k=skills_k)
    domain_parts = [*dom["service_names"], *dom["oasf_domains"], *dom["oasf_skills"], *dom["tags"]]
    agent_domain = ", ".join(p for p in domain_parts if p)

    return {
        "name": agent.name,
        "summary": summary,
        "services": format_service_names(agent.services or []),
        "domains": ", ".join(dom["oasf_domains"]),
        "skills": ", ".join(dom["oasf_skills"]),
        "tags": ", ".join(dom["tags"]),
        "agent_domain": agent_domain,
    }


def feedback_block(
    fb: FeedbackRecord,
    *,
    agent: AgentMeta | None = None,
    include_offchain_when_empty_tags: bool = True,
) -> dict:
    """Compact feedback context dict.

    When `agent` is provided, this also computes `endpoint_matched`: set to the
    matched service's "name:endpoint" string when the feedback endpoint matches
    one of the agent's registered services, absent otherwise. Downstream, a
    match means the feedback targets a real service the agent owns, so
    classification is restricted to {quality, quantity} (junk removed).

    feedbackParsed is included ONLY when both tag1 and tag2 are empty (the user's
    rule: "production classifies on tag1/tag2 if they exist; offchain is the
    fallback signal").
    """
    tag1 = (fb.tag1 or "").strip()
    tag2 = (fb.tag2 or "").strip()
    out: dict = {
        "tag1": tag1,
        "tag2": tag2,
        "comment": fb.feedback_parsed.get("comment", "") if fb.feedback_parsed else "",
        "scale": fb.value_scale or "",
        "value": (float(fb.value) if fb.value else 0.0) / 10 ** fb.value_decimals,
    }
    if agent is not None:
        matched = find_matched_service(fb.endpoint or "", agent.services or [])
        if matched:
            name = (matched.get("name") or "").strip()
            ep = (matched.get("endpoint") or "").strip()
            out["endpoint_matched"] = f"{name}:{ep}" if name else ep
    if include_offchain_when_empty_tags and not tag1 and not tag2 and fb.feedback_parsed:
        snippet = json.dumps(fb.feedback_parsed, ensure_ascii=False)
        if len(snippet) > 600:
            snippet = snippet[:597] + "…"
        out["offchain"] = snippet
    return out


def to_xml_block(name: str, fields: dict) -> str:
    """Render a dict as `<name><k>v</k>...</name>`. Skips empty values."""
    parts = [f"<{name}>"]
    for k, v in fields.items():
        if v is None or v == "" or v is False:
            continue
        parts.append(f"  <{k}>{v}</{k}>")
    parts.append(f"</{name}>")
    return "\n".join(parts)


def build_user_message(agent: AgentMeta, fb: FeedbackRecord) -> str:
    """Full XML user message ready for the LLM."""
    return "\n".join([
        to_xml_block("agent", agent_block(agent)),
        to_xml_block("feedback", feedback_block(fb, agent=agent)),
    ])


def build_embedding_text(agent: AgentMeta, fb: FeedbackRecord) -> str:
    """One-line text representation for sentence-transformer embedding.

    Format: 'tag1=X | tag2=Y | endpoint=Z | agent=NAME — SUMMARY | services=...'
    Kept flat (no XML) because sentence-transformers tokenisation works
    better on natural-ish text than on markup.
    """
    a = agent_block(agent)
    f = feedback_block(fb)
    parts = [
        f"tag1={f['tag1']}",
        f"tag2={f['tag2']}",
        f"endpoint={f.get('endpoint','')}",
        f"agent={a['name']} — {a['summary']}",
        f"services={a.get('services', '')}",
        f"domains={a['domains']}",
    ]
    return " | ".join(p for p in parts if p.split("=", 1)[1])
