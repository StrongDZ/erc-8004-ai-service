"""shared/three_tier.py — production 3-tier feedback classifier.

Mirrors the benchmark pipeline (benchmarks/pipeline_3tier_v2.py, Run 4 = rule +
per-tag SVM + agent-domain cosine, NO LLM — the benchmark's best config on the
de-duplicated, self-feedback-excluded test set: MacroF1 0.814).

The rule cascade (Stage 0 self-feedback gate, Stage 1 rule lookups) runs in the
Go classifier; records Go escalates to "others" reach this module, which runs:

  Stage 2   per-tag binary SVM      TF-IDF + calibrated LinearSVC, symmetric vote.
  Stage 3   agent-domain cosine     max cos(tag, agent_domain_text), embedded on
                                    the fly with bge-small (no prebuilt index,
                                    no agent_key dependency). in-domain bounded
                                    -> quality, in-domain unbounded -> quantity.
  Stage 4   LLM                     no agent metadata OR a borderline cosine goes
                                    straight to the LLM. No ML default or scale
                                    heuristic — the LLM resolves these residuals.

Convention (gold-aligned): quality is only a positive bounded score, so an
unbounded scale (or a negative value, which maps to unbounded) is NEVER quality.
It is NOT, however, automatically quantity: a gibberish tag on an unbounded scale
is junk. The Go rule engine therefore escalates unknown unbounded tags to this
module rather than force-labelling them quantity; here the junk-vs-quantity split
is decided by content (agent-domain cosine, or the LLM), not by scale.

The per-tag SVM inference (predict_quality_prob, vote_per_tag) is duplicated from
benchmarks/per_tag_svm.py on purpose: production must not import research code.
Both share the serialized artifact data/models/per_tag_svm.joblib.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import joblib
import numpy as np

from .oasf_enrich import agent_domain_text

ROOT = Path(__file__).resolve().parent.parent
SVM_MODEL_PATH = ROOT / "data/models/per_tag_svm.joblib"

# Cosine thresholds — empirically tuned on bge-small (benchmark stage3_domain.py).
DOMAIN_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
SVM_VOTE_THRESH = 0.70
THRESH_IN_DOMAIN = 0.55

# Mirrors Go infraTagSet (rule_patterns.go) — generic infra/protocol signals that apply
# to ANY agent regardless of business domain → feature = "infrastructure".
_INFRA_TAGS: frozenset[str] = frozenset({
    "reachable", "liveness", "liveness-check", "uptime", "successrate", "success-rate",
    "responsetime", "response-time", "ping", "health-check",
    "blocktimefreshness", "blocktime-freshness", "blocktime freshness", "a2a", "mcp",
    "web", "trust-oracle", "oracle-screening", "trust-score",
    "trustscore", "reputation", "sentinel8004", "agentguard",
    "safety-score", "ownerverified", "owner verified", "sentinelnet-v1",
    # additional infra keywords caught by substring matching
    "automated-screening", "contractrisk", "counterparty",
})


def _infer_feature(tag1: str, tag2: str, category: str, source: str) -> str:
    """Heuristic feature inference for 3-tier results (no LLM context available).

    Logic mirrors Go featureOf() and adds source-based signals:
    - junk                → ""  (no feature — noise)
    - tag in infraTagSet  → "infrastructure"  (generic protocol/infra signal)
    - classified via agent-domain cosine (source contains 'domain') → "agent_domain"
    - fallback            → "agent_domain"  (default: business-specific)
    """
    if category == "junk":
        return ""
    t1 = tag1.strip().lower()
    t2 = tag2.strip().lower()
    if t1 in _INFRA_TAGS or t2 in _INFRA_TAGS:
        return "infrastructure"
    # Source signal: agent-domain cosine stage explicitly matched the agent domain
    if "domain" in source:
        return "agent_domain"
    # Default for quality/quantity determined by SVM or scale heuristic
    return "agent_domain"


@dataclass
class ThreeTierResult:
    category: str
    confidence: float
    reason: str
    source: str
    feature: str = ""  # "infrastructure" | "agent_domain" | "" (for junk)


@lru_cache(maxsize=1)
def load_svm():
    """Load the serialized per-tag SVM (internal artifact written by train())."""
    return joblib.load(SVM_MODEL_PATH)


def _quality_prob(pipe, tag: str, scale: str) -> float:
    """Quality probability [0,1] for one tag+scale (same feature text as training)."""
    text = f"tag={tag} | scale={scale}"
    proba = pipe.predict_proba([text])[0]
    quality_idx = list(pipe.classes_).index(1)  # classes_[1] == 1 (quality)
    return float(proba[quality_idx])


def _vote(p1: float, p2: float, t2_empty: bool, thresh: float = SVM_VOTE_THRESH) -> str | None:
    """Symmetric per-tag vote: quality if p>=thresh, non_quality if p<=1-thresh.
    Returns 'quality', 'non_quality', or None (escalate to Stage 3)."""

    def _confident(p: float) -> str | None:
        if p >= thresh:
            return "quality"
        if p <= 1.0 - thresh:
            return "non_quality"
        return None

    c1 = _confident(p1)
    c2 = None if t2_empty else _confident(p2)
    if t2_empty:
        return c1
    if c1 is not None and c2 is not None:
        return c1 if c1 == c2 else None  # real conflict -> Stage 3
    return c1 if c1 is not None else c2


def _scale_to_label(scale: str) -> str:
    """In-domain: scale decides. Unbounded -> quantity, bounded -> quality."""
    return "quantity" if scale == "unbounded" else "quality"


def _llm_resolve(
    llm_classify_fn: "Callable[[], tuple[str, float, str, str | None]] | None",
    reason_prefix: str,
) -> "ThreeTierResult":
    """Resolve a residual record with the LLM. No ML default / scale heuristic:
    records with no agent metadata or a borderline cosine go straight here.
    Requires an LLM fallback to be supplied."""
    if llm_classify_fn is None:
        raise ValueError(
            f"three-tier residual ({reason_prefix}) requires an LLM fallback; "
            "no ML default or scale heuristic is used"
        )
    llm_cat, llm_conf, llm_reason, llm_feat = llm_classify_fn()
    return ThreeTierResult(
        category=llm_cat,
        confidence=llm_conf,
        reason=f"[{reason_prefix}] {llm_reason}",
        source="llm",
        feature=llm_feat,
    )


def _domain_best_cos(encoder, tags: list[str], agent_text: str) -> float:
    """Max cosine(tag, agent_text), both L2-normalized (bge encode)."""
    texts = [agent_text] + tags
    vecs = encoder.encode(texts, normalize_embeddings=True)
    agent_vec = np.asarray(vecs[0], dtype="float32")
    return max(float(np.dot(np.asarray(v, dtype="float32"), agent_vec)) for v in vecs[1:])


def build_agent_text(description: str, oasf_domains, oasf_skills) -> str:
    """Agent-domain text for the cosine check (description + expanded OASF)."""
    return agent_domain_text(description or "", list(oasf_domains or []), list(oasf_skills or []))


def classify_three_tier(
    *,
    encoder,
    tag1: str,
    tag2: str,
    scale: str,
    value_norm: float,
    agent_text: str,
    value_decimals: int = 0,
    llm_classify_fn: Callable[[], tuple[str, float, str, str | None]] | None = None,
) -> ThreeTierResult:
    """Run Stage 0.5 -> 2 -> 3 -> 4 for an "others" record. `encoder` is the
    bge-small SentenceTransformer used for the agent-domain cosine check.
    Returns ThreeTierResult with feature inferred from tag signals and classifier source.
    """
    t1, t2 = (tag1 or "").strip(), (tag2 or "").strip()
    sc = (scale or "").strip().lower()
    pipe = load_svm()

    def _result(cat, conf, reason, source):
        feat = _infer_feature(t1, t2, cat, source)
        return ThreeTierResult(cat, float(max(0.0, min(1.0, conf))), reason, source, feature=feat)

    # Stage 2 — per-tag SVM vote.
    p1 = _quality_prob(pipe, t1, sc) if t1 else 0.5
    p2 = _quality_prob(pipe, t2, sc) if t2 else 0.5
    vote = _vote(p1, p2, t2_empty=not t2)
    if vote == "quality":
        return _result("quality", max(p1, p2), f"svm vote quality (p1={p1:.2f},p2={p2:.2f})", "three_tier_svm")

    # Stage 3 — agent-domain cosine (vote == non_quality or None).
    tags = [t for t in (t1, t2) if t]
    agent_text = (agent_text or "").strip()

    # No agent metadata -> straight to the LLM (no scale heuristic, no ML default).
    if not agent_text:
        return _llm_resolve(llm_classify_fn, "no_agent_metadata")

    best_cos = _domain_best_cos(encoder, tags, agent_text)
    if best_cos > THRESH_IN_DOMAIN:
        return _result(_scale_to_label(sc), min(1.0, best_cos), f"in-domain cos={best_cos:.3f}", "three_tier_domain")

    # Stage 4 — borderline cosine -> LLM (no ML default).
    return _llm_resolve(llm_classify_fn, f"borderline_cos={best_cos:.3f}")
