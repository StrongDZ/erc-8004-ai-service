"""shared/three_tier.py — production 3-tier feedback classifier.

Mirrors the benchmark pipeline (benchmarks/pipeline_run13.py, "mandatory
escalation" — the configuration that wins on every reliably-measurable
accuracy axis (two-class Macro F1, quantity F1, quantity recall) once the
3-record junk class is set aside as too small to rank configurations by; see
the thesis evaluation chapter for the full audit).

The rule cascade (Stage 0 self-feedback gate, Stage 1 rule lookups) runs in the
Go classifier; records Go escalates to "others" reach this module, which runs:

  Stage 2   per-tag BGE quality gate   one-directional: the SVM may only ever
                                        assert "quality", at a high threshold.
                                        It is NEVER used to assert "quantity" or
                                        "non_quality" — an audited failure mode
                                        (low confidence on unfamiliar business-
                                        /service-domain vocabulary, not evidence
                                        of a metric) made every symmetric or
                                        scale-default alternative tried less
                                        accurate than just escalating instead.
  Stage 3   agent-domain cosine        max cos(tag, agent_domain_text), embedded
                                        on the fly with bge-small (no prebuilt
                                        index, no agent_key dependency).
                                        in-domain + unbounded -> quantity (safe:
                                        the scale convention makes this
                                        structural, not a guess). in-domain +
                                        bounded, with Stage 2 not confident,
                                        is NOT resolved here — see Stage 4.
  Stage 4   LLM                        every record Stage 2/3 did not resolve
                                        with genuine evidence goes here: no
                                        agent metadata, a borderline cosine, OR
                                        an in-domain-but-bounded record with no
                                        confident "quality" signal. No ML
                                        default or scale heuristic ever guesses
                                        "quantity" in this module.

Convention (gold-aligned): quality is only a positive bounded score, so an
unbounded scale (or a negative value, which maps to unbounded) is NEVER quality.
It is NOT, however, automatically quantity: a gibberish tag on an unbounded scale
is junk. The Go rule engine therefore escalates unknown unbounded tags to this
module rather than force-labelling them quantity; here the junk-vs-quantity split
is decided by content (agent-domain cosine, or the LLM), not by scale.

The per-tag SVM inference (predict_quality_gate_prob) is duplicated from
benchmarks/per_tag_svm.py on purpose: production must not import research code.
Both share the serialized artifact data/models/per_tag_svm_bge_quality_gate.joblib.
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
SVM_MODEL_PATH = ROOT / "data/models/per_tag_svm_bge_quality_gate.joblib"

# Cosine/SVM thresholds — empirically tuned on bge-small (benchmark stage3_domain.py,
# benchmarks/pipeline_run13.py). SVM_QUALITY_THRESH is a one-directional bar: it is
# never compared against its complement to assert the opposite class.
DOMAIN_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
SVM_QUALITY_THRESH = 0.80
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
    """Load the serialized BGE quality-gate classifier (artifact written by
    benchmarks.per_tag_svm.train_bge_quality_gate()).

    joblib.load is safe here: SVM_MODEL_PATH is a fixed, internal artifact path
    written only by this codebase's own training script, never derived from
    user-supplied input or an external/untrusted source."""
    return joblib.load(SVM_MODEL_PATH)


def _quality_prob(encoder, clf, tag: str, scale: str) -> float:
    """One-directional quality probability via BGE embedding of "<tag> <scale>".

    Only ever compared against SVM_QUALITY_THRESH to assert "quality"; a low
    value must NOT be read as evidence for "quantity" (see module docstring)."""
    vec = encoder.encode([f"{tag.strip().lower()} {scale.strip().lower()}"], normalize_embeddings=True)[0]
    proba = clf.predict_proba([np.asarray(vec, dtype="float32")])[0]
    quality_idx = list(clf.classes_).index(1)  # classes_[1] == 1 (quality)
    return float(proba[quality_idx])


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
    is_unbounded = sc == "unbounded"
    clf = load_svm()

    def _result(cat, conf, reason, source):
        feat = _infer_feature(t1, t2, cat, source)
        return ThreeTierResult(cat, float(max(0.0, min(1.0, conf))), reason, source, feature=feat)

    # Stage 2 — one-directional BGE quality gate. May only ever assert "quality";
    # never used to assert "quantity" (unbounded is structurally never quality,
    # so the gate cannot fire there either — see module docstring).
    p1 = _quality_prob(encoder, clf, t1, sc) if t1 else 0.0
    p2 = _quality_prob(encoder, clf, t2, sc) if t2 else 0.0
    quality_prob = max(p1, p2) if t2 else p1
    if quality_prob > SVM_QUALITY_THRESH and not is_unbounded:
        return _result("quality", quality_prob, f"svm quality_prob={quality_prob:.2f}", "three_tier_svm")

    # Stage 3 — agent-domain cosine (Stage 2 not confident).
    tags = [t for t in (t1, t2) if t]
    agent_text = (agent_text or "").strip()

    # No agent metadata -> straight to the LLM (no scale heuristic, no ML default).
    if not agent_text:
        return _llm_resolve(llm_classify_fn, "no_agent_metadata")

    best_cos = _domain_best_cos(encoder, tags, agent_text)
    if best_cos > THRESH_IN_DOMAIN:
        if is_unbounded:
            # Safe to resolve here: the scale convention makes "quantity" structural,
            # not a probabilistic guess (unbounded can never be "quality").
            return _result("quantity", min(1.0, best_cos), f"in-domain unbounded cos={best_cos:.3f}", "three_tier_domain")
        # In-domain but bounded, with Stage 2 not confidently "quality": mandatory
        # escalation. Asserting "quality" here by default, or "quantity" via a
        # low-confidence tie-break, were both audited and found less accurate
        # than escalating (see module docstring) — never guess in this branch.
        return _llm_resolve(llm_classify_fn, f"in_domain_bounded_low_conf cos={best_cos:.3f},prob={quality_prob:.2f}")

    # Stage 4 — borderline/not-in-domain cosine -> LLM (no ML default).
    return _llm_resolve(llm_classify_fn, f"borderline_cos={best_cos:.3f}")
