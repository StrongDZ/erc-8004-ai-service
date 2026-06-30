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

from .oasf_enrich import agent_domain_text_full

ROOT = Path(__file__).resolve().parent.parent
SVM_MODEL_PATH = ROOT / "data/models/per_tag_svm_bge_quality_gate.joblib"

# Cosine/SVM thresholds — empirically tuned on bge-small (benchmark stage3_domain.py,
# benchmarks/pipeline_run13.py). SVM_QUALITY_THRESH is a one-directional bar: it is
# never compared against its complement to assert the opposite class.
DOMAIN_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
SVM_QUALITY_THRESH = 0.70
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


def _infer_feature(tag1: str, tag2: str, category: str, in_domain: bool | None) -> str:
    """Derive feature from FAISS in_domain signal + tag heuristics.

    Priority order:
    - junk                           → ""   (noise, no feature)
    - tag in infraTagSet             → "infrastructure"  (protocol/infra signal)
    - in_domain=True  (cos > 0.55)   → "agent_domain"
    - in_domain=False, quantity      → "infrastructure"  (metric outside agent domain)
    - fallback                       → "agent_domain"  (default: assume business-specific)
    """
    if category == "junk":
        return ""
    t1 = tag1.strip().lower()
    t2 = tag2.strip().lower()
    if t1 in _INFRA_TAGS or t2 in _INFRA_TAGS:
        return "infrastructure"
    if in_domain is True:
        return "agent_domain"
    if in_domain is False and category == "quantity":
        return "infrastructure"
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


def _scale_heuristic(scale: str) -> str | None:
    """No-agent-signal fallback, decided by scale alone (mirrors benchmark
    stage3_domain.scale_heuristic): unbounded -> quantity; star5/star10/binary ->
    quality; else (pct100) -> None (escalate to LLM)."""
    s = (scale or "").strip().lower()
    if s == "unbounded":
        return "quantity"
    if s in ("star5", "star10", "binary"):
        return "quality"
    return None


def build_agent_text(description: str, oasf_domains, oasf_skills, service_names=None, tags=None) -> str:
    """Canonical agent-domain text for the cosine check: description + expanded
    OASF domains/skills + (non-generic) service names + tags. Empty components
    contribute nothing; returns '' only when every component is empty."""
    return agent_domain_text_full(
        description or "",
        list(oasf_domains or []),
        list(oasf_skills or []),
        list(service_names or []),
        list(tags or []),
    )


def classify_three_tier(
    *,
    encoder,
    tag1: str,
    tag2: str,
    scale: str,
    value_norm: float,
    agent_text: str,
    llm_classify_fn: Callable[[], tuple[str, float, str, str | None]] | None = None,
) -> ThreeTierResult:
    """Run Stage 0.5 -> 2 -> 3 -> 4 for an "others" record. `encoder` is the
    bge-small SentenceTransformer used for the agent-domain cosine check.
    Returns ThreeTierResult with feature derived from FAISS in_domain signal.
    """
    t1, t2 = (tag1 or "").strip(), (tag2 or "").strip()
    sc = (scale or "").strip().lower()
    is_unbounded = sc == "unbounded"
    clf = load_svm()
    agent_str = (agent_text or "").strip()
    tags = [t for t in (t1, t2) if t]

    # Early domain cosine: compute for ALL records so Stage 2 (SVM quality gate)
    # records also receive an in_domain signal for feature assignment, not only
    # Stage 3/4 records. Adds one encoder.encode() call (~20 ms on BGE-small/CPU)
    # for Stage 2 records; negligible vs LLM latency for Stage 4 records.
    if agent_str and tags:
        best_cos = _domain_best_cos(encoder, tags, agent_str)
        in_domain: bool | None = best_cos > THRESH_IN_DOMAIN
    else:
        best_cos = 0.0
        in_domain = None

    def _result(cat: str, conf: float, reason: str, source: str) -> ThreeTierResult:
        feat = _infer_feature(t1, t2, cat, in_domain)
        return ThreeTierResult(cat, float(max(0.0, min(1.0, conf))), reason, source, feature=feat)

    def _llm_with_feature(reason_prefix: str) -> ThreeTierResult:
        """Call LLM fallback and override its feature with our in_domain signal."""
        res = _llm_resolve(llm_classify_fn, reason_prefix)
        res.feature = _infer_feature(t1, t2, res.category, in_domain)
        return res

    # Stage 2 — one-directional BGE quality gate. May only ever assert "quality";
    # never used to assert "quantity" (unbounded is structurally never quality,
    # so the gate cannot fire there either — see module docstring).
    p1 = _quality_prob(encoder, clf, t1, sc) if t1 else 0.0
    p2 = _quality_prob(encoder, clf, t2, sc) if t2 else 0.0
    quality_prob = max(p1, p2) if t2 else p1
    if quality_prob > SVM_QUALITY_THRESH and not is_unbounded:
        return _result("quality", quality_prob, f"svm quality_prob={quality_prob:.2f}", "three_tier_svm")

    # Stage 3 — agent-domain cosine (Stage 2 not confident).
    # No agent-domain signal at all -> scale_heuristic (matches benchmark
    # run13.resolve); only when the heuristic abstains do we escalate to the LLM.
    if not agent_str:
        sh = _scale_heuristic(sc)
        if sh is not None:
            return _result(sh, 0.60, f"scale_heuristic:{sc}", "three_tier_scale")
        return _llm_with_feature("no_agent_metadata")

    if best_cos > THRESH_IN_DOMAIN:
        if is_unbounded:
            # Safe to resolve here: the scale convention makes "quantity" structural,
            # not a probabilistic guess (unbounded can never be "quality").
            return _result("quantity", min(1.0, best_cos), f"in-domain unbounded cos={best_cos:.3f}", "three_tier_domain")
        # In-domain but bounded, with Stage 2 not confidently "quality": mandatory
        # escalation. Asserting "quality" here by default, or "quantity" via a
        # low-confidence tie-break, were both audited and found less accurate
        # than escalating (see module docstring) — never guess in this branch.
        return _llm_with_feature(f"in_domain_bounded_low_conf cos={best_cos:.3f},prob={quality_prob:.2f}")

    # Stage 4 — borderline/not-in-domain cosine -> LLM (no ML default).
    return _llm_with_feature(f"borderline_cos={best_cos:.3f}")
