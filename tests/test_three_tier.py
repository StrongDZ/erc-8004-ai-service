"""Unit tests for the production 3-tier classifier (shared/three_tier.py).

Uses the real bge-small encoder and the real serialized BGE quality-gate SVM
(both are loaded once per stage, not mocked) so that Stage 2's quality
probability and Stage 3's agent-domain cosine are genuine model outputs, not
fabricated values that could drift from what production actually computes.
Tag/agent_text pairs below were chosen and verified empirically to land in
the intended bucket relative to SVM_QUALITY_THRESH (0.80) and
THRESH_IN_DOMAIN (0.55); if the underlying model or thresholds change, these
fixtures may need re-verifying.

Mandatory-escalation design: Stage 2 may only ever assert "quality"; Stage 3's
bounded+in-domain branch never defaults to a label when Stage 2 is not
confident — it always calls the LLM fallback instead. Tests therefore always
supply an llm_classify_fn for any case Stage 2/3 cannot resolve on its own.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.three_tier import classify_three_tier


@lru_cache(maxsize=1)
def _real_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


def _llm_stub(category: str, confidence: float = 0.6):
    return lambda: (category, confidence, "stub llm reason", "agent_domain")


def test_quality_tag_svm_gate():
    # "professional"/"trustworthy" score well above SVM_QUALITY_THRESH (0.80).
    r = classify_three_tier(encoder=_real_encoder(), tag1="professional", tag2="trustworthy",
                            scale="star5", value_norm=5.0, agent_text="x")
    assert r.category == "quality" and r.source == "three_tier_svm"


def test_stage3_not_in_domain_escalates_to_llm():
    # Low SVM quality_prob (0.04) and cosine below THRESH_IN_DOMAIN (0.51 vs 0.55)
    # against an unrelated agent -> no confident signal anywhere -> LLM.
    agent_text = "A protocol providing MEV protection and security audits for DeFi traders."
    r = classify_three_tier(encoder=_real_encoder(), tag1="weather forecast accuracy", tag2="",
                            scale="pct100", value_norm=99.0, agent_text=agent_text,
                            llm_classify_fn=_llm_stub("junk"))
    assert r.category == "junk" and r.source == "llm"
    assert "borderline_cos" in r.reason


def test_stage3_in_domain_bounded_never_defaults_always_escalates():
    # In-domain (cos ~0.78) but SVM quality_prob (~0.79) just under threshold:
    # mandatory escalation means this is NEVER resolved by a blind default,
    # even though it's in-domain — it must go to the LLM.
    agent_text = "A protocol providing MEV protection and security audits for DeFi traders."
    r = classify_three_tier(encoder=_real_encoder(), tag1="MEV Protection", tag2="Security Audit",
                            scale="pct100", value_norm=90.0, agent_text=agent_text,
                            llm_classify_fn=_llm_stub("quality"))
    assert r.category == "quality" and r.source == "llm"
    assert "in_domain_bounded_low_conf" in r.reason


def test_stage3_in_domain_unbounded_is_safe_quantity_no_llm():
    # In-domain (cos ~0.63) + unbounded is structurally never "quality" by
    # convention, so this resolves directly to "quantity" without the LLM.
    agent_text = "A protocol providing MEV protection and security audits for DeFi traders."
    r = classify_three_tier(encoder=_real_encoder(), tag1="trade-volume", tag2="",
                            scale="unbounded", value_norm=0.0, agent_text=agent_text)
    assert r.category == "quantity" and r.source == "three_tier_domain"


def test_no_agent_metadata_escalates_to_llm():
    # Low SVM quality_prob (0.03) and no agent metadata -> straight to LLM,
    # skipping the cosine check entirely.
    r = classify_three_tier(encoder=_real_encoder(), tag1="winRate", tag2="",
                            scale="pct100", value_norm=99.0, agent_text="",
                            llm_classify_fn=_llm_stub("quality"))
    assert r.category == "quality" and r.source == "llm"
    assert "no_agent_metadata" in r.reason


def test_missing_llm_fallback_raises():
    # A record neither Stage 2 nor Stage 3 can resolve with evidence, and no
    # LLM fallback supplied, must raise rather than silently guess.
    agent_text = "A protocol providing MEV protection and security audits for DeFi traders."
    try:
        classify_three_tier(encoder=_real_encoder(), tag1="weather forecast accuracy", tag2="",
                            scale="pct100", value_norm=99.0, agent_text=agent_text)
    except ValueError:
        return
    raise AssertionError("expected ValueError when no llm_classify_fn is supplied")
