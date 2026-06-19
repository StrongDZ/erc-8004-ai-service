"""Dataclasses + category constants shared across notebooks.

Two-axis taxonomy (matches the Go classifier + prompts SYSTEM_V6):
  category ∈ {junk, quality, quantity}   (+ ``others`` as the rule fallback bucket)
  feature  ∈ {infrastructure, agent_domain, both}   (analysis axis, not scored)

The LLM predicts 3 real categories via a junk→quantity→quality cascade. ``others``
is retained only as the rule-based fallback/source bucket for rows the rules could
not cover. Only ``quality`` feeds the trust score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Category(str, Enum):
    JUNK = "junk"          # meaningless / spam / noise / placeholder
    QUALITY = "quality"    # subjective sentiment + domain service/trust evaluations (scored)
    QUANTITY = "quantity"  # measured outcome/metric; unbounded excludes quality but not auto-quantity
    OTHERS = "others"      # rule fallback / escalation bucket — NOT a semantic class


class Feature(str, Enum):
    INFRASTRUCTURE = "infrastructure"  # generic signal applicable to ANY agent
    AGENT_DOMAIN = "agent_domain"      # specific to this agent's business
    BOTH = "both"                      # generic metric on a domain service


LLM_OUTPUT_CATEGORIES: list[str] = [
    Category.JUNK.value,
    Category.QUALITY.value,
    Category.QUANTITY.value,
]

ALL_CATEGORIES: list[str] = LLM_OUTPUT_CATEGORIES + [Category.OTHERS.value]

# Categories actually scored by F1 / precision / recall. Identical to the LLM
# output schema because `others` is not a semantic class.
SCORED_CATEGORIES: list[str] = LLM_OUTPUT_CATEGORIES

FEATURES: list[str] = [Feature.INFRASTRUCTURE.value, Feature.AGENT_DOMAIN.value, Feature.BOTH.value]


# Mapping from rule labels stored in Mongo to the data labels. The runtime rule
# engine writes junk/quality/quantity/others directly; legacy 5-class keys are
# folded in for any pre-migration rows (service→quality, config→quality,
# app_specific→quantity, spam/noise→junk). `others` stays as an LLM source bucket.
RULE_TO_CAT: dict[str, str] = {
    "junk": Category.JUNK.value,
    "quality": Category.QUALITY.value,
    "quantity": Category.QUANTITY.value,
    "others": Category.OTHERS.value,
    # legacy 5-class fallbacks
    "service_feedback": Category.QUALITY.value,
    "config_feedback": Category.QUALITY.value,
    "app_specific": Category.QUANTITY.value,
    "spam": Category.JUNK.value,
    "noise": Category.JUNK.value,
}

# Backwards-compatible alias (old name used across notebooks/benchmarks).
RULE_TO_5CAT: dict[str, str] = RULE_TO_CAT

# Mongo `classification.rule.category` values to query per canonical label.
# Runtime writes junk/quality/quantity/others; legacy rows may still use the
# 5-class keys — always query with $in aliases so corpus sampling stays balanced.
MONGO_CATEGORY_ALIASES: dict[str, list[str]] = {
    Category.JUNK.value: ["junk", "spam", "noise"],
    Category.QUALITY.value: ["quality", "service_feedback", "config_feedback"],
    Category.QUANTITY.value: ["quantity", "app_specific"],
    Category.OTHERS.value: ["others"],
}


@dataclass
class FeedbackRecord:
    """One row from feedback_history. Mirrors the Mongo schema, only fields used downstream."""
    id: str
    agent_id: str
    chain_id: int
    tag1: str
    tag2: str
    endpoint: str
    value: str
    value_decimals: int
    value_scale: str
    feedback_parsed: dict | None
    rule_category: str  # data label after RULE_TO_5CAT mapping
    is_self_feedback: bool = False


@dataclass
class AgentMeta:
    """Subset of an agent document used for prompt context."""
    chain_id: int
    agent_id: str
    name: str = ""
    description: str = ""
    summary: str = ""              # filled by 02_agent_summary notebook
    services: list[dict] = field(default_factory=list)
    oasf_domains: list[str] = field(default_factory=list)
    oasf_skills: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """One classification result regardless of which approach produced it."""
    category: str
    confidence: float
    reason: str = ""
    source: str = "llm"            # "llm" | "embedding" | "fallback"
    latency_ms: int = 0
    raw_output: str = ""           # for debugging
    feature: str | None = None
