"""Dataclasses + category constants shared across notebooks.

The LLM predicts 4 real categories. ``others`` is retained only as the
rule-based fallback/source bucket for rows the rules could not cover.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Category(str, Enum):
    JUNK = "junk"                          # spam + noise merged
    SERVICE_FEEDBACK = "service_feedback"
    CONFIG_FEEDBACK = "config_feedback"
    APP_SPECIFIC = "app_specific"
    OTHERS = "others"


LLM_OUTPUT_CATEGORIES: list[str] = [
    Category.JUNK.value,
    Category.SERVICE_FEEDBACK.value,
    Category.CONFIG_FEEDBACK.value,
    Category.APP_SPECIFIC.value,
]

ALL_CATEGORIES: list[str] = LLM_OUTPUT_CATEGORIES + [Category.OTHERS.value]

# Categories actually scored by F1 / precision / recall. This is identical to
# the LLM output schema because `others` is not a semantic class.
SCORED_CATEGORIES: list[str] = LLM_OUTPUT_CATEGORIES


# Mapping from rule labels stored in Mongo to the data labels.
# The runtime rule engine writes `junk` directly (spam/noise are merged before
# persistence); the legacy spam/noise keys are kept for any pre-merge rows.
# `others` stays as a source bucket so those rows can be handed to the LLM.
RULE_TO_5CAT: dict[str, str] = {
    "junk": Category.JUNK.value,
    "service_feedback": Category.SERVICE_FEEDBACK.value,
    "config_feedback": Category.CONFIG_FEEDBACK.value,
    "app_specific": Category.APP_SPECIFIC.value,
    "others": Category.OTHERS.value,
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
