"""Unit tests for the production 3-tier classifier (shared/three_tier.py).

Rule + SVM stages use the real serialized model; Stage 3 (agent-domain cosine)
is exercised with a fake encoder that returns controlled vectors so the
in-domain / not-in-domain / borderline branches are deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.three_tier import classify_three_tier


class FakeEncoder:
    """Returns a fixed cosine between every tag and the agent vector.

    texts[0] is the agent text; texts[1:] are tags. We return unit vectors so
    that dot(tag, agent) == cos_target for the tags and 1.0 for the agent.
    """

    def __init__(self, cos_target: float):
        self.cos = cos_target

    def encode(self, texts, normalize_embeddings=True):
        agent = np.array([1.0, 0.0], dtype="float32")
        out = [agent]
        # tag vector with dot(tag, agent) == cos_target
        x = self.cos
        y = float(np.sqrt(max(0.0, 1.0 - x * x)))
        for _ in texts[1:]:
            out.append(np.array([x, y], dtype="float32"))
        return out


def test_quality_tag_svm_vote():
    # "reliable"/"fast" are confidently quality in the per-tag SVM.
    r = classify_three_tier(encoder=FakeEncoder(0.1), tag1="reliable", tag2="fast",
                            scale="star5", value_norm=5.0, agent_text="x")
    assert r.category == "quality" and r.source == "three_tier_svm"


def test_stage3_not_in_domain_is_junk():
    # A non-quality-SVM tag with low agent cosine -> junk.
    r = classify_three_tier(encoder=FakeEncoder(0.10), tag1="uptime", tag2="",
                            scale="pct100", value_norm=99.0, agent_text="defi trading agent")
    assert r.category == "junk" and r.source == "three_tier_domain"


def test_stage3_in_domain_bounded_is_quality():
    # Non-quality-SVM tag but high agent cosine on a bounded scale -> quality.
    r = classify_three_tier(encoder=FakeEncoder(0.80), tag1="uptime", tag2="",
                            scale="pct100", value_norm=99.0, agent_text="defi trading agent")
    assert r.category == "quality" and r.source == "three_tier_domain"


def test_stage3_borderline_ml_default():
    # Borderline cosine (0.35-0.55) -> deterministic ML default.
    r = classify_three_tier(encoder=FakeEncoder(0.45), tag1="uptime", tag2="",
                            scale="pct100", value_norm=99.0, agent_text="defi trading agent")
    assert r.source == "three_tier_ml_default"


def test_no_agent_metadata_scale_heuristic():
    r = classify_three_tier(encoder=FakeEncoder(0.9), tag1="uptime", tag2="",
                            scale="star5", value_norm=5.0, agent_text="")
    assert r.category == "quality" and r.source == "three_tier_heuristic"
