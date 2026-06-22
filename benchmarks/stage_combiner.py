#!/usr/bin/env python3
"""Unified Stage 2+3 combiner: replaces the hard-gated per-tag-SVM-vote ->
FAISS-in-domain-scale-override cascade with continuous signals fed into one
multinomial logistic regression.

Why: the original cascade makes each stage's decision final and throws away
sub-threshold evidence (an SVM quality-prob of 0.65, just under the 0.70 vote
threshold, carries no information forward to Stage 3). Stage 3 then treats
`value_scale` as an absolute override (bounded -> quality, unbounded ->
quantity) regardless of any metric-keyword evidence in the tag text. A lexical
whitelist patch on Stage 3 would only fix the exact words seen in one audit and
re-break on the next agent's vocabulary.

This module instead:
  1. Keeps svm_p1, svm_p2 (quality-vs-quantity SVM probs per tag) continuous.
     SVM is trained on quality/quantity only (junk excluded), so junk inputs
     produce low-confidence predictions and fall through to LLM.
  2. Treats domain-cosine, scale, and value_decimals as ordinary features.
  3. Lets a logistic regression (trained on a held-out silver set, never the
     gold eval set) learn the combination weights instead of hand-tuned cutoffs.
  4. Combiner output classes: quality and quantity only. Junk is handled by
     LLM fallback when max_proba < threshold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.stage3_domain import _encode, _load_index  # reuse FAISS index + bge encoder

# svm_p1 = P(quality | tag1), svm_p2 = P(quality | tag2) from quality-vs-quantity SVM.
# High value → quality, low value → quantity. Junk items score near 0.5 (OOD) → LLM.
FEATURE_NAMES = ["svm_p1", "svm_p2", "cos_domain", "has_domain", "is_unbounded", "value_decimals"]


def cos_domain(tags: list[str], agent_key: str) -> tuple[float, bool]:
    """Best cosine to the agent's domain vector. Returns (cos, has_domain)."""
    index, key_to_pos = _load_index()
    pos = key_to_pos.get(agent_key)
    if pos is None or not tags:
        return 0.0, False
    agent_vec = index.reconstruct(pos)
    sims = [float(np.dot(_encode(t), agent_vec)) for t in tags]
    return max(sims), True


def build_feature_row(
    p1: float,
    p2: float,
    cosd: float,
    has_dom: bool,
    scale: str,
    value_decimals: int = 0,
) -> list[float]:
    is_unbounded = 1.0 if (scale or "").strip().lower() == "unbounded" else 0.0
    return [p1, p2, cosd, float(has_dom), is_unbounded, float(value_decimals)]
