"""Evaluation metrics consistent across approaches.

Produces three things every approach should report:
  - macro-F1 + per-category P/R/F1 (sklearn classification_report)
  - confusion matrix (pandas DataFrame for nice notebook display)
  - latency stats (mean / p50 / p95 / p99)

All metrics computed over the SAME test split so cross-approach comparison is fair.

**`others` is excluded from F1 scoring.** Rule-based "others" means "rule didn't
know" and handed the row to the LLM — not a true semantic class. When the LLM reclassifies those rows into
junk/service/config/app, that's the goal, not an error. Concretely:

  - rows where gold == "others" are dropped before computing F1 / P / R
  - the label set passed to sklearn is `SCORED_CATEGORIES` (4 categories)

The confusion matrix keeps the rule `others` row so the gold-others → pred-X
migration is still visible for diagnostics. A pred-others column only indicates
client fallback/invalid output, not an allowed LLM category.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .types import ALL_CATEGORIES, SCORED_CATEGORIES


def _drop_others_gold(y_true: list[str], y_pred: list[str]) -> tuple[list[str], list[str]]:
    """Drop rows where gold == 'others' so they don't penalize any class.

    See module docstring for rationale.
    """
    yt, yp = [], []
    for t, p in zip(y_true, y_pred, strict=True):
        if t == "others":
            continue
        yt.append(t)
        yp.append(p)
    return yt, yp


def per_class_report(y_true: list[str], y_pred: list[str]) -> pd.DataFrame:
    """Per-class precision/recall/F1 + support + macro/weighted averages.

    Computed over `SCORED_CATEGORIES` only (excludes `others`).
    """
    yt, yp = _drop_others_gold(y_true, y_pred)
    report = classification_report(
        yt, yp,
        labels=SCORED_CATEGORIES, output_dict=True, zero_division=0,
    )
    rows = []
    for cat in SCORED_CATEGORIES + ["macro avg", "weighted avg"]:
        if cat not in report:
            continue
        r = report[cat]
        rows.append({
            "category": cat,
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1-score"], 4),
            "support": int(r["support"]),
        })
    # sklearn drops the top-level 'accuracy' key when `labels` doesn't cover every
    # value present in y_true/y_pred (fallback predictions may be 'others').
    # Compute it manually over the filtered rows so the report still shows it.
    correct = sum(1 for t, p in zip(yt, yp, strict=True) if t == p)
    acc = correct / len(yt) if yt else 0.0
    rows.append({
        "category": "accuracy",
        "precision": np.nan, "recall": np.nan,
        "f1": round(acc, 4),
        "support": len(yt),
    })
    return pd.DataFrame(rows)


def confusion_df(y_true: list[str], y_pred: list[str]) -> pd.DataFrame:
    """Confusion matrix over 4 LLM categories + rule/fallback `others`.

    Use to see *how* the model reclassified rule-others rows.
    """
    cm = confusion_matrix(y_true, y_pred, labels=ALL_CATEGORIES)
    return pd.DataFrame(cm, index=[f"true_{c}" for c in ALL_CATEGORIES],
                        columns=[f"pred_{c}" for c in ALL_CATEGORIES])


def latency_stats(latencies_ms: list[int]) -> dict[str, float]:
    if not latencies_ms:
        return {"n": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0}
    arr = np.array(latencies_ms)
    return {
        "n": len(arr),
        "mean": float(arr.mean().round(1)),
        "p50": float(np.percentile(arr, 50).round(1)),
        "p95": float(np.percentile(arr, 95).round(1)),
        "p99": float(np.percentile(arr, 99).round(1)),
    }


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    """Macro-F1 over `SCORED_CATEGORIES` (excludes `others`)."""
    yt, yp = _drop_others_gold(y_true, y_pred)
    return float(f1_score(yt, yp, labels=SCORED_CATEGORIES,
                          average="macro", zero_division=0))


def summarize_run(
    approach: str,
    y_true: list[str],
    y_pred: list[str],
    latencies_ms: list[int],
) -> dict:
    """One-row summary suitable for the comparison table in 06_evaluation.

    `n_scored` counts only rows used in F1 (excludes gold-others); `n_total`
    keeps the raw test-set size for latency context.
    """
    yt, _ = _drop_others_gold(y_true, y_pred)
    stats = latency_stats(latencies_ms)
    return {
        "approach": approach,
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "n_scored": len(yt),
        "n_total": len(y_true),
        "mean_ms": stats["mean"],
        "p50_ms": stats["p50"],
        "p95_ms": stats["p95"],
        "p99_ms": stats["p99"],
    }
