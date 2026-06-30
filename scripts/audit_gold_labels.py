#!/usr/bin/env python3
"""Audit gold label consistency for the pure-others labelled set.

Surfaces rows whose human_label contradicts (a) a hard structural invariant,
(b) the production rule cascade, or (c) the V8 prompt's quality/quantity
boundary. The script NEVER overwrites the gold: by default it only writes a
review CSV of (current_label -> suggested_label) with the rule that fired and a
one-line reason. Pass --apply to additionally write a *corrected copy* to a new
file (the original is left untouched).

Rules, in precedence order (first match wins per row):

  R0  unbounded_quality      HARD INVARIANT. quality is NEVER valid on an
                             `unbounded` scale (no normalised ceiling exists).
                             -> suggest quantity (or junk if tags look like noise).
  R1  rule_engine_disagree   The production rule cascade (rule_classify, the port
                             of classifier.go) returns a definite label that
                             differs from human_label. Strong signal: either a
                             gold mislabel or Go/Python rule drift.
  R2  metric_as_quality      A dashboard-metric tag (accuracy, speed,
                             credit_score, *-rate, *-coefficient, ...) on a
                             MEASURED scale (pct100/binary) is labelled quality,
                             with no quality-adjective co-tag. The V8 prompt's
                             Layer 2 ("metric outranks domain") says quantity.
                             Star scales are excluded: on star5/star10 a
                             metric-named tag is a subjective N-star rating, not
                             a measured statistic. A metric paired with a praise
                             word (accuracy|accurate, accuracy|top-notch) is a
                             subjective rating too -> stays quality.
  R4  quality_concept_as_qty A BOUNDED record whose only semantic content is a
                             quality concept (reliability, transparency, ...) with
                             NO metric and NO count tag is labelled quantity.
                             -> suggest quality.
  R5  junk_review            A spam/promo phrase survived into a non-junk label.
                             Flagged for human review only (no auto-suggestion).

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m scripts.audit_gold_labels \\
        --gold data/labelled/pure_others_to_label.csv

    # also write a corrected copy (new file, original untouched):
    .venv/bin/python3 -m scripts.audit_gold_labels \\
        --gold data/labelled/pure_others_to_label.csv --apply
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the production rule cascade + its canonical keyword sets so the audit
# never drifts from what the live classifier actually does.
from benchmarks.pipeline_3tier import (
    _QUALITY_T1,
    _SPAM_RANK,
    _SPAM_URL,
    rule_classify,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD = ROOT / "data/labelled/pure_others_to_label.csv"

BOUNDED_SCALES = {"pct100", "binary", "star5", "star10"}
# Scales on which a metric-named tag reads as a measured statistic rather than a
# subjective N-star rating. R2 (metric -> quantity) only fires on these.
MEASURED_SCALES = {"pct100", "binary"}

# Dashboard-metric tokens that the V8 prompt's Layer 2 treats as quantity but
# that are NOT already in the production _QUANTITY_T1/T2 sets. These are the
# boundary cases the gold actually mislabels as quality.
_METRIC_EXTRA_EXACT = {
    "accuracy", "speed", "latency", "throughput", "freshness", "volume",
    "frequency", "yield", "pnl", "p&l", "downtime", "runtime", "tps", "qps",
    "credit_score", "credit-score", "creditscore", "risk-score", "risk_score",
    "viral-coefficient", "readability-score", "viral-score", "stitch-rate",
}
# Suffixes that mark a metric. A token ending in one of these is a metric UNLESS
# it is a known quality keyword (e.g. trust-score, reputation) — those live in
# the production _QUALITY_T1 set and are excluded below.
_METRIC_SUFFIX = ("-rate", "-ratio", "-count", "-coefficient", "-yield",
                  "-score", "-index", "-percentile", "-throughput", "-freq")

# Pure quality concepts (no metric, no count) used by R2's praise-guard and
# R4's reverse check.
_QUALITY_CONCEPT = {
    "reliability", "reliable", "trust", "trustworthy", "trustworthiness",
    "transparency", "transparent", "fairness", "fair", "helpful", "helpfulness",
    "friendly", "professional", "professionalism", "excellent", "satisfaction",
    "responsive", "responsiveness", "consistency", "consistent", "correctness",
    "integrity", "honesty", "quality", "supportive", "knowledgeable",
    "courteous", "accurate", "peer-review",
}

# Praise / sentiment phrases that mark a SUBJECTIVE rating even when a metric tag
# is also present (e.g. accuracy|top-notch). Used by R2's praise-guard.
_PRAISE = {
    "top notch", "top noth", "top-notch", "spot on", "spot-on", "awesome",
    "amazing", "excellent", "outstanding", "great", "best", "perfect",
    "flawless", "superb", "impressive", "solid",
}

# Marketing / promo nouns that the rule spam regex does not catch.
_PROMO = re.compile(r"(?i)(promo|airdrop|giveaway|shill|moon\b|pump\b)")


def _norm(s) -> str:
    return str(s or "").strip().lower()


def _sep_norm(tok: str) -> str:
    """Underscores and spaces -> hyphens, so 'success_rate', 'success rate' and
    'success-rate' all match the metric suffix rules alike."""
    return tok.replace("_", "-").replace(" ", "-")


def _is_number(s) -> bool:
    return bool(re.fullmatch(r"-?[0-9]+(\.[0-9]+)?", str(s).strip()))


def _is_metric_tok(tok: str) -> bool:
    if not tok:
        return False
    hy = _sep_norm(tok)
    if tok in _QUALITY_T1 or hy in _QUALITY_T1:   # trust-score / reputation stay quality
        return False
    if tok in _METRIC_EXTRA_EXACT or hy in _METRIC_EXTRA_EXACT:
        return True
    return hy.endswith(_METRIC_SUFFIX)


def _is_quality_word(tok: str) -> bool:
    """A pure quality concept or praise phrase. Overrides a co-occurring metric:
    'accuracy|accurate' / 'accuracy|top-notch' read as subjective ratings."""
    return tok in _QUALITY_CONCEPT or tok in _PRAISE


def _looks_noise(t1: str, t2: str) -> bool:
    """Tags with no constructible meaning (used to pick junk over quantity in R0)."""
    blob = f"{t1} {t2}".strip()
    if not blob:
        return True
    if _SPAM_URL.search(blob) or _SPAM_RANK.search(blob):
        return True
    # all-consonant gibberish on either tag
    for t in (t1, t2):
        if t and re.fullmatch(r"[bcdfghjklmnpqrstvwxz]{4,}", t):
            return True
    return False


def audit_row(row: pd.Series) -> tuple[str, str, str] | None:
    """Return (rule_id, suggested_label, reason) or None if the row looks fine."""
    t1, t2 = _norm(row.get("tag1")), _norm(row.get("tag2"))
    scale = _norm(row.get("scale"))
    cur = _norm(row.get("human_label"))
    bounded = scale in BOUNDED_SCALES

    # R0 — hard invariant: quality is never unbounded.
    if cur == "quality" and scale == "unbounded":
        if _looks_noise(t1, t2):
            return ("R0_unbounded_quality", "junk",
                    "quality is impossible on unbounded scale; tags look like noise -> junk")
        return ("R0_unbounded_quality", "quantity",
                "quality is impossible on unbounded scale -> quantity")

    # R1 — production rule cascade returns a different definite label.
    rule_pred = rule_classify(row)
    if rule_pred is not None and rule_pred != cur:
        return ("R1_rule_engine_disagree", rule_pred,
                f"production rule cascade classifies this as {rule_pred}")

    # R2 — a dashboard metric on a MEASURED scale (pct100/binary), labelled
    # quality, with NO quality-adjective co-tag. Excludes star5/star10 (there a
    # metric-named tag is a subjective N-star rating) and metric+praise pairs
    # (accuracy|accurate, accuracy|top-notch are subjective ratings).
    if cur == "quality" and scale in MEASURED_SCALES:
        metric_present = _is_metric_tok(t1) or _is_metric_tok(t2)
        quality_word = _is_quality_word(t1) or _is_quality_word(t2)
        if metric_present and not quality_word:
            which = t1 if _is_metric_tok(t1) else t2
            return ("R2_metric_as_quality", "quantity",
                    f"'{which}' is a dashboard metric on a measured scale; "
                    "V8 Layer 2 (metric outranks domain) -> quantity")

    # R4 — bounded record, only quality-concept content, labelled quantity.
    if cur == "quantity" and bounded:
        has_metric = _is_metric_tok(t1) or _is_metric_tok(t2)
        has_number = _is_number(t1) or _is_number(t2)
        has_concept = _is_quality_word(t1) or _is_quality_word(t2)
        if has_concept and not has_metric and not has_number:
            return ("R4_quality_concept_as_qty", "quality",
                    "only semantic content is a quality concept (no metric/count) -> quality")

    # R5 — promo/spam phrase in a non-junk label (review only).
    if cur != "junk":
        blob = f"{row.get('tag1','')} {row.get('tag2','')} {row.get('offchain_note','')}"
        if _SPAM_URL.search(blob) or _SPAM_RANK.search(blob) or _PROMO.search(blob):
            return ("R5_junk_review", "REVIEW",
                    "promo/spam phrase survived into a non-junk label; review for junk")

    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--apply", action="store_true",
                    help="also write a corrected copy to a NEW file (original untouched)")
    args = ap.parse_args()

    df = pd.read_csv(args.gold).fillna("")
    if "human_label" not in df.columns:
        sys.exit(f"ERROR: {args.gold} has no 'human_label' column")
    n = len(df)
    print(f"Loaded {n} rows from {args.gold.name}")
    print(f"Label distribution: {dict(df['human_label'].value_counts())}\n")

    # Audit every row.
    suggestions = []
    for idx, row in df.iterrows():
        res = audit_row(row)
        if res is None:
            continue
        rule_id, suggested, reason = res
        suggestions.append({
            "idx": idx,
            "feedback_id": row.get("feedback_id", ""),
            "rule": rule_id,
            "current_label": _norm(row.get("human_label")),
            "suggested_label": suggested,
            "tag1": row.get("tag1", ""),
            "tag2": row.get("tag2", ""),
            "scale": _norm(row.get("scale")),
            "reason": reason,
        })

    sug = pd.DataFrame(suggestions)
    if sug.empty:
        print("No inconsistencies found. Gold looks clean against all rules.")
        return

    # Per-rule summary (row-level impact).
    print(f"=== Flagged {len(sug)} / {n} rows ({len(sug)/n*100:.1f}%) ===\n")
    print(f"{'rule':28} {'current -> suggested':24} rows")
    print("-" * 64)
    for rule_id in sorted(sug["rule"].unique()):
        sub = sug[sug["rule"] == rule_id]
        for (cur, sug_label), grp in sub.groupby(["current_label", "suggested_label"]):
            arrow = f"{cur} -> {sug_label}"
            print(f"{rule_id:28} {arrow:24} {len(grp)}")
    print()

    # Collapse to unique (tag1, tag2, scale, current, suggested) for a readable
    # review list, with how many rows each combo affects.
    report = (sug.groupby(["rule", "current_label", "suggested_label",
                           "tag1", "tag2", "scale", "reason"], as_index=False)
                 .agg(n_rows=("idx", "count"),
                      example_feedback_id=("feedback_id", "first")))
    report = report.sort_values(["rule", "n_rows"], ascending=[True, False])

    out = args.out or (ROOT / "data/labelled" /
                       f"gold_audit_{datetime.now():%Y%m%d_%H%M%S}.csv")
    report.to_csv(out, index=False)
    print(f"Review CSV ({len(report)} unique combos) -> {out}")

    # Top combos by impact, for a quick look in the console.
    print("\n=== Top 15 combos by affected rows ===")
    cols = ["rule", "current_label", "suggested_label", "tag1", "tag2", "scale", "n_rows"]
    print(report.sort_values("n_rows", ascending=False)[cols].head(15).to_string(index=False))

    if args.apply:
        corrected = df.copy()
        applied = 0
        for s in suggestions:
            if s["suggested_label"] == "REVIEW":
                continue  # R5 is review-only, never auto-applied
            corrected.at[s["idx"], "human_label"] = s["suggested_label"]
            applied += 1
        cp = args.gold.with_name(args.gold.stem + "_audited.csv")
        corrected.to_csv(cp, index=False)
        print(f"\n--apply: wrote corrected copy with {applied} relabels -> {cp}")
        print(f"         (original {args.gold.name} left untouched; "
              f"{len(sug) - applied} R5 review rows NOT auto-applied)")


if __name__ == "__main__":
    main()
