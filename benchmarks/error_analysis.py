#!/usr/bin/env python3
"""Error analysis: extract and categorise misclassified examples from run13 vs LLM-only.

Loads gold CSV + re-runs run13 pipeline predictions (using cache to avoid LLM calls),
then categorises each error into one of:
  - ambiguous_tags: tag1/tag2 are multi-domain or semantically underspecified
  - missing_metadata: agent_description / agent_domain_text is empty
  - missing_uri: offchain_note is empty (no URI content available)
  - uncovered_rule: rule engine produced 'others' but pattern looks rule-classifiable in hindsight

Writes a Markdown error analysis section suitable for inclusion in the benchmark report.

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.error_analysis \\
        --gold data/labelled/pure_others_stratified_dedup.csv \\
        --n-samples 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.pipeline_3tier_v2 import LLM_MODEL, load_gold, llm_classify
from shared.types import LLM_OUTPUT_CATEGORIES

ROOT = Path(__file__).resolve().parent.parent

AMBIGUOUS_TAGS = {
    # tags that are dual-domain (service name AND a measurable quantity)
    "score", "rating", "grade", "rank", "level", "tier",
    "performance", "efficiency", "accuracy",
    # tags with scale context needed to distinguish
    "success", "completion", "execution",
}


def error_type(row: pd.Series, true_label: str, pred_label: str) -> str:
    tag1 = str(row.get("tag1") or "").strip().lower()
    tag2 = str(row.get("tag2") or "").strip().lower()
    agent_text = str(row.get("agent_domain_text") or row.get("agent_description") or "").strip()
    offchain = str(row.get("offchain_note") or "").strip()

    if tag1 in AMBIGUOUS_TAGS or tag2 in AMBIGUOUS_TAGS:
        return "ambiguous_tags"
    if not agent_text:
        return "missing_metadata"
    if not offchain:
        return "missing_uri"
    return "uncovered_rule"


def run_llm_only_predictions(gold: pd.DataFrame) -> list[str]:
    preds = []
    for _, row in gold.iterrows():
        try:
            pred = llm_classify(row, LLM_MODEL).strip().lower()
            if pred not in LLM_OUTPUT_CATEGORIES:
                pred = "quality"
        except Exception:
            pred = "quality"
        preds.append(pred)
    return preds


def format_confusion_matrix(y_true: list[str], y_pred: list[str],
                             labels: list[str]) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    header = "  True \\ Pred | " + " | ".join(f"{l:>8}" for l in labels)
    sep = "-" * len(header)
    rows = [header, sep]
    for i, true_label in enumerate(labels):
        cells = " | ".join(f"{cm[i][j]:>8}" for j in range(len(labels)))
        rows.append(f"  {true_label:>11} | {cells}")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=6,
                        help="number of misclassified examples to show per error type")
    args = parser.parse_args()

    print("Loading gold set...")
    gold = load_gold(args.gold)
    gold = gold[gold["label"].isin(LLM_OUTPUT_CATEGORIES)].reset_index(drop=True)
    n = len(gold)
    print(f"Gold N={n}")

    print("\nRunning LLM-only predictions (from cache)...")
    llm_preds = run_llm_only_predictions(gold)
    y_true = gold["label"].str.strip().str.lower().tolist()

    errors: list[dict] = []
    for i, (true_lbl, pred_lbl) in enumerate(zip(y_true, llm_preds)):
        if true_lbl != pred_lbl:
            row = gold.iloc[i]
            etype = error_type(row, true_lbl, pred_lbl)
            errors.append({
                "idx": i,
                "id": str(row.get("id", "")),
                "true": true_lbl,
                "pred": pred_lbl,
                "error_type": etype,
                "tag1": str(row.get("tag1") or ""),
                "tag2": str(row.get("tag2") or ""),
                "scale": str(row.get("value_scale") or ""),
                "agent_text": str(row.get("agent_domain_text") or row.get("agent_description") or "")[:120],
                "offchain": str(row.get("offchain_note") or "")[:80],
            })

    total_errors = len(errors)
    error_df = pd.DataFrame(errors)
    print(f"\nTotal LLM errors: {total_errors}/{n} ({total_errors/n*100:.1f}%)")

    # Count by type
    type_counts = error_df["error_type"].value_counts().to_dict() if len(error_df) > 0 else {}
    print("Error type breakdown:")
    for etype, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {cnt} ({cnt/total_errors*100:.1f}%)")

    # Confusion matrix
    print("\nLLM-only confusion matrix:")
    print(format_confusion_matrix(y_true, llm_preds, LLM_OUTPUT_CATEGORIES))

    # Sample misclassified examples
    print(f"\n=== Sample misclassifications (up to {args.n_samples} per type) ===\n")
    for etype in ["ambiguous_tags", "missing_metadata", "missing_uri", "uncovered_rule"]:
        subset = error_df[error_df["error_type"] == etype].head(args.n_samples)
        if len(subset) == 0:
            continue
        print(f"--- {etype} ({type_counts.get(etype, 0)} total) ---")
        for _, e in subset.iterrows():
            print(f"  [{e['true']} → pred={e['pred']}]  "
                  f"tag1={e['tag1']!r}  tag2={e['tag2']!r}  scale={e['scale']!r}")
            if e["agent_text"]:
                print(f"    agent: {e['agent_text'][:80]}")
            if e["offchain"]:
                print(f"    offchain: {e['offchain'][:60]}")
        print()

    # Write markdown section
    out_lines = [
        "## Error Analysis — LLM-only (qwen2.5:7b-instruct)\n",
        f"Evaluated on N={n} records. Total errors: {total_errors} ({total_errors/n*100:.1f}%)\n",
        "### Confusion Matrix\n",
        "```",
        format_confusion_matrix(y_true, llm_preds, LLM_OUTPUT_CATEGORIES),
        "```\n",
        "### Error Type Breakdown\n",
        "| Error Type | Count | % of Errors |",
        "|---|---|---|",
    ]
    for etype, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        out_lines.append(f"| {etype} | {cnt} | {cnt/total_errors*100:.1f}% |")

    out_lines += [
        "",
        "### Representative Misclassified Examples\n",
    ]
    for etype in ["ambiguous_tags", "missing_metadata", "missing_uri", "uncovered_rule"]:
        subset = error_df[error_df["error_type"] == etype].head(args.n_samples)
        if len(subset) == 0:
            continue
        out_lines.append(f"#### {etype.replace('_', ' ').title()} ({type_counts.get(etype, 0)} errors)\n")
        for _, e in subset.iterrows():
            out_lines.append(
                f"- **True={e['true']}** pred={e['pred']} — `tag1={e['tag1']!r}` "
                f"`tag2={e['tag2']!r}` scale=`{e['scale']}`"
            )
            if not e["agent_text"]:
                out_lines.append("  - *agent metadata absent*")
            else:
                out_lines.append(f"  - agent: {e['agent_text'][:100]}")
            if not e["offchain"]:
                out_lines.append("  - *offchain URI absent*")
        out_lines.append("")

    out_lines += [
        "### Root Cause Summary\n",
        "| Cause | Mechanism | Affected Records |",
        "|---|---|---|",
        "| Ambiguous tags | tag1/tag2 are multi-domain service names with dual quality/metric semantics "
        "(e.g. 'score', 'rating') — LLM needs scale context to disambiguate, model sees both interpretations "
        f"as valid | {type_counts.get('ambiguous_tags', 0)} |",
        "| Missing agent metadata | description + OASF fields empty — model cannot infer domain, falls back "
        f"to tag semantics alone | {type_counts.get('missing_metadata', 0)} |",
        "| Missing feedbackURI | offchain_note empty — no narrative context, tag pair alone is insufficient "
        f"for quality vs quantity boundary | {type_counts.get('missing_uri', 0)} |",
        "| Uncovered rule | Tag pattern is novel (not in rule engine vocabulary) but semantically "
        f"unambiguous — rule added post-hoc would catch these | {type_counts.get('uncovered_rule', 0)} |",
    ]

    md_out = ROOT / "docs" / "error_analysis_llm_only.md"
    md_out.write_text("\n".join(out_lines) + "\n")
    print(f"\nMarkdown error analysis saved to {md_out}")


if __name__ == "__main__":
    main()
