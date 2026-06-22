#!/usr/bin/env python3
"""Run 9: Rule + 3-group per-tag SVM (tag__X scale__Y pair__X__Y) + FAISS + LLM.

One change vs Run 5 (pipeline_3tier_v2.py): per-tag SVM uses explicit 3-group
feature tokens instead of the plain 'X Y' bigram format.

Feature design (one row per tag, same expansion as Run 5):
  plain (Run 5 after fix):  'winRate pct100'
    → tokens: winrate, pct100
    → bigram: winrate pct100   (interaction present when only 1 tag in doc)

  3-group (Run 9):           'tag__winrate scale__pct100 pair__winrate__pct100'
    → tokens: tag__winrate, scale__pct100, pair__winrate__pct100
    → bigrams: tag__winrate scale__pct100, scale__pct100 pair__winrate__pct100
    → pair__ token is the explicit interaction UNIGRAM (high IDF if rare combo)

Key hypothesis: because pair__winrate__pct100 is a distinct unigram, unseen
combos have IDF=0 for that token but the tag__ and scale__ signals still exist
independently, giving the model cleaner individual feature weights without the
n-gram coincidence problem.

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.pipeline_run9 \
        --gold data/labelled/pure_others_to_label.csv --exclude-self
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.per_tag_svm import (
    MODEL_3GROUP_PATH,
    load_per_tag_svm,
    predict_quality_prob_3group,
    train_3group,
    vote_per_tag,
)
from benchmarks.pipeline_3tier import rule_classify
from benchmarks.pipeline_3tier_v2 import (
    LLM_MODEL,
    SVM_VOTE_THRESH,
    enrich_gold_with_agent_meta,
    llm_classify,
    load_gold,
)
from benchmarks.stage3_domain import DomainClassifier
from shared.types import LLM_OUTPUT_CATEGORIES

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--exclude-self", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    print("Training 3-group SVM (tag__X scale__Y pair__X__Y)...")
    train_3group(save=True)
    svm_pipe = load_per_tag_svm(model_path=MODEL_3GROUP_PATH)

    print(f"\nLoading gold from {args.gold} ...")
    gold = load_gold(args.gold)
    print(f"  Gold N={len(gold)}")
    gold = enrich_gold_with_agent_meta(gold)
    if args.exclude_self:
        n_self = int(gold["is_self"].sum())
        gold = gold[~gold["is_self"]].reset_index(drop=True)
        print(f"  --exclude-self: dropped {n_self} -> N={len(gold)}")

    dc = DomainClassifier()
    use_llm = not args.skip_llm

    y_true = gold["label"].tolist()
    rich_mask = gold["has_agent_metadata"].tolist()
    poor_mask = [not m for m in rich_mask]

    preds, sources, audit_rows = [], [], []
    llm_count = 0
    t0 = time.time()

    for _, row in gold.iterrows():
        tag1 = str(row.get("tag1", "") or "").strip()
        tag2 = str(row.get("tag2", "") or "").strip()
        scale = str(row.get("value_scale", "") or "").strip()
        decimals = int(row.get("value_decimals", 0) or 0)
        agent_key = str(row.get("agent_key", "") or "")
        true_label = row.get("label")
        has_meta = bool(row.get("has_agent_metadata"))

        def _rec(pred, source, reason=""):
            preds.append(pred); sources.append(source)
            audit_rows.append({
                "id": row.get("id"), "tag1": tag1, "tag2": tag2,
                "value_scale": scale, "agent_key": agent_key,
                "has_agent_metadata": has_meta, "true_label": true_label,
                "pred": pred, "stage": source, "reason": reason,
                "correct": pred == true_label,
            })

        # Stage 0.5: empty-tag rule (unchanged)
        if not tag1 and not tag2:
            lab = "junk" if scale.lower() == "unbounded" else "quality"
            _rec(lab, "empty_tag_rule"); continue

        # Stage 1: rule (unchanged)
        cat = rule_classify(row)
        if cat:
            _rec(cat, "rule"); continue

        # Stage 2: 3-group per-tag SVM — same voting logic as Run 5
        t2_empty = not bool(tag2)
        p1 = predict_quality_prob_3group(svm_pipe, tag1, scale) if tag1 else 0.5
        p2 = predict_quality_prob_3group(svm_pipe, tag2, scale) if tag2 else 0.5
        stage2 = vote_per_tag(p1, p2, t2_empty=t2_empty, thresh=SVM_VOTE_THRESH)

        if stage2 == "quality":
            _rec("quality", "svm_3group", f"p1={p1:.2f},p2={p2:.2f}"); continue
        # non_quality → fall through to Stage 3 (same as Run 5)

        # Stage 3: FAISS domain cosine (unchanged)
        label3, reason3 = dc.classify(tag1, tag2, scale, decimals, agent_key)
        if label3 is not None:
            _rec(label3, "faiss", reason3); continue

        # Stage 4: LLM (unchanged)
        if use_llm:
            llm_cat = llm_classify(row, LLM_MODEL)
            _rec(llm_cat, "llm"); llm_count += 1
        else:
            guess = "quality" if p1 >= 0.50 else "quantity"
            _rec(guess, "ml_default", f"p1={p1:.2f}")

    elapsed = time.time() - t0
    mf1 = f1_score(y_true, preds, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  Run 9: Rule + 3-group SVM + FAISS + LLM")
    print(f"  Macro F1: {mf1:.4f}   N={len(gold)}")
    print(f"{'='*60}")
    print(classification_report(y_true, preds, labels=LLM_OUTPUT_CATEGORIES, zero_division=0))

    stage_counts: dict[str, int] = {}
    for s in sources:
        stage_counts[s] = stage_counts.get(s, 0) + 1
    for s, n in sorted(stage_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {n} ({n/len(sources)*100:.1f}%)")
    print(f"  LLM calls: {llm_count} ({llm_count/len(gold)*100:.1f}%)  "
          f"avg latency: {elapsed/max(llm_count,1)*1000:.0f}ms")

    def _sub_f1(mask, name):
        st = [y for y, m in zip(y_true, mask) if m]
        sp = [p for p, m in zip(preds, mask) if m]
        f = f1_score(st, sp, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
        print(f"  [{name} N={sum(mask)}] Macro F1: {f:.4f}")
        return f

    f_rich = _sub_f1(rich_mask, "Gold-Rich")
    f_poor = _sub_f1(poor_mask, "Gold-Poor")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_path = OUT_DIR / f"audit_run9_{ts}.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    print(f"  Audit saved to {audit_path}")

    out_path = OUT_DIR / f"pipeline_run9_{ts}.json"
    out_path.write_text(json.dumps({
        "name": "Run 9: Rule + 3-group SVM + FAISS + LLM",
        "macro_f1": mf1, "stage_counts": stage_counts,
        "f1_rich": f_rich, "f1_poor": f_poor,
        "llm_calls": llm_count, "audit_csv": str(audit_path),
    }, indent=2))
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
