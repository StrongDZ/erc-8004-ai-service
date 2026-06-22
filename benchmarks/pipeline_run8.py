#!/usr/bin/env python3
"""Run 8: Rule + Triplet SVM (no prefix, no per-tag expansion) + FAISS + LLM.

Two changes vs the official Run 5 baseline (pipeline_3tier_v2.py --run 5):

  1. Drop 'tag=', 'scale=' prefixes — these are pure noise (IDF=1.0 in every
     document). Content words and their TF-IDF weights are unchanged; the prefix
     tokens are dropped entirely.

  2. Per-tag expansion → triplet: instead of expanding each record into two
     per-tag training rows "(tag, scale)" and voting p1 vs p2, each record
     becomes ONE training example "(tag1, tag2, scale)". With no separator
     between tag and scale, the bigram 'winrate pct100' now forms directly —
     the interaction bigram that was structurally missing in the old format
     (the '| scale=' separator prevented it).

Everything else (Stage 0 self-gate, Stage 0.5 empty-tag rule, Stage 1 keywords,
Stage 3 FAISS cosine, Stage 4 LLM, thresholds) is identical to Run 5.

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.pipeline_run8 \
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
    TRIPLET_MODEL_PATH,
    load_per_tag_svm,
    predict_quality_prob_triplet,
    train_triplet,
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

    print("Training triplet SVM (no prefix, one row per record)...")
    train_triplet(save=True)
    svm_pipe = load_per_tag_svm(model_path=TRIPLET_MODEL_PATH)

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

        # Stage 2: triplet SVM — one probability for the full (tag1, tag2, scale) record
        prob = predict_quality_prob_triplet(svm_pipe, tag1, tag2, scale)
        if prob >= SVM_VOTE_THRESH:
            _rec("quality", "triplet_svm", f"prob={prob:.2f}"); continue
        # non_quality (prob <= 0.30) falls through to Stage 3 — FAISS resolves qty vs junk
        # uncertain (0.30 < prob < 0.70) also goes to Stage 3

        # Stage 3: FAISS domain cosine (unchanged)
        label3, reason3 = dc.classify(tag1, tag2, scale, decimals, agent_key)
        if label3 is not None:
            _rec(label3, "faiss", reason3); continue

        # Stage 4: LLM (unchanged)
        if use_llm:
            llm_cat = llm_classify(row, LLM_MODEL)
            _rec(llm_cat, "llm"); llm_count += 1
        else:
            guess = "quality" if prob >= 0.50 else "quantity"
            _rec(guess, "ml_default", f"prob={prob:.2f}")

    elapsed = time.time() - t0
    mf1 = f1_score(y_true, preds, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  Run 8: Rule + Triplet SVM (no prefix) + FAISS + LLM")
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
    audit_path = OUT_DIR / f"audit_run8_{ts}.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    print(f"  Audit saved to {audit_path}")

    result = {
        "name": "Run 8: Rule + Triplet SVM (no prefix) + FAISS + LLM",
        "macro_f1": mf1,
        "stage_counts": stage_counts,
        "f1_rich": f_rich,
        "f1_poor": f_poor,
        "llm_calls": llm_count,
        "audit_csv": str(audit_path),
    }
    out_path = OUT_DIR / f"pipeline_run8_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
