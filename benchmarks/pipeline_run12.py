#!/usr/bin/env python3
"""Run 12: Hybrid SVM gate → FAISS unbounded → LLM.

Pipeline design:
  Stage 0.5  empty-tag rule
  Stage 1    keyword rules
  Stage 2    Hybrid SVM (TF-IDF + bge, junk included as non_quality)
             • unbounded scale → SKIP Stage 2 entirely
             • bounded + quality_prob > threshold → quality
             • else → Stage 3
  Stage 3    FAISS in-domain
             • no metadata → scale_heuristic fallback, else LLM
             • in_domain + unbounded → quantity
             • in_domain + bounded   → LLM (ambiguous, SVM wasn't confident)
             • not in_domain         → LLM (OOD / junk candidates)
  Stage 4    LLM (V8 prompt with improved junk layer)

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.pipeline_run12 \\
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

from benchmarks.per_tag_svm import predict_quality_prob_hybrid, train_hybrid
from benchmarks.pipeline_3tier import rule_classify
from benchmarks.pipeline_3tier_v2 import (
    LLM_MODEL,
    SVM_VOTE_THRESH,
    enrich_gold_with_agent_meta,
    llm_classify,
    load_gold,
)
from benchmarks.stage3_domain import DomainClassifier, scale_heuristic
from shared.types import LLM_OUTPUT_CATEGORIES

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--exclude-self", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--thresh", type=float, default=SVM_VOTE_THRESH,
                        help="SVM quality_prob threshold (default 0.70)")
    args = parser.parse_args()

    print("Training hybrid SVM (TF-IDF + bge, junk included)...")
    hybrid_bundle = train_hybrid(save=True)

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
    thresh = args.thresh
    print(f"  SVM quality threshold: {thresh}")

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
        is_unbounded = scale.lower() == "unbounded"

        def _rec(pred, source, reason=""):
            preds.append(pred); sources.append(source)
            audit_rows.append({
                "id": row.get("id"), "tag1": tag1, "tag2": tag2,
                "value_scale": scale, "agent_key": agent_key,
                "has_agent_metadata": has_meta, "true_label": true_label,
                "pred": pred, "stage": source, "reason": reason,
                "correct": pred == true_label,
            })

        # Stage 0.5: empty-tag rule
        if not tag1 and not tag2:
            lab = "junk" if is_unbounded else "quality"
            _rec(lab, "empty_tag_rule"); continue

        # Stage 1: keyword rules
        cat = rule_classify(row)
        if cat:
            _rec(cat, "rule"); continue

        # Stage 2: Hybrid SVM — skip entirely for unbounded scale
        quality_prob = 0.5
        if not is_unbounded:
            p1 = predict_quality_prob_hybrid(hybrid_bundle, tag1) if tag1 else 0.5
            p2 = predict_quality_prob_hybrid(hybrid_bundle, tag2) if tag2 else p1
            quality_prob = max(p1, p2) if tag2 else p1
            if quality_prob > thresh:
                _rec("quality", "hybrid_svm", f"prob={quality_prob:.2f}"); continue

        # Stage 3: FAISS in-domain check
        in_domain, best_cos = dc.check_in_domain(tag1, tag2, agent_key)

        if in_domain is None:
            # No FAISS metadata — scale heuristic, else LLM
            label_h = scale_heuristic(scale, decimals)
            if label_h is not None:
                _rec(label_h, "scale_heuristic", f"scale={scale}"); continue
        elif in_domain:
            if is_unbounded:
                _rec("quantity", "faiss_unbounded", f"cos={best_cos:.3f}"); continue
            # bounded + in_domain: SVM was not confident → LLM decides
        # not in_domain or (in_domain + bounded) → fall to LLM

        # Stage 4: LLM
        if use_llm:
            llm_cat = llm_classify(row, LLM_MODEL)
            _rec(llm_cat, "llm"); llm_count += 1
        else:
            guess = "quality" if quality_prob >= 0.50 else "quantity"
            _rec(guess, "ml_default", f"prob={quality_prob:.2f}")

    elapsed = time.time() - t0
    mf1 = f1_score(y_true, preds, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  Run 12: Hybrid SVM → FAISS unbounded → LLM")
    print(f"  Macro F1: {mf1:.4f}   N={len(gold)}")
    print(f"{'='*60}")
    print(classification_report(y_true, preds, labels=LLM_OUTPUT_CATEGORIES, zero_division=0))

    stage_counts: dict[str, int] = {}
    for s in sources:
        stage_counts[s] = stage_counts.get(s, 0) + 1
    for s, n in sorted(stage_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {n} ({n/len(sources)*100:.1f}%)")
    print(f"  LLM calls: {llm_count} ({llm_count/len(gold)*100:.1f}%)  "
          f"elapsed: {elapsed:.1f}s")

    def _sub_f1(mask, name):
        st = [y for y, m in zip(y_true, mask) if m]
        sp = [p for p, m in zip(preds, mask) if m]
        f = f1_score(st, sp, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
        print(f"  [{name} N={sum(mask)}] Macro F1: {f:.4f}")
        return f

    f_rich = _sub_f1(rich_mask, "Gold-Rich")
    f_poor = _sub_f1(poor_mask, "Gold-Poor")

    audit_df = pd.DataFrame(audit_rows)
    junk_rows = audit_df[audit_df["true_label"] == "junk"]
    print(f"\n  Junk records ({len(junk_rows)}):")
    for _, jr in junk_rows.iterrows():
        mark = "✓" if jr["correct"] else "✗"
        print(f"    {mark} [{jr['stage']}] {jr['tag1']!r}|{jr['tag2']!r} → {jr['pred']}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_path = OUT_DIR / f"audit_run12_{ts}.csv"
    audit_df.to_csv(audit_path, index=False)
    print(f"\n  Audit saved to {audit_path}")

    out_path = OUT_DIR / f"pipeline_run12_{ts}.json"
    out_path.write_text(json.dumps({
        "name": "Run 12: Hybrid SVM → FAISS unbounded → LLM",
        "macro_f1": mf1, "stage_counts": stage_counts,
        "f1_rich": f_rich, "f1_poor": f_poor,
        "llm_calls": llm_count, "audit_csv": str(audit_path),
    }, indent=2))
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
