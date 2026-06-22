#!/usr/bin/env python3
"""Run 7: Rule + 2-class per-tag SVM (quality-vs-quantity only) + Stage-3 tie-break.

Two surgical changes vs the official Run 5 cascade (pipeline_3tier_v2.py), proposed
and scoped by the user -- deliberately NOT touching Poor-agent routing, NOT adding
any new feature (no cos_metric, no combiner), NOT adding a confidence-threshold LLM
gate:

  1. Stage 2 SVM is retrained excluding junk rows entirely (quality vs quantity
     only, instead of quality vs non_quality=quantity+junk). Junk vocabulary mixed
     into the "non_quality" negative class adds noise to the quality/quantity
     boundary; removing it should give a cleaner lean.

  2. Stage 3's in-domain + bounded-scale branch no longer blindly defaults to
     "quality" (the cascade's biggest single error source -- ~49% of all
     quantity->quality misclassifications, see thesis_summary_3tier_tuning.md).
     It now uses Stage 2's own (cleaner) quality-lean p1/p2 as the tie-break,
     instead of throwing that evidence away.

Known, accepted limitation (explicitly flagged by the user before running this):
junk that Stage 2 ALREADY confidently (and wrongly) resolves with p>=0.70 -- e.g.
gibberish like "Jesus"/"nsjak|asdjck" that the SVM has never seen and defaults
toward the majority class -- is untouched by this change, since Stage 2's vote
still short-circuits before Stage 3 ever runs. Only junk that reaches Stage 3
(low-confidence Stage 2 vote) benefits, and of those, only the ones that fail the
in-domain cosine check (most junk, since it's semantically unrelated to any
agent's specific business domain) get routed onward to the LLM -- junk that
coincidentally scores high in-domain cosine still gets the SVM-lean tie-break,
not a junk label (Stage 2/3 still cannot emit "junk").

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.pipeline_qty_svm \
        --gold data/labelled/pure_others_to_label.csv --exclude-self
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.per_tag_svm import load_per_tag_svm, predict_quality_prob, train, vote_per_tag
from benchmarks.pipeline_3tier import rule_classify
from benchmarks.pipeline_3tier_v2 import LLM_MODEL, enrich_gold_with_agent_meta, llm_classify, load_gold
from benchmarks.stage3_domain import _encode, _load_index
from shared.types import LLM_OUTPUT_CATEGORIES

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SVM_QTY_MODEL_PATH = ROOT / "data/models/per_tag_svm_quality_vs_quantity.joblib"

SVM_VOTE_THRESH = 0.70
THRESH_IN_DOMAIN = 0.55


def best_cos_to_domain(tags: list[str], agent_key: str) -> tuple[float | None, bool]:
    """Mirrors stage3_domain.DomainClassifier's cosine check. Returns (cos, has_domain)."""
    index, key_to_pos = _load_index()
    pos = key_to_pos.get(agent_key)
    if pos is None or not tags:
        return None, False
    agent_vec = index.reconstruct(pos)
    sims = [float(np.dot(_encode(t), agent_vec)) for t in tags]
    return max(sims), True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--exclude-self", action="store_true")
    args = parser.parse_args()

    print("Training Stage 2 SVM on quality-vs-quantity only (junk excluded)...")
    train(save=True, exclude_junk=True, model_path=SVM_QTY_MODEL_PATH)
    per_tag_pipe = load_per_tag_svm(model_path=SVM_QTY_MODEL_PATH)

    print(f"\nLoading gold from {args.gold} ...")
    gold = load_gold(args.gold)
    print(f"  Gold N={len(gold)}")
    gold = enrich_gold_with_agent_meta(gold)
    if args.exclude_self:
        n_self = int(gold["is_self"].sum())
        gold = gold[~gold["is_self"]].reset_index(drop=True)
        print(f"  --exclude-self: dropped {n_self} -> N={len(gold)}")

    (OUT_DIR / "llm_classification.log").write_text("", encoding="utf-8")

    rich_mask = gold["has_agent_metadata"].tolist()
    poor_mask = [not m for m in rich_mask]
    y_true = gold["label"].tolist()

    preds, sources, audit_rows = [], [], []
    llm_count = 0
    t0 = time.time()

    for _, row in gold.iterrows():
        tag1 = str(row.get("tag1", "") or "").strip()
        tag2 = str(row.get("tag2", "") or "").strip()
        scale = str(row.get("value_scale", "") or "").strip()
        agent_key = str(row.get("agent_key", "") or "")
        true_label = row.get("label")
        has_meta = bool(row.get("has_agent_metadata"))

        def _rec(pred, source, reason=""):
            preds.append(pred); sources.append(source)
            audit_rows.append({
                "id": row.get("id"), "tag1": tag1, "tag2": tag2, "value_scale": scale,
                "agent_key": agent_key, "has_agent_metadata": has_meta, "true_label": true_label,
                "pred": pred, "stage": source, "reason": reason, "correct": pred == true_label,
            })

        # Stage 0.5: empty-tag rule (unchanged)
        if not tag1 and not tag2:
            lab = "junk" if scale.lower() == "unbounded" else "quality"
            _rec(lab, "empty_tag_rule"); continue

        # Stage 1: rule (unchanged)
        cat = rule_classify(row)
        if cat:
            _rec(cat, "rule"); continue

        # Stage 2: 2-class SVM (quality-vs-quantity only)
        t2_empty = not tag2
        p1 = predict_quality_prob(per_tag_pipe, tag1, scale) if tag1 else 0.5
        p2 = predict_quality_prob(per_tag_pipe, tag2, scale) if tag2 else 0.5
        vote = vote_per_tag(p1, p2, t2_empty=t2_empty, thresh=SVM_VOTE_THRESH)

        if vote == "quality":
            _rec("quality", "per_tag_svm", f"p1={p1:.2f},p2={p2:.2f}"); continue

        # Stage 3: agent-domain cosine. NEW: bounded branch uses SVM lean, not a
        # blind "quality" default -- the one change that targets the cascade's
        # single biggest error source.
        tags = [t for t in (tag1, tag2) if t]
        if not has_meta:
            llm_cat = llm_classify(row, LLM_MODEL)
            _rec(llm_cat, "llm", "no_agent_metadata"); llm_count += 1; continue

        best_cos, has_dom = best_cos_to_domain(tags, agent_key)
        if has_dom and best_cos is not None and best_cos > THRESH_IN_DOMAIN:
            if scale.lower() == "unbounded":
                label = "quantity"  # hard convention: unbounded is never quality
            else:
                quality_lean = max(p1, p2) if not t2_empty else p1
                label = "quality" if quality_lean >= 0.5 else "quantity"
            _rec(label, "faiss_tiebreak", f"in_domain cos={best_cos:.3f}, p1={p1:.2f},p2={p2:.2f}")
            continue

        # Borderline/not-in-domain -> LLM (unchanged)
        llm_cat = llm_classify(row, LLM_MODEL)
        _rec(llm_cat, "llm", f"borderline_cos={best_cos}"); llm_count += 1

    mf1 = f1_score(y_true, preds, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
    print(f"\n{'='*60}\n  Run 7: Rule + 2-class SVM + Stage3 tie-break + LLM\n  Macro F1: {mf1:.4f}\n{'='*60}")
    print(classification_report(y_true, preds, labels=LLM_OUTPUT_CATEGORIES, zero_division=0))
    stage_counts: dict[str, int] = {}
    for s in sources:
        stage_counts[s] = stage_counts.get(s, 0) + 1
    for s, n in sorted(stage_counts.items()):
        print(f"  {s}: {n} ({n/len(sources)*100:.1f}%)")
    elapsed = time.time() - t0
    print(f"  LLM calls: {llm_count} ({llm_count/len(gold)*100:.1f}%)  avg latency: {elapsed/max(llm_count,1)*1000:.0f}ms")

    def _sub_f1(mask, name):
        st = [y for y, m in zip(y_true, mask) if m]
        sp = [p for p, m in zip(preds, mask) if m]
        f = f1_score(st, sp, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
        print(f"  [{name} N={len(st)}] Macro F1: {f:.4f}")
        return f

    f_rich = _sub_f1(rich_mask, "Gold-Rich")
    f_poor = _sub_f1(poor_mask, "Gold-Poor")

    audit_path = OUT_DIR / f"audit_run7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    print(f"  Audit saved to {audit_path}")

    result = {
        "name": "Run 7: Rule + 2-class SVM + Stage3 tie-break + LLM", "macro_f1": mf1,
        "stage_counts": stage_counts, "f1_rich": f_rich, "f1_poor": f_poor,
        "audit_csv": str(audit_path),
    }
    out_path = OUT_DIR / f"pipeline_run7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
