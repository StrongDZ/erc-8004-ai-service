#!/usr/bin/env python3
"""Systematic 3-parameter grid sweep for Run 12 (hybrid SVM gate -> FAISS -> LLM).

Sweeps (thresh, qty_thresh, llm_hi) on a 5x5x5 grid. Stage 0.5/1/2/3 outcomes
are threshold-independent per record except for the SVM quality/quantity gate
(thresh, qty_thresh) and the LLM gate (llm_hi), so everything is precomputed
once and the grid is evaluated by cheap re-combination. LLM calls are cached
per record index across the whole sweep -- each record is sent to the LLM at
most once, regardless of how many (thresh, qty_thresh) cells route it to
Stage 4.

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.sweep_run12 \\
        --gold data/labelled/pure_others_to_label.csv --exclude-self
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.per_tag_svm import predict_qtag_proba, train_qtag_hybrid
from benchmarks.pipeline_3tier import rule_classify
from benchmarks.pipeline_3tier_v2 import (
    LLM_MODEL,
    enrich_gold_with_agent_meta,
    llm_classify,
    load_gold,
)
from benchmarks.stage3_domain import DomainClassifier, scale_heuristic
from shared.types import LLM_OUTPUT_CATEGORIES

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESH_GRID = [0.60, 0.65, 0.70, 0.75, 0.80]
QTY_THRESH_GRID = [0.20, 0.25, 0.30, 0.35, 0.40]
LLM_HI_GRID = [0.50, 0.55, 0.60, 0.65, 0.70]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--exclude-self", action="store_true")
    args = parser.parse_args()

    print("Training hybrid SVM (TF-IDF + bge, junk excluded)...")
    qtag_bundle = train_qtag_hybrid(save=True)

    print(f"Loading gold from {args.gold} ...")
    gold = load_gold(args.gold)
    gold = enrich_gold_with_agent_meta(gold)
    if args.exclude_self:
        n_self = int(gold["is_self"].sum())
        gold = gold[~gold["is_self"]].reset_index(drop=True)
        print(f"  --exclude-self: dropped {n_self} -> N={len(gold)}")

    dc = DomainClassifier()

    # --- Precompute everything that is threshold-independent, once. ---
    print("Precomputing per-record signals (SVM probs, FAISS, scale heuristic)...")
    records = []
    for idx, row in gold.iterrows():
        tag1 = str(row.get("tag1", "") or "").strip()
        tag2 = str(row.get("tag2", "") or "").strip()
        scale = str(row.get("value_scale", "") or "").strip()
        decimals = int(row.get("value_decimals", 0) or 0)
        agent_key = str(row.get("agent_key", "") or "")
        true_label = row.get("label")
        has_meta = bool(row.get("has_agent_metadata"))
        is_unbounded = scale.lower() == "unbounded"

        rec = {
            "idx": idx, "row": row, "tag1": tag1, "tag2": tag2, "scale": scale,
            "true_label": true_label, "has_meta": has_meta,
            "is_unbounded": is_unbounded, "llm_label": None,
        }

        if not tag1 and not tag2:
            rec["fixed_pred"] = "junk" if is_unbounded else "quality"
            rec["fixed_stage"] = "empty_tag_rule"
            records.append(rec)
            continue

        cat = rule_classify(row)
        if cat:
            rec["fixed_pred"] = cat
            rec["fixed_stage"] = "rule"
            records.append(rec)
            continue

        p1, _ = predict_qtag_proba(qtag_bundle, tag1) if tag1 else (0.5, 0.5)
        p2, _ = predict_qtag_proba(qtag_bundle, tag2) if tag2 else (p1, 1 - p1)
        quality_prob = max(p1, p2) if tag2 else p1
        rec["quality_prob"] = quality_prob

        in_domain, best_cos = dc.check_in_domain(tag1, tag2, agent_key)
        rec["in_domain"] = in_domain
        rec["best_cos"] = best_cos
        rec["scale_h"] = scale_heuristic(scale, decimals)
        records.append(rec)

    n_total = len(records)
    n_fixed = sum(1 for r in records if "fixed_pred" in r)
    print(f"  N={n_total}  resolved by Stage0.5/1 (threshold-independent): {n_fixed}")

    def stage234(rec: dict, thresh: float, qty_thresh: float) -> tuple[str | None, str]:
        """Returns (pred_or_None, stage). pred is None if it must fall to Stage 4."""
        if "fixed_pred" in rec:
            return rec["fixed_pred"], rec["fixed_stage"]

        qp = rec["quality_prob"]
        if qp > thresh and not rec["is_unbounded"]:
            return "quality", "hybrid_svm"
        if qp < qty_thresh and rec["is_unbounded"]:
            return "quantity", "hybrid_svm_qty_unbounded"

        in_domain = rec["in_domain"]
        if in_domain is None:
            if rec["scale_h"] is not None:
                return rec["scale_h"], "scale_heuristic"
        elif in_domain:
            if rec["is_unbounded"]:
                return "quantity", "faiss_unbounded"
            # bounded + in_domain -> ambiguous, falls to Stage 4
        return None, "stage4"

    # --- Determine, for every (thresh, qty_thresh) cell, which records need the LLM. ---
    print(f"\nResolving Stage 2/3 across {len(THRESH_GRID)}x{len(QTY_THRESH_GRID)} (thresh, qty_thresh) grid...")
    cell_results: dict[tuple[float, float], list[tuple[str | None, str]]] = {}
    llm_needed_idx: set[int] = set()
    for thresh in THRESH_GRID:
        for qty_thresh in QTY_THRESH_GRID:
            outcomes = []
            for rec in records:
                pred, stage = stage234(rec, thresh, qty_thresh)
                outcomes.append((pred, stage))
                if pred is None:
                    llm_needed_idx.add(rec["idx"])
            cell_results[(thresh, qty_thresh)] = outcomes

    print(f"  Unique records needing an LLM call across the whole grid: {len(llm_needed_idx)}")

    # --- Call the LLM once per unique record, cache by idx. ---
    print("Calling LLM for the union of Stage-4 records (cached, called once each)...")
    t0 = time.time()
    idx_to_rec = {rec["idx"]: rec for rec in records}
    for n, idx in enumerate(sorted(llm_needed_idx), 1):
        rec = idx_to_rec[idx]
        rec["llm_label"] = llm_classify(rec["row"], LLM_MODEL)
        if n % 50 == 0 or n == len(llm_needed_idx):
            elapsed = time.time() - t0
            print(f"    {n}/{len(llm_needed_idx)}  ({elapsed:.0f}s elapsed)")
    print(f"  LLM calls completed in {time.time() - t0:.0f}s")

    y_true_all = [rec["true_label"] for rec in records]
    rich_mask = [rec["has_meta"] for rec in records]
    poor_mask = [not rec["has_meta"] for rec in records]

    # --- Final sweep: combine cell_results with llm_hi (cheap, no LLM calls). ---
    print(f"\nSweeping llm_hi over {LLM_HI_GRID} for each (thresh, qty_thresh) cell...")
    results = []
    for (thresh, qty_thresh), outcomes in cell_results.items():
        for llm_hi in LLM_HI_GRID:
            preds = []
            llm_count = 0
            for rec, (pred, stage) in zip(records, outcomes):
                if pred is not None:
                    preds.append(pred)
                    continue
                qp = rec["quality_prob"]
                if qp < llm_hi:
                    preds.append(rec["llm_label"])
                    llm_count += 1
                else:
                    preds.append("quality" if qp >= 0.50 else "quantity")

            mf1 = f1_score(y_true_all, preds, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
            wf1 = f1_score(y_true_all, preds, labels=LLM_OUTPUT_CATEGORIES, average="weighted", zero_division=0)

            rich_true = [y for y, m in zip(y_true_all, rich_mask) if m]
            rich_pred = [p for p, m in zip(preds, rich_mask) if m]
            poor_true = [y for y, m in zip(y_true_all, poor_mask) if m]
            poor_pred = [p for p, m in zip(preds, poor_mask) if m]
            mf1_rich = f1_score(rich_true, rich_pred, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
            wf1_rich = f1_score(rich_true, rich_pred, labels=LLM_OUTPUT_CATEGORIES, average="weighted", zero_division=0)
            mf1_poor = f1_score(poor_true, poor_pred, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
            wf1_poor = f1_score(poor_true, poor_pred, labels=LLM_OUTPUT_CATEGORIES, average="weighted", zero_division=0)

            results.append({
                "thresh": thresh, "qty_thresh": qty_thresh, "llm_hi": llm_hi,
                "macro_f1": mf1, "weighted_f1": wf1,
                "macro_f1_rich": mf1_rich, "weighted_f1_rich": wf1_rich,
                "macro_f1_poor": mf1_poor, "weighted_f1_poor": wf1_poor,
                "llm_calls": llm_count, "llm_pct": llm_count / n_total * 100,
            })

    best_wf1 = max(results, key=lambda r: r["weighted_f1"])
    best_mf1 = max(results, key=lambda r: r["macro_f1"])
    # Best Weighted F1 among configs with LLM cost <= Run 7's 19.7% for a fair-cost comparison
    cheap = [r for r in results if r["llm_pct"] <= 19.7]
    best_wf1_cheap = max(cheap, key=lambda r: r["weighted_f1"]) if cheap else None

    for label, r in [
        ("BEST BY WEIGHTED F1 (any cost)", best_wf1),
        ("BEST BY MACRO F1 (any cost)", best_mf1),
        ("BEST BY WEIGHTED F1 (LLM cost <= 19.7%, fair vs Run 7)", best_wf1_cheap),
    ]:
        print(f"\n{'='*70}\n{label}:")
        if r is None:
            print("  (no config found under this constraint)")
            continue
        print(f"  thresh={r['thresh']}  qty_thresh={r['qty_thresh']}  llm_hi={r['llm_hi']}")
        print(f"  Macro F1: {r['macro_f1']:.4f}   Weighted F1: {r['weighted_f1']:.4f}")
        print(f"  Rich -- Macro: {r['macro_f1_rich']:.4f}  Weighted: {r['weighted_f1_rich']:.4f}")
        print(f"  Poor -- Macro: {r['macro_f1_poor']:.4f}  Weighted: {r['weighted_f1_poor']:.4f}")
        print(f"  LLM calls: {r['llm_calls']} ({r['llm_pct']:.1f}%)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"sweep_run12_{ts}.json"
    out_path.write_text(json.dumps({
        "n_total": n_total,
        "grid": {"thresh": THRESH_GRID, "qty_thresh": QTY_THRESH_GRID, "llm_hi": LLM_HI_GRID},
        "best_weighted_f1": best_wf1,
        "best_macro_f1": best_mf1,
        "best_weighted_f1_cost_capped": best_wf1_cheap,
        "all_results": results,
    }, indent=2))
    print(f"\nFull sweep saved to {out_path}")


if __name__ == "__main__":
    main()
