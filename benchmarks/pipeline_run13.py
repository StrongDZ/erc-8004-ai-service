#!/usr/bin/env python3
"""Run 13: BGE-SVM quality-only gate + mandatory LLM escalation at bounded+in-domain.

Motivated by an audited failure mode shared by every prior run that let the SVM
(or a blind scale default) assert "quantity": when Stage 2's quality_prob is
low, that is NOT evidence of quantity -- it is frequently just an unfamiliar
business-domain service name (e.g. "crypto-payments|instant-settlement",
"MEV Protection|Security Audit") that the SVM has never seen. Measured on the
current gold:
  - Run 7 (canonical) tie-break: 42.7% accuracy when it asserts "quantity"
    (vs. 82.8% when it asserts "quality") -- worse than chance in that branch.
  - Run 10's analogous branch: 25.9% accuracy.
  - Run 7' "no tie-break" blind default: the ambiguous bucket's true-label
    mix (82.8% quality) is statistically indistinguishable from the dataset's
    overall prior (82.4%) -- the default adds zero signal, it just exploits
    class imbalance.

Run 13's rule, derived directly from that evidence: the SVM is a one-directional
detector. It may only ever assert "quality" (high threshold). It is NEVER used,
and no blind scale default is used, to assert "quantity" in the bounded+
in-domain branch -- that branch always escalates to the LLM instead. The only
place "quantity" is asserted without an LLM call is the unbounded+in-domain
branch, which is safe because unbounded->quality is structurally impossible by
convention (the only ambiguity there is quantity-vs-junk, not quality-vs-
quantity, and junk is reduced by Stage 1).

Embedding: BGE-small embedding of "<tag> <scale>" (Run 7''s embedding, the best
performing one found in the sweep), trained quality-vs-quantity only (junk
excluded from SVM training, per the separately-confirmed finding that mixing
junk into the SVM's negative class hurts even the one-directional decision).

Only one threshold remains: SVM_THRESH (the quality-assertion bar). There is no
qty_thresh and no tie-break threshold -- both were the mechanisms that let the
model guess "quantity" without real evidence.

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.pipeline_run13 \\
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.pipeline_3tier import rule_classify
from benchmarks.pipeline_3tier_v2 import LLM_MODEL, enrich_gold_with_agent_meta, llm_classify, load_gold
from benchmarks.stage3_domain import DomainClassifier, _load_model, scale_heuristic
from shared.types import LLM_OUTPUT_CATEGORIES

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits" / "agent_enriched"
OUT_DIR = DATA_DIR / "benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESH_GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def load_train_data() -> pd.DataFrame:
    group_a = pd.read_parquet(SPLITS_DIR / "group_a.parquet")
    group_b = pd.read_parquet(SPLITS_DIR / "group_b.parquet")
    df = pd.concat([group_a, group_b], ignore_index=True)
    return df[df["label"] != "junk"].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--exclude-self", action="store_true")
    args = parser.parse_args()

    print("Loading datasets...")
    df_train = load_train_data()
    gold = load_gold(args.gold)
    gold = enrich_gold_with_agent_meta(gold)
    if args.exclude_self:
        n_self = int(gold["is_self"].sum())
        gold = gold[~gold["is_self"]].reset_index(drop=True)
    print(f"Train N={len(df_train)} (junk excluded)  Gold N={len(gold)}")

    print("Loading BGE model + batch-encoding tag+scale strings...")
    model = _load_model()

    unique_texts = set()
    train_rows = []
    for _, r in df_train.iterrows():
        binary = 1 if r["label"] == "quality" else 0
        scale = str(r.get("value_scale") or "").strip().lower()
        t1 = str(r.get("tag1") or "").strip().lower()
        t2 = str(r.get("tag2") or "").strip().lower()
        r1 = f"{t1} {scale}" if t1 else None
        r2 = f"{t2} {scale}" if t2 else None
        if r1: unique_texts.add(r1)
        if r2: unique_texts.add(r2)
        train_rows.append({"t1_text": r1, "t2_text": r2, "label_binary": binary})

    for _, r in gold.iterrows():
        tag1 = str(r.get("tag1", "") or "").strip().lower()
        tag2 = str(r.get("tag2", "") or "").strip().lower()
        scale = str(r.get("value_scale", "") or "").strip().lower()
        if tag1: unique_texts.add(f"{tag1} {scale}")
        if tag2: unique_texts.add(f"{tag2} {scale}")

    all_to_encode = list(unique_texts)
    embeddings = model.encode(all_to_encode, normalize_embeddings=True, show_progress_bar=True)
    cache = {t: v for t, v in zip(all_to_encode, embeddings)}

    X_train, y_train = [], []
    for row in train_rows:
        if row["t1_text"]:
            X_train.append(cache[row["t1_text"]]); y_train.append(row["label_binary"])
        if row["t2_text"]:
            X_train.append(cache[row["t2_text"]]); y_train.append(row["label_binary"])
    X_train, y_train = np.array(X_train), np.array(y_train)

    print(f"Training SVM on {len(X_train)} single-tag samples (quality-vs-quantity, junk excluded)...")
    clf = CalibratedClassifierCV(LinearSVC(C=0.3, max_iter=2000), cv=3, method="sigmoid")
    clf.fit(X_train, y_train)
    quality_idx = list(clf.classes_).index(1)

    def get_quality_prob(tag: str, scale: str) -> float:
        vec = cache[f"{tag.strip().lower()} {scale.strip().lower()}"]
        return float(clf.predict_proba([vec])[0][quality_idx])

    dc = DomainClassifier()

    # --- Precompute everything threshold-independent ---
    print("Precomputing per-record signals...")
    records = []
    for idx, row in gold.iterrows():
        tag1 = str(row.get("tag1", "") or "").strip()
        tag2 = str(row.get("tag2", "") or "").strip()
        scale = str(row.get("value_scale", "") or "").strip()
        decimals = int(row.get("value_decimals", 0) or 0)
        agent_key = str(row.get("agent_key", "") or "")
        has_meta = bool(row.get("has_agent_metadata"))
        is_unbounded = scale.lower() == "unbounded"

        rec = {"row": row, "true_label": row["label"], "has_meta": has_meta,
               "is_unbounded": is_unbounded}

        if not tag1 and not tag2:
            rec["fixed_pred"] = "junk" if is_unbounded else "quality"
            rec["fixed_stage"] = "empty_tag_rule"
            records.append(rec); continue

        cat = rule_classify(row)
        if cat:
            rec["fixed_pred"] = cat
            rec["fixed_stage"] = "rule"
            records.append(rec); continue

        p1 = get_quality_prob(tag1, scale) if tag1 else 0.5
        p2 = get_quality_prob(tag2, scale) if tag2 else 0.5
        quality_prob = max(p1, p2) if tag2 else p1
        rec["quality_prob"] = quality_prob

        # Domain cosine uses the same tag-only embedding convention as every
        # other run (DomainClassifier), NOT the tag+scale text used for the SVM.
        in_domain, best_cos = dc.check_in_domain(tag1, tag2, agent_key)
        rec["in_domain"] = in_domain
        rec["best_cos"] = best_cos
        rec["scale_h"] = scale_heuristic(scale, decimals)
        records.append(rec)

    n_total = len(records)

    def resolve(rec: dict, thresh: float) -> tuple[str | None, str]:
        if "fixed_pred" in rec:
            return rec["fixed_pred"], rec["fixed_stage"]
        qp = rec["quality_prob"]
        if qp > thresh and not rec["is_unbounded"]:
            return "quality", "svm_quality_gate"
        in_domain = rec["in_domain"]
        if in_domain is None:
            if rec["scale_h"] is not None:
                return rec["scale_h"], "scale_heuristic"
            return None, "stage4"  # no metadata, no scale signal -> LLM
        if in_domain:
            if rec["is_unbounded"]:
                return "quantity", "faiss_unbounded"  # safe: unbounded structurally != quality
            return None, "stage4"  # bounded + in_domain + SVM not confident -> ALWAYS escalate
        return None, "stage4"  # not in_domain -> escalate

    print(f"\nResolving Stage 2/3 across thresh in {THRESH_GRID}...")
    cell_results: dict[float, list[tuple[str | None, str]]] = {}
    llm_needed: set[int] = set()
    for thresh in THRESH_GRID:
        outcomes = []
        for i, rec in enumerate(records):
            pred, stage = resolve(rec, thresh)
            outcomes.append((pred, stage))
            if pred is None:
                llm_needed.add(i)
        cell_results[thresh] = outcomes
    print(f"  Unique records needing LLM across all thresholds: {len(llm_needed)}")

    print("Calling LLM (cached) for the union of Stage-4 records...")
    t0 = time.time()
    for n, i in enumerate(sorted(llm_needed), 1):
        rec = records[i]
        rec["llm_label"] = llm_classify(rec["row"], LLM_MODEL)
        if n % 100 == 0 or n == len(llm_needed):
            print(f"    {n}/{len(llm_needed)}  ({time.time()-t0:.0f}s)")
    print(f"  Done in {time.time()-t0:.0f}s")

    y_true_all = [r["true_label"] for r in records]
    rich_mask = [r["has_meta"] for r in records]
    poor_mask = [not r["has_meta"] for r in records]

    print(f"\n{'THRESH':8} | {'MacroF1':8} | {'WtdF1':8} | {'QualF1':7} | {'QtyF1':7} | {'QualRec':8} | {'QtyRec':7} | {'LLM%':6}")
    print("-" * 80)
    results = []
    for thresh, outcomes in cell_results.items():
        preds = []
        llm_count = 0
        for i, (pred, stage) in enumerate(outcomes):
            if pred is not None:
                preds.append(pred)
            else:
                preds.append(records[i]["llm_label"])
                llm_count += 1

        mf1 = f1_score(y_true_all, preds, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
        wf1 = f1_score(y_true_all, preds, labels=LLM_OUTPUT_CATEGORIES, average="weighted", zero_division=0)
        rep = classification_report(y_true_all, preds, labels=LLM_OUTPUT_CATEGORIES, output_dict=True, zero_division=0)

        rich_true = [y for y, m in zip(y_true_all, rich_mask) if m]
        rich_pred = [p for p, m in zip(preds, rich_mask) if m]
        poor_true = [y for y, m in zip(y_true_all, poor_mask) if m]
        poor_pred = [p for p, m in zip(preds, poor_mask) if m]
        mf1_rich = f1_score(rich_true, rich_pred, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
        wf1_rich = f1_score(rich_true, rich_pred, labels=LLM_OUTPUT_CATEGORIES, average="weighted", zero_division=0)
        mf1_poor = f1_score(poor_true, poor_pred, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
        wf1_poor = f1_score(poor_true, poor_pred, labels=LLM_OUTPUT_CATEGORIES, average="weighted", zero_division=0)

        llm_pct = llm_count / n_total * 100
        print(f"{thresh:<8.2f} | {mf1:<8.4f} | {wf1:<8.4f} | {rep['quality']['f1-score']:<7.3f} | "
              f"{rep['quantity']['f1-score']:<7.3f} | {rep['quality']['recall']:<8.3f} | "
              f"{rep['quantity']['recall']:<7.3f} | {llm_pct:<6.1f}")

        results.append({
            "thresh": thresh, "macro_f1": mf1, "weighted_f1": wf1,
            "quality_f1": rep["quality"]["f1-score"], "quantity_f1": rep["quantity"]["f1-score"],
            "quality_recall": rep["quality"]["recall"], "quantity_recall": rep["quantity"]["recall"],
            "junk_f1": rep.get("junk", {}).get("f1-score", 0.0),
            "macro_f1_rich": mf1_rich, "weighted_f1_rich": wf1_rich,
            "macro_f1_poor": mf1_poor, "weighted_f1_poor": wf1_poor,
            "llm_calls": llm_count, "llm_pct": llm_pct,
            "preds": preds,
        })

    best_wf1 = max(results, key=lambda r: r["weighted_f1"])
    best_mf1 = max(results, key=lambda r: r["macro_f1"])
    for label, r in [("BEST BY WEIGHTED F1", best_wf1), ("BEST BY MACRO F1", best_mf1)]:
        print(f"\n{'='*70}\n{label}: thresh={r['thresh']}")
        print(f"  Macro F1={r['macro_f1']:.4f}  Weighted F1={r['weighted_f1']:.4f}")
        print(f"  quality F1={r['quality_f1']:.3f}  quantity F1={r['quantity_f1']:.3f}  junk F1={r['junk_f1']:.3f}")
        print(f"  quality recall={r['quality_recall']:.3f}  quantity recall={r['quantity_recall']:.3f}")
        print(f"  Rich -- Macro:{r['macro_f1_rich']:.4f} Weighted:{r['weighted_f1_rich']:.4f}")
        print(f"  Poor -- Macro:{r['macro_f1_poor']:.4f} Weighted:{r['weighted_f1_poor']:.4f}")
        print(f"  LLM calls: {r['llm_calls']} ({r['llm_pct']:.1f}%)")
        print(classification_report(y_true_all, r["preds"], labels=LLM_OUTPUT_CATEGORIES, zero_division=0))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"pipeline_run13_{ts}.json"
    out_path.write_text(json.dumps({
        "name": "Run 13: BGE-SVM quality-only gate + mandatory LLM escalation at bounded+in-domain",
        "n_total": n_total,
        "best_weighted_f1": {k: v for k, v in best_wf1.items() if k != "preds"},
        "best_macro_f1": {k: v for k, v in best_mf1.items() if k != "preds"},
        "all_results": [{k: v for k, v in r.items() if k != "preds"} for r in results],
    }, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
