#!/usr/bin/env python3
"""Run 14: Unified fused logreg + Chow reject rule.

Replaces Run 13's two-threshold asymmetric cascade (SVM quality-gate + cosine
domain stage) with a single 3-class logistic-regression model trained on
late-fused embeddings: [tag+scale embedding ‖ agent_description embedding].

Key design choices:
- SAME BGE-small encoder as Run 13 (frozen — SetFit not available in venv).
  The "unified" part is fusing both vectors before the classifier head, and
  training a symmetric 3-class head instead of a 1-directional binary SVM.
- Training data: group_a + group_b (agent_enriched) — same as Run 13, but
  now trains 3-class directly (quality/quantity/junk) using the full record
  with agent_description as second embedding tower.
- Reject rule: Chow's rule with a single threshold τ = max(P(·)) ≥ τ.
  τ swept on a held-out validation fold (20% of training data), picking the
  τ that maximises coverage subject to ≤10% error rate on retained records.
  Records below τ escalate to Stage 4 LLM (same as Run 13).
- Structural constraint preserved: unbounded scale → quantity is always safe
  (same "structural rule" as Run 13's faiss_unbounded branch). Applied as a
  post-filter: if scale==unbounded, P(quality) is zeroed and renormalized
  before argmax.

Comparison target: pipeline_run13 @ thresh=0.80 (best weighted F1 in Run 13).

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.pipeline_run14 \\
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.pipeline_3tier import rule_classify
from benchmarks.pipeline_3tier_v2 import LLM_MODEL, enrich_gold_with_agent_meta, llm_classify, load_gold
from benchmarks.stage3_domain import _load_model

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits" / "agent_enriched"
OUT_DIR = DATA_DIR / "benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["junk", "quality", "quantity"]
TAU_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
MAX_ERROR_RATE = 0.10  # Chow criterion: pick highest coverage τ with error ≤ 10%


# ── Training data ────────────────────────────────────────────────────────────

def load_train_data() -> pd.DataFrame:
    group_a = pd.read_parquet(SPLITS_DIR / "group_a.parquet")
    group_b = pd.read_parquet(SPLITS_DIR / "group_b.parquet")
    df = pd.concat([group_a, group_b], ignore_index=True)
    # keep junk (unlike Run 13 which excluded it from SVM training)
    return df.reset_index(drop=True)


# ── Fused-text builder ────────────────────────────────────────────────────────

def fused_tag_text(tag1: str, tag2: str, scale: str) -> str:
    parts = [p for p in [tag1.strip().lower(), tag2.strip().lower(), scale.strip().lower()] if p]
    return " ".join(parts) if parts else ""


# ── Encode helper: [tag_vec ‖ agent_vec] ─────────────────────────────────────

def encode_fused(model, tag_texts: list[str], agent_texts: list[str]) -> np.ndarray:
    """Return (N, 2D) matrix: tag embedding concatenated with agent embedding."""
    tag_vecs = model.encode(tag_texts, normalize_embeddings=True, show_progress_bar=False,
                            batch_size=256)
    dim = tag_vecs.shape[1]

    nonempty_idx = [i for i, t in enumerate(agent_texts) if t.strip()]
    ag_vecs = np.zeros((len(agent_texts), dim), dtype="float32")
    if nonempty_idx:
        enc = model.encode(
            [agent_texts[i] for i in nonempty_idx],
            normalize_embeddings=True, show_progress_bar=False, batch_size=256,
        )
        for j, i in enumerate(nonempty_idx):
            ag_vecs[i] = enc[j]

    return np.hstack([tag_vecs, ag_vecs])  # (N, 2D)


# ── Chow reject threshold selection ──────────────────────────────────────────

def select_tau(clf, X_val: np.ndarray, y_val: list[str],
               is_unbounded_val: list[bool]) -> float:
    """Pick τ via Chow's rule on validation fold.

    For each candidate τ in TAU_GRID:
      - Compute max(P(·)) for each record (after applying unbounded constraint).
      - Retain records where max(P) ≥ τ.
      - Measure accuracy on retained records.
    Return the smallest τ such that accuracy_retained ≥ (1 - MAX_ERROR_RATE),
    falling back to the largest τ if none qualifies.
    """
    proba = clf.predict_proba(X_val)  # (N, 3)
    class_idx = {c: i for i, c in enumerate(clf.classes_)}

    # Apply unbounded structural constraint: zero P(quality) for unbounded records
    proba_adj = proba.copy()
    q_idx = class_idx.get("quality", -1)
    if q_idx >= 0:
        for i, ub in enumerate(is_unbounded_val):
            if ub:
                proba_adj[i, q_idx] = 0.0
                s = proba_adj[i].sum()
                if s > 0:
                    proba_adj[i] /= s

    conf = proba_adj.max(axis=1)  # max probability after constraint

    best_tau = TAU_GRID[-1]
    for tau in TAU_GRID:
        mask = conf >= tau
        n_retained = mask.sum()
        if n_retained == 0:
            continue
        preds_retained = [clf.classes_[np.argmax(proba_adj[i])] for i, m in enumerate(mask) if m]
        true_retained = [y_val[i] for i, m in enumerate(mask) if m]
        err = sum(p != t for p, t in zip(preds_retained, true_retained)) / n_retained
        coverage = n_retained / len(y_val)
        if err <= MAX_ERROR_RATE:
            best_tau = tau
            print(f"  τ={tau:.2f}  coverage={coverage:.2f}  error={err:.3f}  ← qualifies")
            break
        else:
            print(f"  τ={tau:.2f}  coverage={coverage:.2f}  error={err:.3f}")

    return best_tau


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--exclude-self", action="store_true")
    args = parser.parse_args()

    print("Loading datasets...")
    df_train_all = load_train_data()
    gold = load_gold(args.gold)
    gold = enrich_gold_with_agent_meta(gold)
    if args.exclude_self:
        n_self = int(gold["is_self"].sum())
        gold = gold[~gold["is_self"]].reset_index(drop=True)
        print(f"  Excluded {n_self} self-feedback records")
    print(f"Train N={len(df_train_all)}  Gold N={len(gold)}")

    # Train/val split for τ selection
    df_train, df_val = train_test_split(
        df_train_all, test_size=0.20, random_state=42, stratify=df_train_all["label"]
    )
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    print(f"  Train split: {len(df_train)}  Val split: {len(df_val)}")

    print("Loading BGE model...")
    model = _load_model()

    # Build training fused embeddings
    print("Encoding training data (fused tag+scale || agent_description)...")
    train_tag_texts = [
        fused_tag_text(
            str(r.get("tag1") or ""),
            str(r.get("tag2") or ""),
            str(r.get("value_scale") or ""),
        )
        for _, r in df_train.iterrows()
    ]
    train_ag_texts = [
        str(r.get("agent_description") or "")[:500]
        for _, r in df_train.iterrows()
    ]
    X_train = encode_fused(model, train_tag_texts, train_ag_texts)
    y_train = df_train["label"].tolist()

    # Build validation fused embeddings
    val_tag_texts = [
        fused_tag_text(
            str(r.get("tag1") or ""),
            str(r.get("tag2") or ""),
            str(r.get("value_scale") or ""),
        )
        for _, r in df_val.iterrows()
    ]
    val_ag_texts = [
        str(r.get("agent_description") or "")[:500]
        for _, r in df_val.iterrows()
    ]
    X_val = encode_fused(model, val_tag_texts, val_ag_texts)
    y_val = df_val["label"].tolist()
    is_ub_val = [str(r.get("value_scale") or "").lower() == "unbounded"
                 for _, r in df_val.iterrows()]

    # Train unified 3-class logreg
    print(f"Training 3-class LogisticRegression on {len(X_train)} records...")
    clf = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=3000, random_state=42,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    print(f"  Classes: {clf.classes_}")

    # Select τ via Chow's rule on validation fold
    print(f"\nSelecting τ via Chow's rule (target error ≤ {MAX_ERROR_RATE:.0%}):")
    best_tau = select_tau(clf, X_val, y_val, is_ub_val)
    print(f"  Selected τ = {best_tau:.2f}")

    # Precompute gold embeddings
    print("\nEncoding gold records...")
    gold_tag_texts = [
        fused_tag_text(
            str(r.get("tag1", "") or ""),
            str(r.get("tag2", "") or ""),
            str(r.get("value_scale", "") or ""),
        )
        for _, r in gold.iterrows()
    ]
    gold_ag_texts = [
        str(r.get("agent_context", "") or "")[:500]
        for _, r in gold.iterrows()
    ]
    X_gold = encode_fused(model, gold_tag_texts, gold_ag_texts)

    class_idx = {c: i for i, c in enumerate(clf.classes_)}
    q_idx = class_idx.get("quality", -1)

    # Precompute per-record signals
    print("Precomputing per-record predictions...")
    records = []
    for i, (idx, row) in enumerate(gold.iterrows()):
        tag1 = str(row.get("tag1", "") or "").strip()
        tag2 = str(row.get("tag2", "") or "").strip()
        scale = str(row.get("value_scale", "") or "").strip()
        is_unbounded = scale.lower() == "unbounded"
        has_meta = bool(row.get("has_agent_metadata"))

        rec = {
            "row": row,
            "true_label": row["label"],
            "has_meta": has_meta,
            "is_unbounded": is_unbounded,
            "gold_idx": i,
        }

        # Rule layer (same as Run 13 — always applied first)
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

        # Unified classifier: get probabilities
        proba = clf.predict_proba(X_gold[i:i+1])[0]  # shape (3,)

        # Structural constraint: unbounded → P(quality)=0
        if is_unbounded and q_idx >= 0:
            proba = proba.copy()
            proba[q_idx] = 0.0
            s = proba.sum()
            if s > 0:
                proba /= s

        rec["proba"] = proba
        rec["conf"] = float(proba.max())
        rec["argmax_class"] = clf.classes_[int(np.argmax(proba))]
        records.append(rec)

    n_total = len(records)

    def resolve_tau(rec: dict, tau: float) -> tuple[str | None, str]:
        if "fixed_pred" in rec:
            return rec["fixed_pred"], rec["fixed_stage"]
        if rec["conf"] >= tau:
            return rec["argmax_class"], "unified_logreg"
        return None, "stage4"

    print(f"\nResolving across τ grid...")
    cell_results: dict[float, list[tuple[str | None, str]]] = {}
    llm_needed: set[int] = set()
    for tau in TAU_GRID:
        outcomes = []
        for i, rec in enumerate(records):
            pred, stage = resolve_tau(rec, tau)
            outcomes.append((pred, stage))
            if pred is None:
                llm_needed.add(i)
        cell_results[tau] = outcomes
    print(f"  Unique records needing LLM across all τ: {len(llm_needed)}")

    # LLM calls (cached across τ values)
    print("Calling LLM for Stage 4 records...")
    t0 = time.time()
    for n, i in enumerate(sorted(llm_needed), 1):
        rec = records[i]
        rec["llm_label"] = llm_classify(rec["row"], LLM_MODEL)
        if n % 50 == 0 or n == len(llm_needed):
            print(f"    {n}/{len(llm_needed)}  ({time.time()-t0:.0f}s)")
    if llm_needed:
        print(f"  Done in {time.time()-t0:.0f}s")

    y_true_all = [r["true_label"] for r in records]
    rich_mask = [r["has_meta"] for r in records]
    poor_mask = [not r["has_meta"] for r in records]

    print(f"\n{'TAU':6} | {'MacroF1':8} | {'WtdF1':8} | {'QualF1':7} | {'QtyF1':7} "
          f"| {'QualRec':8} | {'QtyRec':7} | {'LLM%':6}")
    print("-" * 80)
    results = []
    for tau in TAU_GRID:
        outcomes = cell_results[tau]
        preds, llm_count = [], 0
        for i, (pred, stage) in enumerate(outcomes):
            if pred is not None:
                preds.append(pred)
            else:
                preds.append(records[i].get("llm_label", "quality"))
                llm_count += 1

        mf1 = f1_score(y_true_all, preds, labels=CLASSES, average="macro", zero_division=0)
        wf1 = f1_score(y_true_all, preds, labels=CLASSES, average="weighted", zero_division=0)
        rep = classification_report(y_true_all, preds, labels=CLASSES, output_dict=True, zero_division=0)

        rich_true = [y for y, m in zip(y_true_all, rich_mask) if m]
        rich_pred = [p for p, m in zip(preds, rich_mask) if m]
        poor_true = [y for y, m in zip(y_true_all, poor_mask) if m]
        poor_pred = [p for p, m in zip(preds, poor_mask) if m]
        mf1_rich = f1_score(rich_true, rich_pred, labels=CLASSES, average="macro", zero_division=0)
        mf1_poor = f1_score(poor_true, poor_pred, labels=CLASSES, average="macro", zero_division=0)

        llm_pct = llm_count / n_total * 100
        star = " ←" if abs(tau - best_tau) < 1e-9 else ""
        print(f"{tau:<6.2f} | {mf1:<8.4f} | {wf1:<8.4f} | {rep['quality']['f1-score']:<7.3f} | "
              f"{rep['quantity']['f1-score']:<7.3f} | {rep['quality']['recall']:<8.3f} | "
              f"{rep['quantity']['recall']:<7.3f} | {llm_pct:<6.1f}{star}")

        results.append({
            "tau": tau,
            "macro_f1": mf1, "weighted_f1": wf1,
            "quality_f1": rep["quality"]["f1-score"],
            "quantity_f1": rep["quantity"]["f1-score"],
            "junk_f1": rep.get("junk", {}).get("f1-score", 0.0),
            "quality_recall": rep["quality"]["recall"],
            "quantity_recall": rep["quantity"]["recall"],
            "quality_precision": rep["quality"]["precision"],
            "quantity_precision": rep["quantity"]["precision"],
            "quality_support": rep["quality"].get("support", 0),
            "quantity_support": rep["quantity"].get("support", 0),
            "junk_support": rep.get("junk", {}).get("support", 0),
            "macro_f1_rich": mf1_rich, "macro_f1_poor": mf1_poor,
            "llm_calls": llm_count, "llm_pct": llm_pct,
            "preds": preds,
        })

    # Best by Chow-selected τ, and by grid search for reference
    best_chow = next(r for r in results if abs(r["tau"] - best_tau) < 1e-9)
    best_wf1 = max(results, key=lambda r: r["weighted_f1"])
    best_mf1 = max(results, key=lambda r: r["macro_f1"])

    for label, r in [
        ("CHOW-SELECTED τ", best_chow),
        ("BEST BY MACRO F1 (grid)", best_mf1),
        ("BEST BY WEIGHTED F1 (grid)", best_wf1),
    ]:
        print(f"\n{'='*70}\n{label}: τ={r['tau']:.2f}")
        print(f"  Macro F1={r['macro_f1']:.4f}  Weighted F1={r['weighted_f1']:.4f}")
        print(f"  quality F1={r['quality_f1']:.3f}  qty F1={r['quantity_f1']:.3f}  junk F1={r['junk_f1']:.3f}")
        print(f"  quality recall={r['quality_recall']:.3f}  qty recall={r['quantity_recall']:.3f}")
        print(f"  Rich macro={r['macro_f1_rich']:.4f}  Poor macro={r['macro_f1_poor']:.4f}")
        print(f"  LLM calls: {r['llm_calls']} ({r['llm_pct']:.1f}%)")
        print(classification_report(y_true_all, r["preds"], labels=CLASSES, zero_division=0))

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"pipeline_run14_{ts}.json"
    payload = {
        "run": 14,
        "description": "Unified fused logreg [tag||agent] + Chow reject rule",
        "gold_path": str(args.gold),
        "gold_n": len(gold),
        "train_n": len(df_train),
        "val_n": len(df_val),
        "chow_tau": best_tau,
        "chow_max_error_rate": MAX_ERROR_RATE,
        "tau_grid": TAU_GRID,
        "results": [
            {k: v for k, v in r.items() if k != "preds"}
            for r in results
        ],
        "best_chow": {k: v for k, v in best_chow.items() if k != "preds"},
        "best_macro": {k: v for k, v in best_mf1.items() if k != "preds"},
        "best_weighted": {k: v for k, v in best_wf1.items() if k != "preds"},
        "timestamp": ts,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
