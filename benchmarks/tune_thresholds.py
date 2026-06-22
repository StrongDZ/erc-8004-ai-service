#!/usr/bin/env python3
"""Threshold grid-search for the 3-tier pipeline (Run 4 configuration).

Pre-computes all threshold-independent intermediate values (p1, p2, best_cos)
for every record in gold_combined_clean_v2.csv, then sweeps threshold
combinations on a stratified 80% val split. The 20% test split is only
touched once to report final numbers — never during the sweep.

Two thresholds are searched jointly (junk excluded from eval — caught upstream
by Go rule engine or LLM; THRESH_NOT_DOMAIN removed as dead code):
  SVM_VOTE_THRESH   — Stage 2 quality confidence gate
  THRESH_IN_DOMAIN  — Stage 3 cosine gate  (in-domain → quality/quantity)

SVM training data (agent_enriched splits with rule-based silver labels) and
the gold test set stay strictly separate — the SVM is not retrained here.

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.tune_thresholds
    .venv/bin/python3 -m benchmarks.tune_thresholds --no-cache   # force re-compute
    .venv/bin/python3 -m benchmarks.tune_thresholds --gold data/labelled/gold_combined_clean_v2.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.per_tag_svm import load_per_tag_svm, predict_quality_prob
from benchmarks.pipeline_3tier import rule_classify
from benchmarks.pipeline_3tier_v2 import enrich_gold_with_agent_meta, load_gold
from benchmarks.stage3_domain import (
    _cosine_to_agent,
    _load_index,
    _load_model,
    scale_heuristic,
)
ROOT = Path(__file__).resolve().parent.parent
GOLD_CSV = ROOT / "data/labelled/gold_combined_clean_v2.csv"
CACHE_PATH = ROOT / "data/benchmark_results/tune_cache.parquet"
OUT_DIR = ROOT / "data/benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["quality", "quantity"]  # junk excluded: caught upstream by Go/LLM


# ── Cascade helpers ────────────────────────────────────────────────────────────

def _svm_vote(p1: float, p2: float, t2_empty: bool, thresh: float) -> str | None:
    """Replicate vote_per_tag from per_tag_svm.py with a variable threshold."""
    def _c(p: float) -> str | None:
        if p >= thresh:
            return "quality"
        if p <= 1.0 - thresh:
            return "non_quality"
        return None

    c1 = _c(p1)
    c2 = None if t2_empty else _c(p2)
    if t2_empty:
        return c1
    if c1 is not None and c2 is not None:
        return c1 if c1 == c2 else None  # conflict → Stage 3
    return c1 if c1 is not None else c2


def _scale_label(scale: str) -> str:
    return "quantity" if scale == "unbounded" else "quality"


def cascade_predict(rec: pd.Series, svm_thresh: float, thresh_in: float) -> str:
    """Apply the Run-5 cascade (with LLM fallback) on one precomputed record."""
    # Stage 0.5 + Stage 1 (deterministic — captured in stage_early)
    se = rec["stage_early"]
    if se is not None and se != "":
        return se

    p1, p2, t2_empty = float(rec["p1"]), float(rec["p2"]), bool(rec["t2_empty"])
    scale = str(rec["scale"])

    # Stage 2: SVM vote
    vote = _svm_vote(p1, p2, t2_empty, svm_thresh)
    if vote == "quality":
        return "quality"
    # non_quality or uncertain → Stage 3

    # Helper function to get LLM prediction lazily
    def get_llm_pred():
        from benchmarks.pipeline_3tier_v2 import LLM_MODEL, llm_classify
        # Pass rec (which is a pd.Series representing the row) to llm_classify
        return llm_classify(rec, LLM_MODEL)

    # Stage 3: cosine gate
    best_cos = rec["best_cos"]
    if pd.isna(best_cos):
        # No FAISS entry (Poor records) -> LLM fallback
        return get_llm_pred()

    best_cos = float(best_cos)
    if best_cos > thresh_in:
        return _scale_label(scale)

    # Stage 4: LLM fallback
    return get_llm_pred()


def macro_f1(cache: pd.DataFrame, svm_thresh: float, thresh_in: float) -> float:
    preds = [cascade_predict(r, svm_thresh, thresh_in) for _, r in cache.iterrows()]
    return f1_score(cache["label"].tolist(), preds, labels=LABELS, average="macro", zero_division=0)


# ── Pre-computation ────────────────────────────────────────────────────────────

def precompute(gold: pd.DataFrame) -> pd.DataFrame:
    """Compute threshold-independent features for every record once."""
    print("Loading SVM model...")
    svm = load_per_tag_svm()

    print("Loading FAISS index + bge-small encoder (first call downloads if needed)...")
    index, key_to_pos = _load_index()
    _load_model()  # warm up cache

    rows = []
    n = len(gold)
    t0 = time.monotonic()

    for i, (_, row) in enumerate(gold.iterrows()):
        if i % 100 == 0:
            print(f"  {i}/{n} ({i/n*100:.0f}%)...", end="\r", flush=True)

        tag1 = str(row.get("tag1", "") or "").strip()
        tag2 = str(row.get("tag2", "") or "").strip()
        scale = str(row.get("value_scale", "") or "").strip().lower()
        decimals = int(row.get("value_decimals", 0) or 0)
        agent_key = str(row.get("agent_key", "") or "")
        label = str(row.get("label", "") or "").strip()
        has_meta = bool(row.get("has_agent_metadata", False))

        # Stage 0.5: empty-tag rule (same logic as pipeline_3tier_v2.py)
        if not tag1 and not tag2:
            stage_early = "junk" if scale == "unbounded" else "quality"
        else:
            # Stage 1: deterministic rule engine
            stage_early = rule_classify(row)  # None → escalate to Python AI service

        # SVM probabilities (computed for all records; used in Stage 2 and Stage 4)
        p1 = predict_quality_prob(svm, tag1, scale) if tag1 else 0.5
        p2 = predict_quality_prob(svm, tag2, scale) if tag2 else 0.5
        t2_empty = not bool(tag2)

        # Stage 3: max cosine similarity to agent domain embedding
        best_cos = None
        scale_heur = None
        if stage_early is None:
            pos = key_to_pos.get(agent_key)
            if pos is not None:
                tags = [t for t in (tag1, tag2) if t]
                if tags:
                    agent_vec = index.reconstruct(pos)
                    cos_scores = [_cosine_to_agent(t, agent_vec) for t in tags]
                    best_cos = float(max(cos_scores))
            else:
                # No FAISS entry (Poor record) — scale heuristic
                scale_heur = scale_heuristic(scale, decimals)

        rows.append({
            "id": str(row.get("id", "")),
            "label": label,
            "has_meta": has_meta,
            "scale": scale,
            "stage_early": stage_early if stage_early is not None else "",
            "p1": p1,
            "p2": p2,
            "t2_empty": t2_empty,
            "best_cos": best_cos,
            "scale_heur": scale_heur if scale_heur is not None else "",
            
            # Fields for LLM (with safe types to avoid pyarrow overflow)
            "agent_key": str(row.get("agent_key", "") or ""),
            "fb_parsed": json.dumps(row.get("fb_parsed")) if isinstance(row.get("fb_parsed"), dict) else str(row.get("fb_parsed") or ""),
            "value_scale": str(row.get("value_scale", "") or ""),
            "tag1": tag1,
            "tag2": tag2,
            "endpoint": str(row.get("endpoint", "") or ""),
            "value": str(row.get("value", "") or ""),
            "value_decimals": int(row.get("value_decimals", 0) or 0),
            "is_self": bool(row.get("is_self", False)),
        })

    elapsed = time.monotonic() - t0
    print(f"\nPre-computed {n} records in {elapsed:.1f}s  ({n/elapsed:.0f} records/s)")
    return pd.DataFrame(rows)


# ── Grid search ────────────────────────────────────────────────────────────────

def grid_search(val: pd.DataFrame) -> dict:
    # Search space — coarse grid first, refine after if needed
    svm_range = [0.60, 0.65, 0.70, 0.75, 0.80]
    in_range  = [0.45, 0.50, 0.55, 0.60, 0.65]

    combos = [(sv, ti) for sv in svm_range for ti in in_range]
    print(f"\nGrid search: {len(combos)} valid combinations on val set (N={len(val)})...")

    best: dict = {"macro_f1": -1.0}
    for idx, (sv, ti) in enumerate(combos):
        mf1 = macro_f1(val, sv, ti)
        if mf1 > best["macro_f1"]:
            best = {"macro_f1": mf1, "svm_thresh": sv, "thresh_in": ti}
            print(f"  [{idx+1}/{len(combos)}] New best: svm={sv}  in={ti}  →  MacroF1={mf1:.4f}")

    return best


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold grid-search for 3-tier pipeline")
    parser.add_argument("--no-cache", action="store_true", help="Force re-compute intermediates")
    parser.add_argument("--gold", type=Path, default=GOLD_CSV, help="Benchmark CSV path")
    args = parser.parse_args()

    # Load + enrich
    print(f"Loading gold benchmark from {args.gold} ...")
    gold = load_gold(args.gold)
    print(f"  N={len(gold)}  dist={dict(gold['label'].value_counts())}")

    print("Enriching with MongoDB agent metadata...")
    gold = enrich_gold_with_agent_meta(gold)
    rich = gold["has_agent_metadata"].sum()
    print(f"  Rich={rich}  Poor={len(gold)-rich}")

    # Pre-compute or load cache
    if not args.no_cache and CACHE_PATH.exists():
        print(f"\nLoading cached intermediates from {CACHE_PATH} ...")
        cache = pd.read_parquet(CACHE_PATH)
        print(f"  Loaded {len(cache)} records")
    else:
        print("\nPre-computing intermediates (SVM probs + cosine similarities)...")
        cache = precompute(gold)
        cache.to_parquet(CACHE_PATH, index=False)
        print(f"Saved cache → {CACHE_PATH}")

    # Stratified val/test split (fixed seed for reproducibility)
    val_idx, test_idx = train_test_split(
        range(len(cache)), test_size=0.20, random_state=42,
        stratify=cache["label"].tolist(),
    )
    val  = cache.iloc[list(val_idx)].reset_index(drop=True)
    test = cache.iloc[list(test_idx)].reset_index(drop=True)

    # Exclude junk from evaluation — caught upstream by Go rule engine or LLM
    val  = val[val["label"].isin(LABELS)].reset_index(drop=True)
    test = test[test["label"].isin(LABELS)].reset_index(drop=True)

    print(f"\nSplit (junk excluded): val={len(val)}  test={len(test)}")
    print(f"  Val  dist: {dict(val['label'].value_counts())}")
    print(f"  Test dist: {dict(test['label'].value_counts())}")

    # Production baseline on val
    prod_val = macro_f1(val, svm_thresh=0.70, thresh_in=0.55)
    print(f"\nProduction thresholds (svm=0.70, in=0.55):")
    print(f"  Val  Macro F1 = {prod_val:.4f}")

    # Grid search on val
    best = grid_search(val)
    print(f"\nBest val result: svm={best['svm_thresh']}  in={best['thresh_in']}")
    print(f"  Val  Macro F1 = {best['macro_f1']:.4f}  (+{best['macro_f1']-prod_val:+.4f} vs production)")

    # Final evaluation on test — one shot, never used during sweep
    prod_test = macro_f1(test, svm_thresh=0.70, thresh_in=0.55)
    opt_test  = macro_f1(test, best["svm_thresh"], best["thresh_in"])

    prod_preds = [cascade_predict(r, 0.70, 0.55) for _, r in test.iterrows()]
    opt_preds  = [cascade_predict(r, best["svm_thresh"], best["thresh_in"])
                  for _, r in test.iterrows()]

    print(f"\n{'='*60}")
    print(f"  FINAL TEST RESULTS (N={len(test)}, one-shot, junk excluded)")
    print(f"{'='*60}")
    print(f"\nProduction (svm=0.70, in=0.55):")
    print(f"  Test Macro F1 = {prod_test:.4f}")
    print(classification_report(test["label"].tolist(), prod_preds, labels=LABELS, zero_division=0))

    print(f"\nOptimal (svm={best['svm_thresh']}, in={best['thresh_in']}):")
    print(f"  Test Macro F1 = {opt_test:.4f}  ({opt_test-prod_test:+.4f} vs production)")
    print(classification_report(test["label"].tolist(), opt_preds, labels=LABELS, zero_division=0))

    # Rich / Poor breakdown for optimal thresholds
    rich_mask = test["has_meta"].tolist()
    poor_mask = [not m for m in rich_mask]
    rich_true  = [l for l, m in zip(test["label"].tolist(), rich_mask) if m]
    rich_pred  = [p for p, m in zip(opt_preds, rich_mask) if m]
    poor_true  = [l for l, m in zip(test["label"].tolist(), poor_mask) if m]
    poor_pred  = [p for p, m in zip(opt_preds, poor_mask) if m]
    rich_f1 = f1_score(rich_true, rich_pred, labels=LABELS, average="macro", zero_division=0)
    poor_f1 = f1_score(poor_true, poor_pred, labels=LABELS, average="macro", zero_division=0)
    print(f"  Rich (N={len(rich_true)}) Macro F1 = {rich_f1:.4f}")
    print(f"  Poor (N={len(poor_true)}) Macro F1 = {poor_f1:.4f}")

    # Save results
    result = {
        "prod_val_f1":   prod_val,
        "opt_val_f1":    best["macro_f1"],
        "opt_svm_thresh": best["svm_thresh"],
        "opt_thresh_in":  best["thresh_in"],
        "prod_test_f1":  prod_test,
        "opt_test_f1":   opt_test,
        "improvement":   round(opt_test - prod_test, 4),
        "rich_test_f1":  rich_f1,
        "poor_test_f1":  poor_f1,
    }
    out = OUT_DIR / "tune_thresholds_result.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
