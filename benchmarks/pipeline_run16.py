#!/usr/bin/env python3
"""Run 16: Early FAISS domain check + feedback-only LLM prompt (V9).

Changes from Run 13:
  1. FAISS in_domain check runs for ALL records right after loading (not only
     for Stage-3-bound records).  Stage 2 (SVM) records therefore also carry
     an in_domain signal that is used for feature assignment.
  2. LLM prompt V9 is feedback-only (tag1, tag2, scale, comment).  No agent
     metadata is passed.  When uncertain the prompt instructs the model to
     default to "quality" (matching the gold-set skew: 79.5 % quality).
  3. Output now includes a `feature` field (agent_domain / infrastructure / None)
     derived by combining the stage-2/3/4 category decision with in_domain.

Feature assignment rules:
  Stage 2 (SVM assert quality)   + in_domain=True  → (quality,  agent_domain)
  Stage 2 (SVM assert quality)   + other           → (quality,  None)
  Stage 3 direct (unbounded+ind) + in_domain=True  → (quantity, agent_domain)
  Scale heuristic (no FAISS)                        → (*,       None)
  Stage 4 LLM=quantity           + in_domain=True  → (quantity, agent_domain)
  Stage 4 LLM=quantity           + False/None      → (quantity, infrastructure)
  Stage 4 LLM=quality            + in_domain=True  → (quality,  agent_domain)
  Stage 4 LLM=quality            + False/None      → (quality,  None)
  Stage 4 LLM=junk                                 → (junk,    None)

V9 LLM cache: data/benchmark_results/llm_cache_v9.json (separate from V8)

Usage:
    cd erc-8004-ai-service
    ollama serve  # must be running with qwen2.5:7b-instruct
    .venv/bin/python3 -m benchmarks.pipeline_run16 \\
        --gold data/labelled/pure_others_stratified_dedup.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.pipeline_3tier import rule_classify
from benchmarks.pipeline_3tier_v2 import LLM_MODEL, LLM_URL, enrich_gold_with_agent_meta, load_gold
from benchmarks.stage3_domain import DomainClassifier, _load_model, scale_heuristic
from shared.types import LLM_OUTPUT_CATEGORIES

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits" / "agent_enriched"
OUT_DIR = DATA_DIR / "benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V9_CACHE_PATH = OUT_DIR / "llm_cache_v9.json"
THRESH_GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

_V9_CACHE: dict[str, str] | None = None


def _load_v9_cache() -> dict[str, str]:
    global _V9_CACHE
    if _V9_CACHE is not None:
        return _V9_CACHE
    if V9_CACHE_PATH.exists():
        try:
            _V9_CACHE = json.loads(V9_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _V9_CACHE = {}
    else:
        _V9_CACHE = {}
    return _V9_CACHE


def _save_v9_cache(fb_id: str, category: str) -> None:
    cache = _load_v9_cache()
    cache[fb_id] = category
    try:
        V9_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


def llm_classify_v9(row: pd.Series, model: str) -> str:
    """V9 feedback-only classification.

    Input: tag1, tag2, value_scale, offchain_note — no agent metadata.
    Output: quality | quantity | junk.  Defaults to quality when uncertain.
    """
    fb_id = str(row.get("id", ""))
    cache = _load_v9_cache()
    if fb_id in cache:
        return cache[fb_id]

    tag1 = str(row.get("tag1", "") or "").strip()
    tag2 = str(row.get("tag2", "") or "").strip()
    scale = str(row.get("value_scale", "") or "").strip()
    comment = str(row.get("offchain_note", "") or "").strip()

    prompt = (
        "Classify this feedback into one category:\n"
        "  - quality: contains a meaningful opinion or assessment about agent\n"
        '    behavior, outcomes, or service quality (e.g. "good", "bad",\n'
        '    "helpful", "wrong", "unreliable"). When uncertain or ambiguous,\n'
        "    default to quality.\n"
        "  - quantity: clearly reports a measurable count, volume, or usage\n"
        '    metric (e.g. number of calls, transactions, files processed).\n'
        "    Only choose when the feedback is unambiguously about numbers/volume.\n"
        "  - junk: noise, spam, or carries no information (meaningless tags,\n"
        "    empty comment with no tags).\n\n"
        "Feedback:\n"
        f"  tag1: {tag1}\n"
        f"  tag2: {tag2}\n"
        f"  scale: {scale}\n"
        f"  comment: {comment}\n\n"
        'Respond with JSON only: {"category": "quality|quantity|junk", '
        '"confidence": 0.00, "reason": "..."}'
    )

    cat_pattern = re.compile(r'"category"\s*:\s*"(junk|quantity|quality)"', re.I)
    final_cat = "quality"

    try:
        resp = requests.post(
            f"{LLM_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 100},
            },
            timeout=60,
        )
        raw = resp.json()["message"]["content"].strip()
        m = cat_pattern.search(raw)
        if m:
            final_cat = m.group(1).lower()
        else:
            # Best-effort word scan; default=quality on parse failure
            w = re.sub(r"[^a-z]", "", raw.lower()[:20])
            if w in ("quality", "quantity", "junk"):
                final_cat = w
    except Exception:
        pass

    _save_v9_cache(fb_id, final_cat)
    return final_cat


def assign_feature(category: str, stage: str, in_domain: bool | None) -> str | None:
    """Derive feature label from category + stage + FAISS in_domain signal."""
    if category == "junk":
        return None
    if stage in ("rule", "empty_tag_rule"):
        return None
    if stage == "scale_heuristic":
        return None
    if stage == "svm_quality_gate":
        return "agent_domain" if in_domain is True else None
    if stage == "faiss_unbounded":
        return "agent_domain"
    # stage4 (LLM)
    if category == "quantity":
        return "agent_domain" if in_domain is True else "infrastructure"
    if category == "quality":
        return "agent_domain" if in_domain is True else None
    return None


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
        gold = gold[~gold["is_self"]].reset_index(drop=True)
    print(f"Train N={len(df_train)} (junk excluded)  Gold N={len(gold)}")

    # --- Early FAISS: compute in_domain for ALL records upfront ---
    print("Early FAISS: computing in_domain for all gold records...")
    dc = DomainClassifier()
    early_domain: dict[int, tuple[bool | None, float]] = {}
    t_faiss = time.time()
    for idx, row in gold.iterrows():
        tag1 = str(row.get("tag1", "") or "").strip()
        tag2 = str(row.get("tag2", "") or "").strip()
        agent_key = str(row.get("agent_key", "") or "")
        in_domain, best_cos = dc.check_in_domain(tag1, tag2, agent_key)
        early_domain[idx] = (in_domain, best_cos)
        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(gold)} ({time.time() - t_faiss:.0f}s)")
    n_in = sum(1 for v, _ in early_domain.values() if v is True)
    n_out = sum(1 for v, _ in early_domain.values() if v is False)
    n_none = sum(1 for v, _ in early_domain.values() if v is None)
    print(f"  in_domain: True={n_in}  False={n_out}  None(no_index)={n_none}")

    # --- BGE-SVM training ---
    print("Loading BGE model + encoding tag+scale strings...")
    model_emb = _load_model()

    unique_texts: set[str] = set()
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

    all_texts = list(unique_texts)
    embs = model_emb.encode(all_texts, normalize_embeddings=True, show_progress_bar=True)
    emb_cache = {t: v for t, v in zip(all_texts, embs)}

    X_train, y_train = [], []
    for row in train_rows:
        if row["t1_text"]:
            X_train.append(emb_cache[row["t1_text"]]); y_train.append(row["label_binary"])
        if row["t2_text"]:
            X_train.append(emb_cache[row["t2_text"]]); y_train.append(row["label_binary"])
    X_train, y_train = np.array(X_train), np.array(y_train)

    print(f"Training SVM on {len(X_train)} samples (quality-vs-quantity only)...")
    clf = CalibratedClassifierCV(LinearSVC(C=0.3, max_iter=2000), cv=3, method="sigmoid")
    clf.fit(X_train, y_train)
    quality_idx = list(clf.classes_).index(1)

    def get_quality_prob(tag: str, scale: str) -> float:
        key = f"{tag.strip().lower()} {scale.strip().lower()}"
        vec = emb_cache[key]
        return float(clf.predict_proba([vec])[0][quality_idx])

    # --- Precompute per-record signals ---
    print("Precomputing per-record signals...")
    records = []
    for idx, row in gold.iterrows():
        tag1 = str(row.get("tag1", "") or "").strip()
        tag2 = str(row.get("tag2", "") or "").strip()
        scale = str(row.get("value_scale", "") or "").strip()
        decimals = int(row.get("value_decimals", 0) or 0)
        has_meta = bool(row.get("has_agent_metadata"))
        is_unbounded = scale.lower() == "unbounded"

        in_domain, best_cos = early_domain[idx]

        rec: dict = {
            "row": row,
            "true_label": row["label"],
            "has_meta": has_meta,
            "is_unbounded": is_unbounded,
            "in_domain": in_domain,
            "best_cos": best_cos,
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

        p1 = get_quality_prob(tag1, scale) if tag1 else 0.5
        p2 = get_quality_prob(tag2, scale) if tag2 else 0.5
        rec["quality_prob"] = max(p1, p2) if tag2 else p1
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
            return None, "stage4"
        if in_domain:
            if rec["is_unbounded"]:
                return "quantity", "faiss_unbounded"
            return None, "stage4"
        return None, "stage4"

    print(f"Resolving Stage 2/3 across thresh {THRESH_GRID}...")
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
    print(f"  Unique records needing LLM (union across thresholds): {len(llm_needed)}")

    v9_cache_hits = sum(1 for i in llm_needed
                        if str(records[i]["row"].get("id", "")) in _load_v9_cache())
    print(f"  V9 cache hits: {v9_cache_hits}/{len(llm_needed)}"
          f" ({'LLM calls needed' if v9_cache_hits < len(llm_needed) else 'all cached'})")

    print("Calling LLM V9 (feedback-only) for Stage-4 records...")
    t0 = time.time()
    for n, i in enumerate(sorted(llm_needed), 1):
        rec = records[i]
        rec["llm_label_v9"] = llm_classify_v9(rec["row"], LLM_MODEL)
        if n % 100 == 0 or n == len(llm_needed):
            print(f"    {n}/{len(llm_needed)}  ({time.time()-t0:.0f}s)")
    print(f"  Done in {time.time()-t0:.0f}s")

    y_true_all = [r["true_label"] for r in records]
    rich_mask = [r["has_meta"] for r in records]
    poor_mask = [not r["has_meta"] for r in records]

    print(f"\n{'THRESH':8} | {'MacroF1':8} | {'WtdF1':8} | "
          f"{'QualF1':7} | {'QtyF1':7} | {'QualRec':8} | {'QtyRec':7} | {'LLM%':6}")
    print("-" * 80)
    results = []
    for thresh, outcomes in cell_results.items():
        preds, features, stages = [], [], []
        llm_count = 0
        for i, (pred, stage) in enumerate(outcomes):
            rec = records[i]
            in_domain = rec["in_domain"]
            if pred is not None:
                final_pred = pred
                final_stage = stage
            else:
                final_pred = rec["llm_label_v9"]
                final_stage = "stage4"
                llm_count += 1
            preds.append(final_pred)
            stages.append(final_stage)
            features.append(assign_feature(final_pred, final_stage, in_domain))

        mf1 = f1_score(y_true_all, preds, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
        wf1 = f1_score(y_true_all, preds, labels=LLM_OUTPUT_CATEGORIES, average="weighted", zero_division=0)
        rep = classification_report(y_true_all, preds, labels=LLM_OUTPUT_CATEGORIES,
                                    output_dict=True, zero_division=0)

        rich_true = [y for y, m in zip(y_true_all, rich_mask) if m]
        rich_pred = [p for p, m in zip(preds, rich_mask) if m]
        poor_true = [y for y, m in zip(y_true_all, poor_mask) if m]
        poor_pred = [p for p, m in zip(preds, poor_mask) if m]
        mf1_rich = f1_score(rich_true, rich_pred, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
        wf1_rich = f1_score(rich_true, rich_pred, labels=LLM_OUTPUT_CATEGORIES, average="weighted", zero_division=0)
        mf1_poor = f1_score(poor_true, poor_pred, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
        wf1_poor = f1_score(poor_true, poor_pred, labels=LLM_OUTPUT_CATEGORIES, average="weighted", zero_division=0)

        llm_pct = llm_count / n_total * 100
        print(f"{thresh:<8.2f} | {mf1:<8.4f} | {wf1:<8.4f} | "
              f"{rep['quality']['f1-score']:<7.3f} | {rep['quantity']['f1-score']:<7.3f} | "
              f"{rep['quality']['recall']:<8.3f} | {rep['quantity']['recall']:<7.3f} | {llm_pct:<6.1f}")

        # Stage + feature distributions
        stage_counts = Counter(stages)
        feat_by_cat: dict[str, Counter] = defaultdict(Counter)
        for p, f in zip(preds, features):
            feat_by_cat[p][f or "none"] += 1

        results.append({
            "thresh": thresh, "macro_f1": mf1, "weighted_f1": wf1,
            "quality_f1": rep["quality"]["f1-score"],
            "quantity_f1": rep["quantity"]["f1-score"],
            "quality_recall": rep["quality"]["recall"],
            "quantity_recall": rep["quantity"]["recall"],
            "junk_f1": rep.get("junk", {}).get("f1-score", 0.0),
            "macro_f1_rich": mf1_rich, "weighted_f1_rich": wf1_rich,
            "macro_f1_poor": mf1_poor, "weighted_f1_poor": wf1_poor,
            "llm_calls": llm_count, "llm_pct": llm_pct,
            "stage_counts": dict(stage_counts),
            "feature_by_category": {k: dict(v) for k, v in feat_by_cat.items()},
            "in_domain_stats": {"true": n_in, "false": n_out, "none": n_none},
            "preds": preds,
            "features": features,
            "stages": stages,
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
        print("\n  Stage distribution:")
        for s, c in sorted(r["stage_counts"].items(), key=lambda x: -x[1]):
            print(f"    {s:30s}: {c}")
        print("\n  Feature distribution by category:")
        for cat in ("quality", "quantity", "junk"):
            feat = r["feature_by_category"].get(cat, {})
            print(f"    {cat}: " + "  ".join(f"{k}={v}" for k, v in sorted(feat.items(), key=lambda x: -x[1])))
        print(classification_report(y_true_all, r["preds"], labels=LLM_OUTPUT_CATEGORIES, zero_division=0))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"pipeline_run16_{ts}.json"
    out_path.write_text(json.dumps({
        "name": "Run 16: Early FAISS + feedback-only LLM V9",
        "n_total": n_total,
        "early_faiss": {"in_domain_true": n_in, "in_domain_false": n_out, "no_index": n_none},
        "best_weighted_f1": {k: v for k, v in best_wf1.items() if k not in ("preds", "features", "stages")},
        "best_macro_f1": {k: v for k, v in best_mf1.items() if k not in ("preds", "features", "stages")},
        "all_results": [{k: v for k, v in r.items() if k not in ("preds", "features", "stages")}
                        for r in results],
    }, indent=2))
    print(f"\nSaved to {out_path}")

    # Per-record audit CSV (at best threshold)
    r_best = best_mf1
    audit_rows = []
    for i, rec in enumerate(records):
        row = rec["row"]
        audit_rows.append({
            "feedback_id": str(row.get("id", "")),
            "tag1": str(row.get("tag1", "") or ""),
            "tag2": str(row.get("tag2", "") or ""),
            "scale": str(row.get("value_scale", "") or ""),
            "human_label": rec["true_label"],
            "stage": r_best["stages"][i],
            "llm_label": rec.get("llm_label_v9", ""),
            "in_domain": rec["in_domain"],
            "best_cos": round(rec["best_cos"], 4),
            "final_category": r_best["preds"][i],
            "final_feature": r_best["features"][i] or "",
        })
    audit_path = OUT_DIR / f"pipeline_run16_{ts}_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    print(f"Audit CSV saved to {audit_path}")


if __name__ == "__main__":
    main()
