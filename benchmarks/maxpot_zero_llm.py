#!/usr/bin/env python3
"""Best-possible ZERO-LLM pipeline: the cascade's ML tiers, forced to decide.

The production cascade reaches 2-class Macro-F1 0.814 but spends the LLM on
~38% of records. This strips the LLM out and forces every escalation to a
deterministic decision, to answer: how much of the cascade's score comes from
its cheap ML tiers, and how much from the LLM?

Pipeline (no LLM, on the unified stratified_dedup gold — no Mongo needed, the
gold already carries agent_key + agent metadata):
  Stage 0.5  empty tags        -> scale rule (unbounded->junk, else quality)
  Stage 2    per-tag BGE-SVM    -> vote(tau); confident quality -> quality
  Stage 3    agent-domain cos   -> in-domain (cos>0.55) -> scale label
  Fallback   (would be LLM)     -> scale convention (unbounded->quantity, else quality)

Same gold + 2-class Macro-F1 (junk excluded) as the rest of the comparison.

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.maxpot_zero_llm \
        --gold data/labelled/pure_others_stratified_dedup.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.pipeline_3tier_v2 import LLM_MODEL, llm_classify, load_gold
from benchmarks.per_tag_svm import vote_per_tag
from benchmarks.stage3_domain import _load_index, _load_model, _scale_to_label
from shared.types import LLM_OUTPUT_CATEGORIES

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits" / "agent_enriched"
OUT_DIR = ROOT / "data" / "benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CLASSES = ["junk", "quality", "quantity"]
THRESH_IN_DOMAIN = 0.55


def _norm(x) -> str:
    return str(x or "").strip().lower()


def two_class_macro_f1(y_true, y_pred) -> tuple[float, dict]:
    rep = classification_report(y_true, y_pred, labels=CLASSES,
                                output_dict=True, zero_division=0)
    return (rep["quality"]["f1-score"] + rep["quantity"]["f1-score"]) / 2.0, rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--use-llm", action="store_true",
                    help="escalate the residual to the real LLM (cascade+LLM) instead of "
                         "the scale-convention fallback (zero-LLM); runs tau=0.70 only")
    args = ap.parse_args()
    np.random.seed(SEED)

    # ── train pool (quality-vs-non-quality, junk excluded like the cascade) ──
    ga = pd.read_parquet(SPLITS / "group_a.parquet")
    gb = pd.read_parquet(SPLITS / "group_b.parquet")
    tr = pd.concat([ga, gb], ignore_index=True)
    tr = tr[tr["label"] != "junk"].reset_index(drop=True)

    gold = load_gold(args.gold)
    gold = gold[gold["label"].isin(LLM_OUTPUT_CATEGORIES)].reset_index(drop=True)
    y_true = gold["label"].tolist()
    print(f"Train N={len(tr)} (junk excluded)  Gold N={len(gold)}")
    print(f"  gold dist: {gold['label'].value_counts().to_dict()}")

    enc = _load_model()
    index, key_to_pos = _load_index()

    # ── batch-encode every unique "{tag} {scale}" (SVM) and bare tag (cosine) ──
    svm_texts, bare_tags = set(), set()

    def add_row(row, scale_key):
        sc = _norm(row.get(scale_key))
        for tc in ("tag1", "tag2"):
            t = _norm(row.get(tc))
            if t:
                svm_texts.add(f"{t} {sc}")
                bare_tags.add(t)

    for _, r in tr.iterrows():
        add_row(r, "value_scale")
    for _, r in gold.iterrows():
        add_row(r, "value_scale")

    to_encode = sorted(svm_texts | bare_tags)
    vecs = np.asarray(enc.encode(to_encode, normalize_embeddings=True,
                                 show_progress_bar=False), dtype="float32")
    cache = {t: v for t, v in zip(to_encode, vecs)}

    # ── train calibrated per-tag BGE-SVM quality gate ──
    X, y = [], []
    for _, r in tr.iterrows():
        sc = _norm(r.get("value_scale"))
        binv = 1 if r["label"] == "quality" else 0
        for tc in ("tag1", "tag2"):
            t = _norm(r.get(tc))
            if t:
                X.append(cache[f"{t} {sc}"])
                y.append(binv)
    clf = CalibratedClassifierCV(LinearSVC(C=0.3, max_iter=2000, random_state=SEED),
                                 cv=3, method="sigmoid")
    clf.fit(np.array(X), np.array(y))
    q_idx = list(clf.classes_).index(1)

    def qprob(tag: str, scale: str) -> float:
        return float(clf.predict_proba([cache[f"{tag} {scale}"]])[0][q_idx])

    # ── precompute per-record signals (SVM probs + best cosine) ──
    n_indexed = 0
    recs = []
    for _, row in gold.iterrows():
        t1, t2 = _norm(row.get("tag1")), _norm(row.get("tag2"))
        sc = _norm(row.get("value_scale"))
        akey = str(row.get("agent_key") or "")
        pos = key_to_pos.get(akey)
        best_cos = None
        if pos is not None:
            tags = [t for t in (t1, t2) if t]
            if tags:
                av = index.reconstruct(pos)
                best_cos = max(float(np.dot(cache[t], av)) for t in tags)
                n_indexed += 1
        recs.append({
            "t1": t1, "t2": t2, "scale": sc,
            "p1": qprob(t1, sc) if t1 else 0.5,
            "p2": qprob(t2, sc) if t2 else 0.5,
            "t2_empty": not t2,
            "best_cos": best_cos,
            "row": row,
        })
    print(f"  agents found in FAISS index: {n_indexed}/{len(gold)} "
          f"({n_indexed / len(gold) * 100:.1f}%)  [Stage-3 cosine coverage]\n")

    # ── sweep the vote threshold; residual -> scale convention (zero-LLM) or LLM ──
    mode = "cascade+LLM" if args.use_llm else "zero-LLM"
    taus = [0.70] if args.use_llm else [0.60, 0.65, 0.70, 0.75, 0.80]
    print(f"mode={mode}")
    print(f"{'vote_tau':>9} {'2cls-MacroF1':>12} {'quality-F1':>11} {'quantity-F1':>12} {'qty-recall':>11} {'LLM%':>7}")
    print("-" * 68)
    results = {}
    last_preds = None
    for tau in taus:
        preds, llm_calls = [], 0
        for r in recs:
            if not r["t1"] and not r["t2"]:                    # Stage 0.5
                preds.append("junk" if r["scale"] == "unbounded" else "quality")
                continue
            vote = vote_per_tag(r["p1"], r["p2"], t2_empty=r["t2_empty"], thresh=tau)
            if vote == "quality":                              # Stage 2
                preds.append("quality")
                continue
            if r["best_cos"] is not None and r["best_cos"] > THRESH_IN_DOMAIN:  # Stage 3 in-domain
                if r["scale"] == "unbounded":                  # unbounded in-domain -> quantity (rule)
                    preds.append("quantity")
                    continue
                # bounded in-domain: thesis cascade escalates these to the LLM (mandatory);
                # the zero-LLM variant has no LLM, so it falls back to the scale convention.
                if args.use_llm:
                    preds.append(llm_classify(r["row"], LLM_MODEL))
                    llm_calls += 1
                else:
                    preds.append(_scale_to_label(r["scale"]))  # = "quality"
                continue
            if args.use_llm:                                   # residual (out-of-domain / no meta) -> LLM
                preds.append(llm_classify(r["row"], LLM_MODEL))
                llm_calls += 1
            else:                                              # residual -> scale convention
                preds.append(_scale_to_label(r["scale"]))
        two, rep = two_class_macro_f1(y_true, preds)
        last_preds = preds
        llm_pct = llm_calls / len(recs) * 100
        tag = "  <- production knee" if abs(tau - 0.70) < 1e-9 else ""
        print(f"{tau:>9.2f} {two:>12.4f} {rep['quality']['f1-score']:>11.4f} "
              f"{rep['quantity']['f1-score']:>12.4f} {rep['quantity']['recall']:>11.4f} "
              f"{llm_pct:>6.1f}%{tag}")
        results[f"{tau:.2f}"] = {"macro_f1_2cls": round(two, 4),
                                 "quality_f1": round(rep["quality"]["f1-score"], 4),
                                 "quantity_f1": round(rep["quantity"]["f1-score"], 4),
                                 "quantity_recall": round(rep["quantity"]["recall"], 4),
                                 "llm_pct": round(llm_pct, 1), "llm_calls": llm_calls}

    if last_preds is not None:
        cm = confusion_matrix(y_true, last_preds, labels=CLASSES)  # rows=true, cols=pred
        qi, qly = CLASSES.index("quantity"), CLASSES.index("quality")
        n_qty = sum(1 for y in y_true if y == "quantity")
        q2q = int(cm[qi][qly])
        print(f"\nCONFUSION (last tau={taus[-1]}): quantity->quality = {q2q}/{n_qty} "
              f"({q2q / n_qty * 100:.1f}% of quantity read as quality)")
        print(f"  full CM rows=true[{'/'.join(CLASSES)}] cols=pred:\n{cm}")

    best = max(results, key=lambda k: results[k]["macro_f1_2cls"])
    print(f"\nBest {mode}: tau={best}  2cls-MacroF1={results[best]['macro_f1_2cls']}  "
          f"LLM={results[best]['llm_pct']}%")
    print("Reference (unified gold): NB 0.724 | LLM-only 0.810 @100%")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT_DIR / f"maxpot_{'cascade_llm' if args.use_llm else 'zero_llm'}_{ts}.json"
    p.write_text(json.dumps({"pipeline": f"cascade-tiers-{mode}", "n_train": len(tr),
                             "n_gold": len(gold), "faiss_coverage": n_indexed,
                             "seed": SEED, "by_tau": results, "best_tau": best}, indent=2))
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
