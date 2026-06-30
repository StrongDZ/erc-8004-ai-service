#!/usr/bin/env python3
"""Clean Group-A / Group-B baseline benchmark.

Fixes the feature-engineering defects in comprehensive_bench_gold.py:
  * NO `endpoint` token. endpoint is populated in 48.6% of the gold (test) rows
    but the group_a+b training parquet has no endpoint column at all, so the
    token appears at test time but never in training -- a train/test mismatch.
  * NO `offchain` token. The training parquet has no feedback_parsed/offchain
    column, and the gold has a non-empty offchain_note in only 1/1486 rows, so
    the feature is dead and the feedback_parsed-vs-offchain_note field swap is
    moot. Removing it matches the documented methodology.

The feature is exactly the documented tag1 (+) tag2 (+) scale string.

Group A: word-level TF-IDF (1-2 grams), thesis hyper-parameters.
Group B: frozen BAAI/bge-small-en-v1.5 embeddings -> Logistic Regression + kNN.
         kNN neighbours are retrieved from the TRAINING pool (not the gold set).
Train: group_a + group_b (N=1032).  Test: audited gold (N=1486).

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.clean_baselines_ab \\
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.stage3_domain import _load_model
from shared.types import LLM_OUTPUT_CATEGORIES, RULE_TO_CAT

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits" / "agent_enriched"
OUT_DIR = ROOT / "data" / "benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def feature_text(row: pd.Series) -> str:
    """Documented clean feature: tag1 + tag2 + scale only."""
    t1 = str(row.get("tag1", "") or "").strip().lower()
    t2 = str(row.get("tag2", "") or "").strip().lower()
    sc = str(row.get("value_scale", "") or "").strip().lower()
    return " ".join(p for p in (t1, t2, sc) if p) or "<empty>"


def score(y_true: list[str], y_pred) -> dict:
    rep = classification_report(y_true, y_pred, labels=LLM_OUTPUT_CATEGORIES,
                                output_dict=True, zero_division=0)
    two = (rep["quality"]["f1-score"] + rep["quantity"]["f1-score"]) / 2
    wf1 = f1_score(y_true, y_pred, labels=LLM_OUTPUT_CATEGORIES,
                   average="weighted", zero_division=0)
    return {
        "macro_f1_2cls": two,
        "weighted_f1": wf1,
        "per_class": {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"],
                          "f1": rep[c]["f1-score"], "support": rep[c]["support"]}
                      for c in LLM_OUTPUT_CATEGORIES},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, required=True)
    args = ap.parse_args()

    # Training pool: group_a + group_b (agent-enriched, N=1032), same corpus the
    # cascade / D / E train on.
    ga = pd.read_parquet(SPLITS / "group_a.parquet")
    gb = pd.read_parquet(SPLITS / "group_b.parquet")
    tr = pd.concat([ga, gb], ignore_index=True)

    # Gold (audited): rename to a common schema and keep the three real classes.
    g = pd.read_csv(args.gold).fillna("")
    g = g.rename(columns={"human_label": "label", "scale": "value_scale"})
    g["label"] = g["label"].str.strip().str.lower().map(lambda x: RULE_TO_CAT.get(x, x))
    g = g[g["label"].isin(LLM_OUTPUT_CATEGORIES)].reset_index(drop=True)

    Xtr_text = tr.apply(feature_text, axis=1).tolist()
    ytr = tr["label"].tolist()
    Xte_text = g.apply(feature_text, axis=1).tolist()
    yte = g["label"].tolist()
    print(f"Train N={len(tr)}  Gold N={len(g)}  feature = tag1 + tag2 + scale (clean)")
    print(f"  train dist: {pd.Series(ytr).value_counts().to_dict()}")
    print(f"  gold  dist: {pd.Series(yte).value_counts().to_dict()}\n")

    results: dict[str, dict] = {}

    # ── Group A: word-level TF-IDF (1-2 grams) ──────────────────────────────
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    Xtr = vec.fit_transform(Xtr_text)
    Xte = vec.transform(Xte_text)
    # Plain classifiers with the documented hyper-parameters. The training pool
    # is already near-balanced (44/43/13), so class_weight=balanced gives no
    # gain on the linear models and degrades the bge-small logistic head; plain
    # configs are the simpler, stronger baseline here.
    group_a = {
        "A1_logreg_tfidf": LogisticRegression(C=1.0, max_iter=3000),
        "A2_linsvm_tfidf": LinearSVC(C=1.0),
        "A3_naivebayes_tfidf": MultinomialNB(alpha=1.0),
        "A4_gbt_tfidf": GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
        "A5_rf_tfidf": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    print("Group A (TF-IDF, tag+scale):")
    for name, clf in group_a.items():
        clf.fit(Xtr, ytr)
        results[name] = score(yte, clf.predict(Xte))
        print(f"  {name:22} 2cls={results[name]['macro_f1_2cls']:.4f}  wf1={results[name]['weighted_f1']:.4f}")

    # ── Group B: frozen bge-small-en-v1.5 ───────────────────────────────────
    print("\nGroup B (frozen bge-small, tag+scale):")
    enc = _load_model()
    Etr = np.asarray(enc.encode(Xtr_text, normalize_embeddings=True, show_progress_bar=False), dtype="float32")
    Ete = np.asarray(enc.encode(Xte_text, normalize_embeddings=True, show_progress_bar=False), dtype="float32")
    group_b = {
        "B1_bge_logreg": LogisticRegression(C=1.0, max_iter=3000),
        "B2_bge_knn_k5": KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance"),
    }
    for name, clf in group_b.items():
        clf.fit(Etr, ytr)
        results[name] = score(yte, clf.predict(Ete))
        print(f"  {name:22} 2cls={results[name]['macro_f1_2cls']:.4f}  wf1={results[name]['weighted_f1']:.4f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"clean_baselines_ab_{ts}.json"
    out.write_text(json.dumps({
        "feature": "tag1+tag2+scale (no endpoint, no offchain)",
        "n_train": len(tr), "n_gold": len(g),
        "results": results,
    }, indent=2))
    print(f"\nSaved {out}")
    print("\n=== SUMMARY (2-cls Macro F1 | Weighted F1) ===")
    for name, r in results.items():
        print(f"  {name:22} {r['macro_f1_2cls']:.3f} | {r['weighted_f1']:.3f}")


if __name__ == "__main__":
    main()
