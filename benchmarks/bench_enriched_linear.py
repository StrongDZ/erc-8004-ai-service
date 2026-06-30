#!/usr/bin/env python3
"""Benchmark EnrichedLinearClassifier against the hand-labelled gold set.

EnrichedLinearClassifier trains on rule-labelled Mongo corpus, then predicts via
late-fusion [feedback_vec ‖ agent_vec]. Agent text comes from the precomputed
agent_domain_text column already in the gold CSV, so no Mongo lookup is needed at
evaluation time.

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.bench_enriched_linear \\
        --gold data/labelled/pure_others_stratified_dedup.csv
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

from benchmarks.pipeline_3tier_v2 import load_gold
from shared.knn_classifier import feedback_embed_text
from shared.linear_classifier import EnrichedLinearClassifier
from shared.types import LLM_OUTPUT_CATEGORIES

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument(
        "--per-category", type=int, default=1000,
        help="training records per category sampled from Mongo (default: 1000)",
    )
    args = parser.parse_args()

    print("Loading gold set...")
    gold = load_gold(args.gold)
    gold = gold[gold["label"].isin(LLM_OUTPUT_CATEGORIES)].reset_index(drop=True)
    print(f"Gold N={len(gold)}")
    print(gold["label"].value_counts().to_string())

    print("\nBuilding EnrichedLinearClassifier (trains from live Mongo)...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    clf = EnrichedLinearClassifier(embedder=embedder, per_category=args.per_category)
    clf.build()

    print("\nBatch-encoding gold feedback texts...")
    fb_texts: list[str] = []
    for _, row in gold.iterrows():
        fb_texts.append(feedback_embed_text(
            str(row.get("tag1") or ""),
            str(row.get("tag2") or ""),
            str(row.get("endpoint") or ""),
            str(row.get("offchain_note") or ""),
        ))

    ag_texts: list[str] = [str(row.get("agent_domain_text") or "") for _, row in gold.iterrows()]

    t0 = time.monotonic()
    fb_vecs = embedder.encode(fb_texts, normalize_embeddings=True, show_progress_bar=True)
    ag_vecs = clf._encode_agent(ag_texts, clf._dim)
    X = np.hstack([fb_vecs, ag_vecs])
    y_pred_arr = clf._clf.predict(X)
    elapsed = time.monotonic() - t0

    y_true = gold["label"].str.strip().str.lower().tolist()
    y_pred = y_pred_arr.tolist()

    report_str = classification_report(y_true, y_pred, labels=LLM_OUTPUT_CATEGORIES, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=LLM_OUTPUT_CATEGORIES, average="weighted", zero_division=0)

    print("\n=== EnrichedLinearClassifier — Gold Benchmark ===")
    print(report_str)
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")
    print(f"Inference:    {elapsed:.1f}s for {len(y_true)} records ({elapsed/len(y_true)*1000:.1f}ms/rec)")

    per_class = classification_report(
        y_true, y_pred, labels=LLM_OUTPUT_CATEGORIES, zero_division=0, output_dict=True,
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "model": "EnrichedLinearClassifier",
        "gold_csv": str(args.gold),
        "n_gold": len(y_true),
        "timestamp": ts,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "elapsed_s": round(elapsed, 1),
        "per_category_train": args.per_category,
    }
    out_path = OUT_DIR / f"enriched_linear_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
