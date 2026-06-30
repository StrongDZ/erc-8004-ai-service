#!/usr/bin/env python3
"""Benchmark: LLM-only classification on the hand-labelled gold set.

Uses the existing llm_cache.json for speed (1483/1486 cache hits expected).
The 3 cache misses attempt a live Ollama call; records that still fail are
logged and excluded from metrics (counted separately).

Produces a JSON result in data/benchmark_results/ comparable to pipeline_run13.

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.bench_llm_only \\
        --gold data/labelled/pure_others_stratified_dedup.csv
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

from benchmarks.pipeline_3tier_v2 import LLM_MODEL, load_gold, llm_classify
from shared.types import LLM_OUTPUT_CATEGORIES

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--model", default=LLM_MODEL)
    args = parser.parse_args()

    print("Loading gold set...")
    gold = load_gold(args.gold)
    gold = gold[gold["label"].isin(LLM_OUTPUT_CATEGORIES)].reset_index(drop=True)
    print(f"Gold N={len(gold)}")
    print(gold["label"].value_counts().to_string())

    print(f"\nRunning LLM-only classification (model={args.model}) — uses cache where available...")
    y_true, y_pred = [], []
    n_cached, n_live, n_failed = 0, 0, 0

    cache_path = ROOT / "data/benchmark_results/llm_cache.json"
    cache: dict[str, str] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    t0 = time.monotonic()
    for _, row in gold.iterrows():
        fb_id = str(row.get("id", ""))
        is_cached = fb_id in cache
        try:
            pred = llm_classify(row, args.model)
            pred = pred.strip().lower()
            if pred not in LLM_OUTPUT_CATEGORIES:
                pred = "quality"
            if is_cached:
                n_cached += 1
            else:
                n_live += 1
            y_true.append(str(row["label"]).strip().lower())
            y_pred.append(pred)
        except Exception as exc:
            n_failed += 1
            print(f"  FAILED id={fb_id}: {exc}")

    elapsed = time.monotonic() - t0

    print(f"\nCached: {n_cached}  Live LLM: {n_live}  Failed/skipped: {n_failed}")
    print(f"Evaluated: {len(y_true)} records in {elapsed:.1f}s")

    report_str = classification_report(y_true, y_pred, labels=LLM_OUTPUT_CATEGORIES, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=LLM_OUTPUT_CATEGORIES, average="weighted", zero_division=0)

    print("\n=== LLM-Only Benchmark (qwen2.5:7b-instruct) ===")
    print(report_str)
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")

    per_class = classification_report(
        y_true, y_pred, labels=LLM_OUTPUT_CATEGORIES, zero_division=0, output_dict=True,
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "model": f"LLM-only ({args.model})",
        "gold_csv": str(args.gold),
        "n_gold": len(y_true),
        "n_cached": n_cached,
        "n_live": n_live,
        "n_failed": n_failed,
        "timestamp": ts,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "elapsed_s": round(elapsed, 1),
    }
    out_path = OUT_DIR / f"llm_only_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
