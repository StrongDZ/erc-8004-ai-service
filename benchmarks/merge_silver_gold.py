#!/usr/bin/env python3
"""Merge Opus-labeled silver records with the existing gold CSV.

The silver CSV must have `opus_label` column filled. This script:
  1. Validates that opus_label ∈ {quality, quantity, junk}
  2. Copies opus_label → category (gold-compatible column)
  3. Concatenates with gold_final.csv (deduplicating on feedback_id)
  4. Writes gold_combined.csv in the same format as gold_final.csv

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.merge_silver_gold \\
        --silver data/labelled/others_to_label.csv \\
        --gold   ../erc-8004-benchmarking-be/scripts/labelled/gold_final.csv \\
        --out    data/labelled/gold_combined.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

VALID_LABELS = {"quality", "quantity", "junk"}

# Columns that must appear in the final combined CSV (gold_final.csv format)
GOLD_COLUMNS = ["feedback_id", "tag1", "tag2", "value", "value_decimals",
                "scale", "category", "feature", "agent_domains", "source"]


def merge(silver_path: Path, gold_path: Path, out_path: Path, min_labeled: int = 1) -> None:
    # Load silver
    silver = pd.read_csv(silver_path).fillna("")
    print(f"Silver loaded: {len(silver)} rows")

    # Only keep rows where Opus actually labeled
    labeled = silver[silver["opus_label"].str.strip().str.lower().isin(VALID_LABELS)].copy()
    unlabeled = silver[~silver["opus_label"].str.strip().str.lower().isin(VALID_LABELS)]
    print(f"  Labeled   : {len(labeled)}")
    print(f"  Unlabeled : {len(unlabeled)} (will be EXCLUDED)")

    if len(labeled) < min_labeled:
        raise SystemExit(f"Only {len(labeled)} labeled rows — run labeling first")

    # Copy opus_label → category
    labeled = labeled.copy()
    labeled["category"] = labeled["opus_label"].str.strip().str.lower()

    # Show distribution
    dist = labeled["category"].value_counts()
    print(f"\nSilver label distribution:")
    for cat, n in dist.items():
        print(f"  {cat:10s}: {n:4d} ({n/len(labeled)*100:.1f}%)")

    # Keep only gold-compatible columns (fill missing with "")
    for col in GOLD_COLUMNS:
        if col not in labeled.columns:
            labeled[col] = ""
    silver_final = labeled[GOLD_COLUMNS].copy()

    # Load gold
    gold = pd.read_csv(gold_path).fillna("")
    print(f"\nGold loaded: {len(gold)} rows")
    gold_dist = gold["category"].value_counts()
    for cat, n in gold_dist.items():
        print(f"  {cat:10s}: {n:4d} ({n/len(gold)*100:.1f}%)")

    # Dedup: gold wins over silver for any overlapping feedback_id
    existing_ids = set(gold["feedback_id"].astype(str))
    silver_new = silver_final[~silver_final["feedback_id"].astype(str).isin(existing_ids)]
    duplicates = len(silver_final) - len(silver_new)
    if duplicates:
        print(f"\n[warn] {duplicates} silver rows already in gold — dropped")

    # Combine
    combined = pd.concat([gold, silver_new], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    print(f"\nCombined: {len(gold)} gold + {len(silver_new)} silver = {len(combined)} total")
    combined_dist = combined["category"].value_counts()
    print("Combined distribution:")
    for cat, n in combined_dist.items():
        print(f"  {cat:10s}: {n:4d} ({n/len(combined)*100:.1f}%)")
    print(f"\nSaved → {out_path}")


def main() -> None:
    ROOT = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver", type=Path,
                        default=ROOT / "data/labelled/others_to_label.csv")
    parser.add_argument("--gold", type=Path,
                        default=ROOT.parent / "erc-8004-benchmarking-be/scripts/labelled/gold_final.csv")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "data/labelled/gold_combined.csv")
    args = parser.parse_args()
    merge(args.silver, args.gold, args.out)


if __name__ == "__main__":
    main()
