#!/usr/bin/env python3
"""Build a tag-pair/chain stratified subsample of the existing hand-labelled gold pool.

The full pool (data/labelled/pure_others_to_label.csv, N=2,206) is concentrated:
chain 8453 alone accounts for ~68% of records, and a handful of tag-pairs (e.g.
"tip"/"agent" with 284 records) dominate the rest. This script does NOT draw new
records or request new human labels -- every record here already has a gold
human_label from the original annotation pass. It caps each (chain_id, tag1,
tag2) cluster at --cap records (seeded random choice within the cluster) so no
single chain or tag-pair can dominate the evaluation, producing a more
representative subsample of the same gold pool for a robustness check against
the full-pool numbers reported in Chapter 6 (tab:bench-bge).

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.stratified_resample_gold \\
        --gold data/labelled/pure_others_to_label.csv \\
        --cap 5 --seed 42 \\
        --out data/labelled/pure_others_stratified_cap5.csv
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def cluster_key(row: pd.Series) -> tuple[str, str, str]:
    chain_id = str(row["feedback_id"]).split(":")[0]
    tag1 = str(row.get("tag1") or "").strip().lower()
    tag2 = str(row.get("tag2") or "").strip().lower()
    return (chain_id, tag1, tag2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=5, help="max records kept per (chain, tag1, tag2) cluster")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.gold)
    df["_cluster"] = df.apply(cluster_key, axis=1)

    rng = random.Random(args.seed)
    kept_idx: list[int] = []
    for _, group in df.groupby("_cluster"):
        idx = list(group.index)
        if len(idx) <= args.cap:
            kept_idx.extend(idx)
        else:
            kept_idx.extend(rng.sample(idx, args.cap))

    out = df.loc[sorted(kept_idx)].drop(columns=["_cluster"])
    n_clusters = df["_cluster"].nunique()
    print(f"Source pool: N={len(df)}, {n_clusters} (chain, tag1, tag2) clusters")
    print(f"Stratified subsample (cap={args.cap}): N={len(out)} ({len(out) / len(df) * 100:.1f}% of pool)")
    print(out["human_label"].value_counts())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
