#!/usr/bin/env python3
"""Build a stratified subsample of the existing hand-labelled gold pool.

Cluster key: (tag1, tag2, scale, agent_key) — the four factors that determine
classifier output. tag1/tag2/scale drive the rule-based layers; agent_key fixes
the offchain content seen by the LLM fallback. Two records that share all four
values will always receive the same classification, so keeping more than one per
cluster adds no new evaluation signal. The default cap is therefore 1
(deduplication), producing the smallest representative subsample.

This script does NOT draw new records or request new human labels — every record
here already has a gold human_label from the original annotation pass.

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.stratified_resample_gold \\
        --gold data/labelled/pure_others_to_label.csv \\
        --cap 1 --seed 42 \\
        --out data/labelled/pure_others_stratified_dedup.csv
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def cluster_key(row: pd.Series) -> tuple[str, str, str, str]:
    tag1  = str(row.get("tag1")      or "").strip().lower()
    tag2  = str(row.get("tag2")      or "").strip().lower()
    scale = str(row.get("scale")     or "").strip().lower()
    agent = str(row.get("agent_key") or "").strip().lower()
    return (tag1, tag2, scale, agent)


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
    print(f"Source pool: N={len(df)}, {n_clusters} (tag1, tag2, scale, agent_key) clusters")
    print(f"Stratified subsample (cap={args.cap}): N={len(out)} ({len(out) / len(df) * 100:.1f}% of pool)")
    print(out["human_label"].value_counts())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
