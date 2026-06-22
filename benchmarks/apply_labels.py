#!/usr/bin/env python3
"""Apply per-feedback batch label files into others_to_label.csv.

Each batch file in data/labelled/label_batches/ is a JSON object mapping
feedback_id -> [label, reason].  This script ingests every *.json batch and
writes opus_label / opus_reason into the records CSV, matching on feedback_id.

Idempotent: re-running re-applies all batches from scratch each time.

Usage:
    .venv/bin/python3 -m benchmarks.apply_labels
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

VALID_LABELS = {"quality", "quantity", "junk"}


def apply(rec_path: Path, batch_dir: Path) -> None:
    df = pd.read_csv(rec_path).fillna("")
    df["feedback_id"] = df["feedback_id"].astype(str)
    if "opus_label" not in df.columns:
        df["opus_label"] = ""
    if "opus_reason" not in df.columns:
        df["opus_reason"] = ""

    labels: dict[str, list] = {}
    batch_files = sorted(batch_dir.glob("*.json"))
    for bf in batch_files:
        data = json.loads(bf.read_text())
        for fid, payload in data.items():
            if isinstance(payload, list):
                label, reason = payload[0], (payload[1] if len(payload) > 1 else "")
            elif isinstance(payload, dict):
                label, reason = payload.get("label", ""), payload.get("reason", "")
            else:
                label, reason = str(payload), ""
            labels[str(fid)] = [str(label).strip().lower(), reason]
    print(f"Ingested {len(labels)} labels from {len(batch_files)} batch files")

    bad = {k: v for k, v in labels.items() if v[0] not in VALID_LABELS}
    if bad:
        print(f"[warn] {len(bad)} labels have invalid category (ignored): "
              f"{list(bad.items())[:5]}")

    valid_ids = set(df["feedback_id"])
    unknown = [fid for fid in labels if fid not in valid_ids]
    if unknown:
        print(f"[warn] {len(unknown)} feedback_ids in batches not found in CSV: {unknown[:5]}")

    def _set(row, idx):
        entry = labels.get(row["feedback_id"])
        if entry and entry[0] in VALID_LABELS:
            return entry[idx]
        return row["opus_label"] if idx == 0 else row["opus_reason"]

    df["opus_label"] = df.apply(lambda r: _set(r, 0), axis=1)
    df["opus_reason"] = df.apply(lambda r: _set(r, 1), axis=1)

    done = df["opus_label"].str.lower().isin(VALID_LABELS)
    print(f"Records labeled: {done.sum()}/{len(df)} ({(~done).sum()} remaining)")
    print("\nRecord-level label distribution:")
    print(df.loc[done, "opus_label"].value_counts().to_string())

    df.to_csv(rec_path, index=False)
    print(f"\nUpdated → {rec_path}")


def main() -> None:
    ROOT = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path,
                        default=ROOT / "data/labelled/others_to_label.csv")
    parser.add_argument("--batch-dir", type=Path,
                        default=ROOT / "data/labelled/label_batches")
    args = parser.parse_args()
    args.batch_dir.mkdir(parents=True, exist_ok=True)
    apply(args.records, args.batch_dir)


if __name__ == "__main__":
    main()
