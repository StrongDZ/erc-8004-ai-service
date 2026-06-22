"""Sync gold_combined_clean.csv scale + category against current MongoDB state.

Steps:
1. Re-fetch valueScale and category from feedback_history for every feedback_id.
2. Update the 'scale' column where MongoDB differs.
3. Use MongoDB's production 'category' field as ground truth (reflects the Go
   rule engine + LLM decisions that are live in production).  Python
   convention_label.classify() is used ONLY as a fallback when MongoDB has
   'others' or the record is missing.
4. Report diff summary and save gold_combined_clean_v2.csv.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.convention_label import classify as convention_classify  # noqa: E402

load_dotenv()


def _safe_float(v) -> float | None:
    try:
        return float(str(v).strip())
    except (ValueError, TypeError):
        return None


def main():
    gold_path = ROOT / "data/labelled/gold_combined_clean.csv"
    out_path = ROOT / "data/labelled/gold_combined_clean_v2.csv"

    df = pd.read_csv(gold_path)
    print(f"Loaded {len(df)} records from {gold_path.name}")
    print(f"Original scale dist:\n{df['scale'].value_counts()}\n")

    # Connect to MongoDB
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("MONGO_DATABASE_ANALYZED_AGENTS")]
    col = db["feedback_history"]

    all_ids = df["feedback_id"].tolist()
    print(f"Fetching {len(all_ids)} records from MongoDB...")
    docs = {
        d["_id"]: d
        for d in col.find(
            {"_id": {"$in": all_ids}},
            {"valueScale": 1, "value": 1, "tag1": 1, "tag2": 1, "category": 1},
        )
    }
    print(f"Found {len(docs)} / {len(all_ids)} in MongoDB\n")

    scale_changes: list[dict] = []
    label_changes: list[dict] = []
    not_found = 0

    new_scales = []
    new_categories = []

    for _, row in df.iterrows():
        fid = row["feedback_id"]
        doc = docs.get(fid)

        if doc is None:
            not_found += 1
            new_scales.append(row["scale"])
            new_categories.append(row["category"])
            continue

        # Scale from MongoDB (authoritative current state)
        mongo_scale = str(doc.get("valueScale") or "").strip()
        csv_scale = str(row["scale"]).strip()
        scale = mongo_scale if mongo_scale else csv_scale

        if scale != csv_scale:
            scale_changes.append({
                "feedback_id": fid,
                "tag1": row["tag1"],
                "tag2": row["tag2"],
                "old_scale": csv_scale,
                "new_scale": scale,
            })

        # Re-derive value from MongoDB (in case CSV value is also stale)
        mongo_val = _safe_float(doc.get("value"))
        csv_val = _safe_float(row.get("value"))
        value = mongo_val if mongo_val is not None else csv_val

        # Category: use MongoDB's production value as ground truth.
        # Fall back to convention_classify only when MongoDB has 'others'
        # (not yet resolved by LLM) or the field is missing.
        old_cat = str(row["category"]).strip()
        mongo_cat = str(doc.get("category") or "").strip().lower()
        if mongo_cat and mongo_cat != "others":
            new_cat = mongo_cat
        else:
            result = convention_classify(
                tag1=str(row["tag1"] or ""),
                tag2=str(row["tag2"] or ""),
                scale=scale,
                is_self=False,
                value=value,
            )
            new_cat = result[0] if isinstance(result, tuple) else result

        if new_cat != old_cat:
            label_changes.append({
                "feedback_id": fid,
                "tag1": row["tag1"],
                "tag2": row["tag2"],
                "scale": scale,
                "value": value,
                "old_cat": old_cat,
                "new_cat": new_cat,
            })

        new_scales.append(scale)
        new_categories.append(new_cat)

    df["scale"] = new_scales
    df["category"] = new_categories

    print(f"Scale changes:   {len(scale_changes)}")
    print(f"Label changes:   {len(label_changes)}")
    print(f"Not found in DB: {not_found}\n")

    if scale_changes:
        print("=== Scale changes (first 20) ===")
        for c in scale_changes[:20]:
            print(f"  {c['tag1']!r}/{c['tag2']!r}: {c['old_scale']} → {c['new_scale']}")
        print()

    if label_changes:
        print("=== Label changes (first 20) ===")
        for c in label_changes[:20]:
            print(f"  {c['tag1']!r}/{c['tag2']!r} scale={c['scale']!r} val={c['value']}: {c['old_cat']} → {c['new_cat']}")
        print()

    print(f"New category dist:\n{df['category'].value_counts()}\n")
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path.name}")


if __name__ == "__main__":
    main()
