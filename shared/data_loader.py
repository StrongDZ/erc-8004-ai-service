"""Stratified sampling + train/val/test splits.

Pulls labelled feedback_history records, applies the rule-label remap
(spam + noise → junk), and returns balanced DataFrames.

Augments with hand-labelled `others_gold_v1.csv` so the "others" pool gets
human-verified labels rather than just the rule's noisy fallback.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import pandas as pd

from .mongo_client import feedback_coll
from .types import LLM_OUTPUT_CATEGORIES, MONGO_CATEGORY_ALIASES, RULE_TO_5CAT


def _record_to_row(doc: dict) -> dict:
    rule = doc.get("classification", {}).get("rule", {}).get("category", "others")
    return {
        "id": doc["_id"],
        "agent_id": doc.get("agentId", ""),
        "chain_id": doc.get("chainId", 0),
        "tag1": doc.get("tag1", "") or "",
        "tag2": doc.get("tag2", "") or "",
        "endpoint": doc.get("endpoint", "") or "",
        "value": str(doc.get("value", "")),
        "value_decimals": int(doc.get("valueDecimals", 0) or 0),
        "value_scale": doc.get("valueScale", "") or "",
        "feedback_parsed": doc.get("feedbackParsed"),
        "rule_category": RULE_TO_5CAT.get(rule, "others"),
        "is_self_feedback": bool(doc.get("isSelfFeedback", False)),
    }


def stratified_sample(
    per_category: int = 2000,
    seed: int = 42,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """Sample `per_category` records per rule label using $sample aggregation.

    `noise` (33 records) gets all available rows; small categories are not
    upsampled at this stage — handle imbalance in the classifier (class_weight)
    or in a later augmentation pass.
    """
    if categories is None:
        categories = list(MONGO_CATEGORY_ALIASES.keys())

    rng = random.Random(seed)
    frames = []
    coll = feedback_coll()
    for cat in categories:
        aliases = MONGO_CATEGORY_ALIASES.get(cat, [cat])
        cursor = coll.aggregate([
            {"$match": {"classification.rule.category": {"$in": aliases}}},
            {"$sample": {"size": per_category}},
        ])
        rows = [_record_to_row(d) for d in cursor]
        rng.shuffle(rows)
        frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def merge_hand_labels(df: pd.DataFrame, gold_csv: Path) -> pd.DataFrame:
    """Overwrite `rule_category` with hand-labelled gold where available.

    The hand-labels in `scripts/labelled/others_gold_v1.csv` are
    authoritative for the "others" pool — replace the noisy rule label with
    the human verdict, mapping spam/noise to junk.
    """
    df = df.copy()
    gold: dict[str, str] = {}
    if gold_csv.exists():
        with gold_csv.open() as f:
            for row in csv.DictReader(f):
                fid = (
                    row.get("feedback_id") or row.get("id") or row.get("_id") or ""
                ).strip()
                gcat = (row.get("gold_category") or "").strip().lower()
                if fid and gcat:
                    gold[fid] = RULE_TO_5CAT.get(gcat, gcat)

    if gold and "rule_category" in df.columns:
        mask = df["id"].isin(gold)
        df.loc[mask, "rule_category"] = df.loc[mask, "id"].map(gold)

    return df.rename(columns={"rule_category": "label"})


def load_hand_labelled_csv(gold_csv: Path) -> pd.DataFrame:
    """Load the manually labelled rule-others CSV as a 4-category benchmark set."""
    raw = pd.read_csv(gold_csv).fillna("")
    raw = raw.rename(columns={
        "feedback_id": "id",
        "value_raw": "value",
        "scale": "value_scale",
        "gold_category": "gold_label",
        "category": "gold_label_from_category",
    })
    if "gold_label" not in raw.columns or raw["gold_label"].astype(str).str.strip().eq("").all():
        if "gold_label_from_category" in raw.columns:
            raw["gold_label"] = raw["gold_label_from_category"]
    raw["label"] = (
        raw["gold_label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: RULE_TO_5CAT.get(x, x))
    )
    raw = raw[raw["label"].isin(LLM_OUTPUT_CATEGORIES)].copy()
    raw["feedback_parsed"] = None
    raw["is_self_feedback"] = False
    raw["value_decimals"] = pd.to_numeric(raw["value_decimals"], errors="coerce").fillna(0).astype(int)
    raw["bench"] = "hand_labelled"

    cols = [
        "id", "chain_id", "agent_id", "tag1", "tag2", "endpoint", "value",
        "value_decimals", "value_scale", "feedback_parsed", "is_self_feedback",
        "label", "rule_category", "agent_description", "agent_services", "agent_tags",
        "bench",
    ]
    for col in cols:
        if col not in raw.columns:
            raw[col] = ""
    return raw[cols].reset_index(drop=True)


def split_train_val_test(
    df: pd.DataFrame,
    label_col: str = "label",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split by `label_col`. Remaining fraction → test."""
    rng = random.Random(seed)
    train_parts, val_parts, test_parts = [], [], []
    for label, group in df.groupby(label_col):
        ids = group.index.tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_parts.append(group.loc[ids[:n_train]])
        val_parts.append(group.loc[ids[n_train:n_train + n_val]])
        test_parts.append(group.loc[ids[n_train + n_val:]])
    return (
        pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True),
        pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True),
        pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True),
    )
