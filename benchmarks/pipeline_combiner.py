#!/usr/bin/env python3
"""Run 6: Rule + unified Stage2/3 combiner (multinomial logistic regression),
replacing the hard-gated per-tag-SVM-vote -> FAISS-in-domain-scale-override
cascade (Run 5 in pipeline_3tier_v2.py) with one model over continuous signals.

SVM redesign vs the original Run 6:
  - Trained on tag text only (TF-IDF + BGE hybrid, no scale feature).
  - Junk excluded from SVM training and from combiner training.
  - Output: svm_p1 = P(quality|tag1), svm_p2 = P(quality|tag2) from a
    quality-vs-quantity binary SVM (not quality-vs-non_quality).
  - Combiner output classes: quality and quantity only.
  - Junk items are OOD for the SVM → low max_proba → LLM fallback.
  - cos_metric feature removed; value_decimals added.

Trains the combiner on data/splits/rule_based_diverse_v2/train.parquet (junk
rows filtered out, verified 0 ID overlap with the gold eval set).

Two variants reported:
  Run 6a (--skip-llm): combiner only, no LLM (fully deterministic)
  Run 6b (default):    combiner + LLM fallback when max-proba < 0.50, a fixed
                        a-priori threshold (not tuned against the gold set)

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.pipeline_combiner \
        --gold data/labelled/pure_others_to_label.csv --exclude-self
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.per_tag_svm import predict_qtag_proba, train_qtag_hybrid
from benchmarks.pipeline_3tier import rule_classify
from benchmarks.pipeline_3tier_v2 import LLM_MODEL, enrich_gold_with_agent_meta, llm_classify, load_gold
from benchmarks.stage_combiner import FEATURE_NAMES, build_feature_row, cos_domain
from shared.types import LLM_OUTPUT_CATEGORIES

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data/splits/rule_based_diverse_v2/train.parquet"
OUT_DIR = ROOT / "data/benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOW_CONF_THRESH = 0.50  # fixed a priori, NOT tuned against gold


def _row_features(
    qtag_bundle: dict,
    tag1: str,
    tag2: str,
    scale: str,
    agent_key: str,
    value_decimals: int = 0,
) -> list[float]:
    # p1, p2 are P(quality) from quality-vs-quantity SVM (junk-excluded).
    # Empty tag → neutral 0.5 (combiner treats missing tag as uninformative).
    p1, _ = predict_qtag_proba(qtag_bundle, tag1) if tag1 else (0.5, 0.5)
    p2, _ = predict_qtag_proba(qtag_bundle, tag2) if tag2 else (0.5, 0.5)
    tags = [t for t in (tag1, tag2) if t]
    cosd, has_dom = cos_domain(tags, agent_key)
    return build_feature_row(p1, p2, cosd, has_dom, scale, value_decimals)


def train_combiner(qtag_bundle: dict) -> LogisticRegression:
    print(f"Training combiner on {TRAIN_PATH.name} (junk excluded, 0 ID overlap with gold verified)...")
    df = pd.read_parquet(TRAIN_PATH).fillna("")
    # Combiner only classifies quality and quantity — junk excluded.
    n_before = len(df)
    df = df[df["label"].isin(["quality", "quantity"])].reset_index(drop=True)
    print(f"  Dropped {n_before - len(df)} junk rows -> {len(df)} records")
    df["agent_key"] = df["chain_id"].astype(str) + ":" + df["agent_id"].astype(str)
    X, y = [], []
    for _, r in df.iterrows():
        feats = _row_features(
            qtag_bundle,
            str(r["tag1"]), str(r["tag2"]), str(r["value_scale"]),
            str(r["agent_key"]),
            int(r.get("value_decimals") or 0),
        )
        X.append(feats)
        y.append(r["label"])
    X = np.array(X)
    print(f"  N={len(X)}  label dist: {pd.Series(y).value_counts().to_dict()}")
    clf = LogisticRegression(class_weight=None, max_iter=2000, C=1.0)
    clf.fit(X, y)
    preds = clf.predict(X)
    print("  Self-fit sanity check (training data itself, NOT held-out -- diagnostic only):")
    print(classification_report(y, preds, zero_division=0))
    for cls, coefs in zip(clf.classes_, clf.coef_):
        print(f"  coef[{cls}]: " + ", ".join(f"{n}={c:.2f}" for n, c in zip(FEATURE_NAMES, coefs)))
    return clf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--exclude-self", action="store_true")
    parser.add_argument("--skip-llm", action="store_true", help="Run 6a only (no LLM fallback)")
    args = parser.parse_args()

    print(f"Loading gold from {args.gold} ...")
    gold = load_gold(args.gold)
    print(f"  Gold N={len(gold)}")
    gold = enrich_gold_with_agent_meta(gold)
    if args.exclude_self:
        n_self = int(gold["is_self"].sum())
        gold = gold[~gold["is_self"]].reset_index(drop=True)
        print(f"  --exclude-self: dropped {n_self} -> N={len(gold)}")

    (OUT_DIR / "llm_classification.log").write_text("", encoding="utf-8")

    qtag_bundle = train_qtag_hybrid(save=True)
    combiner = train_combiner(qtag_bundle)
    joblib.dump(combiner, ROOT / "data/models/stage_combiner.joblib")

    rich_mask = gold["has_agent_metadata"].tolist()
    poor_mask = [not m for m in rich_mask]
    y_true = gold["label"].tolist()

    variants = [("6a_no_llm", False)] if args.skip_llm else [("6a_no_llm", False), ("6b_llm_fallback", True)]

    for variant, use_llm in variants:
        preds, sources, audit_rows = [], [], []
        llm_count = 0
        t0 = time.time()
        for _, row in gold.iterrows():
            tag1 = str(row.get("tag1", "") or "").strip()
            tag2 = str(row.get("tag2", "") or "").strip()
            scale = str(row.get("value_scale", "") or "").strip()
            decimals = int(row.get("value_decimals", 0) or 0)
            agent_key = str(row.get("agent_key", "") or "")
            true_label = row.get("label")

            def _rec(pred, source, conf=None):
                preds.append(pred); sources.append(source)
                audit_rows.append({
                    "id": row.get("id"), "tag1": tag1, "tag2": tag2, "value_scale": scale,
                    "value_decimals": decimals, "agent_key": agent_key,
                    "true_label": true_label, "pred": pred,
                    "stage": source, "max_proba": conf, "correct": pred == true_label,
                })

            if not tag1 and not tag2:
                lab = "junk" if scale.lower() == "unbounded" else "quality"
                _rec(lab, "empty_tag_rule"); continue

            cat = rule_classify(row)
            if cat:
                _rec(cat, "rule"); continue

            feats = _row_features(qtag_bundle, tag1, tag2, scale, agent_key, decimals)
            proba = combiner.predict_proba([feats])[0]
            classes = combiner.classes_
            best_idx = int(np.argmax(proba))
            best_label, best_p = classes[best_idx], float(proba[best_idx])

            if use_llm and best_p < LOW_CONF_THRESH:
                llm_cat = llm_classify(row, LLM_MODEL)
                _rec(llm_cat, "llm", best_p); llm_count += 1
            else:
                _rec(best_label, "combiner", best_p)

        mf1 = f1_score(y_true, preds, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
        print(f"\n{'='*60}\n  Run {variant}\n  Macro F1: {mf1:.4f}\n{'='*60}")
        print(classification_report(y_true, preds, labels=LLM_OUTPUT_CATEGORIES, zero_division=0))
        stage_counts: dict[str, int] = {}
        for s in sources:
            stage_counts[s] = stage_counts.get(s, 0) + 1
        for s, n in sorted(stage_counts.items()):
            print(f"  {s}: {n} ({n/len(sources)*100:.1f}%)")
        if use_llm:
            elapsed = time.time() - t0
            print(f"  LLM calls: {llm_count}  avg latency: {elapsed/max(llm_count,1)*1000:.0f}ms")

        def _sub_f1(mask, name):
            st = [y for y, m in zip(y_true, mask) if m]
            sp = [p for p, m in zip(preds, mask) if m]
            f = f1_score(st, sp, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
            print(f"  [{name} N={len(st)}] Macro F1: {f:.4f}")
            return f

        f_rich = _sub_f1(rich_mask, "Gold-Rich")
        f_poor = _sub_f1(poor_mask, "Gold-Poor")

        audit_path = OUT_DIR / f"audit_combiner_{variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
        print(f"  Audit saved to {audit_path}")

        result = {
            "name": f"Run {variant}", "macro_f1": mf1, "stage_counts": stage_counts,
            "f1_rich": f_rich, "f1_poor": f_poor, "audit_csv": str(audit_path),
        }
        out_path = OUT_DIR / f"pipeline_combiner_{variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
