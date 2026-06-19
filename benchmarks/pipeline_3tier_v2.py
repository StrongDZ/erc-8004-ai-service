#!/usr/bin/env python3
"""Ablation benchmark: 4-stage per-tag agent-domain pipeline.

Ablation runs:
  1. Rule only
  2. Rule + SVM pair-tag (existing baseline)
  3. Rule + SVM per-tag (Stage 1+2)
  4. Rule + SVM per-tag + FAISS (Stage 1+2+3, no LLM)
  5. Rule + SVM per-tag + FAISS + LLM (full pipeline)

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.pipeline_3tier_v2 [--skip-llm] [--run 3]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.per_tag_svm import load_per_tag_svm, predict_quality_prob, vote_per_tag
from benchmarks.pipeline_3tier import build_text, rule_classify
from benchmarks.stage3_domain import DomainClassifier
from shared.types import LLM_OUTPUT_CATEGORIES, RULE_TO_CAT

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
GOLD_CSV = ROOT.parent / "erc-8004-benchmarking-be/scripts/labelled/gold_final.csv"
PAIR_TRAIN = ROOT / "data/splits/rule_based_diverse_v2/train.parquet"
OUT_DIR = ROOT / "data/benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SVM_VOTE_THRESH = 0.70
LLM_MODEL = "qwen2.5:7b-instruct"
LLM_URL = "http://localhost:11434"


def load_gold() -> pd.DataFrame:
    df = pd.read_csv(GOLD_CSV).fillna("")
    df = df.rename(columns={"feedback_id": "id", "value_raw": "value", "scale": "value_scale", "category": "label"})
    df["label"] = df["label"].str.strip().str.lower().map(lambda x: RULE_TO_CAT.get(x, x))
    df = df[df["label"].isin(LLM_OUTPUT_CATEGORIES)].copy()
    for col in ("tag1", "tag2", "value_scale", "feedback_parsed", "value_decimals"):
        if col not in df.columns:
            df[col] = "" if col != "value_decimals" else 0
    df["value_decimals"] = pd.to_numeric(df["value_decimals"], errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)


def enrich_gold_with_agent_meta(gold: pd.DataFrame) -> pd.DataFrame:
    """Add agent_key + has_agent_metadata columns by MongoDB lookup."""
    from shared.mongo_client import agents_coll, feedback_coll
    fb_coll = feedback_coll()
    ag_coll = agents_coll()

    agent_keys = []
    has_meta = []
    for _, row in gold.iterrows():
        doc = fb_coll.find_one({"_id": row["id"]}, {"agentId": 1, "chainId": 1})
        if doc:
            key = f"{doc.get('chainId',0)}:{doc.get('agentId','')}"
            ag = ag_coll.find_one({"_id": key}, {"description": 1, "summarizedDescription": 1, "services": 1}) or {}
            desc = (ag.get("summarizedDescription") or ag.get("description") or "").strip()
            svcs = [s.get("name","") for s in (ag.get("services") or []) if s.get("name")]
            agent_keys.append(key)
            has_meta.append(bool(desc) or bool(svcs))
        else:
            agent_keys.append("")
            has_meta.append(False)
    gold = gold.copy()
    gold["agent_key"] = agent_keys
    gold["has_agent_metadata"] = has_meta
    return gold


def llm_classify(row: pd.Series, model: str) -> str:
    _SYSTEM = "You are a feedback classifier for ERC-8004 on-chain agent feedback. Classify into exactly ONE of: quality, quantity, junk. Respond with ONLY the category word."
    _USER = "tag1: {tag1}\ntag2: {tag2}\nvalue_scale: {scale}\noffchain: {offchain}"

    def _offchain(fp):
        if fp is None or (isinstance(fp, float) and np.isnan(fp)): return ""
        if isinstance(fp, str) and fp not in ("", "null", "None"): return fp[:200]
        return ""

    msg = _USER.format(
        tag1=str(row.get("tag1","") or "") or "(empty)",
        tag2=str(row.get("tag2","") or "") or "(empty)",
        scale=str(row.get("value_scale","") or "") or "(unknown)",
        offchain=_offchain(row.get("feedback_parsed")) or "(none)",
    )
    try:
        resp = requests.post(f"{LLM_URL}/api/chat", json={
            "model": model,
            "messages": [{"role":"system","content":_SYSTEM},{"role":"user","content":msg}],
            "stream": False, "options": {"temperature": 0, "num_predict": 16},
        }, timeout=60)
        raw = re.sub(r"[^a-z]", "", resp.json()["message"]["content"].strip().lower()[:20])
        return raw if raw in ("quality","quantity","junk") else "junk"
    except Exception:
        return "junk"


def _print_results(name: str, y_true: list, y_pred: list, sources: list[str]) -> dict:
    mf1 = f1_score(y_true, y_pred, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Macro F1: {mf1:.4f}")
    print(classification_report(y_true, y_pred, labels=LLM_OUTPUT_CATEGORIES, zero_division=0))
    stage_counts = {}
    for s in sources:
        stage_counts[s] = stage_counts.get(s, 0) + 1
    for s, n in sorted(stage_counts.items()):
        print(f"  {s}: {n} ({n/len(sources)*100:.1f}%)")
    return {"name": name, "macro_f1": mf1, "stage_counts": stage_counts}


def _sub_group_f1(y_true: list, y_pred: list, mask: list[bool], name: str) -> float:
    sub_true = [y for y, m in zip(y_true, mask) if m]
    sub_pred = [p for p, m in zip(y_pred, mask) if m]
    if not sub_true:
        return 0.0
    f1 = f1_score(sub_true, sub_pred, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
    print(f"  [{name} N={len(sub_true)}] Macro F1: {f1:.4f}")
    return f1


def run_ablation(gold: pd.DataFrame, run: int, skip_llm: bool) -> list[dict]:
    results = []
    y_true = gold["label"].tolist()
    rich_mask = gold["has_agent_metadata"].tolist()
    poor_mask = [not m for m in rich_mask]

    # ── Run 1: Rule only ──────────────────────────────────────────────────────
    if run in (0, 1):
        preds = []
        sources = []
        for _, row in gold.iterrows():
            cat = rule_classify(row)
            preds.append(cat if cat else "others")
            sources.append("rule" if cat else "default_others")
        results.append(_print_results("Run 1: Rule Only", y_true, preds, sources))
        results[-1]["f1_rich"] = _sub_group_f1(y_true, preds, rich_mask, "Gold-Rich")
        results[-1]["f1_poor"] = _sub_group_f1(y_true, preds, poor_mask, "Gold-Poor")

    # ── Run 2: Rule + SVM pair-tag (baseline) ─────────────────────────────────
    if run in (0, 2):
        train_df = pd.read_parquet(PAIR_TRAIN)
        X_tr = train_df.apply(build_text, axis=1).tolist()
        y_tr = train_df["label"].tolist()
        pair_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=8000, sublinear_tf=True)),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000), cv=3, method="sigmoid")),
        ])
        pair_pipe.fit(X_tr, y_tr)

        preds = []; sources = []
        for _, row in gold.iterrows():
            cat = rule_classify(row)
            if cat:
                preds.append(cat); sources.append("rule")
            else:
                text = build_text(row)
                preds.append(pair_pipe.predict([text])[0]); sources.append("pair_svm")
        results.append(_print_results("Run 2: Rule + SVM Pair (baseline)", y_true, preds, sources))
        results[-1]["f1_rich"] = _sub_group_f1(y_true, preds, rich_mask, "Gold-Rich")
        results[-1]["f1_poor"] = _sub_group_f1(y_true, preds, poor_mask, "Gold-Poor")

    # ── Runs 3–5: per-tag SVM + FAISS + LLM ──────────────────────────────────
    if run in (0, 3, 4, 5):
        per_tag_pipe = load_per_tag_svm()

    if run in (0, 4, 5):
        dc = DomainClassifier()

    for run_id in ([run] if run in (3, 4, 5) else [3, 4, 5]):
        use_faiss = run_id >= 4
        use_llm = run_id == 5 and not skip_llm

        preds = []; sources = []; llm_count = 0
        llm_t0 = time.time()

        for _, row in gold.iterrows():
            tag1 = str(row.get("tag1","") or "").strip()
            tag2 = str(row.get("tag2","") or "").strip()
            scale = str(row.get("value_scale","") or "").strip()
            decimals = int(row.get("value_decimals", 0) or 0)
            agent_key = str(row.get("agent_key","") or "")

            # Stage 1: rule
            cat = rule_classify(row)
            if cat:
                preds.append(cat); sources.append("rule"); continue

            # Stage 2: per-tag SVM voting (single source of truth: per_tag_svm.vote_per_tag)
            p1 = predict_quality_prob(per_tag_pipe, tag1, scale) if tag1 else 0.5
            p2 = predict_quality_prob(per_tag_pipe, tag2, scale) if tag2 else 0.5
            t2_empty = not bool(tag2)

            stage2_result = vote_per_tag(p1, p2, t2_empty=t2_empty, thresh=SVM_VOTE_THRESH)

            if stage2_result == "quality":
                preds.append("quality"); sources.append("per_tag_svm"); continue
            elif stage2_result == "non_quality":
                # SVM says non-quality but doesn't know if quantity or junk → Stage 3 resolves it
                if not use_faiss:
                    preds.append("quantity"); sources.append("per_tag_svm_non_quality"); continue

            # Stage 3: FAISS domain check
            if use_faiss:
                label3, reason = dc.classify(tag1, tag2, scale, decimals, agent_key)
                if label3 is not None:
                    preds.append(label3); sources.append(f"faiss:{reason[:20]}"); continue

            # Stage 4: LLM
            if use_llm:
                llm_cat = llm_classify(row, LLM_MODEL)
                preds.append(llm_cat); sources.append("llm"); llm_count += 1
            else:
                # No LLM → use ML best guess
                preds.append("quality" if p1 >= 0.50 else "quantity")
                sources.append("ml_default")

        run_name = f"Run {run_id}: Rule + Per-Tag SVM" + (" + FAISS" if use_faiss else "") + (" + LLM" if use_llm else "")
        res = _print_results(run_name, y_true, preds, sources)
        if use_llm:
            elapsed = time.time() - llm_t0
            print(f"  LLM calls: {llm_count} ({llm_count/len(gold)*100:.1f}%)  avg latency: {elapsed/max(llm_count,1)*1000:.0f}ms")
        res["f1_rich"] = _sub_group_f1(y_true, preds, rich_mask, "Gold-Rich")
        res["f1_poor"] = _sub_group_f1(y_true, preds, poor_mask, "Gold-Poor")
        results.append(res)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=0, help="0=all, 1-5=specific run")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    print("Loading gold test set...")
    gold = load_gold()
    print(f"  Gold N={len(gold)}")

    print("Enriching gold with agent metadata (MongoDB lookup)...")
    gold = enrich_gold_with_agent_meta(gold)
    rich = gold["has_agent_metadata"].sum()
    print(f"  Gold-Rich: {rich} | Gold-Poor: {len(gold)-rich}")

    results = run_ablation(gold, args.run, args.skip_llm)

    print("\n\n=== SUMMARY: MacroF1 by run (full / rich / poor) ===")
    for res in results:
        print(f"  {res['name']:55s} full={res['macro_f1']:.4f}  rich={res['f1_rich']:.4f}  poor={res['f1_poor']:.4f}")

    # Save results
    out_path = OUT_DIR / f"pipeline_3tier_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
