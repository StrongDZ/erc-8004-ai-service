#!/usr/bin/env python3
"""Conservative fair-max for the fine-tuned ModernBERT baseline (config 4).

The canonical run has two defensible flaws that the memo attributes 100% of its
weakness to "distribution shift":
  1. train/test featurization skew — train fuses agent_description, but gold
     fuses agent_domain_text (a different, richer field absent from train).
  2. unseeded fine-tuning — +-0.05 Macro-F1 between runs, a single illustrative
     number.

This isolates both, measuring the PURE MODEL (argmax, NO LLM fallback — so it
needs no Ollama) 2-class Macro-F1 across seeds. Because canonical and symmetric
share identical TRAIN text, one fine-tune per seed serves both evals:

  agent-desc model  --eval-->  gold(agent_domain_text)  = canonical (skewed)
                    --eval-->  gold(agent_description)   = symmetric (fixed)
  tag-only model    --eval-->  gold(tag+scale only)      = no-agent-context

Same gold + 2-class Macro-F1 (junk excluded) as the rest of the comparison.

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.maxpot_modernbert \
        --gold data/labelled/pure_others_stratified_dedup.csv --seeds 42
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.pipeline_run15 import (BACKBONES, CLASS2IDX, CLASSES, TAU_GRID,
                                       apply_unbounded_constraint,
                                       build_fused_text, finetune_encoder)
from benchmarks.pipeline_3tier_v2 import LLM_MODEL, llm_classify, load_gold
from shared.types import LLM_OUTPUT_CATEGORIES

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits" / "agent_enriched"
OUT_DIR = ROOT / "data" / "benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(s: int) -> None:
    import random
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(s)


def two_class_macro_f1(y_true, y_pred) -> float:
    rep = classification_report(y_true, y_pred, labels=CLASSES,
                                output_dict=True, zero_division=0)
    return (rep["quality"]["f1-score"] + rep["quantity"]["f1-score"]) / 2.0


def pure_model_2cls(clf, ft_model, texts: list[str], scales: list[str],
                    y_true: list[str]) -> float:
    """Argmax prediction, unbounded constraint applied, NO LLM. 2-class Macro-F1."""
    X = ft_model.encode(texts, normalize_embeddings=True, batch_size=64,
                        show_progress_bar=False)
    proba = clf.predict_proba(X)                       # columns == CLASS2IDX order
    proba = apply_unbounded_constraint(proba, scales)
    preds = [CLASSES[int(i)] for i in proba.argmax(axis=1)]
    return two_class_macro_f1(y_true, preds)


def eval_with_llm(clf, ft_model, texts, gold_df, gold_scales, y_true) -> dict:
    """Full pipeline: max(P)>=tau -> argmax, else LLM fallback (cached). Sweep the
    tau grid and return the best-Macro operating point (matches the thesis, which
    reports the fine-tuned encoder at its best-Macro tau=0.90)."""
    X = ft_model.encode(texts, normalize_embeddings=True, batch_size=64,
                        show_progress_bar=False)
    proba = apply_unbounded_constraint(clf.predict_proba(X), gold_scales)
    maxp, argm = proba.max(axis=1), proba.argmax(axis=1)
    rows = [row for _, row in gold_df.iterrows()]
    best = {"f1": -1.0, "tau": None, "llm_pct": None, "preds": None}
    for tau in TAU_GRID:
        preds, nllm = [], 0
        for i, row in enumerate(rows):
            if maxp[i] >= tau:
                preds.append(CLASSES[int(argm[i])])
            else:
                preds.append(llm_classify(row, LLM_MODEL))
                nllm += 1
        f1 = two_class_macro_f1(y_true, preds)
        if f1 > best["f1"]:
            best = {"f1": f1, "tau": float(tau), "llm_pct": round(nllm / len(rows) * 100, 1),
                    "preds": list(preds)}
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, best["preds"], labels=["quality", "quantity"])
    best["cm2x2"] = cm.tolist()  # rows=true[quality,quantity] cols=pred[quality,quantity]
    print(f"  CM2x2 (rows=true q/qty, cols=pred q/qty): {cm.tolist()}")
    return best


def fused(row, scale_key, agent_field: str | None) -> str:
    agent = ""
    if agent_field == "domain":
        agent = str(row.get("agent_domain_text") or "") or str(row.get("agent_description") or "")
    elif agent_field == "desc":
        agent = str(row.get("agent_description") or "")
    return build_fused_text(str(row.get("tag1") or ""), str(row.get("tag2") or ""),
                            str(row.get(scale_key) or ""), agent)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--seeds", default="42", help="comma-separated seeds")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--with-llm", action="store_true",
                    help="report the FULL pipeline (Chow escalation + LLM fallback) 2-class "
                         "Macro-F1 at best-Macro tau, matching the thesis 0.747; needs the "
                         "warm LLM cache. Skips the tag-only model.")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    ga = pd.read_parquet(SPLITS / "group_a.parquet")
    gb = pd.read_parquet(SPLITS / "group_b.parquet")
    df = pd.concat([ga, gb], ignore_index=True)
    df = df[df["label"].isin(CLASSES)].reset_index(drop=True)

    gold = load_gold(args.gold)
    gold = gold[gold["label"].isin(LLM_OUTPUT_CATEGORIES)].reset_index(drop=True)
    y_true = gold["label"].str.strip().str.lower().tolist()
    gold_scales = gold["value_scale"].tolist()
    print(f"Train N={len(df)}  Gold N={len(gold)}  backbone=ModernBERT-base  (pure model, no LLM)\n")

    # gold texts for the three eval featurizations (built once)
    g_domain = [fused(r, "value_scale", "domain") for _, r in gold.iterrows()]  # canonical
    g_desc = [fused(r, "value_scale", "desc") for _, r in gold.iterrows()]      # symmetric
    g_tag = [fused(r, "value_scale", None) for _, r in gold.iterrows()]         # no-agent

    mode = "with-LLM (best-Macro tau)" if args.with_llm else "pure model, no LLM"
    variants = {"canonical_skewed": [], "symmetric_desc": [], "tag_only": []}
    llm_pcts = {"canonical_skewed": [], "symmetric_desc": []}
    for seed in seeds:
        # fixed 80/20 split (as canonical); only the model seed varies
        train_df, _ = train_test_split(df, test_size=0.20, stratify=df["label"],
                                       random_state=42)
        y_train = train_df["label"].map(CLASS2IDX).values

        # ── model A: agent_description train text (serves canonical + symmetric) ──
        seed_everything(seed)
        t0 = time.monotonic()
        tr_desc = [fused(r, "value_scale", "desc") for _, r in train_df.iterrows()]
        m_desc = finetune_encoder(BACKBONES["modernbert"], tr_desc, list(y_train),
                                  epochs=args.epochs)
        Xtr = m_desc.encode(tr_desc, normalize_embeddings=True, batch_size=64,
                            show_progress_bar=False)
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000,
                                 random_state=42).fit(Xtr, y_train)

        if args.with_llm:
            bc = eval_with_llm(clf, m_desc, g_domain, gold, gold_scales, y_true)
            bs = eval_with_llm(clf, m_desc, g_desc, gold, gold_scales, y_true)
            variants["canonical_skewed"].append(bc["f1"]); llm_pcts["canonical_skewed"].append(bc["llm_pct"])
            variants["symmetric_desc"].append(bs["f1"]); llm_pcts["symmetric_desc"].append(bs["llm_pct"])
            print(f"[seed {seed}] canonical={bc['f1']:.4f}@tau{bc['tau']:.2f}/{bc['llm_pct']:.0f}%LLM  "
                  f"symmetric={bs['f1']:.4f}@tau{bs['tau']:.2f}/{bs['llm_pct']:.0f}%LLM  "
                  f"({time.monotonic()-t0:.0f}s)")
        else:
            c = pure_model_2cls(clf, m_desc, g_domain, gold_scales, y_true)
            s = pure_model_2cls(clf, m_desc, g_desc, gold_scales, y_true)
            variants["canonical_skewed"].append(c)
            variants["symmetric_desc"].append(s)
            # ── model B: tag+scale only train text (pure-model diagnostic only) ──
            seed_everything(seed)
            tr_tag = [fused(r, "value_scale", None) for _, r in train_df.iterrows()]
            m_tag = finetune_encoder(BACKBONES["modernbert"], tr_tag, list(y_train),
                                     epochs=args.epochs)
            Xtr2 = m_tag.encode(tr_tag, normalize_embeddings=True, batch_size=64,
                                show_progress_bar=False)
            clf2 = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000,
                                      random_state=42).fit(Xtr2, y_train)
            variants["tag_only"].append(pure_model_2cls(clf2, m_tag, g_tag, gold_scales, y_true))
            print(f"[seed {seed}] canonical={c:.4f}  symmetric={s:.4f}  "
                  f"tag_only={variants['tag_only'][-1]:.4f}  ({time.monotonic()-t0:.0f}s)")

    print(f"\n{'variant':20} {'mean-2cls':>10} {'std':>8} {'meanLLM%':>9} {'n':>3}  ({mode})")
    print("-" * 58)
    out = {}
    for name, vals in variants.items():
        if not vals:
            continue
        m, sd = float(np.mean(vals)), float(np.std(vals))
        lp = f"{np.mean(llm_pcts[name]):>8.1f}%" if llm_pcts.get(name) else f"{'--':>9}"
        print(f"{name:20} {m:>10.4f} {sd:>8.4f} {lp} {len(vals):>3}")
        out[name] = {"mean_2cls": round(m, 4), "std": round(sd, 4),
                     "runs": [round(v, 4) for v in vals],
                     "mean_llm_pct": round(float(np.mean(llm_pcts[name])), 1) if llm_pcts.get(name) else None}
    if args.with_llm:
        print("\nThesis reference: config-4 = 0.747 @ 53% LLM (single un-seeded run).")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "withllm" if args.with_llm else "puremodel"
    p = OUT_DIR / f"maxpot_modernbert_{tag}_{ts}.json"
    p.write_text(json.dumps({"backbone": "modernbert-base", "n_train": len(df),
                             "n_gold": len(gold), "seeds": seeds, "epochs": args.epochs,
                             "mode": mode, "variants": out}, indent=2))
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
