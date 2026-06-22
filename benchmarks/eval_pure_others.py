#!/usr/bin/env python3
"""Evaluate the production 3-tier pipeline on the hand-labelled pure-others gold.

`data/labelled/pure_others_to_label.csv` is the de-circularised AI-service
benchmark: every record genuinely escaped the Go rule engine
(classification.rule.category == "others"), so there is no Stage-1 label leakage
and no self-feedback leak. Each record's `human_label` was assigned by hand using
both the feedback (tag1/tag2/scale/value) and the agent context.

The pipeline (shared/three_tier.classify_three_tier) is the SAME code production
runs. It is invoked here with NO LLM (llm_classify_fn=None):

  - Stage 2 (per-tag SVM) and Stage 3 (agent-domain cosine) resolve records
    deterministically -> compared against human_label.
  - Records with no agent metadata or a borderline cosine require the LLM in
    production; with no LLM supplied they raise, and are reported here as the
    "LLM residual" (broken down by true label) rather than guessed with an ML
    default (which the pipeline no longer uses).

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.eval_pure_others
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.three_tier import DOMAIN_EMBED_MODEL, classify_three_tier  # noqa: E402

GOLD = ROOT / "data/labelled/pure_others_to_label.csv"
# 3-class: junk is kept in scope even though the no-LLM deterministic stages
# (SVM + cosine) structurally cannot emit it — that gap is the point being
# measured here, not a reason to exclude it.
LABELS = ["junk", "quality", "quantity"]


def main() -> None:
    df = pd.read_csv(GOLD).fillna("")
    print(f"Loaded {len(df)} records from {GOLD.name}")
    print(f"  gold dist: {dict(df['human_label'].value_counts())}")

    enc = SentenceTransformer(DOMAIN_EMBED_MODEL, device="cpu")
    enc.max_seq_length = 256

    preds, sources = [], []
    for i, (_, r) in enumerate(df.iterrows()):
        if i % 250 == 0:
            print(f"  {i}/{len(df)}...", end="\r", flush=True)
        agent_text = str(r.get("agent_domain_text", "")).strip()
        try:
            res = classify_three_tier(
                encoder=enc,
                tag1=str(r["tag1"]), tag2=str(r["tag2"]),
                scale=str(r["scale"]), value_norm=0.0,
                agent_text=agent_text, llm_classify_fn=None,
            )
            preds.append(res.category)
            sources.append(res.source)
        except ValueError:
            preds.append("RESIDUAL")   # no metadata / borderline cosine -> needs LLM
            sources.append("needs_llm")
    df["pred"] = preds
    df["src"] = sources

    det = df[df["pred"] != "RESIDUAL"]
    resid = df[df["pred"] == "RESIDUAL"]
    print(f"\n\nResolved deterministically (SVM + cosine): {len(det)} ({100*len(det)/len(df):.1f}%)")
    print(f"  source: {dict(det['src'].value_counts())}")
    print(f"LLM residual (no metadata / borderline cosine): {len(resid)} ({100*len(resid)/len(df):.1f}%)")
    print(f"  residual by true label: {dict(resid['human_label'].value_counts())}")

    # 3-class metrics on the deterministically resolved records. The
    # deterministic stages (SVM + cosine) never emit "junk", so any
    # true-junk record that lands here is necessarily a miss — that
    # depresses the score honestly rather than hiding the gap.
    mask = det["human_label"].isin(LABELS) & det["pred"].isin(LABELS)
    det2 = det[mask]
    print(f"\n{'='*56}\nDeterministic three-class result (N={len(det2)})\n{'='*56}")
    mf1 = f1_score(det2["human_label"], det2["pred"], labels=LABELS, average="macro", zero_division=0)
    print(f"Macro F1 = {mf1:.4f}")
    print(classification_report(det2["human_label"], det2["pred"], labels=LABELS, zero_division=0))

    # How were the 7 gold-junk records handled by the no-LLM pipeline?
    jk = df[df["human_label"] == "junk"]
    if len(jk):
        print(f"Gold junk (N={len(jk)}) — deterministic pipeline has no junk output, "
              f"so these go to: {dict(jk['pred'].value_counts())}")

    # Rich (has agent metadata) vs Poor split
    df["rich"] = df["agent_domain_text"].astype(str).str.strip() != ""
    for split, g in (("Rich", df[df["rich"]]), ("Poor", df[~df["rich"]])):
        gd = g[(g["pred"] != "RESIDUAL") & g["human_label"].isin(LABELS) & g["pred"].isin(LABELS)]
        nres = (g["pred"] == "RESIDUAL").sum()
        if len(gd):
            f = f1_score(gd["human_label"], gd["pred"], labels=LABELS, average="macro", zero_division=0)
            print(f"{split} (N={len(g)}): deterministic N={len(gd)} MacroF1={f:.4f}, residual={nres}")
        else:
            print(f"{split} (N={len(g)}): deterministic N=0, residual={nres} (all need the LLM)")


if __name__ == "__main__":
    main()
