#!/usr/bin/env python3
"""Conservative fair-max ablation ladder for the frozen bge-small baseline.

Answers: "was the *input design* handicapping the frozen-embedding baseline
(config 3), or is a frozen sentence encoder genuinely weak on ultra-short tag
input?" Every rung keeps the SAME frozen encoder (BAAI/bge-small-en-v1.5) and
the SAME information (tag1, tag2, value_scale) — it only changes HOW that
information is fed to a linear head. Each rung adds exactly one change over the
previous, so the per-rung delta isolates one lever:

  B1  encode "tag1 tag2 scale" -> LogReg            (canonical config 3, ~0.672)
  F1  + per-tag decomposition (encode each tag, average posteriors)  [blending]
  F2  + pull scale OUT of the string, add as one-hot feature         [scale-in-string]
  F3  + class_weight='balanced' head                                 [head]
  F4  + unbounded structural rule (P(quality)=0, renormalise)        [convention]

Same gold + same 2-class Macro-F1 (junk excluded) as clean_baselines_ab.py, so
the numbers drop straight into the Ch6 comparison table.

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.maxpot_frozen_ladder \
        --gold data/labelled/pure_others_stratified_dedup.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.stage3_domain import _load_model
from shared.types import LLM_OUTPUT_CATEGORIES, RULE_TO_CAT

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits" / "agent_enriched"
OUT_DIR = ROOT / "data" / "benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CLASSES = ["junk", "quality", "quantity"]  # sklearn sorts labels -> this order


# ── shared helpers ──────────────────────────────────────────────────────────

def two_class_macro_f1(y_true: list[str], y_pred: list[str]) -> tuple[float, dict]:
    """(F1_quality + F1_quantity)/2 — the ranking metric; junk excluded."""
    rep = classification_report(y_true, y_pred, labels=CLASSES,
                                output_dict=True, zero_division=0)
    two = (rep["quality"]["f1-score"] + rep["quantity"]["f1-score"]) / 2.0
    return two, rep


def _norm(x) -> str:
    return str(x or "").strip().lower()


def load_train() -> pd.DataFrame:
    ga = pd.read_parquet(SPLITS / "group_a.parquet")
    gb = pd.read_parquet(SPLITS / "group_b.parquet")
    tr = pd.concat([ga, gb], ignore_index=True)
    return tr[tr["label"].isin(CLASSES)].reset_index(drop=True)


def load_gold_df(path: Path) -> pd.DataFrame:
    """Identical loading to clean_baselines_ab.py so B1 reproduces exactly."""
    g = pd.read_csv(path).fillna("")
    g = g.rename(columns={"human_label": "label", "scale": "value_scale"})
    g["label"] = g["label"].str.strip().str.lower().map(lambda x: RULE_TO_CAT.get(x, x))
    return g[g["label"].isin(LLM_OUTPUT_CATEGORIES)].reset_index(drop=True)


def encode_all(enc, texts: list[str]) -> dict[str, np.ndarray]:
    """Batch-encode unique strings once, return a cache."""
    uniq = sorted(set(texts))
    vecs = np.asarray(enc.encode(uniq, normalize_embeddings=True,
                                 show_progress_bar=False), dtype="float32")
    return {t: v for t, v in zip(uniq, vecs)}


def proba_by_class(clf, X: np.ndarray) -> np.ndarray:
    """predict_proba re-ordered into fixed CLASSES order."""
    p = clf.predict_proba(X)
    idx = [list(clf.classes_).index(c) for c in CLASSES]
    return p[:, idx]


# ── record-level feature builders ───────────────────────────────────────────

def record_feature_text(row: pd.Series) -> str:
    """B1: the documented tag1 + tag2 + scale string."""
    parts = [_norm(row.get("tag1")), _norm(row.get("tag2")), _norm(row.get("value_scale"))]
    return " ".join(p for p in parts if p) or "<empty>"


def expand_single_tags(df: pd.DataFrame) -> pd.DataFrame:
    """One row per tag, inheriting the record's 3-class label."""
    rows = []
    for _, r in df.iterrows():
        sc, lab = _norm(r.get("value_scale")), r["label"]
        for tc in ("tag1", "tag2"):
            t = _norm(r.get(tc))
            if t:
                rows.append({"tag": t, "scale": sc, "label": lab})
    return pd.DataFrame(rows)


def record_tags(row: pd.Series) -> list[str]:
    return [t for t in (_norm(row.get("tag1")), _norm(row.get("tag2"))) if t]


# ── rungs ────────────────────────────────────────────────────────────────────

def rung_B1(enc, tr, gold):
    """Canonical config 3: emb('tag1 tag2 scale') -> LogReg."""
    Xtr_txt = tr.apply(record_feature_text, axis=1).tolist()
    Xte_txt = gold.apply(record_feature_text, axis=1).tolist()
    cache = encode_all(enc, Xtr_txt + Xte_txt)
    Etr = np.stack([cache[t] for t in Xtr_txt])
    Ete = np.stack([cache[t] for t in Xte_txt])
    clf = LogisticRegression(C=1.0, max_iter=3000, random_state=SEED)
    clf.fit(Etr, tr["label"].tolist())
    return clf.predict(Ete).tolist()


def _per_tag_predict(enc, tr, gold, *, scale_in_text: bool,
                     balanced: bool, unbounded_rule: bool):
    """Per-tag posterior averaging. Shared engine for F1..F4.

    scale_in_text : True  -> encode "{tag} {scale}"          (F1)
                    False -> encode "{tag}" + one-hot(scale)  (F2+)
    balanced      : class_weight='balanced' on the head       (F3+)
    unbounded_rule: zero P(quality) when scale=='unbounded'   (F4)
    """
    st = expand_single_tags(tr)

    def tag_text(tag: str, scale: str) -> str:
        return f"{tag} {scale}" if scale_in_text else tag

    # encode every tag string used in train + gold
    train_texts = [tag_text(r.tag, r.scale) for r in st.itertuples()]
    gold_pairs = [(record_tags(row), _norm(row.get("value_scale")))
                  for _, row in gold.iterrows()]
    gold_texts = [tag_text(t, sc) for tags, sc in gold_pairs for t in tags]
    cache = encode_all(enc, train_texts + gold_texts)

    scale_oh: dict[str, np.ndarray] = {}
    if not scale_in_text:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe.fit(st[["scale"]].to_numpy())
        # precompute the one-hot per distinct scale once (avoids per-tag transform)
        cats = list(ohe.categories_[0])
        seen = set(st["scale"]) | {sc for _, sc in gold_pairs}
        for sc in seen:
            row = np.zeros(len(cats), dtype="float32")
            if sc in cats:
                row[cats.index(sc)] = 1.0
            scale_oh[sc] = row

    def feat(tag: str, scale: str) -> np.ndarray:
        v = cache[tag_text(tag, scale)]
        if scale_in_text:
            return v
        return np.concatenate([v, scale_oh[scale]])

    Xtr = np.stack([feat(r.tag, r.scale) for r in st.itertuples()])
    ytr = st["label"].tolist()
    clf = LogisticRegression(C=1.0, max_iter=3000, random_state=SEED,
                             class_weight="balanced" if balanced else None)
    clf.fit(Xtr, ytr)

    q_idx = CLASSES.index("quality")
    preds = []
    for tags, sc in gold_pairs:
        if not tags:                       # ~2 rows: no tags -> scale convention
            preds.append("quantity" if sc == "unbounded" else "quality")
            continue
        P = proba_by_class(clf, np.stack([feat(t, sc) for t in tags]))
        avg = P.mean(axis=0)               # average posteriors over the record's tags
        if unbounded_rule and sc == "unbounded":
            avg = avg.copy()
            avg[q_idx] = 0.0
            s = avg.sum()
            avg = avg / s if s > 0 else np.eye(len(CLASSES))[CLASSES.index("quantity")]
        preds.append(CLASSES[int(avg.argmax())])
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, required=True)
    args = ap.parse_args()

    np.random.seed(SEED)
    tr = load_train()
    gold = load_gold_df(args.gold)
    yte = gold["label"].tolist()
    print(f"Train N={len(tr)}  Gold N={len(gold)}  encoder=bge-small (frozen)")
    print(f"  train dist: {tr['label'].value_counts().to_dict()}")
    print(f"  gold  dist: {gold['label'].value_counts().to_dict()}\n")

    enc = _load_model()

    # Fair-max path (each rung = one input-design change over the previous):
    #   B1 -> F1 (per-tag) -> F2 (scale as feature) -> F3 (unbounded rule).
    # D_balanced is a labelled NEGATIVE diagnostic, not a progression rung:
    # class_weight='balanced' is applied on top of F2 to show it hurts here.
    rungs = {
        "B1_canonical            ": rung_B1(enc, tr, gold),
        "F1_per_tag              ": _per_tag_predict(enc, tr, gold, scale_in_text=True,  balanced=False, unbounded_rule=False),
        "F2_scale_as_feature     ": _per_tag_predict(enc, tr, gold, scale_in_text=False, balanced=False, unbounded_rule=False),
        "F3_unbounded_rule       ": _per_tag_predict(enc, tr, gold, scale_in_text=False, balanced=False, unbounded_rule=True),
        "D_balanced_head(neg)    ": _per_tag_predict(enc, tr, gold, scale_in_text=False, balanced=True,  unbounded_rule=False),
    }

    print(f"{'rung':24} {'2cls-MacroF1':>12} {'quality-F1':>11} {'quantity-F1':>12} {'qty-recall':>11}")
    print("-" * 74)
    out = {}
    for name, preds in rungs.items():
        two, rep = two_class_macro_f1(yte, preds)
        qf1, qtyf1 = rep["quality"]["f1-score"], rep["quantity"]["f1-score"]
        qtyrec = rep["quantity"]["recall"]
        print(f"{name:24} {two:>12.4f} {qf1:>11.4f} {qtyf1:>12.4f} {qtyrec:>11.4f}")
        out[name.strip()] = {"macro_f1_2cls": round(two, 4),
                             "quality_f1": round(qf1, 4),
                             "quantity_f1": round(qtyf1, 4),
                             "quantity_recall": round(qtyrec, 4)}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT_DIR / f"maxpot_frozen_ladder_{ts}.json"
    p.write_text(json.dumps({"encoder": "bge-small-frozen", "n_train": len(tr),
                             "n_gold": len(gold), "seed": SEED, "rungs": out}, indent=2))
    print(f"\nSaved {p}")


if __name__ == "__main__":
    main()
