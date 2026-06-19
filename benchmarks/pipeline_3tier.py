#!/usr/bin/env python3
"""3-tier pipeline benchmark: Rule-based → ML (SVM TF-IDF) → Real Ollama LLM.

Usage:
    cd erc-8004-ai-service
    python -m benchmarks.pipeline_3tier [--model qwen2.5:3b] [--threshold 0.70]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
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
from shared.types import LLM_OUTPUT_CATEGORIES, RULE_TO_CAT

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
TRAIN_PARQUET = ROOT / "data/splits/rule_based_diverse_v2/train.parquet"
GOLD_CSV = ROOT.parent / "erc-8004-benchmarking-be/scripts/labelled/gold_final.csv"


# ══════════════════════════════════════════════════════════════════════════════
# 1. PYTHON RULE-BASED CASCADE  (port of classifier.go + rule_patterns.go)
# ══════════════════════════════════════════════════════════════════════════════

_SPAM_URL = re.compile(r"(?i)(t\.me/|telegram\.me|https?://|http://)")
_SPAM_RANK = re.compile(r"(?i)(get\s+top|top\s*[0-9]|-{2,}>|#1\s+rank)")
_UUID = re.compile(
    r"(?i)^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_EMOJI_ONLY = re.compile(r"^[\U0001F000-\U0001FFFF☀-⟿\s]+$")
_ALL_DIGITS = re.compile(r"^[0-9]+$")

_NOISE_T1 = {"test", "asd", "custom", "settled", "claudelance", "vibez"}

_QUANTITY_T1 = {
    "reachable", "liveness", "successrate", "success-rate",
    "responsetime", "response-time", "blocktimefreshness",
    "blocktime-freshness", "blocktime freshness", "uptime", "creditscore",
    "attendance-rate", "completion-rate", "execution-speed",
    "payment-speed", "settlement-speed", "win-rate",
    "coverage-rate", "exit-rate", "active", "safety-score", "contractrisk",
    "counterparty", "longevity", "activity",
}
_QUANTITY_T2 = {
    "liveness-check", "win-rate", "coverage-rate", "exit-rate",
    "automated-screening", "completion-rate", "scroll-stop-rate",
}

_QUALITY_T1 = {
    "trustscore", "trust-score", "trust", "trust-oracle",
    "starred", "quality", "performance", "service",
    "helpful", "fast", "reliable", "reliability",
    "excellence", "excellent", "satisfaction", "experience",
    "value", "rating", "good", "robust", "secure",
    "audited", "innovative", "transparent", "trustless",
    "composable", "interoperable", "cool", "great",
    "review", "useful", "smart", "nice", "overall",
    "creative", "support", "intelligent", "analytical",
    "usability", "compliance", "peer-review",
    "content-moderation", "amazing", "awesome", "beautiful",
    "professional", "impressive", "outstanding", "best",
    "miner-vouch",
    "helpfull", "powerfull", "usefull", "reliabel", "excelent",
}
_QUALITY_KW = [
    "helpful", "fast", "reliable", "quality", "excellent", "good",
    "useful", "great", "smart", "simple", "easy", "smooth", "solid",
    "stable", "best", "nice", "clean", "amazing", "awesome", "love", "trust", "fragment",
]


def _is_spam(t1: str, t2: str) -> bool:
    return (
        _SPAM_URL.search(t1) is not None or _SPAM_URL.search(t2) is not None
        or _SPAM_RANK.search(t1) is not None or _SPAM_RANK.search(t2) is not None
    )


def _is_noise(t1: str, t2: str) -> bool:
    return t1 in _NOISE_T1 and (
        t2 == "" or t2 in _NOISE_T1 or _ALL_DIGITS.match(t2) is not None
    )


def _contains_quality_kw(t: str) -> bool:
    return any(kw in t for kw in _QUALITY_KW)


def rule_classify(row: pd.Series) -> str | None:
    """Returns category string or None if rule can't decide (→ escalate)."""
    t1r = str(row.get("tag1", "") or "").strip()
    t2r = str(row.get("tag2", "") or "").strip()
    t1 = t1r.lower()
    t2 = t2r.lower()

    # Layer 1: JUNK
    if _is_spam(t1, t2):
        return "junk"
    if _is_noise(t1, t2):
        return "junk"
    if t1 != "" and t2 != "" and _ALL_DIGITS.match(t1) and _ALL_DIGITS.match(t2):
        return "junk"
    if _UUID.match(t1r) or (t1 == "" and _UUID.match(t2r)):
        return "junk"
    if t1 == "" and t2 == "":
        return None  # escalate: LLM reads offchain
    if _EMOJI_ONLY.match(t1r) and t2 == "":
        return "junk"

    # Layer 2: QUANTITY
    if t1 in _QUANTITY_T1 or t2 in _QUANTITY_T1 or t2 in _QUANTITY_T2:
        return "quantity"

    # Layer 3: QUALITY
    if t1 in _QUALITY_T1:
        return "quality"
    if _contains_quality_kw(t1) or _contains_quality_kw(t2):
        return "quality"

    return None  # escalate to ML


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def build_text(row: pd.Series) -> str:
    t1 = str(row.get("tag1", "") or "").strip()
    t2 = str(row.get("tag2", "") or "").strip()
    sc = str(row.get("value_scale", "") or "").strip()
    parts = []
    if t1:
        parts.append(f"tag1={t1}")
    if t2:
        parts.append(f"tag2={t2}")
    if sc:
        parts.append(f"scale={sc}")
    return " | ".join(parts) if parts else "<empty>"


# ══════════════════════════════════════════════════════════════════════════════
# 3. REAL OLLAMA LLM CALLER (matches prompt V7 used in Go benchmark)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a feedback classifier for ERC-8004 on-chain agent feedback.
Classify the feedback into exactly ONE of these categories:
- quality   : Subjective judgment about service, experience, trustworthiness, or performance.
- quantity  : A measured metric, rate, score, count, or statistical outcome.
- junk      : Spam, noise, meaningless, or placeholder content.

Respond with ONLY the category word: quality, quantity, or junk. No explanation."""

USER_TEMPLATE = """\
tag1: {tag1}
tag2: {tag2}
value_scale: {scale}
offchain: {offchain}"""


def _extract_offchain(fp) -> str:
    if fp is None or (isinstance(fp, float) and np.isnan(fp)):
        return ""
    if isinstance(fp, str):
        if not fp or fp in ("null", "None", "nan"):
            return ""
        try:
            fp = json.loads(fp)
        except Exception:
            return fp[:300]
    if isinstance(fp, dict):
        for k in ("comment", "review", "text", "description", "message", "feedback", "content"):
            if k in fp:
                return str(fp[k])[:300]
        return json.dumps(fp, ensure_ascii=False)[:300]
    return str(fp)[:300]


def llm_classify(row: pd.Series, model: str, base_url: str = "http://localhost:11434") -> str:
    tag1 = str(row.get("tag1", "") or "")
    tag2 = str(row.get("tag2", "") or "")
    scale = str(row.get("value_scale", "") or "")
    offchain = _extract_offchain(row.get("feedback_parsed"))

    user_msg = USER_TEMPLATE.format(
        tag1=tag1 or "(empty)",
        tag2=tag2 or "(empty)",
        scale=scale or "(unknown)",
        offchain=offchain or "(none)",
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 16},
    }
    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip().lower().split()[0]
        # strip punctuation
        raw = re.sub(r"[^a-z]", "", raw)
        if raw in ("quality", "quantity", "junk"):
            return raw
        return "junk"  # fallback on bad output
    except Exception:
        return "junk"


# ══════════════════════════════════════════════════════════════════════════════
# 4. GOLD CSV LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_gold(gold_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(gold_csv).fillna("")
    # Normalise column names
    rename = {
        "feedback_id": "id",
        "value_raw": "value",
        "scale": "value_scale",
        "category": "label",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["label"] = (
        df["label"].astype(str).str.strip().str.lower()
        .map(lambda x: RULE_TO_CAT.get(x, x))
    )
    df = df[df["label"].isin(LLM_OUTPUT_CATEGORIES)].copy()
    # Ensure all needed columns exist
    for col in ("tag1", "tag2", "value_scale", "feedback_parsed", "value_decimals"):
        if col not in df.columns:
            df[col] = ""
    df["feedback_parsed"] = df["feedback_parsed"].replace("", None)
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(model: str, threshold: float, llm_url: str, skip_llm: bool) -> None:
    print(f"\n{'='*70}")
    print(f"3-Tier Pipeline Benchmark")
    print(f"  model={model}  threshold={threshold}  skip_llm={skip_llm}")
    print(f"{'='*70}\n")

    # ── Load & train ──────────────────────────────────────────────────────────
    print("Loading training data...")
    train = pd.read_parquet(TRAIN_PARQUET)
    print(f"  Train N={len(train)}  dist: {dict(train['label'].value_counts())}")

    X_train_text = train.apply(build_text, axis=1).tolist()
    y_train = train["label"].tolist()

    print("Training Calibrated SVM TF-IDF...")
    t0 = time.time()
    svm_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)),
        ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000), cv=3, method="sigmoid")),
    ])
    svm_pipe.fit(X_train_text, y_train)
    print(f"  trained in {time.time()-t0:.1f}s")

    # ── Load gold ─────────────────────────────────────────────────────────────
    print("\nLoading gold test set...")
    gold = load_gold(GOLD_CSV)
    print(f"  Gold N={len(gold)}  dist: {dict(gold['label'].value_counts())}")

    # ── Apply pipeline ────────────────────────────────────────────────────────
    predictions = []
    sources = []
    n_rule = 0
    n_ml = 0
    n_llm = 0

    print(f"\nRunning 3-tier pipeline (threshold={threshold})...")
    llm_rows_idx = []

    # Pass 1: rule + ML, collect LLM candidates
    ml_candidates_idx = []
    rule_results = {}

    for i, row in gold.iterrows():
        cat = rule_classify(row)
        if cat is not None:
            rule_results[i] = ("rule", cat)
        else:
            ml_candidates_idx.append(i)

    # Batch ML on escalated records
    if ml_candidates_idx:
        sub = gold.loc[ml_candidates_idx]
        X_sub = sub.apply(build_text, axis=1).tolist()
        probs = svm_pipe.predict_proba(X_sub)
        classes = svm_pipe.classes_
        for idx, (i, row_idx) in enumerate(zip(ml_candidates_idx, ml_candidates_idx)):
            prob_vec = probs[idx]
            best_prob = prob_vec.max()
            best_cat = classes[prob_vec.argmax()]
            if best_prob >= threshold:
                rule_results[i] = ("ml", best_cat)
            else:
                rule_results[i] = ("llm_pending", best_cat)  # will override with LLM
                llm_rows_idx.append(i)

    print(f"  Rule handled: {len([v for v in rule_results.values() if v[0]=='rule'])}")
    print(f"  ML handled (conf≥{threshold}): {len([v for v in rule_results.values() if v[0]=='ml'])}")
    print(f"  Escalated to LLM: {len(llm_rows_idx)}")

    # LLM pass
    if llm_rows_idx and not skip_llm:
        print(f"\nCalling {model} for {len(llm_rows_idx)} records (this may take a while)...")
        llm_t0 = time.time()
        for j, i in enumerate(llm_rows_idx):
            row = gold.loc[i]
            llm_cat = llm_classify(row, model=model, base_url=llm_url)
            rule_results[i] = ("llm", llm_cat)
            if (j + 1) % 20 == 0:
                elapsed = time.time() - llm_t0
                avg_ms = elapsed * 1000 / (j + 1)
                print(f"    {j+1}/{len(llm_rows_idx)} done  avg={avg_ms:.0f}ms/req")
        total_llm_s = time.time() - llm_t0
        print(f"  LLM total: {total_llm_s:.1f}s  avg: {total_llm_s/len(llm_rows_idx)*1000:.0f}ms/req")
    elif llm_rows_idx and skip_llm:
        print("  [--skip-llm] Using ML prediction for LLM-pending records.")
        for i in llm_rows_idx:
            src, cat = rule_results[i]
            rule_results[i] = ("ml_fallback", cat)

    # Assemble final predictions
    for i in gold.index:
        src, cat = rule_results[i]
        predictions.append(cat)
        sources.append(src)
        if src == "rule":
            n_rule += 1
        elif src in ("ml",):
            n_ml += 1
        elif src in ("llm",):
            n_llm += 1
        else:
            n_ml += 1  # ml_fallback or llm_pending fallback

    y_true = gold["label"].tolist()

    # ── Results ───────────────────────────────────────────────────────────────
    macro_f1 = f1_score(y_true, predictions, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Gold N={len(gold)}  |  Rule={n_rule} ({n_rule/len(gold)*100:.1f}%)  ML={n_ml} ({n_ml/len(gold)*100:.1f}%)  LLM={n_llm} ({n_llm/len(gold)*100:.1f}%)")
    print(f"\n  Macro F1: {macro_f1:.4f}\n")
    print(classification_report(y_true, predictions, labels=LLM_OUTPUT_CATEGORIES, zero_division=0))

    # Confusion breakdown by tier
    gold_copy = gold.copy()
    gold_copy["pred"] = predictions
    gold_copy["source"] = sources

    print("\n--- Prediction source breakdown ---")
    for src in ("rule", "ml", "llm", "ml_fallback", "llm_pending"):
        mask = gold_copy["source"] == src
        if mask.sum() == 0:
            continue
        sub = gold_copy[mask]
        acc = (sub["pred"] == sub["label"]).mean()
        print(f"  {src:15s}: N={mask.sum():3d}  acc={acc:.3f}")

    # Per-category F1
    print("\n--- Per-category F1 ---")
    for cat in LLM_OUTPUT_CATEGORIES:
        tp = ((gold_copy["pred"] == cat) & (gold_copy["label"] == cat)).sum()
        fp = ((gold_copy["pred"] == cat) & (gold_copy["label"] != cat)).sum()
        fn = ((gold_copy["pred"] != cat) & (gold_copy["label"] == cat)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"  {cat:12s}: P={p:.3f}  R={r:.3f}  F1={f:.3f}  support={tp+fn}")

    # Compare against flat SVM baseline
    print("\n--- Flat SVM baseline (no rule, no LLM) ---")
    X_gold = gold.apply(build_text, axis=1).tolist()
    flat_preds = svm_pipe.predict(X_gold)
    flat_f1 = f1_score(y_true, flat_preds, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
    print(f"  Flat SVM MacroF1: {flat_f1:.4f}")
    print(classification_report(y_true, flat_preds, labels=LLM_OUTPUT_CATEGORIES, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:7b-instruct", help="Ollama model name")
    parser.add_argument("--threshold", type=float, default=0.70, help="ML confidence threshold for LLM escalation")
    parser.add_argument("--llm-url", default="http://localhost:11434")
    parser.add_argument("--skip-llm", action="store_true", help="Dry-run without calling LLM")
    args = parser.parse_args()
    run(args.model, args.threshold, args.llm_url, args.skip_llm)
