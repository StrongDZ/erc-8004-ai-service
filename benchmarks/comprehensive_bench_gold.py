#!/usr/bin/env python3
"""
comprehensive_bench_gold.py — Re-benchmark all A/B/C configurations on rich gold set.

Training data : data/splits/agent_enriched/group_a.parquet + group_b.parquet (N≈1032, rule-based)
Test data     : data/labelled/pure_others_stratified_dedup.csv (N=1486, human-labelled gold)
LLM cache     : data/benchmark_results/llm_cache.json

Design decisions per config:
  A1 LogReg    : class_weight="balanced" (handles imbalance natively)
  A2 SVM       : class_weight="balanced"
  A3 NaiveBayes: fit_prior=False + uniform class_prior=[1/3,1/3,1/3] (NB has no class_weight)
  A4 GBT       : sample_weight proportional to inverse class frequency
  A5 RF        : class_weight="balanced"
  B1 kNN       : weights='distance' (distance-weighted voting mitigates imbalance without resampling)
  B2 LogReg Emb: class_weight="balanced"
  C  LLM-only  : LLM cache (feedback_id key → label value)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits" / "agent_enriched"
GOLD_CSV = DATA_DIR / "labelled" / "pure_others_stratified_dedup.csv"
LLM_CACHE_PATH = DATA_DIR / "benchmark_results" / "llm_cache.json"
OUT_DIR = DATA_DIR / "benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["quality", "quantity", "junk"]


# ── Feature builders ──────────────────────────────────────────────────────────

def _offchain_text(val) -> str:
    """Extract text from offchain_note (already plain text in gold)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if isinstance(val, dict):
        for k in ("comment", "review", "text", "description", "message", "feedback", "content"):
            if v := val.get(k):
                return str(v)[:300]
        return str(val)[:300]
    return str(val).strip()[:300]


def build_feature(row: pd.Series, is_gold: bool = False) -> str:
    """Build combined text feature string. Handles both train and gold column names."""
    parts: list[str] = []

    tag1 = str(row.get("tag1") or "").strip().lower()
    tag2 = str(row.get("tag2") or "").strip().lower()
    if tag1:
        parts.append(f"tag1={tag1}")
    if tag2:
        parts.append(f"tag2={tag2}")

    # value_scale in train → scale in gold
    scale = str(row.get("value_scale") or row.get("scale") or "").strip().lower()
    if scale:
        parts.append(f"scale={scale}")

    # endpoint hostname
    endpoint = str(row.get("endpoint") or "").strip()
    if endpoint and endpoint not in ("nan", "none", ""):
        from urllib.parse import urlparse
        host = urlparse(endpoint).hostname or endpoint
        parts.append(f"endpoint={host}")

    # offchain text: feedback_parsed (train) or offchain_note (gold)
    raw = row.get("feedback_parsed") if not is_gold else row.get("offchain_note")
    offchain = _offchain_text(raw)
    if offchain:
        parts.append(f"offchain={offchain[:300]}")

    return " | ".join(parts) if parts else "<empty>"


def build_features(df: pd.DataFrame, is_gold: bool = False) -> pd.Series:
    return df.apply(lambda r: build_feature(r, is_gold), axis=1)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_train() -> pd.DataFrame:
    """Load group_a + group_b (rule-based, N≈1032, labels: quality/quantity/junk)."""
    ga = pd.read_parquet(SPLITS_DIR / "group_a.parquet")
    gb = pd.read_parquet(SPLITS_DIR / "group_b.parquet")
    df = pd.concat([ga, gb], ignore_index=True)
    print(f"Train: {len(df)} records — {df['label'].value_counts().to_dict()}")
    return df


def load_gold() -> pd.DataFrame:
    """Load human-labelled gold set."""
    df = pd.read_csv(GOLD_CSV)
    df = df.rename(columns={"human_label": "label"})
    print(f"Gold : {len(df)} records — {df['label'].value_counts().to_dict()}")
    return df


def load_llm_cache() -> dict[str, str]:
    """Load LLM classification cache: {feedback_id → label}."""
    cache = json.load(open(LLM_CACHE_PATH))
    print(f"LLM cache: {len(cache)} entries")
    return cache


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(y_true, y_pred, labels=CATEGORIES):
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {}
    for c in labels:
        r = report.get(c, {})
        support = int(r.get("support", 0))
        # Balanced accuracy per class = recall (= true positive rate for that class)
        per_class[c] = {
            "precision": round(r.get("precision", 0), 4),
            "recall": round(r.get("recall", 0), 4),
            "f1": round(r.get("f1-score", 0), 4),
            "support": support,
        }

    q_f1 = per_class["quality"]["f1"]
    qty_f1 = per_class["quantity"]["f1"]
    macro_2cls = round((q_f1 + qty_f1) / 2, 4)
    macro_3cls = round(report["macro avg"]["f1-score"], 4)
    weighted = round(report["weighted avg"]["f1-score"], 4)

    # Overall balanced accuracy (macro recall)
    macro_recall = round(report["macro avg"]["recall"], 4)

    return {
        "macro_f1_2cls": macro_2cls,
        "macro_f1_3cls": macro_3cls,
        "weighted_f1": weighted,
        "balanced_accuracy": macro_recall,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "cm_labels": labels,
    }


def collect_errors(y_true, y_pred, df_test: pd.DataFrame, n=3) -> list[dict]:
    """Collect representative misclassification examples."""
    errors = []
    for true_cat in ["quality", "quantity"]:
        for pred_cat in ["quality", "quantity"]:
            if true_cat == pred_cat:
                continue
            mask = (y_true == true_cat) & (y_pred == pred_cat)
            idxs = np.where(mask)[0][:n]
            for i in idxs:
                row = df_test.iloc[i]
                errors.append({
                    "true": true_cat,
                    "pred": pred_cat,
                    "tag1": str(row.get("tag1", "")),
                    "tag2": str(row.get("tag2", "")),
                    "scale": str(row.get("scale", "")),
                    "offchain": str(row.get("offchain_note", ""))[:120],
                })
    return errors


# ── Group A — TF-IDF classifiers ─────────────────────────────────────────────

def _tfidf_fit(X_train, max_features=20000):
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    return tfidf, tfidf.fit(X_train)


def run_a1_logreg(X_train, y_train, X_test, y_test, df_test):
    print("  [A1] LogReg TF-IDF...")
    t0 = time.time()
    tfidf, _ = _tfidf_fit(X_train)
    Xtr = tfidf.transform(X_train)
    clf = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000)
    clf.fit(Xtr, y_train)
    train_secs = time.time() - t0

    t1 = time.time()
    Xte = tfidf.transform(X_test)
    y_pred = clf.predict(Xte)
    infer_secs = time.time() - t1

    return {
        "train_secs": round(train_secs, 2),
        "infer_secs": round(infer_secs, 3),
        "avg_infer_ms": round(infer_secs / len(y_test) * 1000, 3),
        **metrics(y_test, y_pred),
        "errors": collect_errors(np.array(y_test), y_pred, df_test),
    }


def run_a2_svm(X_train, y_train, X_test, y_test, df_test):
    print("  [A2] SVM TF-IDF...")
    t0 = time.time()
    tfidf, _ = _tfidf_fit(X_train)
    Xtr = tfidf.transform(X_train)
    clf = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", max_iter=2000))
    clf.fit(Xtr, y_train)
    train_secs = time.time() - t0

    t1 = time.time()
    Xte = tfidf.transform(X_test)
    y_pred = clf.predict(Xte)
    infer_secs = time.time() - t1

    return {
        "train_secs": round(train_secs, 2),
        "infer_secs": round(infer_secs, 3),
        "avg_infer_ms": round(infer_secs / len(y_test) * 1000, 3),
        **metrics(y_test, y_pred),
        "errors": collect_errors(np.array(y_test), y_pred, df_test),
    }


def run_a3_naivebayes(X_train, y_train, X_test, y_test, df_test):
    """NaiveBayes with uniform class prior (no class_weight support → fix prior)."""
    print("  [A3] NaiveBayes TF-IDF (uniform prior)...")
    t0 = time.time()
    tfidf, _ = _tfidf_fit(X_train)
    Xtr = tfidf.transform(X_train)
    # Uniform prior forces equal P(class) regardless of training distribution
    n_classes = len(set(y_train))
    clf = MultinomialNB(alpha=0.1, fit_prior=False, class_prior=[1.0 / n_classes] * n_classes)
    clf.fit(Xtr, y_train)
    train_secs = time.time() - t0

    t1 = time.time()
    Xte = tfidf.transform(X_test)
    y_pred = clf.predict(Xte)
    infer_secs = time.time() - t1

    return {
        "train_secs": round(train_secs, 2),
        "infer_secs": round(infer_secs, 3),
        "avg_infer_ms": round(infer_secs / len(y_test) * 1000, 3),
        **metrics(y_test, y_pred),
        "errors": collect_errors(np.array(y_test), y_pred, df_test),
    }


def run_a4_gbt(X_train, y_train, X_test, y_test, df_test):
    """GradientBoosting with sample_weight (no class_weight → use inverse freq weights)."""
    print("  [A4] GradientBoosting TF-IDF (sample_weight)...")
    t0 = time.time()
    tfidf, _ = _tfidf_fit(X_train, max_features=5000)
    Xtr = tfidf.transform(X_train)
    sw = compute_sample_weight("balanced", y_train)
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, random_state=42
    )
    clf.fit(Xtr, y_train, sample_weight=sw)
    train_secs = time.time() - t0

    t1 = time.time()
    Xte = tfidf.transform(X_test)
    y_pred = clf.predict(Xte)
    infer_secs = time.time() - t1

    return {
        "train_secs": round(train_secs, 2),
        "infer_secs": round(infer_secs, 3),
        "avg_infer_ms": round(infer_secs / len(y_test) * 1000, 3),
        **metrics(y_test, y_pred),
        "errors": collect_errors(np.array(y_test), y_pred, df_test),
    }


def run_a5_rf(X_train, y_train, X_test, y_test, df_test):
    print("  [A5] RandomForest TF-IDF...")
    t0 = time.time()
    tfidf, _ = _tfidf_fit(X_train, max_features=10000)
    Xtr = tfidf.transform(X_train)
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(Xtr, y_train)
    train_secs = time.time() - t0

    t1 = time.time()
    Xte = tfidf.transform(X_test)
    y_pred = clf.predict(Xte)
    infer_secs = time.time() - t1

    return {
        "train_secs": round(train_secs, 2),
        "infer_secs": round(infer_secs, 3),
        "avg_infer_ms": round(infer_secs / len(y_test) * 1000, 3),
        **metrics(y_test, y_pred),
        "errors": collect_errors(np.array(y_test), y_pred, df_test),
    }


# ── Group B — Frozen embedding (all-MiniLM-L6-v2) ────────────────────────────

def _encode(texts: list[str], model) -> np.ndarray:
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embs, dtype=np.float32)


def run_b1_knn_distance(X_train_texts, y_train, X_test_texts, y_test, df_test):
    """kNN with distance-weighted voting (weights='distance') to mitigate imbalance."""
    print("  [B1] kNN distance-weighted (loading MiniLM)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    t0 = time.time()
    Xtr = _encode(list(X_train_texts), model)
    Xte = _encode(list(X_test_texts), model)
    encode_secs = time.time() - t0

    t1 = time.time()
    clf = KNeighborsClassifier(n_neighbors=7, metric="cosine", weights="distance")
    clf.fit(Xtr, list(y_train))
    train_secs = time.time() - t1

    t2 = time.time()
    y_pred = clf.predict(Xte)
    infer_secs = time.time() - t2

    return {
        "encode_secs": round(encode_secs, 2),
        "train_secs": round(train_secs, 3),
        "infer_secs": round(infer_secs, 3),
        "avg_infer_ms": round(infer_secs / len(y_test) * 1000, 3),
        **metrics(y_test, y_pred),
        "errors": collect_errors(np.array(y_test), np.array(y_pred), df_test),
    }


def run_b2_logreg_emb(X_train_texts, y_train, X_test_texts, y_test, df_test):
    print("  [B2] LogReg on MiniLM embeddings...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    t0 = time.time()
    Xtr = _encode(list(X_train_texts), model)
    Xte = _encode(list(X_test_texts), model)
    encode_secs = time.time() - t0

    t1 = time.time()
    clf = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000)
    clf.fit(Xtr, list(y_train))
    train_secs = time.time() - t1

    t2 = time.time()
    y_pred = clf.predict(Xte)
    infer_secs = time.time() - t2

    return {
        "encode_secs": round(encode_secs, 2),
        "train_secs": round(train_secs, 3),
        "infer_secs": round(infer_secs, 3),
        "avg_infer_ms": round(infer_secs / len(y_test) * 1000, 3),
        **metrics(y_test, y_pred),
        "errors": collect_errors(np.array(y_test), np.array(y_pred), df_test),
    }


# ── Group C — LLM-only (cache) ────────────────────────────────────────────────

def run_c_llm_cache(gold_df: pd.DataFrame, llm_cache: dict[str, str]):
    """Classify using LLM cache. Records not in cache are skipped (logged)."""
    print("  [C] LLM-only (from cache)...")
    y_true, y_pred = [], []
    missing = 0
    for _, row in gold_df.iterrows():
        fid = str(row.get("feedback_id", ""))
        label_true = row["label"]
        label_pred = llm_cache.get(fid)
        if label_pred is None:
            missing += 1
            continue
        if label_pred not in CATEGORIES:
            missing += 1
            continue
        y_true.append(label_true)
        y_pred.append(label_pred)

    print(f"    cache hits: {len(y_true)}/{len(gold_df)}  missing: {missing}")

    df_test_sub = gold_df[gold_df["feedback_id"].apply(lambda x: llm_cache.get(str(x)) in CATEGORIES)]
    errors = collect_errors(np.array(y_true), np.array(y_pred), df_test_sub.reset_index(drop=True))

    return {
        "n_from_cache": len(y_true),
        "n_missing": missing,
        **metrics(y_true, y_pred),
        "errors": errors,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("COMPREHENSIVE BENCHMARK — RICH GOLD (N=1486)")
    print("=" * 65)

    # Load data
    train_df = load_train()
    gold_df = load_gold()
    llm_cache = load_llm_cache()

    # Build features
    print("\nBuilding features...")
    X_train = build_features(train_df, is_gold=False)
    y_train = train_df["label"].tolist()

    X_gold = build_features(gold_df, is_gold=True)
    y_gold = gold_df["label"].tolist()

    results = {}

    # ── Group A ───────────────────────────────────────────────────────────────
    print("\n--- GROUP A: TF-IDF Classifiers ---")
    results["A1_logreg_tfidf"] = run_a1_logreg(X_train, y_train, X_gold, y_gold, gold_df)
    results["A2_svm_tfidf"] = run_a2_svm(X_train, y_train, X_gold, y_gold, gold_df)
    results["A3_naivebayes_tfidf"] = run_a3_naivebayes(X_train, y_train, X_gold, y_gold, gold_df)
    results["A4_gbt_tfidf"] = run_a4_gbt(X_train, y_train, X_gold, y_gold, gold_df)
    results["A5_rf_tfidf"] = run_a5_rf(X_train, y_train, X_gold, y_gold, gold_df)

    # ── Group B ───────────────────────────────────────────────────────────────
    print("\n--- GROUP B: Frozen Embeddings (MiniLM) ---")
    results["B1_knn_dist_weighted"] = run_b1_knn_distance(X_train, y_train, X_gold, y_gold, gold_df)
    results["B2_logreg_embedding"] = run_b2_logreg_emb(X_train, y_train, X_gold, y_gold, gold_df)

    # ── Group C ───────────────────────────────────────────────────────────────
    print("\n--- GROUP C: LLM-only (cache) ---")
    results["C_llm_only"] = run_c_llm_cache(gold_df, llm_cache)

    # ── Save raw JSON ─────────────────────────────────────────────────────────
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"comprehensive_bench_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved → {out_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Config':<25} {'2cls-MacroF1':>12} {'Q.F1':>7} {'Qty.F1':>7} {'Q.Rec':>7} {'Qty.Rec':>8}")
    print("-" * 65)
    for name, r in results.items():
        pc = r.get("per_class", {})
        qf = pc.get("quality", {}).get("f1", 0)
        qr = pc.get("quality", {}).get("recall", 0)
        qtyf = pc.get("quantity", {}).get("f1", 0)
        qtyr = pc.get("quantity", {}).get("recall", 0)
        m2 = r.get("macro_f1_2cls", 0)
        print(f"{name:<25} {m2:>12.4f} {qf:>7.4f} {qtyf:>7.4f} {qr:>7.4f} {qtyr:>8.4f}")

    return out_path


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
