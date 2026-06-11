#!/usr/bin/env python3
"""Benchmark all classification methods on the ERC-8004 feedback dataset.

Pipeline:
  1. Load rule-based classified feedback from MongoDB (unique tag combos only).
  2. Load hand-labelled "others" gold CSV for benchmark evaluation.
  3. Train each model on rule-based data; evaluate on both rule-based test
     split AND hand-labelled gold.
  4. Save per-model results + summary comparison table.

Usage:
    cd erc-8004-ai-service
    python -m benchmarks.run_all_benchmarks
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# ---------- project imports ------------------------------------------------
# Add parent dir to path so shared/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.types import ALL_CATEGORIES, LLM_OUTPUT_CATEGORIES, RULE_TO_5CAT, SCORED_CATEGORIES

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------- paths ----------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
BENCH_OUTPUT_DIR = DATA_DIR / "benchmark_results"
BENCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def _offchain_to_text(fp) -> str:
    """Extract text from feedbackParsed."""
    if fp is None or (isinstance(fp, float) and np.isnan(fp)):
        return ""
    if isinstance(fp, str):
        try:
            fp = json.loads(fp)
        except Exception:
            return fp[:500] if fp else ""
    if isinstance(fp, dict):
        # Try common text fields
        for key in ("comment", "review", "text", "description", "message", "feedback", "content"):
            if key in fp:
                return str(fp[key])[:500]
        # Fallback to JSON dump
        return json.dumps(fp, ensure_ascii=False)[:500]
    return str(fp)[:500]


def build_feature_text(row: pd.Series) -> str:
    """Combine tag1, tag2, value_scale, and offchain into a single text feature."""
    parts = []
    tag1 = str(row.get("tag1", "") or "").strip()
    tag2 = str(row.get("tag2", "") or "").strip()
    vscale = str(row.get("value_scale", "") or "").strip()
    endpoint = str(row.get("endpoint", "") or "").strip()

    if tag1:
        parts.append(f"tag1={tag1}")
    if tag2:
        parts.append(f"tag2={tag2}")
    if vscale:
        parts.append(f"scale={vscale}")
    if endpoint:
        # Extract just the host for brevity
        import urllib.parse
        try:
            host = urllib.parse.urlparse(endpoint).hostname or endpoint
            parts.append(f"endpoint={host}")
        except Exception:
            parts.append(f"endpoint={endpoint}")

    fp = row.get("feedback_parsed")
    offchain = _offchain_to_text(fp)
    if offchain:
        parts.append(f"offchain={offchain[:300]}")

    return " | ".join(parts) if parts else "<empty>"


def load_rule_based_data_from_parquet() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load existing parquet splits. De-duplicate by (tag1, tag2) to get unique feedback."""
    log.info("Loading rule-based data from parquet files...")
    train = pd.read_parquet(SPLITS_DIR / "rule_based" / "train.parquet")
    val = pd.read_parquet(SPLITS_DIR / "rule_based" / "val.parquet")
    test = pd.read_parquet(SPLITS_DIR / "rule_based" / "test.parquet")

    log.info("Before dedup — train: %d, val: %d, test: %d", len(train), len(val), len(test))

    # De-duplicate: remove rows that share BOTH tag1 AND tag2 AND the same
    # feedback_parsed content (truly duplicate submissions). Keep one per
    # unique (id) if 'id' column exists, otherwise (tag1, tag2, feedback_parsed hash).
    def dedup(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "id" in df.columns:
            deduped = df.drop_duplicates(subset=["id"], keep="first")
        else:
            df["_tag1"] = df["tag1"].fillna("").astype(str)
            df["_tag2"] = df["tag2"].fillna("").astype(str)
            df["_fp_hash"] = df["feedback_parsed"].apply(
                lambda x: str(x)[:200] if x is not None else ""
            )
            deduped = df.drop_duplicates(subset=["_tag1", "_tag2", "_fp_hash", "label"], keep="first")
            deduped = deduped.drop(columns=["_tag1", "_tag2", "_fp_hash"], errors="ignore")
        return deduped.reset_index(drop=True)

    train = dedup(train)
    val = dedup(val)
    test = dedup(test)

    log.info("After dedup — train: %d, val: %d, test: %d", len(train), len(val), len(test))
    log.info("Train label distribution:\n%s", train["label"].value_counts().to_string())
    return train, val, test


def load_rule_based_data_from_mongo() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load from MongoDB with stratified sampling, de-duplicate by (tag1, tag2)."""
    log.info("Loading rule-based data from MongoDB...")
    from shared.data_loader import split_train_val_test, stratified_sample

    df = stratified_sample(
        per_category=3000,
        seed=42,
        categories=["junk", "service_feedback", "config_feedback", "app_specific"],
    )

    # De-duplicate by (tag1, tag2, label)
    df["tag1_str"] = df["tag1"].fillna("").astype(str)
    df["tag2_str"] = df["tag2"].fillna("").astype(str)
    df = df.drop_duplicates(subset=["tag1_str", "tag2_str", "rule_category"], keep="first")
    df = df.drop(columns=["tag1_str", "tag2_str"])

    # Rename to label
    df = df.rename(columns={"rule_category": "label"})
    df = df[df["label"].isin(LLM_OUTPUT_CATEGORIES)].reset_index(drop=True)

    log.info("Total unique records from MongoDB: %d", len(df))
    log.info("Label distribution:\n%s", df["label"].value_counts().to_string())

    train, val, test = split_train_val_test(df, label_col="label", train_frac=0.7, val_frac=0.15, seed=42)
    return train, val, test


def load_hand_labelled_gold() -> pd.DataFrame:
    """Load the hand-labelled gold benchmark set."""
    gold_path = SPLITS_DIR / "hand_labelled" / "test.parquet"
    if gold_path.exists():
        log.info("Loading hand-labelled gold from parquet: %s", gold_path)
        df = pd.read_parquet(gold_path)
        log.info("Gold set: %d records, labels:\n%s", len(df), df["label"].value_counts().to_string())
        return df

    # Fallback to CSV
    csv_candidates = [
        ROOT / "scripts" / "labelled" / "others_gold_v1.csv",
        ROOT / "data" / "others_gold_v1.csv",
    ]
    for csv_path in csv_candidates:
        if csv_path.exists():
            log.info("Loading gold from CSV: %s", csv_path)
            from shared.data_loader import load_hand_labelled_csv
            return load_hand_labelled_csv(csv_path)

    log.warning("No hand-labelled gold found. Using rule-based test split for benchmark.")
    return pd.DataFrame()


def prepare_features(df: pd.DataFrame) -> pd.Series:
    """Build combined text feature column."""
    return df.apply(build_feature_text, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    name: str,
    y_true: list[str],
    y_pred: list[str],
    train_time_s: float,
    inference_time_s: float,
    n_train: int,
) -> dict:
    """Compute full evaluation for one model."""
    # Filter to scored categories only (exclude 'others')
    yt_scored, yp_scored = [], []
    for t, p in zip(y_true, y_pred, strict=False):
        if t in SCORED_CATEGORIES:
            yt_scored.append(t)
            yp_scored.append(p)

    scored_labels = SCORED_CATEGORIES

    # Classification report
    report = classification_report(
        yt_scored, yp_scored,
        labels=scored_labels, output_dict=True, zero_division=0,
    )

    # Confusion matrix (full including others)
    cm = confusion_matrix(y_true, y_pred, labels=ALL_CATEGORIES)
    cm_df = pd.DataFrame(cm,
                         index=[f"true_{c}" for c in ALL_CATEGORIES],
                         columns=[f"pred_{c}" for c in ALL_CATEGORIES])

    # Macro F1
    mf1 = f1_score(yt_scored, yp_scored, labels=scored_labels, average="macro", zero_division=0)
    wf1 = f1_score(yt_scored, yp_scored, labels=scored_labels, average="weighted", zero_division=0)

    # Accuracy over scored
    correct = sum(1 for t, p in zip(yt_scored, yp_scored) if t == p)
    acc = correct / len(yt_scored) if yt_scored else 0.0

    per_class = {}
    for cat in scored_labels:
        if cat in report:
            per_class[cat] = {
                "precision": round(report[cat]["precision"], 4),
                "recall": round(report[cat]["recall"], 4),
                "f1": round(report[cat]["f1-score"], 4),
                "support": int(report[cat]["support"]),
            }

    return {
        "model": name,
        "accuracy": round(acc, 4),
        "macro_f1": round(mf1, 4),
        "weighted_f1": round(wf1, 4),
        "n_train": n_train,
        "n_test": len(y_true),
        "n_scored": len(yt_scored),
        "train_time_s": round(train_time_s, 2),
        "inference_time_s": round(inference_time_s, 2),
        "avg_inference_ms": round(inference_time_s / max(len(y_true), 1) * 1000, 2),
        "per_class": per_class,
        "confusion_matrix": cm_df,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class BaseModel:
    """Base class for all benchmark models."""
    name: str = "base"

    def train(self, X_train_text: pd.Series, y_train: pd.Series):
        raise NotImplementedError

    def predict(self, X_test_text: pd.Series) -> list[str]:
        raise NotImplementedError


class LogisticRegressionTFIDF(BaseModel):
    name = "logistic_regression_tfidf"

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=20000, ngram_range=(1, 2),
                sublinear_tf=True, min_df=2,
            )),
            ("clf", LogisticRegression(
                C=1.0, class_weight="balanced",
                max_iter=2000, random_state=42,
                solver="lbfgs",
            )),
        ])

    def train(self, X_train_text, y_train):
        self.pipeline.fit(X_train_text, y_train)

    def predict(self, X_test_text):
        return self.pipeline.predict(X_test_text).tolist()


class SVMLinearTFIDF(BaseModel):
    name = "svm_linear_tfidf"

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=20000, ngram_range=(1, 2),
                sublinear_tf=True, min_df=2,
            )),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42),
                cv=3,
            )),
        ])

    def train(self, X_train_text, y_train):
        self.pipeline.fit(X_train_text, y_train)

    def predict(self, X_test_text):
        return self.pipeline.predict(X_test_text).tolist()


class NaiveBayesTFIDF(BaseModel):
    name = "naive_bayes_tfidf"

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=20000, ngram_range=(1, 2),
                sublinear_tf=True, min_df=2,
            )),
            ("clf", MultinomialNB(alpha=0.1)),
        ])

    def train(self, X_train_text, y_train):
        self.pipeline.fit(X_train_text, y_train)

    def predict(self, X_test_text):
        return self.pipeline.predict(X_test_text).tolist()


class RandomForestTFIDF(BaseModel):
    name = "random_forest_tfidf"

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=10000, ngram_range=(1, 2),
                sublinear_tf=True, min_df=2,
            )),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=None,
                class_weight="balanced", random_state=42,
                n_jobs=-1,
            )),
        ])

    def train(self, X_train_text, y_train):
        self.pipeline.fit(X_train_text, y_train)

    def predict(self, X_test_text):
        return self.pipeline.predict(X_test_text).tolist()


class GradientBoostingTFIDF(BaseModel):
    name = "gradient_boosting_tfidf"

    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            sublinear_tf=True, min_df=2,
        )
        self.clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.1, random_state=42,
            subsample=0.8,
        )

    def train(self, X_train_text, y_train):
        X = self.tfidf.fit_transform(X_train_text)
        self.clf.fit(X.toarray(), y_train)

    def predict(self, X_test_text):
        X = self.tfidf.transform(X_test_text)
        return self.clf.predict(X.toarray()).tolist()


class KNNEmbedding(BaseModel):
    name = "knn_embedding"

    def __init__(self, k: int = 7):
        self.k = k
        self.embedder = None
        self.vectors = None
        self.labels = None

    def _load_embedder(self):
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self.embedder

    def train(self, X_train_text, y_train):
        emb = self._load_embedder()
        log.info("  [kNN] Encoding %d training texts...", len(X_train_text))
        self.vectors = emb.encode(
            X_train_text.tolist(), normalize_embeddings=True,
            show_progress_bar=True, batch_size=128,
        )
        self.labels = y_train.tolist()

    def predict(self, X_test_text):
        emb = self._load_embedder()
        log.info("  [kNN] Encoding %d test texts...", len(X_test_text))
        q_vectors = emb.encode(
            X_test_text.tolist(), normalize_embeddings=True,
            show_progress_bar=True, batch_size=128,
        )
        sims = q_vectors @ self.vectors.T  # (n_test, n_train)
        preds = []
        for i in range(len(q_vectors)):
            top_k = np.argsort(sims[i])[::-1][:self.k]
            from collections import Counter
            votes = Counter(self.labels[j] for j in top_k)
            preds.append(votes.most_common(1)[0][0])
        return preds


class EmbeddingLogReg(BaseModel):
    name = "embedding_logreg"

    def __init__(self):
        self.embedder = None
        self.clf = LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=2000, random_state=42,
        )

    def _load_embedder(self):
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self.embedder

    def train(self, X_train_text, y_train):
        emb = self._load_embedder()
        log.info("  [EmbedLogReg] Encoding %d training texts...", len(X_train_text))
        vectors = emb.encode(
            X_train_text.tolist(), normalize_embeddings=True,
            show_progress_bar=True, batch_size=128,
        )
        self.clf.fit(vectors, y_train)

    def predict(self, X_test_text):
        emb = self._load_embedder()
        log.info("  [EmbedLogReg] Encoding %d test texts...", len(X_test_text))
        vectors = emb.encode(
            X_test_text.tolist(), normalize_embeddings=True,
            show_progress_bar=True, batch_size=128,
        )
        return self.clf.predict(vectors).tolist()


# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    model: BaseModel,
    X_train_text: pd.Series,
    y_train: pd.Series,
    test_sets: dict[str, tuple[pd.Series, pd.Series]],
) -> dict[str, dict]:
    """Train one model and evaluate on all test sets."""
    log.info("=" * 60)
    log.info("Training: %s", model.name)
    log.info("  Train size: %d", len(X_train_text))

    # Train
    t0 = time.time()
    model.train(X_train_text, y_train)
    train_time = time.time() - t0
    log.info("  Train time: %.2fs", train_time)

    results = {}
    for test_name, (X_test_text, y_test) in test_sets.items():
        if len(X_test_text) == 0:
            log.warning("  Skipping empty test set: %s", test_name)
            continue

        log.info("  Evaluating on: %s (%d records)", test_name, len(X_test_text))
        t0 = time.time()
        y_pred = model.predict(X_test_text)
        infer_time = time.time() - t0

        result = evaluate_model(
            name=model.name,
            y_true=y_test.tolist(),
            y_pred=y_pred,
            train_time_s=train_time,
            inference_time_s=infer_time,
            n_train=len(X_train_text),
        )

        log.info("  [%s] Macro-F1=%.4f | Accuracy=%.4f | Inference=%.2fs",
                 test_name, result["macro_f1"], result["accuracy"], infer_time)

        results[test_name] = result

    return results


def save_results(all_results: dict[str, dict[str, dict]], output_dir: Path):
    """Save all results to files."""
    # Per-model detail files
    for model_name, test_results in all_results.items():
        for test_name, result in test_results.items():
            fname = f"{model_name}__{test_name}.json"
            detail = {k: v for k, v in result.items() if k != "confusion_matrix"}
            with open(output_dir / fname, "w") as f:
                json.dump(detail, f, indent=2, ensure_ascii=False)

            # Confusion matrix CSV
            cm_fname = f"{model_name}__{test_name}__confusion.csv"
            result["confusion_matrix"].to_csv(output_dir / cm_fname)

    # Summary comparison table
    rows = []
    for model_name, test_results in all_results.items():
        for test_name, result in test_results.items():
            rows.append({
                "model": model_name,
                "test_set": test_name,
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "weighted_f1": result["weighted_f1"],
                "n_train": result["n_train"],
                "n_test": result["n_test"],
                "train_time_s": result["train_time_s"],
                "inference_time_s": result["inference_time_s"],
                "avg_inference_ms": result["avg_inference_ms"],
            })

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["test_set", "macro_f1"], ascending=[True, False])
    summary_df.to_csv(output_dir / "summary_comparison.csv", index=False)
    log.info("Summary saved to %s", output_dir / "summary_comparison.csv")

    return summary_df


def generate_report(
    all_results: dict[str, dict[str, dict]],
    summary_df: pd.DataFrame,
    train_info: dict,
    output_dir: Path,
):
    """Generate a comprehensive markdown report."""
    report_lines = [
        "# 📊 Feedback Classification — Benchmark Results",
        "",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "---",
        "",
        "## 1. Data Summary",
        "",
        f"- **Training data source**: Rule-based classified feedback (unique tag combinations)",
        f"- **Training records**: {train_info['n_train']}",
        f"- **De-duplication**: Records de-duplicated by (tag1, tag2, label) to ensure uniqueness",
        f"- **Categories (4 scored)**: junk, service_feedback, config_feedback, app_specific",
        "",
        "### Training Data Distribution",
        "",
        "| Category | Count |",
        "|----------|-------|",
    ]

    for cat, count in train_info["label_distribution"].items():
        report_lines.append(f"| {cat} | {count} |")

    report_lines += [
        "",
        "### Test Sets",
        "",
    ]
    for test_name, info in train_info["test_sets"].items():
        report_lines.append(f"- **{test_name}**: {info['n']} records")

    report_lines += [
        "",
        "---",
        "",
        "## 2. Overall Comparison",
        "",
    ]

    # Per test set comparison table
    for test_name in summary_df["test_set"].unique():
        subset = summary_df[summary_df["test_set"] == test_name].copy()
        subset = subset.sort_values("macro_f1", ascending=False)

        report_lines += [
            f"### Test Set: `{test_name}`",
            "",
            "| Rank | Model | Macro-F1 | Weighted-F1 | Accuracy | Train (s) | Avg Inference (ms) |",
            "|:----:|-------|:--------:|:-----------:|:--------:|:---------:|:------------------:|",
        ]

        for rank, (_, row) in enumerate(subset.iterrows(), 1):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))
            report_lines.append(
                f"| {medal} | {row['model']} | {row['macro_f1']:.4f} | "
                f"{row['weighted_f1']:.4f} | {row['accuracy']:.4f} | "
                f"{row['train_time_s']:.2f} | {row['avg_inference_ms']:.2f} |"
            )

        report_lines.append("")

    # Per model detailed results
    report_lines += [
        "---",
        "",
        "## 3. Per-Model Detailed Results",
        "",
    ]

    for model_name, test_results in all_results.items():
        report_lines += [
            f"### {model_name}",
            "",
        ]

        for test_name, result in test_results.items():
            report_lines += [
                f"#### Test Set: `{test_name}`",
                "",
                f"- Macro-F1: **{result['macro_f1']:.4f}**",
                f"- Weighted-F1: **{result['weighted_f1']:.4f}**",
                f"- Accuracy: **{result['accuracy']:.4f}**",
                f"- N-train: {result['n_train']} | N-test: {result['n_test']} | N-scored: {result['n_scored']}",
                f"- Train time: {result['train_time_s']:.2f}s | Inference: {result['inference_time_s']:.2f}s",
                "",
                "**Per-class metrics:**",
                "",
                "| Category | Precision | Recall | F1 | Support |",
                "|----------|:---------:|:------:|:--:|:-------:|",
            ]

            for cat, metrics in result["per_class"].items():
                report_lines.append(
                    f"| {cat} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                    f"{metrics['f1']:.4f} | {metrics['support']} |"
                )

            # Confusion matrix
            cm = result["confusion_matrix"]
            report_lines += [
                "",
                "**Confusion Matrix:**",
                "",
                "| | " + " | ".join(cm.columns) + " |",
                "|" + "|".join(["---"] * (len(cm.columns) + 1)) + "|",
            ]
            for idx, row in cm.iterrows():
                report_lines.append(
                    "| " + str(idx) + " | " + " | ".join(str(v) for v in row) + " |"
                )
            report_lines.append("")

    # File locations
    report_lines += [
        "---",
        "",
        "## 4. File Locations",
        "",
        "### Input Data",
        "",
        f"- Training data (rule-based): `{SPLITS_DIR / 'rule_based'}/`",
        f"  - `train.parquet` — {train_info['n_train']} records",
        f"  - `val.parquet`",
        f"  - `test.parquet`",
        f"- Hand-labelled gold: `{SPLITS_DIR / 'hand_labelled' / 'test.parquet'}`",
        "",
        "### Output Files",
        "",
        f"- Summary CSV: `{output_dir / 'summary_comparison.csv'}`",
        f"- This report: `{output_dir / 'BENCHMARK_REPORT.md'}`",
        "",
        "**Per-model result files:**",
        "",
    ]

    for model_name, test_results in all_results.items():
        for test_name in test_results:
            report_lines.append(f"- `{output_dir / f'{model_name}__{test_name}.json'}`")
            report_lines.append(f"- `{output_dir / f'{model_name}__{test_name}__confusion.csv'}`")

    report_lines += [
        "",
        "---",
        "",
        "## 5. Methodology Notes",
        "",
        "1. **Data preparation**: Feedback records de-duplicated by (tag1, tag2, label) to avoid",
        "   inflating accuracy with repeated tag patterns.",
        "2. **Feature engineering**: Combined text feature from tag1, tag2, value_scale, endpoint,",
        "   and offchain content (feedbackParsed). Format: `tag1=X | tag2=Y | scale=Z | ...`",
        "3. **Scoring**: Only 4 semantic categories scored (junk, service_feedback, config_feedback,",
        "   app_specific). 'others' excluded from F1 as it's a fallback bucket, not a real class.",
        "4. **Class balancing**: All models use `class_weight='balanced'` where supported.",
        "5. **Embedding models** (kNN, EmbedLogReg): Use `all-MiniLM-L6-v2` sentence transformer.",
        "6. **TF-IDF models**: Use (1,2)-gram features with sublinear TF weighting.",
        "",
    ]

    report_path = output_dir / "BENCHMARK_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    log.info("Report saved to %s", report_path)
    return report_path


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("FEEDBACK CLASSIFICATION BENCHMARK")
    log.info("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────
    # Try parquet first (faster, no Mongo needed); fall back to Mongo
    if (SPLITS_DIR / "rule_based" / "train.parquet").exists():
        train_df, val_df, test_rb_df = load_rule_based_data_from_parquet()
    else:
        train_df, val_df, test_rb_df = load_rule_based_data_from_mongo()
        # Save for future runs
        rb_dir = SPLITS_DIR / "rule_based"
        rb_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(rb_dir / "train.parquet", index=False)
        val_df.to_parquet(rb_dir / "val.parquet", index=False)
        test_rb_df.to_parquet(rb_dir / "test.parquet", index=False)

    # Combine train + val for final training (standard practice)
    full_train = pd.concat([train_df, val_df], ignore_index=True)

    # Load hand-labelled gold
    gold_df = load_hand_labelled_gold()

    # ── 2. Feature engineering ────────────────────────────────────────────
    log.info("Building text features...")
    X_train_text = prepare_features(full_train)
    y_train = full_train["label"]

    test_sets: dict[str, tuple[pd.Series, pd.Series]] = {}

    X_test_rb = prepare_features(test_rb_df)
    y_test_rb = test_rb_df["label"]
    test_sets["rule_based_test"] = (X_test_rb, y_test_rb)

    if len(gold_df) > 0:
        X_test_gold = prepare_features(gold_df)
        y_test_gold = gold_df["label"]
        test_sets["hand_labelled_gold"] = (X_test_gold, y_test_gold)

    # Save processed features for inspection
    feat_dir = BENCH_OUTPUT_DIR / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": X_train_text, "label": y_train}).to_parquet(
        feat_dir / "train_features.parquet", index=False
    )
    for tname, (xt, yt) in test_sets.items():
        pd.DataFrame({"text": xt, "label": yt}).to_parquet(
            feat_dir / f"{tname}_features.parquet", index=False
        )

    # ── 3. Define models ──────────────────────────────────────────────────
    models: list[BaseModel] = [
        LogisticRegressionTFIDF(),
        SVMLinearTFIDF(),
        NaiveBayesTFIDF(),
        RandomForestTFIDF(),
        GradientBoostingTFIDF(),
        KNNEmbedding(k=7),
        EmbeddingLogReg(),
    ]

    # ── 4. Run benchmarks ─────────────────────────────────────────────────
    all_results: dict[str, dict[str, dict]] = {}

    for model in models:
        try:
            results = run_benchmark(model, X_train_text, y_train, test_sets)
            all_results[model.name] = results
        except Exception as e:
            log.error("FAILED: %s — %s", model.name, e, exc_info=True)
            all_results[model.name] = {}

    # ── 5. Save results ───────────────────────────────────────────────────
    summary_df = save_results(all_results, BENCH_OUTPUT_DIR)

    train_info = {
        "n_train": len(full_train),
        "label_distribution": y_train.value_counts().to_dict(),
        "test_sets": {
            name: {"n": len(xt)} for name, (xt, _) in test_sets.items()
        },
    }

    report_path = generate_report(all_results, summary_df, train_info, BENCH_OUTPUT_DIR)

    # ── 6. Print summary ──────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("BENCHMARK COMPLETE")
    log.info("=" * 70)
    print("\n" + summary_df.to_string(index=False))
    print(f"\n📄 Full report: {report_path}")
    print(f"📁 All outputs: {BENCH_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
