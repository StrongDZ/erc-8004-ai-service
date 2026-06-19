#!/usr/bin/env python3
"""Per-tag binary SVM: train on single (tag, scale) features, binary quality vs non-quality.

Training data: Group A + Group B from agent_enriched dataset.
Each record (tag1, tag2, scale, label) → 2 rows:
  (tag1, scale, binary_label) and (tag2, scale, binary_label) if tag2 non-empty.

Binary label: 1 = quality, 0 = non-quality (quantity or junk).
Uses soft-margin (C=0.3) to tolerate label noise from per-tag expansion.

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.per_tag_svm
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data/splits/agent_enriched"
MODEL_DIR = ROOT / "data/models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "per_tag_svm.joblib"


def expand_to_single_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Expand each record to 2 single-tag training rows."""
    rows = []
    for _, r in df.iterrows():
        binary = 1 if r["label"] == "quality" else 0
        scale = str(r.get("value_scale") or "").strip()
        t1 = str(r.get("tag1") or "").strip()
        t2 = str(r.get("tag2") or "").strip()
        if t1:
            rows.append({"text": f"tag={t1} | scale={scale}", "label_binary": binary})
        if t2:
            rows.append({"text": f"tag={t2} | scale={scale}", "label_binary": binary})
    return pd.DataFrame(rows)


def train(save: bool = True) -> Pipeline:
    print("Loading training data...")
    group_a = pd.read_parquet(DATA_DIR / "group_a.parquet")
    group_b = pd.read_parquet(DATA_DIR / "group_b.parquet")
    df_all = pd.concat([group_a, group_b], ignore_index=True)
    print(f"  Combined: {len(df_all)} records")

    print("Expanding to single-tag rows...")
    train_df = expand_to_single_tags(df_all)
    print(f"  Single-tag rows: {len(train_df)}")
    print(f"  Binary label dist:\n{train_df['label_binary'].value_counts().to_string()}")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=0.3, max_iter=2000),
            cv=3, method="sigmoid",
        )),
    ])
    X = train_df["text"].tolist()
    y = train_df["label_binary"].tolist()
    print("Training calibrated SVM (C=0.3)...")
    pipe.fit(X, y)

    # Quick self-eval on training data (sanity only, not held-out)
    preds = pipe.predict(X)
    print(classification_report(y, preds, target_names=["non_quality", "quality"]))

    if save:
        joblib.dump(pipe, MODEL_PATH)
        print(f"Saved to {MODEL_PATH}")
    return pipe


def load_per_tag_svm() -> Pipeline:
    """Load serialized per-tag SVM. Raises FileNotFoundError if not trained yet."""
    # joblib.load is safe here: MODEL_PATH is an internal artifact written by
    # this same module's train() — never loaded from user-supplied input.
    return joblib.load(MODEL_PATH)


def predict_quality_prob(pipe: Pipeline, tag: str, scale: str) -> float:
    """Return quality probability [0,1] for a single tag+scale feature."""
    text = f"tag={tag} | scale={scale}"
    proba = pipe.predict_proba([text])[0]
    # classes_[1] == 1 (quality)
    quality_idx = list(pipe.classes_).index(1)
    return float(proba[quality_idx])


def vote_per_tag(
    p1: float,
    p2: float,
    t2_empty: bool = False,
    thresh: float = 0.70,
) -> str | None:
    """Stage 2 voting combiner — single source of truth, imported by the pipeline runner.

    p1, p2: quality probability for tag1, tag2 (p2 ignored when t2_empty).
    Confidence is symmetric: a tag is confidently 'quality' at p >= thresh, confidently
    'non_quality' at p <= 1 - thresh, and "not confident" in between.
    Returns 'quality', 'non_quality', or None (escalate to Stage 3).
    """

    def _confident_class(p: float) -> str | None:
        if p >= thresh:
            return "quality"
        if p <= 1.0 - thresh:
            return "non_quality"
        return None

    c1 = _confident_class(p1)
    c2 = None if t2_empty else _confident_class(p2)

    if t2_empty:
        return c1

    if c1 is not None and c2 is not None:
        return c1 if c1 == c2 else None  # real conflict (e.g. quality vs non_quality) → Stage 3

    if c1 is not None:
        return c1
    if c2 is not None:
        return c2
    return None


if __name__ == "__main__":
    train(save=True)
