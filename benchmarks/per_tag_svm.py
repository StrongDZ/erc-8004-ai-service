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
            rows.append({"text": f"{t1} {scale}", "label_binary": binary})
        if t2:
            rows.append({"text": f"{t2} {scale}", "label_binary": binary})
    return pd.DataFrame(rows)


def train(save: bool = True, exclude_junk: bool = False, model_path: Path | None = None) -> Pipeline:
    print("Loading training data...")
    group_a = pd.read_parquet(DATA_DIR / "group_a.parquet")
    group_b = pd.read_parquet(DATA_DIR / "group_b.parquet")
    df_all = pd.concat([group_a, group_b], ignore_index=True)
    print(f"  Combined: {len(df_all)} records")

    if exclude_junk:
        n_before = len(df_all)
        df_all = df_all[df_all["label"] != "junk"].reset_index(drop=True)
        print(f"  exclude_junk: dropped {n_before - len(df_all)} junk rows -> {len(df_all)} "
              "(quality-vs-quantity boundary trained without junk-vocabulary noise)")

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
        out_path = model_path or MODEL_PATH
        joblib.dump(pipe, out_path)
        print(f"Saved to {out_path}")
    return pipe


def load_per_tag_svm(model_path: Path | None = None) -> Pipeline:
    """Load serialized per-tag SVM. Raises FileNotFoundError if not trained yet."""
    # joblib.load is safe here: the path is always an internal artifact written by
    # this same module's train() — never loaded from user-supplied input.
    return joblib.load(model_path or MODEL_PATH)


def predict_quality_prob(pipe: Pipeline, tag: str, scale: str) -> float:
    """Return quality probability [0,1] for a single tag+scale feature."""
    text = f"{tag} {scale}"
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


TRIPLET_MODEL_PATH = MODEL_DIR / "per_tag_svm_triplet.joblib"
MODEL_3GROUP_PATH  = MODEL_DIR / "per_tag_svm_3group.joblib"


def record_to_text(tag1: str, tag2: str, scale: str) -> str:
    """Build one input text from a full (tag1, tag2, scale) record.

    No prefixes — just content words separated by spaces so that TF-IDF n-grams
    form direct tag-scale interaction bigrams (e.g. 'winrate pct100') rather than
    mediated ones ('winrate scale', 'scale pct100') produced by the old
    'tag={t} | scale={s}' format.
    """
    parts = [t.strip() for t in [tag1, tag2] if t and t.strip()]
    if scale and scale.strip():
        parts.append(scale.strip())
    return " ".join(parts)


def train_triplet(
    save: bool = True,
    exclude_junk: bool = False,
    model_path: Path | None = None,
) -> Pipeline:
    """Train SVM on (tag1, tag2, scale) triplets — one row per record, no per-tag expansion.

    Each record contributes exactly one training example. The text is built by
    record_to_text(), which puts tag1, tag2 (if present), and scale side-by-side
    so that TF-IDF bigrams capture direct tag–scale interactions.
    """
    print("Loading training data...")
    group_a = pd.read_parquet(DATA_DIR / "group_a.parquet")
    group_b = pd.read_parquet(DATA_DIR / "group_b.parquet")
    df_all = pd.concat([group_a, group_b], ignore_index=True)
    print(f"  Combined: {len(df_all)} records")

    if exclude_junk:
        n_before = len(df_all)
        df_all = df_all[df_all["label"] != "junk"].reset_index(drop=True)
        print(f"  exclude_junk: dropped {n_before - len(df_all)} junk rows -> {len(df_all)}")

    rows = []
    for _, r in df_all.iterrows():
        binary = 1 if r["label"] == "quality" else 0
        t1 = str(r.get("tag1") or "").strip()
        t2 = str(r.get("tag2") or "").strip()
        sc = str(r.get("value_scale") or "").strip()
        text = record_to_text(t1, t2, sc)
        if text:
            rows.append({"text": text, "label_binary": binary})

    train_df = pd.DataFrame(rows)
    print(f"  Triplet rows: {len(train_df)}")
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

    preds_tr = pipe.predict(X)
    print(classification_report(y, preds_tr, target_names=["non_quality", "quality"]))

    if save:
        out_path = model_path or TRIPLET_MODEL_PATH
        joblib.dump(pipe, out_path)
        print(f"Saved to {out_path}")
    return pipe


def predict_quality_prob_triplet(pipe: Pipeline, tag1: str, tag2: str, scale: str) -> float:
    """Return quality probability [0,1] for a full (tag1, tag2, scale) triplet."""
    text = record_to_text(tag1, tag2, scale)
    proba = pipe.predict_proba([text])[0]
    quality_idx = list(pipe.classes_).index(1)
    return float(proba[quality_idx])


def _3group_text(tag: str, scale: str) -> str:
    """Build 3-group feature string for one (tag, scale) pair.

    Uses __ as internal separator so each group stays a single token after
    sklearn's default word tokenizer (which treats _ as a word character):
      tag__winrate      — tag identity alone
      scale__pct100     — scale identity alone
      pair__winrate__pct100 — explicit tag×scale interaction (direct unigram)

    With ngram_range=(1,2) the bigram 'tag__X scale__Y' also forms, but the
    key gain over plain 'X Y' is that 'pair__X__Y' exists as a distinct unigram
    even when the plain bigram 'X Y' would be blocked by a separator token.
    """
    t = tag.lower().replace(" ", "_")
    s = scale.lower().replace(" ", "_") if scale else "noscale"
    return f"tag__{t} scale__{s} pair__{t}__{s}"


def train_3group(
    save: bool = True,
    exclude_junk: bool = False,
    model_path: Path | None = None,
) -> Pipeline:
    """Train per-tag SVM with explicit 3-group features (tag, scale, pair)."""
    print("Loading training data...")
    group_a = pd.read_parquet(DATA_DIR / "group_a.parquet")
    group_b = pd.read_parquet(DATA_DIR / "group_b.parquet")
    df_all = pd.concat([group_a, group_b], ignore_index=True)
    print(f"  Combined: {len(df_all)} records")

    if exclude_junk:
        n_before = len(df_all)
        df_all = df_all[df_all["label"] != "junk"].reset_index(drop=True)
        print(f"  exclude_junk: dropped {n_before - len(df_all)} -> {len(df_all)}")

    rows = []
    for _, r in df_all.iterrows():
        binary = 1 if r["label"] == "quality" else 0
        scale = str(r.get("value_scale") or "").strip()
        t1 = str(r.get("tag1") or "").strip()
        t2 = str(r.get("tag2") or "").strip()
        if t1:
            rows.append({"text": _3group_text(t1, scale), "label_binary": binary})
        if t2:
            rows.append({"text": _3group_text(t2, scale), "label_binary": binary})

    train_df = pd.DataFrame(rows)
    print(f"  Single-tag rows (3-group): {len(train_df)}")
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
    print("Training calibrated SVM 3-group (C=0.3)...")
    pipe.fit(X, y)
    preds_tr = pipe.predict(X)
    print(classification_report(y, preds_tr, target_names=["non_quality", "quality"]))

    if save:
        out_path = model_path or MODEL_3GROUP_PATH
        joblib.dump(pipe, out_path)
        print(f"Saved to {out_path}")
    return pipe


def predict_quality_prob_3group(pipe: Pipeline, tag: str, scale: str) -> float:
    """Quality probability using 3-group feature representation."""
    text = _3group_text(tag, scale)
    proba = pipe.predict_proba([text])[0]
    quality_idx = list(pipe.classes_).index(1)
    return float(proba[quality_idx])


MODEL_BGE_PATH = MODEL_DIR / "per_tag_svm_bge.joblib"


def train_bge(
    save: bool = True,
    exclude_junk: bool = False,
    model_path: Path | None = None,
) -> object:
    """Train LinearSVC on bge-small-en-v1.5 tag embeddings (no scale feature).

    Scale is intentionally excluded: the binary scale distribution is inverted
    between the training set (agent_enriched: 89% quantity) and the deployment
    population (others pool gold: 84% quality), making scale actively harmful.
    Stage 3 already handles the unbounded→quantity hard rule; the SVM's job is
    purely tag-vocabulary discrimination.

    bge embeddings generalise to unseen tags (winRate ≈ successRate in embedding
    space), which is the core failure mode of TF-IDF on the others pool.
    """
    import numpy as np
    from sklearn.svm import LinearSVC

    # Import here to avoid circular import at module load time
    from benchmarks.stage3_domain import _encode

    print("Loading training data...")
    group_a = pd.read_parquet(DATA_DIR / "group_a.parquet")
    group_b = pd.read_parquet(DATA_DIR / "group_b.parquet")
    df_all = pd.concat([group_a, group_b], ignore_index=True)
    print(f"  Combined: {len(df_all)} records")

    if exclude_junk:
        n_before = len(df_all)
        df_all = df_all[df_all["label"] != "junk"].reset_index(drop=True)
        print(f"  exclude_junk: dropped {n_before - len(df_all)} -> {len(df_all)}")

    texts, labels = [], []
    for _, r in df_all.iterrows():
        binary = 1 if r["label"] == "quality" else 0
        t1 = str(r.get("tag1") or "").strip()
        t2 = str(r.get("tag2") or "").strip()
        if t1:
            texts.append(t1); labels.append(binary)
        if t2:
            texts.append(t2); labels.append(binary)

    print(f"  Per-tag rows: {len(texts)}  (quality={sum(labels)}, non_quality={len(labels)-sum(labels)})")
    print("  Encoding tags with bge-small-en-v1.5 ...")
    X = np.array([_encode(t) for t in texts])
    y = np.array(labels)

    # CalibratedClassifierCV on bge embeddings
    clf = CalibratedClassifierCV(
        LinearSVC(C=0.3, max_iter=2000),
        cv=3, method="sigmoid",
    )
    print("  Training calibrated LinearSVC on 384-dim embeddings (C=0.3)...")
    clf.fit(X, y)

    preds_tr = clf.predict(X)
    print(classification_report(y, preds_tr, target_names=["non_quality", "quality"]))

    if save:
        out_path = model_path or MODEL_BGE_PATH
        joblib.dump(clf, out_path)
        print(f"Saved to {out_path}")
    return clf


def predict_quality_prob_bge(clf: object, tag: str) -> float:
    """Quality probability using bge-small embedding. No scale — tag only."""
    import numpy as np
    from benchmarks.stage3_domain import _encode

    vec = _encode(tag).reshape(1, -1)
    proba = clf.predict_proba(vec)[0]
    quality_idx = list(clf.classes_).index(1)
    return float(proba[quality_idx])


MODEL_HYBRID_PATH = MODEL_DIR / "per_tag_svm_hybrid.joblib"


def train_hybrid(
    save: bool = True,
    model_path: Path | None = None,
) -> dict:
    """Train SVM on concatenated [TF-IDF | bge] features per tag.

    Hybrid feature fusion:
      - TF-IDF (8000-dim, word n-grams): exact keyword signal; OOD tokens → all-zero row.
      - bge-small (384-dim, normalized): semantic signal; unseen tags map near known concepts.

    Concatenation gives the SVM an implicit OOD detector:
      (TF-IDF=0, bge≈quality) differs from (TF-IDF≠0, bge≈quality), so the boundary
      can learn that all-zero TF-IDF + bge-near-quality = junk, not real quality.

    Junk is included in training (label → non_quality=0) so the SVM sees negative examples
    with TF-IDF=0, anchoring the OOD→non_quality decision surface.
    """
    import numpy as np
    from sklearn.svm import LinearSVC
    from benchmarks.stage3_domain import _encode

    print("Loading training data (junk included for OOD anchoring)...")
    group_a = pd.read_parquet(DATA_DIR / "group_a.parquet")
    group_b = pd.read_parquet(DATA_DIR / "group_b.parquet")
    df_all = pd.concat([group_a, group_b], ignore_index=True)
    print(f"  Combined: {len(df_all)} records")

    texts, labels = [], []
    for _, r in df_all.iterrows():
        binary = 1 if r["label"] == "quality" else 0
        t1 = str(r.get("tag1") or "").strip()
        t2 = str(r.get("tag2") or "").strip()
        if t1: texts.append(t1); labels.append(binary)
        if t2: texts.append(t2); labels.append(binary)

    y = np.array(labels)
    print(f"  Per-tag rows: {len(texts)}  (quality={y.sum()}, non_quality={(y==0).sum()})")

    print("  Fitting TF-IDF vectorizer ...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(texts).toarray()  # (N, 8000)

    print("  Encoding tags with bge-small-en-v1.5 ...")
    X_bge = np.array([_encode(t) for t in texts])   # (N, 384)

    X = np.hstack([X_tfidf, X_bge])                 # (N, 8384)
    print(f"  Hybrid feature dim: {X.shape[1]}")

    clf = CalibratedClassifierCV(
        LinearSVC(C=0.3, max_iter=2000),
        cv=3, method="sigmoid",
    )
    print("  Training calibrated LinearSVC on hybrid features (C=0.3)...")
    clf.fit(X, y)

    preds_tr = clf.predict(X)
    print(classification_report(y, preds_tr, target_names=["non_quality", "quality"]))

    bundle = {"tfidf": tfidf, "clf": clf}
    if save:
        out_path = model_path or MODEL_HYBRID_PATH
        joblib.dump(bundle, out_path)
        print(f"Saved to {out_path}")
    return bundle


def predict_quality_prob_hybrid(bundle: dict, tag: str) -> float:
    """Quality probability using hybrid [TF-IDF | bge] features. No scale."""
    import numpy as np
    from benchmarks.stage3_domain import _encode

    tfidf = bundle["tfidf"]
    clf   = bundle["clf"]
    X_tfidf = tfidf.transform([tag]).toarray()       # (1, 8000)
    X_bge   = _encode(tag).reshape(1, -1)            # (1, 384)
    X = np.hstack([X_tfidf, X_bge])                  # (1, 8384)
    proba = clf.predict_proba(X)[0]
    quality_idx = list(clf.classes_).index(1)
    return float(proba[quality_idx])


MODEL_QTAG_HYBRID_PATH = MODEL_DIR / "per_tag_svm_qtag_hybrid.joblib"


def train_qtag_hybrid(
    save: bool = True,
    model_path: Path | None = None,
) -> dict:
    """Train quality-vs-quantity SVM on tag text only (TF-IDF + BGE), junk excluded.

    Key differences from train_hybrid():
    - Only quality and quantity records are used (junk excluded entirely).
    - Features: tag text only — scale is intentionally excluded.
    - Labels: quality=1, quantity=0 (no non_quality catch-all).

    Because junk is OOD for this SVM, junk inputs produce low max_proba in the
    combiner and fall through to LLM fallback rather than being silently
    misclassified as quality/quantity.
    """
    import numpy as np
    from benchmarks.stage3_domain import _encode

    print("Loading training data (junk excluded, tag-only features)...")
    group_a = pd.read_parquet(DATA_DIR / "group_a.parquet")
    group_b = pd.read_parquet(DATA_DIR / "group_b.parquet")
    df_all = pd.concat([group_a, group_b], ignore_index=True)
    n_before = len(df_all)
    df_all = df_all[df_all["label"].isin(["quality", "quantity"])].reset_index(drop=True)
    print(f"  Dropped {n_before - len(df_all)} junk rows -> {len(df_all)} records")
    print(f"  Label dist: {df_all['label'].value_counts().to_dict()}")

    texts, labels = [], []
    for _, r in df_all.iterrows():
        lbl = 1 if r["label"] == "quality" else 0
        t1 = str(r.get("tag1") or "").strip()
        t2 = str(r.get("tag2") or "").strip()
        if t1:
            texts.append(t1); labels.append(lbl)
        if t2:
            texts.append(t2); labels.append(lbl)

    y = np.array(labels)
    print(f"  Per-tag rows: {len(texts)}  (quality={y.sum()}, quantity={(y==0).sum()})")

    print("  Fitting TF-IDF vectorizer ...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(texts).toarray()

    print("  Encoding tags with bge-small-en-v1.5 ...")
    X_bge = np.array([_encode(t) for t in texts])

    X = np.hstack([X_tfidf, X_bge])
    print(f"  Hybrid feature dim: {X.shape[1]}")

    clf = CalibratedClassifierCV(
        LinearSVC(C=0.3, max_iter=2000),
        cv=3, method="sigmoid",
    )
    print("  Training calibrated LinearSVC (C=0.3) ...")
    clf.fit(X, y)

    preds_tr = clf.predict(X)
    print(classification_report(y, preds_tr, target_names=["quantity", "quality"]))

    bundle = {"tfidf": tfidf, "clf": clf}
    if save:
        out_path = model_path or MODEL_QTAG_HYBRID_PATH
        joblib.dump(bundle, out_path)
        print(f"Saved to {out_path}")
    return bundle


def predict_qtag_proba(bundle: dict, tag: str) -> tuple[float, float]:
    """Return (p_quality, p_quantity) for one tag using hybrid [TF-IDF | BGE] features."""
    import numpy as np
    from benchmarks.stage3_domain import _encode

    tfidf = bundle["tfidf"]
    clf = bundle["clf"]
    X_tfidf = tfidf.transform([tag]).toarray()
    X_bge = _encode(tag).reshape(1, -1)
    X = np.hstack([X_tfidf, X_bge])
    proba = clf.predict_proba(X)[0]
    quality_idx = list(clf.classes_).index(1)
    quantity_idx = list(clf.classes_).index(0)
    return float(proba[quality_idx]), float(proba[quantity_idx])


MODEL_BGE_QUALITY_GATE_PATH = MODEL_DIR / "per_tag_svm_bge_quality_gate.joblib"


def train_bge_quality_gate(
    save: bool = True,
    model_path: Path | None = None,
) -> object:
    """Train the one-directional "quality-only" Stage-2 gate (production design,
    mirrors benchmarks/pipeline_run13.py).

    Unlike train(), vote_per_tag(), and every other per-tag SVM in this module,
    this classifier's probability is NEVER used to assert "quantity" or
    "non_quality" — only ever to assert "quality" at a high threshold. Auditing
    the symmetric tie-breaks used elsewhere in this module showed they are wrong
    more often than right when they fire on the "quantity" side (low confidence
    reflects unfamiliar business/service-domain vocabulary, not evidence of a
    metric), so the caller must escalate every record this gate does not
    confidently mark "quality" rather than use a complementary low-confidence
    threshold.

    Features: BGE embedding of "<tag> <scale>" (scale included, unlike
    train_bge()/train_hybrid(), because the inverted scale distribution that
    motivated dropping it there does not apply once the classifier is never
    asked to assert "quantity"). Junk is excluded from training (quality-vs-
    quantity binary only), consistent with train_qtag_hybrid() and the
    confirmed finding that mixing junk into this SVM's training set degrades
    even the one-directional "quality" decision.
    """
    import numpy as np
    from benchmarks.stage3_domain import _encode

    print("Loading training data (junk excluded, tag+scale features)...")
    group_a = pd.read_parquet(DATA_DIR / "group_a.parquet")
    group_b = pd.read_parquet(DATA_DIR / "group_b.parquet")
    df_all = pd.concat([group_a, group_b], ignore_index=True)
    n_before = len(df_all)
    df_all = df_all[df_all["label"].isin(["quality", "quantity"])].reset_index(drop=True)
    print(f"  Dropped {n_before - len(df_all)} junk rows -> {len(df_all)} records")

    texts, labels = [], []
    for _, r in df_all.iterrows():
        lbl = 1 if r["label"] == "quality" else 0
        scale = str(r.get("value_scale") or "").strip().lower()
        t1 = str(r.get("tag1") or "").strip().lower()
        t2 = str(r.get("tag2") or "").strip().lower()
        if t1:
            texts.append(f"{t1} {scale}"); labels.append(lbl)
        if t2:
            texts.append(f"{t2} {scale}"); labels.append(lbl)

    y = np.array(labels)
    print(f"  Per-tag rows: {len(texts)}  (quality={y.sum()}, quantity={(y==0).sum()})")

    print("  Encoding with bge-small-en-v1.5 ...")
    X = np.array([_encode(t) for t in texts])

    clf = CalibratedClassifierCV(LinearSVC(C=0.3, max_iter=2000), cv=3, method="sigmoid")
    print("  Training calibrated LinearSVC (C=0.3) ...")
    clf.fit(X, y)

    preds_tr = clf.predict(X)
    print(classification_report(y, preds_tr, target_names=["quantity", "quality"]))

    if save:
        out_path = model_path or MODEL_BGE_QUALITY_GATE_PATH
        joblib.dump(clf, out_path)
        print(f"Saved to {out_path}")
    return clf


def predict_quality_gate_prob(clf: object, tag: str, scale: str) -> float:
    """Quality probability for the one-directional gate. Caller must treat any
    value below the deployed threshold as "not confident", never as evidence
    for "quantity"."""
    from benchmarks.stage3_domain import _encode

    vec = _encode(f"{tag.strip().lower()} {scale.strip().lower()}").reshape(1, -1)
    proba = clf.predict_proba(vec)[0]
    quality_idx = list(clf.classes_).index(1)
    return float(proba[quality_idx])


if __name__ == "__main__":
    train(save=True)
