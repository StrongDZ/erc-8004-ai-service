"""Linear classifier head (logistic regression) over feedback embeddings.

Trains on the exact same corpus (vectors + labels) that KNNCorpus builds, so the
only difference vs the kNN baseline is the decision rule: a learned linear
boundary with balanced class weights instead of a cosine majority vote. This
directly targets kNN's majority-class bias on the imbalanced 'others' pool.

predict_proba gives a calibrated confidence; inference is a single matmul so it
is faster than kNN's full-corpus similarity scan.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .types import ClassificationResult

log = logging.getLogger(__name__)


class EmbeddingLinearClassifier:
    """Logistic-regression head over precomputed (L2-normalised) embeddings."""

    def __init__(self, embedder, vectors, labels, c: float = 1.0, seed: int = 42) -> None:
        self.embedder = embedder
        self._vectors = vectors          # (N, D) float32, L2-normalised (shared with kNN)
        self._labels = labels            # list[str], same order as vectors
        self.c = c
        self.seed = seed
        self._clf = None
        self._classes: list[str] = []
        self._built = False

    def build(self) -> None:
        """Fit LogisticRegression(class_weight='balanced') on the corpus."""
        from sklearn.linear_model import LogisticRegression

        t0 = time.monotonic()
        clf = LogisticRegression(
            C=self.c,
            class_weight="balanced",
            max_iter=2000,
            random_state=self.seed,
        )
        clf.fit(self._vectors, self._labels)
        self._clf = clf
        self._classes = list(clf.classes_)
        self._built = True
        log.info(
            "Linear head trained on %d records in %.1fs (classes=%s)",
            len(self._labels), time.monotonic() - t0, self._classes,
        )

    def classify(self, text: str) -> ClassificationResult:
        """Encode one record and return the argmax-probability class."""
        if not self._built:
            self.build()
        t0 = time.monotonic()
        q = self.embedder.encode([text], normalize_embeddings=True)
        proba = self._clf.predict_proba(q)[0]
        order = np.argsort(proba)[::-1]
        best = int(order[0])
        reason = "logreg: " + ", ".join(
            f"{self._classes[i]}={proba[i]:.2f}" for i in order[:3]
        )
        return ClassificationResult(
            category=self._classes[best],
            confidence=float(proba[best]),
            reason=reason,
            source="linear",
            latency_ms=int((time.monotonic() - t0) * 1000),
        )


class EnrichedLinearClassifier:
    """Late-fusion logistic-regression head: [feedback_vec ‖ agent_vec].

    feedback_vec = embed(tag1|tag2|endpoint|offchain)   — same tower as kNN/linear
    agent_vec    = embed(description + OASF caption/desc) — zero when absent

    Trains on the rule-labelled corpus enriched with each record's agent context
    (fetched from Mongo + OASF schema), so the head can learn to lean on the
    transferable agent-domain signal for the app_specific vs service_feedback
    boundary that plain feedback embeddings collapse on for the 'others' pool.
    """

    def __init__(self, embedder, per_category: int = 1000, c: float = 1.0, seed: int = 42) -> None:
        self.embedder = embedder
        self.per_category = per_category
        self.c = c
        self.seed = seed
        self._clf = None
        self._classes: list[str] = []
        self._dim = 0
        self._built = False

    def build(self) -> None:
        import json

        from sklearn.linear_model import LogisticRegression

        from .data_loader import stratified_sample
        from .knn_classifier import feedback_embed_text
        from .mongo_client import fetch_agents_by_keys
        from .oasf_enrich import agent_domain_text
        from .types import LLM_OUTPUT_CATEGORIES

        t0 = time.monotonic()
        df = stratified_sample(
            per_category=self.per_category,
            seed=self.seed,
            categories=["junk", "service_feedback", "config_feedback", "app_specific"],
        )
        df = df[df["rule_category"].isin(set(LLM_OUTPUT_CATEGORIES))].reset_index(drop=True)

        keys = {(int(r["chain_id"]), str(r["agent_id"])) for _, r in df.iterrows()}
        agents = fetch_agents_by_keys(keys)

        fb_texts: list[str] = []
        ag_texts: list[str] = []
        for _, row in df.iterrows():
            off = ""
            fp = row.get("feedback_parsed")
            if fp:
                try:
                    off = json.dumps(fp, ensure_ascii=False)
                except Exception:
                    pass
            fb_texts.append(feedback_embed_text(
                row.get("tag1", "") or "", row.get("tag2", "") or "",
                row.get("endpoint", "") or "", off,
            ))
            ag = agents.get(f"{int(row['chain_id'])}:{row['agent_id']}", {})
            ag_texts.append(agent_domain_text(
                ag.get("description", "") or "",
                ag.get("oasfDomains") or [],
                ag.get("oasfSkills") or [],
            ))

        fb_vecs = self.embedder.encode(fb_texts, normalize_embeddings=True, show_progress_bar=False)
        self._dim = fb_vecs.shape[1]
        ag_vecs = self._encode_agent(ag_texts, self._dim)
        X = np.hstack([fb_vecs, ag_vecs])

        clf = LogisticRegression(
            C=self.c, class_weight="balanced", max_iter=2000, random_state=self.seed,
        )
        clf.fit(X, df["rule_category"].tolist())
        self._clf = clf
        self._classes = list(clf.classes_)
        self._built = True
        nonempty = sum(1 for t in ag_texts if t.strip())
        log.info(
            "Enriched linear head: %d records (dim=%d×2, agent_text present=%d) in %.1fs",
            len(df), self._dim, nonempty, time.monotonic() - t0,
        )

    def _encode_agent(self, texts: list[str], dim: int) -> np.ndarray:
        """Encode agent texts; rows with empty text become zero vectors."""
        idx = [i for i, t in enumerate(texts) if t.strip()]
        out = np.zeros((len(texts), dim), dtype="float32")
        if idx:
            enc = self.embedder.encode(
                [texts[i] for i in idx], normalize_embeddings=True, show_progress_bar=False,
            )
            for j, i in enumerate(idx):
                out[i] = enc[j]
        return out

    def classify(self, feedback_text: str, agent_text: str = "") -> ClassificationResult:
        if not self._built:
            self.build()
        t0 = time.monotonic()
        fb = self.embedder.encode([feedback_text], normalize_embeddings=True)
        if (agent_text or "").strip():
            ag = self.embedder.encode([agent_text], normalize_embeddings=True)
        else:
            ag = np.zeros((1, fb.shape[1]), dtype="float32")
        X = np.hstack([fb, ag])
        proba = self._clf.predict_proba(X)[0]
        order = np.argsort(proba)[::-1]
        best = int(order[0])
        reason = "logreg-enriched: " + ", ".join(
            f"{self._classes[i]}={proba[i]:.2f}" for i in order[:3]
        )
        return ClassificationResult(
            category=self._classes[best],
            confidence=float(proba[best]),
            reason=reason,
            source="linear_enriched",
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
