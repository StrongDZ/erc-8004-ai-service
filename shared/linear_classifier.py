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
