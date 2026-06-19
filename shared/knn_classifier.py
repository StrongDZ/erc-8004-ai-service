"""Embedding-based kNN classifier for 'others' feedback records.

Builds a corpus of labelled feedback embeddings sampled from the
rule-annotated MongoDB and classifies query records by cosine similarity.

The corpus is populated lazily on first use and kept in memory for the
lifetime of the process. Thread safety: FastAPI runs single-process by
default so no concurrent build races arise in practice.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
from collections import Counter

import numpy as np

from .data_loader import stratified_sample
from .types import LLM_OUTPUT_CATEGORIES, RULE_TO_CAT, ClassificationResult

log = logging.getLogger(__name__)

# Only the 4 real output categories go into the corpus — "others" is excluded
# because those records have no definitive label.
_SCORED_CATS: set[str] = set(LLM_OUTPUT_CATEGORIES)


# ── Value-tier helpers (Python port of Go AssignTier, new semantics) ───────────

def _assign_tier_v2(real: float) -> str:
    """New-semantics tier: only exact 0 or 1 → binary; (0,1) exclusive → unbounded."""
    abs_val = abs(real)
    if real == 0.0 or real == 1.0:
        return "binary"
    if abs_val < 1.0:
        return "unbounded"
    if real <= 5.0:
        return "star5"
    if real <= 10.0:
        return "star10"
    if abs_val <= 100.0:
        return "pct100"
    return "unbounded"


def _normalize_value(real: float, tier: str) -> float:
    """Normalize real value to [-1, 1] given its tier (matches NormalizeValueWithScale)."""
    def clamp(v: float) -> float:
        return max(-1.0, min(1.0, v))
    if tier == "binary":
        return 1.0 if real >= 0.5 else 0.0
    if tier == "star5":
        return clamp(real / 5.0)
    if tier == "star10":
        return clamp(real / 10.0)
    if tier == "unbounded":
        return 0.0
    return clamp(real / 100.0)  # pct100


def _row_value_fields(row: dict) -> tuple[float, int, str]:
    """Compute (value_norm, value_decimals, score_tier) from a corpus row."""
    try:
        raw = str(row.get("value", "") or "")
        dec = int(row.get("value_decimals", 0) or 0)
        real = float(raw) / (10 ** dec) if dec > 0 else float(raw)
        tier = _assign_tier_v2(real)
        norm = _normalize_value(real, tier)
        return norm, dec, tier
    except (ValueError, ZeroDivisionError, OverflowError):
        return 0.0, 0, ""


def _host(url: str) -> str:
    """Extract hostname from a URL; return the original string if not a URL."""
    try:
        h = urllib.parse.urlparse(url).hostname
        return h or url
    except Exception:
        return url


def feedback_embed_text(
    tag1: str,
    tag2: str,
    endpoint: str = "",
    offchain_content: str = "",
    value_norm: float = 0.0,
    value_decimals: int = 0,
    score_tier: str = "",
) -> str:
    """Flat embedding text for one feedback record (no agent context needed).

    Uses only the fields available without a per-record agent look-up so the
    corpus can be built from a plain MongoDB scan. The classify endpoint
    passes the same fields in the same format so corpus and query live in the
    same embedding space.

    score_tier is the scale tier string (binary/star5/star10/pct100/unbounded)
    computed by AssignTier. value_decimals signals how many decimal places the
    on-chain value was stored with — a key discriminator: decimals=0 + large
    value → pct100/quantity; decimals=18 → fractional ETH-like value → unbounded.
    """
    ep = _host(endpoint) if (endpoint or "").strip() else ""
    off = (offchain_content or "")[:300]
    parts = [
        f"tag1={tag1.strip()}" if (tag1 or "").strip() else "",
        f"tag2={tag2.strip()}" if (tag2 or "").strip() else "",
        f"scale={score_tier}" if (score_tier or "").strip() else "",
        f"decimals={value_decimals}" if value_decimals else "",
        f"value={value_norm:.2f}" if value_norm != 0.0 else "",
        f"endpoint={ep}" if ep else "",
        f"offchain={off}" if off else "",
    ]
    return " | ".join(p for p in parts if p)


class KNNCorpus:
    """In-memory kNN retrieval index over rule-labelled feedback records.

    Usage:
        corpus = KNNCorpus(embedder)
        corpus.build()            # once — samples Mongo, encodes, stores
        result = corpus.classify(text)
    """

    def __init__(
        self,
        embedder,
        per_category: int = 1000,
        k: int = 7,
        seed: int = 42,
    ) -> None:
        self.embedder = embedder
        self.per_category = per_category
        self.k = k
        self.seed = seed
        self._vectors: np.ndarray | None = None   # (N, D) float32, L2-normalised
        self._labels: list[str] = []
        self._built = False

    # ── public ────────────────────────────────────────────────────────────────

    @property
    def vectors(self) -> np.ndarray:
        """Corpus embedding matrix (built on first access). Shared with the linear head."""
        if not self._built:
            self.build()
        return self._vectors

    @property
    def labels(self) -> list[str]:
        """Corpus labels aligned with `vectors` (built on first access)."""
        if not self._built:
            self.build()
        return self._labels

    def build(self) -> None:
        """Sample corpus from MongoDB and encode to dense vectors.

        Samples `per_category` records from each scored category (junk, quality,
        quantity). Mongo queries use legacy label aliases so pre-migration rows
        still contribute exemplars.
        """
        t0 = time.monotonic()
        df = stratified_sample(
            per_category=self.per_category,
            seed=self.seed,
            categories=list(LLM_OUTPUT_CATEGORIES),
        )
        df = df[df["rule_category"].isin(_SCORED_CATS)].reset_index(drop=True)

        texts: list[str] = []
        for _, row in df.iterrows():
            off = ""
            fp = row.get("feedback_parsed")
            if fp:
                try:
                    off = json.dumps(fp, ensure_ascii=False)
                except Exception:
                    pass
            val_norm, val_dec, score_tier = _row_value_fields(row.to_dict())
            texts.append(feedback_embed_text(
                row.get("tag1", "") or "",
                row.get("tag2", "") or "",
                row.get("endpoint", "") or "",
                off,
                value_norm=val_norm,
                value_decimals=val_dec,
                score_tier=score_tier,
            ))

        self._vectors = self.embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        self._labels = df["rule_category"].tolist()
        self._built = True
        log.info(
            "KNN corpus built: %d records in %.1fs (embed_dim=%d)",
            len(self._labels),
            time.monotonic() - t0,
            self._vectors.shape[1],
        )

    def classify(self, text: str, k: int | None = None) -> ClassificationResult:
        """Classify one feedback record by cosine similarity to the corpus.

        Returns the majority-vote category among the top-k neighbours, with
        confidence = vote_count / k.
        """
        if not self._built:
            self.build()

        k = k or self.k
        t0 = time.monotonic()
        q = self.embedder.encode([text], normalize_embeddings=True)
        # vectors are L2-normalised → dot product == cosine similarity
        sims: np.ndarray = (q @ self._vectors.T)[0]   # (N,)
        top_k = np.argsort(sims)[::-1][:k].tolist()
        votes: Counter[str] = Counter(self._labels[i] for i in top_k)
        predicted, count = votes.most_common(1)[0]
        predicted = RULE_TO_CAT.get(predicted, predicted)
        confidence = count / k
        reason = "kNN k={}: {}".format(
            k, ", ".join(f"{cat}={n}" for cat, n in votes.most_common())
        )
        return ClassificationResult(
            category=predicted,
            confidence=confidence,
            reason=reason,
            source="embedding",
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
