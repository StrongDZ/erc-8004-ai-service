"""Best-effort MongoDB cache for /classify responses.

The classifier is deterministic (LLM temperature=0 / seed=42; SVM + cosine are
fixed functions), so a given request always yields the same verdict. We memoize
the full ClassifyResponse keyed by a content hash of the verdict-determining
request fields (tag pair + scale + agent domain identity; see
classify._cache_payload) plus a version tag that captures server-side config not
in that payload (default LLM model, SVM quality threshold). A config or
key-shape change flips the version so stale entries are ignored rather than
served.

Stored in a dedicated collection (never touches feedback_history); a TTL index
on `expireAt` lets MongoDB drop stale entries automatically. Mongo is reused
from the existing connection — no extra infrastructure. All ops are best-effort:
on any error the cache silently disables for that call and classification
proceeds normally. Transient failure verdicts (source="fallback", confidence 0,
or a non-output category) are never cached, so an LLM outage cannot poison it.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
from functools import lru_cache

log = logging.getLogger(__name__)

_ENABLED = os.getenv("AI_SERVICE_CACHE_ENABLED", "1").strip().lower() not in ("0", "false", "no", "")
_TTL_SECONDS = int(os.getenv("AI_SERVICE_CACHE_TTL_SECONDS", str(60 * 24 * 3600)))  # 60 days
_COLL_NAME = os.getenv("AI_SERVICE_CACHE_COLLECTION", "classify_cache")
_VALID_CATEGORIES = {"junk", "quantity", "quality"}


@lru_cache(maxsize=1)
def _coll():
    """Lazy cache collection in the analyzed_agents DB, with a TTL index on
    `expireAt`. Returns None when caching is disabled or Mongo can't be reached;
    per-call errors are caught by get/set."""
    if not _ENABLED:
        return None
    try:
        from shared.mongo_client import get_db

        coll = get_db()[_COLL_NAME]
        # Idempotent: TTL index deletes a doc once now() passes its `expireAt`.
        coll.create_index("expireAt", expireAfterSeconds=0)
        return coll
    except Exception as e:  # Mongo unreachable / index error
        log.warning("classify cache disabled: %s", e)
        return None


def cache_key(version: str, payload: dict) -> str:
    """Content-addressed key: sha256 over version + canonicalised request."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(f"{version}\n{blob}".encode("utf-8")).hexdigest()


def cache_get(key: str) -> dict | None:
    """Return the cached response dict, or None on miss / any Mongo error."""
    c = _coll()
    if c is None:
        return None
    try:
        doc = c.find_one({"_id": key}, {"resp": 1})
        return doc["resp"] if doc else None
    except Exception:
        return None


def _cacheable(resp: dict) -> bool:
    """Only memoise genuine verdicts — never transient failures (LLM down)."""
    return (
        resp.get("source") != "fallback"
        and float(resp.get("confidence") or 0.0) > 0.0
        and resp.get("category") in _VALID_CATEGORIES
    )


def cache_set(key: str, resp: dict) -> None:
    """Store a verdict (best-effort, no-op on any error or non-cacheable resp)."""
    c = _coll()
    if c is None or not _cacheable(resp):
        return
    try:
        expire_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=_TTL_SECONDS)
        c.replace_one({"_id": key}, {"_id": key, "resp": resp, "expireAt": expire_at}, upsert=True)
    except Exception:
        pass
