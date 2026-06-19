"""Ollama HTTP client — Python port of internal/domain/classifier/llm_client.go.

Behaviour preserved:
  - temperature=0, seed=42, num_predict=128, repeat_penalty=1.3
  - Structured-output schema on /api/chat (Ollama v0.5+)
  - JSON extraction tolerates leading/trailing text around the {...}
  - LLM output is restricted to the 4 real categories
  - On client/parse/schema errors → category=others, confidence=0.0, source=fallback

V7 uses two sequential calls: category prompt, then feature prompt (skipped for junk).
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import httpx

from .types import LLM_OUTPUT_CATEGORIES, ClassificationResult

_DEFAULT_TIMEOUT = 120.0
_V7_TAXONOMY = frozenset({"v6", "v7"})


def _uses_v7_taxonomy(prompt_version: str) -> bool:
    return prompt_version in _V7_TAXONOMY or prompt_version.startswith("v7_")


class OllamaClient:
    """Synchronous Ollama client.

    Use one instance per model to keep the loaded weights warm; Ollama
    unloads on model switch.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        system_prompt: str = "",
        seed: int = 42,
        num_predict: int = 128,
        repeat_penalty: float = 1.3,
    ):
        self.model = model
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.seed = seed
        self.num_predict = num_predict
        self.repeat_penalty = repeat_penalty
        self._client = httpx.Client(timeout=timeout + 2)

    def health(self) -> bool:
        try:
            r = self._client.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code < 300
        except httpx.HTTPError:
            return False

    def classify(
        self,
        user_message: str,
        *,
        allowed_categories: list[str] | None = None,
        prompt_version: str = "v5",
        feature_client: OllamaClient | None = None,
    ) -> ClassificationResult:
        """Run classification.

        v7: category call on this client (category system prompt); optional
        `feature_client` for the second feature call. If omitted, v7 runs
        category-only (feature stays None).

        v6: single call returning category + feature on this client.
        """
        if prompt_version == "v7":
            return self.classify_v7(
                user_message,
                allowed_categories=allowed_categories,
                feature_client=feature_client,
            )

        t0 = time.monotonic()
        default_cats = ["junk", "quantity", "quality"] if _uses_v7_taxonomy(prompt_version) else list(LLM_OUTPUT_CATEGORIES)
        enum = list(allowed_categories) if allowed_categories else default_cats
        if not enum:
            enum = default_cats

        properties: dict[str, Any] = {
            "category": {"type": "string", "enum": enum},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        }
        required = ["category", "confidence", "reason"]
        if prompt_version == "v6":
            properties["feature"] = {
                "type": ["string", "null"],
                "enum": ["infrastructure", "agent_domain", "both", "null", "", None],
            }

        raw, err = self._chat_json(self.system_prompt, user_message, properties, required)
        if err:
            return _fallback(err, t0)
        return _parse_category_output(raw, t0, allowed=enum, include_feature=(prompt_version == "v6"))

    def classify_v7(
        self,
        user_message: str,
        *,
        allowed_categories: list[str] | None = None,
        feature_client: OllamaClient | None = None,
        category_only: bool = False,
    ) -> ClassificationResult:
        """Two-call v7 path: category → feature (feature skipped for junk)."""
        t0 = time.monotonic()
        enum = list(allowed_categories) if allowed_categories else ["junk", "quantity", "quality"]
        if not enum:
            enum = ["junk", "quantity", "quality"]

        cat_props = {
            "category": {"type": "string", "enum": enum},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        }
        raw, err = self._chat_json(
            self.system_prompt,
            user_message,
            cat_props,
            ["category", "confidence", "reason"],
        )
        if err:
            return _fallback(err, t0)

        cat_result = _parse_category_output(raw, t0, allowed=enum, include_feature=False)
        if cat_result.source == "fallback":
            return cat_result

        if cat_result.category == "junk" or category_only or feature_client is None:
            total_ms = int((time.monotonic() - t0) * 1000)
            return ClassificationResult(
                category=cat_result.category,
                confidence=cat_result.confidence,
                reason=cat_result.reason,
                source=cat_result.source,
                latency_ms=total_ms,
                raw_output=cat_result.raw_output,
                feature=None,
            )

        feat_user = user_message + f"\n<assigned_category>{cat_result.category}</assigned_category>"
        feat_props = {
            "feature": {
                "type": ["string", "null"],
                "enum": ["infrastructure", "agent_domain", "both", "null", "", None],
            },
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        }
        feat_raw, feat_err = feature_client._chat_json(
            feature_client.system_prompt,
            feat_user,
            feat_props,
            ["feature", "confidence", "reason"],
        )
        total_ms = int((time.monotonic() - t0) * 1000)
        if feat_err:
            return ClassificationResult(
                category=cat_result.category,
                confidence=cat_result.confidence,
                reason=f"{cat_result.reason} | feature_fallback: {feat_err}",
                source=cat_result.source,
                latency_ms=total_ms,
                raw_output=cat_result.raw_output,
                feature=None,
            )

        feat = _parse_feature_only(feat_raw, cat_result.confidence)
        reason = cat_result.reason
        if feat.reason:
            reason = f"{reason} | feature: {feat.reason}"

        return ClassificationResult(
            category=cat_result.category,
            confidence=min(cat_result.confidence, feat.confidence) if feat.confidence else cat_result.confidence,
            reason=reason,
            source="llm",
            latency_ms=total_ms,
            raw_output=cat_result.raw_output,
            feature=feat.feature,
        )

    def _chat_json(
        self,
        system_prompt: str,
        user_message: str,
        properties: dict[str, Any],
        required: list[str],
    ) -> tuple[str, str | None]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "format": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
            "options": {
                "temperature": 0.0,
                "seed": self.seed,
                "num_predict": self.num_predict,
                "repeat_penalty": self.repeat_penalty,
            },
        }
        try:
            r = self._client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", ""), None
        except httpx.HTTPError as e:
            return "", f"http_error: {e}"

    def summarize(self, system: str, user: str, num_predict: int = 96) -> str:
        """Generic text generation (no enum schema) — used by agent summarisation."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "seed": self.seed, "num_predict": num_predict},
        }
        try:
            r = self._client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            return (r.json().get("message", {}).get("content") or "").strip()
        except httpx.HTTPError:
            return ""

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_category_output(
    raw: str,
    t0: float,
    *,
    allowed: list[str] | None = None,
    include_feature: bool = False,
) -> ClassificationResult:
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    raw = (raw or "").strip()
    default_cats = ["junk", "quantity", "quality"]
    valid = set(allowed) if allowed else set(default_cats)
    m = _JSON_RE.search(raw)
    if not m:
        return ClassificationResult(
            category="others", confidence=0.0, source="fallback",
            reason="no_json_in_output", latency_ms=elapsed_ms, raw_output=raw,
        )
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return ClassificationResult(
            category="others", confidence=0.0, source="fallback",
            reason=f"json_parse_error: {e}", latency_ms=elapsed_ms, raw_output=raw,
        )

    cat = str(obj.get("category", "")).strip().lower()
    if cat not in valid:
        return ClassificationResult(
            category="others", confidence=0.0, source="fallback",
            reason=f"invalid_category: {cat}", latency_ms=elapsed_ms, raw_output=raw,
        )

    feat = None
    if include_feature:
        feat = _normalize_feature(obj.get("feature"))

    return ClassificationResult(
        category=cat,
        confidence=float(obj.get("confidence", 0.0)),
        reason=str(obj.get("reason", "")).strip(),
        source="llm",
        latency_ms=elapsed_ms,
        raw_output=raw,
        feature=feat,
    )


def _parse_feature_only(raw: str, default_confidence: float) -> ClassificationResult:
    raw = (raw or "").strip()
    m = _JSON_RE.search(raw)
    if not m:
        return ClassificationResult(
            category="", confidence=default_confidence, source="fallback",
            reason="no_json_in_feature_output", raw_output=raw,
        )
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return ClassificationResult(
            category="", confidence=default_confidence, source="fallback",
            reason="feature_json_parse_error", raw_output=raw,
        )
    return ClassificationResult(
        category="",
        confidence=float(obj.get("confidence", default_confidence)),
        reason=str(obj.get("reason", "")).strip(),
        source="llm",
        raw_output=raw,
        feature=_normalize_feature(obj.get("feature")),
    )


def _normalize_feature(feat: Any) -> str | None:
    if feat is None:
        return None
    feat = str(feat).strip().lower()
    if feat in ("null", ""):
        return None
    if feat in ("infrastructure", "agent_domain", "both"):
        return feat
    return None


def _fallback(reason: str, t0: float) -> ClassificationResult:
    return ClassificationResult(
        category="others", confidence=0.0, source="fallback",
        reason=reason, latency_ms=int((time.monotonic() - t0) * 1000),
    )
