"""Ollama HTTP client — Python port of internal/domain/classifier/llm_client.go.

Behaviour preserved:
  - temperature=0, seed=42, num_predict=128, repeat_penalty=1.3
  - Structured-output schema on /api/chat (Ollama v0.5+)
  - JSON extraction tolerates leading/trailing text around the {...}
  - LLM output is restricted to the 4 real categories
  - On client/parse/schema errors → category=others, confidence=0.0, source=fallback
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
    ) -> ClassificationResult:
        """Run classification. `allowed_categories` restricts the structured-output
        enum at the Ollama level — used by the endpoint hard-gate (drop `junk`
        when feedback endpoint matches an agent service).
        """
        t0 = time.monotonic()
        enum = list(allowed_categories) if allowed_categories else list(LLM_OUTPUT_CATEGORIES)
        if not enum:
            enum = list(LLM_OUTPUT_CATEGORIES)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "format": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": enum},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["category", "confidence"],
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
            raw = r.json().get("message", {}).get("content", "")
        except httpx.HTTPError as e:
            return _fallback(f"http_error: {e}", t0)

        return _parse_output(raw, t0, allowed=enum)

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


def _parse_output(raw: str, t0: float, *, allowed: list[str] | None = None) -> ClassificationResult:
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    raw = (raw or "").strip()
    valid = set(allowed) if allowed else set(LLM_OUTPUT_CATEGORIES)
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
    return ClassificationResult(
        category=cat,
        confidence=float(obj.get("confidence", 0.0)),
        reason=str(obj.get("reason", "")).strip(),
        source="llm",
        latency_ms=elapsed_ms,
        raw_output=raw,
    )


def _fallback(reason: str, t0: float) -> ClassificationResult:
    return ClassificationResult(
        category="others", confidence=0.0, source="fallback",
        reason=reason, latency_ms=int((time.monotonic() - t0) * 1000),
    )
