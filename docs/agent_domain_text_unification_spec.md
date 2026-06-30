# Spec — Unify agent-domain text across benchmark and production

## Context

The Stage-3 agent-domain cosine signal is built from **different, incomplete**
agent texts on the two sides, so the production cascade is not the cascade the
thesis benchmark measured:

| Side | Agent text used to embed `agent_vec` | Source |
|---|---|---|
| Benchmark (FAISS index) | `description + service_names` | `scripts/build_agent_index.py:_agent_text` |
| Production (live cosine) | `description + OASF domains + OASF skills` | `shared/oasf_enrich.py:agent_domain_text` via `three_tier.build_agent_text` |

Neither uses the full agent signal. Goal: **one canonical agent-domain text =
`description + oasf_domains + oasf_skills + service_names + tags`**, computed
**on the fly** (no prebuilt FAISS index) and used **identically** in benchmark
and production. Whatever components are present contribute; missing ones are
simply empty. Then **re-run the benchmark** — the 45.4% / 0.816 numbers and the
verified τ-sweep were measured on the old `desc+services` signal and will be
**superseded** by the corrected cascade's numbers. (This matches the agreed
order: fix the cascade for consistency first → re-benchmark → tune τ after.)

Also folded in per request: the **`scale_heuristic` + `value_decimals`** gap so
production's no-domain-signal branch matches the benchmark exactly.

## Canonical function (single source of truth)

`shared/oasf_enrich.py`:

```python
def agent_domain_text_full(description, oasf_domains, oasf_skills, service_names, tags, max_chars=1000):
    parts = []
    if (description or "").strip():       parts.append(description.strip())
    if expand_oasf(oasf_domains):         parts.append(expand_oasf(oasf_domains))
    if expand_oasf(oasf_skills):          parts.append(expand_oasf(oasf_skills))
    names = [n for n in (service_names or []) if n.strip()]   # generic plumbing pre-filtered by caller
    if names:                             parts.append(", ".join(names))
    if tags:                              parts.append(", ".join(t for t in tags[:10] if t.strip()))
    return " | ".join(parts)[:max_chars]
```

Decisions: OASF expanded to descriptions (richer than raw paths); service names
only (generic web/oasf/a2a/email stripped by caller via `domain_service_names`);
tags capped at 10. `agent_text` is empty iff **all** components are empty →
only then the no-domain-signal branch fires.

`scale_heuristic` duplicated inline into `three_tier.py` (production must not
import benchmark code), identical logic to `stage3_domain.py:scale_heuristic`:
unbounded OR `value_decimals>0` → quantity; star5/star10/binary → quality;
else (pct100, decimals=0) → None (LLM).

## Change groups

**G1 — shared** (`shared/oasf_enrich.py`): add `agent_domain_text_full`.

**G2 — production** (`shared/three_tier.py`, `app/routers/classify.py`, `app/schemas.py`):
- `build_agent_text` → call `agent_domain_text_full` (add `service_names`, `tags` params).
- `classify.py:_three_tier_classify` builds `service_names` via `domain_service_names(req.agent_services)`, passes `req.agent_tags`; passes `value_decimals` to `classify_three_tier`.
- `classify_three_tier` gains `value_decimals` param; no-domain-signal branch (`three_tier.py:228`) applies inline `scale_heuristic(sc, value_decimals)` before escalating to LLM. In-domain+bounded **still escalates** (unchanged, matches `run13.resolve:195`).
- `ClassifyRequest` gains `value_decimals: int = 0`.

**G3 — Go plumbing** (`erc-8004-benchmarking-be`, separate repo): ⚠️ `AIClient.Classify` impact = CRITICAL fan-out but **1 direct call site**.
- `ai_client.go`: add `ValueDecimals int json:"value_decimals,omitempty"` to `classifyRequest`; add `valueDecimals int` param to `Classify`, set in payload.
- `hybrid.go`: pass `in.ValueDecimals` (already on `HybridInput`) at the single call site.

**G4 — benchmark** (`benchmarks/pipeline_run13.py`): replace FAISS Stage-3 (`DomainClassifier.check_in_domain`) with live cosine using `agent_domain_text_full` built from `_agent_meta(agent_key)` fields (already loads desc/summary/services/oasf/tags). Keep threshold 0.55, keep `scale_heuristic` for the None branch. Drops the `agent_index.faiss` dependency.

**G5 — re-benchmark**: run `pipeline_run13.py`, regenerate the τ-sweep + F1/LLM% table on the unified signal; record new numbers (these replace 45.4% / 0.816 as the production cascade baseline).

## Verification

- `go build ./...` in be repo after G3.
- AI service: unit-call `/classify` with `model=3tier` on a few records (with/without agent metadata; with value_decimals>0) → confirm no error, sane categories.
- G5 reproduces a self-consistent τ-sweep; sanity-check direction (lower τ → fewer LLM) still holds.
- Compare new F1/LLM% vs old (45.4%/0.816) and report the delta honestly.

## Out of scope (later, the "tune" phase)

τ tuning, the Redis runtime cache, any model swap (3b↔7b).
