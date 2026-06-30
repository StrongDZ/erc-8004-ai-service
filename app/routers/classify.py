"""POST /classify — wrap shared.ollama_client behind the HTTP API the Go side calls.

Hard endpoint gate: when the feedback endpoint matches one of the agent's
service endpoints, the LLM output enum drops `junk` (the feedback clearly
targets a real service, so it must fall in {quantity, quality}).
`others` is never in the LLM enum to start with.

Special model value: model="knn" routes to the embedding kNN classifier instead
of Ollama. The kNN corpus is built lazily on first use from the rule-labelled
MongoDB corpus and kept in memory for the process lifetime.
"""
from __future__ import annotations

import time

from fastapi import APIRouter

from shared.context_builder import build_user_message, domain_service_names, endpoint_matches_services
from shared.knn_classifier import KNNCorpus, feedback_embed_text
from shared.oasf_enrich import agent_domain_text
from shared.three_tier import DOMAIN_EMBED_MODEL, SVM_QUALITY_THRESH, build_agent_text, classify_three_tier
from shared.types import LLM_OUTPUT_CATEGORIES, AgentMeta, Category, FeedbackRecord

from ..deps import (
    DEFAULT_OLLAMA_MODEL,
    get_embedder,
    get_enriched_linear_classifier,
    get_knn_classifier,
    get_linear_classifier,
    get_ollama_client,
)
from ..schemas import ClassifyRequest, ClassifyResponse
from .. import cache

router = APIRouter()

# Cache-key namespace: derived purely from server config (default model + SVM
# threshold) so a model swap or threshold change auto-invalidates stale entries
# with no wipe. Payload-shape or classification-LOGIC changes are NOT auto-
# captured — drop the classify_cache collection on deploy for those.
_CACHE_VERSION = f"{DEFAULT_OLLAMA_MODEL}|tau{SVM_QUALITY_THRESH}"


def _cache_payload(req: ClassifyRequest) -> dict:
    """The fields that actually determine the verdict, at the granularity that
    matters: tag pair + scale + the agent's domain identity. Lists are sorted so
    ordering differences never cause spurious misses.

    Deliberately excluded (measured low-value on this corpus): value_norm
    (inflates keys ~1.2x without changing the category), endpoint and offchain
    (no empty-tag records depend on offchain; junk/endpoint cases negligible),
    and model/prompt_version (constant on the 3tier path; already in the version
    tag). The agent's contribution to the verdict is fully captured by its
    description + OASF + (non-generic) services + tags, so hashing those is
    equivalent to keying on agent identity — but it auto-invalidates when the
    agent's metadata changes (which a raw agent_id would not)."""
    return {
        "tag1": (req.tag1 or "").strip().lower(),
        "tag2": (req.tag2 or "").strip().lower(),
        "scale": (req.scale or "").strip().lower(),
        "agent": {
            "desc": (req.agent_description or "").strip(),
            "services": sorted(domain_service_names(req.agent_services)),
            "oasf_domains": sorted(req.agent_oasf_domains),
            "oasf_skills": sorted(req.agent_oasf_skills),
            "tags": sorted(t.strip().lower() for t in req.agent_tags if (t or "").strip()),
        },
    }


def _to_agent_meta(req: ClassifyRequest) -> AgentMeta:
    """Build AgentMeta from the structured request fields.

    `description` carries whatever the Go side picked (prefer the realtime
    summarized description, fall back to raw on-chain description). We copy it
    into `summary` too so prompt builders that look at `summary` first still work.
    """
    return AgentMeta(
        chain_id=0,
        agent_id="",
        name="",
        description=req.agent_description,
        summary=req.agent_description,
        services=[svc.model_dump() for svc in req.agent_services],
        oasf_domains=list(req.agent_oasf_domains),
        oasf_skills=list(req.agent_oasf_skills),
        tags=list(req.agent_tags),
    )


def _to_feedback_record(req: ClassifyRequest) -> FeedbackRecord:
    return FeedbackRecord(
        id="",
        agent_id="",
        chain_id=0,
        tag1=req.tag1,
        tag2=req.tag2,
        endpoint=req.endpoint,
        value=("" if req.value_norm is None else str(req.value_norm)),
        value_decimals=0,
        value_scale=req.scale,
        feedback_parsed={"offchain": req.offchain_content} if req.offchain_content else None,
        rule_category="others",
    )


def _allowed_categories(req: ClassifyRequest) -> list[str]:
    """Return the LLM output enum for this request.

    Gates applied in order:
      1. Scale=unbounded → remove 'quality' (no normalised ceiling possible).
      2. Endpoint matches agent service → remove 'junk' (feedback targets a real service).
    """
    categories = (
        ["junk", "quantity", "quality"]
        if req.prompt_version in ("v6", "v7", "v8")
        else list(LLM_OUTPUT_CATEGORIES)
    )
    # Unbounded scale gate: quality is structurally impossible, drop it from enum.
    if (req.scale or "").strip().lower() == "unbounded":
        categories = [c for c in categories if c != "quality"]
    # Endpoint gate: feedback targets a registered service → can't be junk.
    services = [svc.model_dump() for svc in req.agent_services]
    if endpoint_matches_services(req.endpoint or "", services):
        categories = [c for c in categories if c != "junk"]
    return categories





def _run_llm_classify(
    req: ClassifyRequest,
    model: str,
    user_msg: str,
    allowed: list[str],
    *,
    category_only: bool = False,
):
    """Dispatch v6 / v7 / v8 classification.

    v8: single-call with a scale-aware system prompt (unbounded variant omits
        Layer 3 quality; bounded variant uses the full cascade).
    v7: two-call split — category prompt then (optionally) feature prompt.
    v6: single-call two-axis.
    """
    sc = (req.scale or "").strip().lower()
    if req.prompt_version == "v8":
        client = get_ollama_client(model, "v8", "default", sc)
        return client.classify(user_msg, allowed_categories=allowed, prompt_version="v8")
    if req.prompt_version == "v7":
        cat_client = get_ollama_client(model, "v7", "category")
        feat_client = None if category_only else get_ollama_client(model, "v7", "feature")
        return cat_client.classify_v7(
            user_msg,
            allowed_categories=allowed,
            feature_client=feat_client,
            category_only=category_only,
        )
    client = get_ollama_client(model, req.prompt_version)
    return client.classify(user_msg, allowed_categories=allowed, prompt_version=req.prompt_version)


def _knn_classify(req: ClassifyRequest) -> ClassifyResponse:
    """Route model='knn' requests to the embedding kNN classifier."""
    text = feedback_embed_text(
        req.tag1 or "",
        req.tag2 or "",
        req.endpoint or "",
        req.offchain_content or "",
        value_norm=req.value_norm,
        score_tier=req.scale,
    )
    corpus: KNNCorpus = get_knn_classifier()
    result = corpus.classify(text)
    return ClassifyResponse(
        category=result.category,
        confidence=result.confidence,
        reason=result.reason,
        source="embedding",
        latency_ms=result.latency_ms,
        model_ver=f"knn-bge-base-k{corpus.k}",
    )


def _linear_classify(req: ClassifyRequest) -> ClassifyResponse:
    """Route model='linear' to the logistic-regression head (same corpus as kNN)."""
    # Same value+scale tokens as the shared KNN corpus this head trains on.
    text = feedback_embed_text(
        req.tag1 or "",
        req.tag2 or "",
        req.endpoint or "",
        req.offchain_content or "",
        value_norm=req.value_norm,
        score_tier=req.scale,
    )
    clf = get_linear_classifier()
    result = clf.classify(text)
    return ClassifyResponse(
        category=result.category,
        confidence=result.confidence,
        reason=result.reason,
        source="linear",
        latency_ms=result.latency_ms,
        model_ver="logreg-bge-base",
    )


def _linear_enriched_classify(req: ClassifyRequest) -> ClassifyResponse:
    """Route model='linear_enriched' to the late-fusion head.

    feedback tower = tag1|tag2|endpoint|offchain; agent tower = agent description
    + expanded OASF domain/skill descriptions (zero vector when none provided).
    """
    fb_text = feedback_embed_text(
        req.tag1 or "",
        req.tag2 or "",
        req.endpoint or "",
        req.offchain_content or "",
    )
    ag_text = agent_domain_text(
        req.agent_description or "",
        list(req.agent_oasf_domains),
        list(req.agent_oasf_skills),
    )
    clf = get_enriched_linear_classifier()
    result = clf.classify(fb_text, ag_text)
    return ClassifyResponse(
        category=result.category,
        confidence=result.confidence,
        reason=result.reason,
        source="linear_enriched",
        latency_ms=result.latency_ms,
        model_ver="logreg-enriched-bge-base",
    )


def _ensemble_classify(req: ClassifyRequest) -> ClassifyResponse:
    """Ensemble: LLM junk-gate → kNN for non-junk.

    Step 1: run LLM (DEFAULT_OLLAMA_MODEL) to detect junk.
            Junk records have clear semantic signals (gibberish, spam, placeholders)
            that kNN cannot reliably identify in embedding space.
    Step 2: if LLM predicts junk → return junk immediately (source=ensemble_llm).
    Step 3: otherwise hand off to kNN for the final category assignment
            (kNN excels at quality vs quantity boundaries in embedding space).
    """
    llm_model = DEFAULT_OLLAMA_MODEL
    user_msg = build_user_message(_to_agent_meta(req), _to_feedback_record(req))
    allowed = _allowed_categories(req)
    llm_result = _run_llm_classify(req, llm_model, user_msg, allowed, category_only=True)

    if llm_result.category == "junk":
        return ClassifyResponse(
            category=llm_result.category,
            confidence=llm_result.confidence,
            reason=llm_result.reason,
            source="ensemble_llm",
            latency_ms=llm_result.latency_ms,
            model_ver=f"ensemble({llm_model}+knn)",
            feature=llm_result.feature,
        )

    # Same value+scale tokens as the KNN corpus the ensemble hands off to.
    text = feedback_embed_text(
        req.tag1 or "",
        req.tag2 or "",
        req.endpoint or "",
        req.offchain_content or "",
        value_norm=req.value_norm,
        score_tier=req.scale,
    )
    corpus: KNNCorpus = get_knn_classifier()
    knn_result = corpus.classify(text)
    return ClassifyResponse(
        category=knn_result.category,
        confidence=knn_result.confidence,
        reason=f"[llm_gate={llm_result.category}] {knn_result.reason}",
        source="ensemble_knn",
        latency_ms=llm_result.latency_ms + knn_result.latency_ms,
        model_ver=f"ensemble({llm_model}+knn)",
    )


def _three_tier_classify(req: ClassifyRequest) -> ClassifyResponse:
    """Route model='3tier' to the production 3-tier classifier (per-tag SVM +
    agent-domain cosine + LLM fallback). Rule Stage 0/1 already ran on the Go side."""
    t0 = time.time()
    service_names = domain_service_names(req.agent_services)
    agent_text = build_agent_text(
        req.agent_description, req.agent_oasf_domains, req.agent_oasf_skills,
        service_names, list(req.agent_tags),
    )
    encoder = get_embedder(DOMAIN_EMBED_MODEL)

    def llm_fallback():
        llm_model = DEFAULT_OLLAMA_MODEL
        # Force prompt_version to v8 for three_tier fallback
        req_v8 = req.model_copy(update={"prompt_version": "v8"})
        user_msg = build_user_message(_to_agent_meta(req_v8), _to_feedback_record(req_v8))
        allowed = _allowed_categories(req_v8)
        result = _run_llm_classify(req_v8, llm_model, user_msg, allowed)
        return result.category, result.confidence, result.reason, result.feature

    res = classify_three_tier(
        encoder=encoder,
        tag1=req.tag1,
        tag2=req.tag2,
        scale=req.scale,
        value_norm=req.value_norm if req.value_norm is not None else 0.0,
        agent_text=agent_text,
        llm_classify_fn=llm_fallback,
    )

    model_ver = "3tier-svm-bge-gate+mandatory-escalation"
    if res.source == "llm":
        model_ver = f"3tier+llm({DEFAULT_OLLAMA_MODEL})"

    return ClassifyResponse(
        category=res.category,
        confidence=res.confidence,
        reason=res.reason,
        source=res.source,
        latency_ms=int((time.time() - t0) * 1000),
        model_ver=model_ver,
        feature=res.feature or None,
    )


@router.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    """Deterministic-memoised entry point: cache lookup -> dispatch -> store.

    The verdict is a pure function of the request + server config, so a cache
    hit returns the exact response the dispatch would produce (no F1 impact).
    Best-effort: cache (Mongo) errors fall through to a normal classification.
    """
    key = cache.cache_key(_CACHE_VERSION, _cache_payload(req))
    hit = cache.cache_get(key)
    if hit is not None:
        return ClassifyResponse(**hit)
    resp = _classify_uncached(req)
    cache.cache_set(key, resp.model_dump())
    return resp


def _classify_uncached(req: ClassifyRequest) -> ClassifyResponse:
    model_lower = (req.model or "").lower()
    if model_lower in ("3tier", "three_tier"):
        return _three_tier_classify(req)
    if model_lower == "knn":
        return _knn_classify(req)
    if model_lower == "linear":
        return _linear_classify(req)
    if model_lower == "linear_enriched":
        return _linear_enriched_classify(req)
    if model_lower == "ensemble":
        return _ensemble_classify(req)

    model = req.model or DEFAULT_OLLAMA_MODEL
    user_msg = build_user_message(_to_agent_meta(req), _to_feedback_record(req))
    allowed = _allowed_categories(req)
    result = _run_llm_classify(req, model, user_msg, allowed)
    return ClassifyResponse(
        category=result.category,
        confidence=result.confidence,
        reason=result.reason,
        source=result.source,
        latency_ms=result.latency_ms,
        model_ver=model,
        feature=result.feature,
    )
