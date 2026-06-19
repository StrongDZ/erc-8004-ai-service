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

from fastapi import APIRouter

from shared.context_builder import build_user_message, endpoint_matches_services
from shared.knn_classifier import KNNCorpus, feedback_embed_text
from shared.oasf_enrich import agent_domain_text
from shared.types import LLM_OUTPUT_CATEGORIES, AgentMeta, Category, FeedbackRecord

from ..deps import (
    DEFAULT_OLLAMA_MODEL,
    get_enriched_linear_classifier,
    get_knn_classifier,
    get_linear_classifier,
    get_ollama_client,
)
from ..schemas import ClassifyRequest, ClassifyResponse

router = APIRouter()


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
        value=str(req.value_norm),
        value_decimals=0,
        value_scale=req.scale,
        feedback_parsed={"offchain": req.offchain_content} if req.offchain_content else None,
        rule_category="others",
    )


def _allowed_categories(req: ClassifyRequest) -> list[str]:
    """Return the LLM enum for this request.

    Default = all four output categories. When the feedback endpoint matches an
    agent service endpoint, junk is removed — that feedback targets a real
    service the agent owns.
    """
    categories = (
        ["junk", "quantity", "quality"]
        if req.prompt_version in ("v6", "v7")
        else list(LLM_OUTPUT_CATEGORIES)
    )
    services = [svc.model_dump() for svc in req.agent_services]
    if endpoint_matches_services(req.endpoint or "", services):
        return [c for c in categories if c != "junk"]
    return categories


def _run_llm_classify(req: ClassifyRequest, model: str, user_msg: str, allowed: list[str], *, category_only: bool = False):
    """Dispatch v6 single-call vs v7 two-call classification."""
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
    text = feedback_embed_text(
        req.tag1 or "",
        req.tag2 or "",
        req.endpoint or "",
        req.offchain_content or "",
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

    text = feedback_embed_text(
        req.tag1 or "",
        req.tag2 or "",
        req.endpoint or "",
        req.offchain_content or "",
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


@router.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    model_lower = (req.model or "").lower()
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
