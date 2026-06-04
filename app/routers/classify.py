"""POST /classify — wrap shared.ollama_client behind the HTTP API the Go side calls.

Hard endpoint gate: when the feedback endpoint matches one of the agent's
service endpoints, the LLM output enum drops `junk` (the feedback clearly
targets a real service, so it must fall in {config, app_specific, service_feedback}).
`others` is never in the LLM enum to start with.
"""
from __future__ import annotations

from fastapi import APIRouter

from shared.context_builder import build_user_message, endpoint_matches_services
from shared.types import LLM_OUTPUT_CATEGORIES, AgentMeta, Category, FeedbackRecord

from ..deps import DEFAULT_OLLAMA_MODEL, get_ollama_client
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
    services = [svc.model_dump() for svc in req.agent_services]
    if endpoint_matches_services(req.endpoint or "", services):
        return [c for c in LLM_OUTPUT_CATEGORIES if c != Category.JUNK.value]
    return list(LLM_OUTPUT_CATEGORIES)


@router.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    model = req.model or DEFAULT_OLLAMA_MODEL
    client = get_ollama_client(model)
    user_msg = build_user_message(_to_agent_meta(req), _to_feedback_record(req))
    allowed = _allowed_categories(req)
    result = client.classify(user_msg, allowed_categories=allowed)
    return ClassifyResponse(
        category=result.category,
        confidence=result.confidence,
        reason=result.reason,
        source=result.source,
        latency_ms=result.latency_ms,
        model_ver=model,
    )
