"""Pydantic request/response schemas for the AI service HTTP API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AgentServicePayload(BaseModel):
    """Subset of an agent's registration services entry used for classification."""

    name: str = ""
    endpoint: str = ""
    version: str = ""
    skills: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)


class ClassifyRequest(BaseModel):
    """Flat classification request — fields match the Go AIClient call site.

    agent_services is the structured list of registration services so the Python
    context builder can filter generic ones (web / OASF / A2A / email) and match
    the feedback endpoint against the agent's own service endpoints.
    agent_oasf_domains / agent_oasf_skills are normalised hierarchical paths
    (e.g. "technology/blockchain", "natural_language_processing/text_classification").
    """

    tag1: str = ""
    tag2: str = ""
    value_norm: float = 0.0
    scale: str = ""
    offchain_content: str = ""
    endpoint: str = ""
    agent_description: str = ""
    agent_services: list[AgentServicePayload] = Field(default_factory=list)
    agent_oasf_domains: list[str] = Field(default_factory=list)
    agent_oasf_skills: list[str] = Field(default_factory=list)
    agent_tags: list[str] = Field(default_factory=list)
    prompt_version: str = "v4_xml"
    model: str | None = None


class ClassifyResponse(BaseModel):
    category: str
    confidence: float
    reason: str = ""
    source: str = "llm"
    latency_ms: int = 0
    model_ver: str = ""
    feature: str | None = None


class EmbedRequest(BaseModel):
    texts: list[str]
    model: str | None = None


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dim: int


class HealthResponse(BaseModel):
    ollama_ok: bool
    default_model: str
    embedder_loaded: bool
