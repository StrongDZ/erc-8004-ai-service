"""Mongo connection + query helpers.

Reads connection settings from the backend's .env (symlinked into AI/).
Collection names are read from MONGO_COLLECTION_* env vars to stay in sync
with the Go backend.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

load_dotenv()


def _env(key: str, default: str | None = None) -> str:
    v = os.getenv(key, default)
    if v is None:
        raise RuntimeError(f"missing env var: {key}")
    return v


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    return MongoClient(_env("MONGO_URI", "mongodb://localhost:27017"))


@lru_cache(maxsize=1)
def get_db() -> Database:
    return get_client()[_env("MONGO_DATABASE_ANALYZED_AGENTS", "analyzed_agents")]


@lru_cache(maxsize=1)
def get_main_db() -> Database:
    """Main backend DB. OASF schema tables (oasf_domains/oasf_skills) live here,
    not in the analyzed_agents DB that holds feedback_history/agents."""
    return get_client()[_env("MONGO_DATABASE", "erc8004")]


def feedback_coll() -> Collection:
    return get_db()[_env("MONGO_COLLECTION_FEEDBACK_HISTORY", "feedback_history")]


def agents_coll() -> Collection:
    return get_db()[_env("MONGO_COLLECTION_AGENTS", "agents")]


def oasf_domains_coll() -> Collection:
    return get_main_db()[_env("MONGO_COLLECTION_OASF_DOMAINS", "oasf_domains")]


def oasf_skills_coll() -> Collection:
    return get_main_db()[_env("MONGO_COLLECTION_OASF_SKILLS", "oasf_skills")]


def fetch_agents_by_keys(keys: Iterable[tuple[int, str]]) -> dict[str, dict]:
    """Bulk-fetch agent docs. Returns dict keyed by '{chainId}:{agentId}'."""
    coll = agents_coll()
    ids = [f"{cid}:{aid}" for cid, aid in keys]
    if not ids:
        return {}
    return {doc["_id"]: doc for doc in coll.find({"_id": {"$in": ids}})}


def count_by_category() -> dict[str, int]:
    """Return rule-category counts. Useful for stratified sampling decisions."""
    pipeline = [
        {"$group": {"_id": "$classification.rule.category", "n": {"$sum": 1}}},
        {"$sort": {"n": -1}},
    ]
    return {r["_id"]: r["n"] for r in feedback_coll().aggregate(pipeline)}
