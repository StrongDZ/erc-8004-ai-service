#!/usr/bin/env python3
"""Ablation benchmark: 4-stage per-tag agent-domain pipeline.

Ablation runs:
  1. Rule only
  2. Rule + SVM pair-tag (existing baseline)
  3. Rule + SVM per-tag (Stage 1+2)
  4. Rule + SVM per-tag + FAISS (Stage 1+2+3, no LLM)
  5. Rule + SVM per-tag + FAISS + LLM (full pipeline)

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.pipeline_3tier_v2 [--skip-llm] [--run 3]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.per_tag_svm import load_per_tag_svm, predict_quality_prob, vote_per_tag
from benchmarks.pipeline_3tier import build_text, rule_classify
from benchmarks.stage3_domain import DomainClassifier
from shared.context_builder import build_user_message
from shared.prompts import system_prompt_v8_category
from shared.types import LLM_OUTPUT_CATEGORIES, RULE_TO_CAT, AgentMeta, FeedbackRecord

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
GOLD_CSV = ROOT.parent / "erc-8004-benchmarking-be/scripts/labelled/gold_final.csv"
PAIR_TRAIN = ROOT / "data/splits/rule_based_diverse_v2/train.parquet"
OUT_DIR = ROOT / "data/benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SVM_VOTE_THRESH = 0.70
LLM_MODEL = "qwen2.5:7b-instruct"
LLM_URL = "http://localhost:11434"


def load_gold(path: Path = GOLD_CSV) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    df = df.rename(columns={"feedback_id": "id", "value_raw": "value", "scale": "value_scale", "category": "label", "human_label": "label"})
    df["label"] = df["label"].str.strip().str.lower().map(lambda x: RULE_TO_CAT.get(x, x))
    df = df[df["label"].isin(LLM_OUTPUT_CATEGORIES)].copy()
    for col in ("tag1", "tag2", "value_scale", "feedback_parsed", "value_decimals"):
        if col not in df.columns:
            df[col] = "" if col != "value_decimals" else 0
    df["value_decimals"] = pd.to_numeric(df["value_decimals"], errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)


def enrich_gold_with_agent_meta(gold: pd.DataFrame) -> pd.DataFrame:
    """Add agent_key + has_agent_metadata + is_self columns by MongoDB lookup.

    is_self mirrors the Go flow (processor_reputation_events.go): a feedback whose
    clientAddress equals the agent owner or registered agentWallet is self-feedback
    and is forced to junk. Addresses are compared lowercased (repo convention).
    """
    from shared.mongo_client import agents_coll, feedback_coll
    from shared.oasf_enrich import expand_oasf
    fb_coll = feedback_coll()
    ag_coll = agents_coll()

    agent_keys, has_meta, is_self_col, agent_ctx_col = [], [], [], []
    endpoint_col, fbparsed_col = [], []
    for _, row in gold.iterrows():
        doc = fb_coll.find_one({"_id": row["id"]}, {"agentId": 1, "chainId": 1, "clientAddress": 1,
                                                    "endpoint": 1, "feedbackParsed": 1})
        if doc:
            key = f"{doc.get('chainId',0)}:{doc.get('agentId','')}"
            ag = ag_coll.find_one({"_id": key}, {"name": 1, "description": 1, "summarizedDescription": 1,
                                                 "services": 1, "owner": 1, "agentWallet": 1,
                                                 "oasfDomains": 1, "oasfSkills": 1}) or {}
            desc = (ag.get("summarizedDescription") or ag.get("description") or "").strip()
            svc_names = [s.get("name","") for s in (ag.get("services") or []) if s.get("name")]
            client = str(doc.get("clientAddress", "") or "").lower()
            owner = str(ag.get("owner", "") or "").lower()
            wallet = str(ag.get("agentWallet", "") or "").lower()
            # Compact agent context for the LLM prompt (description + OASF + services)
            parts = []
            if ag.get("name"): parts.append(str(ag["name"]).strip())
            if desc: parts.append(desc[:400])
            dt = expand_oasf(ag.get("oasfDomains"))
            st = expand_oasf(ag.get("oasfSkills"))
            if dt: parts.append(dt[:200])
            if st: parts.append(st[:200])
            if svc_names: parts.append("services: " + ", ".join(svc_names[:6]))
            agent_keys.append(key)
            has_meta.append(bool(desc) or bool(svc_names))
            is_self_col.append(bool(client) and (client == owner or client == wallet))
            agent_ctx_col.append(" | ".join(parts)[:700])
            endpoint_col.append(str(doc.get("endpoint", "") or ""))
            fbparsed_col.append(doc.get("feedbackParsed"))
        else:
            csv_agent_key = row.get("agent_key", "")
            csv_agent_desc = row.get("agent_description", "")
            csv_agent_name = row.get("agent_name", "")
            csv_agent_services = row.get("agent_services", "")
            csv_agent_domains = row.get("agent_oasf_domains_text", "")
            csv_agent_skills = row.get("agent_oasf_skills_text", "")

            agent_keys.append(str(csv_agent_key or ""))
            has_meta.append(bool(csv_agent_desc or csv_agent_services or csv_agent_domains or csv_agent_skills))
            is_self_col.append(False)
            
            parts = []
            if csv_agent_name: parts.append(str(csv_agent_name).strip())
            if csv_agent_desc: parts.append(str(csv_agent_desc)[:400])
            if csv_agent_domains: parts.append(str(csv_agent_domains)[:200])
            if csv_agent_skills: parts.append(str(csv_agent_skills)[:200])
            if csv_agent_services: parts.append("services: " + str(csv_agent_services)[:100])
            
            agent_ctx_col.append(" | ".join(parts)[:700])
            endpoint_col.append(str(row.get("endpoint", "") or ""))
            
            fp = row.get("feedback_parsed") or row.get("fb_parsed")
            if not fp and row.get("offchain_note"):
                fp = {"offchain": str(row["offchain_note"])}
            fbparsed_col.append(fp)
    gold = gold.copy()
    gold["agent_key"] = agent_keys
    gold["has_agent_metadata"] = has_meta
    gold["is_self"] = is_self_col
    gold["agent_ctx"] = agent_ctx_col
    gold["endpoint"] = endpoint_col
    gold["fb_parsed"] = fbparsed_col
    return gold


# Category prompt factory — built per-call based on scale (cached in practice because
# the same scale values recur: unbounded vs. bounded). Keeps the module-level cache small.
def _llm_system_prompt(scale: str) -> str:
    """Return V8 category system prompt for the given scale.

    unbounded → short prompt (junk|quantity only, quality absent).
    bounded   → full cascade prompt (junk|quantity|quality).
    """
    return system_prompt_v8_category(include_few_shot=True, scale=scale)

_AGENT_META_CACHE: dict[str, AgentMeta] = {}


def _agent_meta(agent_key: str) -> AgentMeta:
    """Cached AgentMeta for an agent_key, so context_builder can render <agent>."""
    if agent_key in _AGENT_META_CACHE:
        return _AGENT_META_CACHE[agent_key]
    from shared.mongo_client import agents_coll
    ag = agents_coll().find_one({"_id": agent_key}, {
        "name": 1, "description": 1, "summarizedDescription": 1, "services": 1,
        "oasfDomains": 1, "oasfSkills": 1, "tags": 1}) or {}
    chain, aid = agent_key.split(":", 1) if ":" in agent_key else ("0", agent_key)
    meta = AgentMeta(
        chain_id=int(chain) if str(chain).isdigit() else 0,
        agent_id=aid,
        name=str(ag.get("name", "") or ""),
        description=str(ag.get("description", "") or ""),
        summary=str(ag.get("summarizedDescription", "") or ""),
        services=ag.get("services") or [],
        oasf_domains=ag.get("oasfDomains") or [],
        oasf_skills=ag.get("oasfSkills") or [],
        tags=[t for t in (ag.get("tags") or []) if t],
    )
    _AGENT_META_CACHE[agent_key] = meta
    return meta


_LLM_CACHE = None

def _load_llm_cache() -> dict[str, str]:
    global _LLM_CACHE
    if _LLM_CACHE is not None:
        return _LLM_CACHE
    cache_path = ROOT / "data/benchmark_results/llm_cache.json"
    if cache_path.exists():
        try:
            _LLM_CACHE = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            _LLM_CACHE = {}
    else:
        _LLM_CACHE = {}
    return _LLM_CACHE


def _save_llm_cache(fb_id: str, category: str) -> None:
    cache = _load_llm_cache()
    cache[fb_id] = category
    cache_path = ROOT / "data/benchmark_results/llm_cache.json"
    try:
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


def llm_classify(row: pd.Series, model: str) -> str:
    """Stage 4: V8 category prompt with scale-aware system message.

    When scale=unbounded the prompt has NO quality layer and the output regex
    is constrained to junk|quantity. This matches the structured-output enum
    used by the production classify.py endpoint.
    """
    fb_id = str(row.get("id", ""))
    cache = _load_llm_cache()
    if fb_id in cache:
        return cache[fb_id]

    agent_key = str(row.get("agent_key", "") or "")
    agent = _agent_meta(agent_key) if agent_key else AgentMeta(chain_id=0, agent_id="")
    
    # Fallback nếu DB không có dữ liệu nhưng file CSV có sẵn
    if not agent.description and row.get("agent_description"):
        agent = AgentMeta(
            chain_id=agent.chain_id,
            agent_id=agent.agent_id,
            name=str(row.get("agent_name", "") or ""),
            description=str(row.get("agent_description", "") or ""),
            summary=str(row.get("agent_description", "") or ""),
            services=[{"name": s.strip()} for s in str(row.get("agent_services", "")).split("|") if s.strip()] if row.get("agent_services") else [],
            oasf_domains=str(row.get("agent_oasf_domains_text", "")).split() if row.get("agent_oasf_domains_text") else [],
            oasf_skills=str(row.get("agent_oasf_skills_text", "")).split() if row.get("agent_oasf_skills_text") else [],
            tags=list(row.get("agent_tags", [])) if isinstance(row.get("agent_tags"), list) else [],
        )
        
    fp = row.get("fb_parsed")
    if isinstance(fp, str) and fp.strip():
        try:
            fp = json.loads(fp)
        except Exception:
            pass
    scale_val = str(row.get("value_scale", "") or "")
    is_unbounded = scale_val.strip().lower() == "unbounded"

    fb = FeedbackRecord(
        id=str(row.get("id", "")), agent_id=agent.agent_id, chain_id=agent.chain_id,
        tag1=str(row.get("tag1", "") or ""), tag2=str(row.get("tag2", "") or ""),
        endpoint=str(row.get("endpoint", "") or ""),
        value=str(row.get("value", "") or ""),
        value_decimals=int(row.get("value_decimals", 0) or 0),
        value_scale=scale_val,
        feedback_parsed=fp if isinstance(fp, dict) else None,
        rule_category="",
        is_self_feedback=bool(row.get("is_self", False)),
    )
    system_prompt = _llm_system_prompt(scale_val)
    msg = build_user_message(agent, fb)

    # Category regex: exclude 'quality' when unbounded (structurally impossible).
    if is_unbounded:
        cat_pattern = re.compile(r'"category"\s*:\s*"(junk|quantity)"', re.I)
    else:
        cat_pattern = re.compile(r'"category"\s*:\s*"(junk|quantity|quality)"', re.I)

    try:
        resp = requests.post(f"{LLM_URL}/api/chat", json={
            "model": model,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": msg}],
            "stream": False, "options": {"temperature": 0, "num_predict": 80},
        }, timeout=60)
        raw = resp.json()["message"]["content"].strip()
        m = cat_pattern.search(raw)
        if m:
            final_cat = m.group(1).lower()
        else:
            w = re.sub(r"[^a-z]", "", raw.lower()[:20])
            valid = ("quality", "quantity", "junk") if not is_unbounded else ("quantity", "junk")
            final_cat = w if w in valid else "junk"

        log_entry = (
            f"\n==================== LLM CLASSIFICATION COMPLETE ====================\n"
            f"--- Feedback Record ---\n"
            f"  id: {fb.id}\n"
            f"  agent_id: {fb.agent_id}\n"
            f"  chain_id: {fb.chain_id}\n"
            f"  tag1: {fb.tag1}\n"
            f"  tag2: {fb.tag2}\n"
            f"  endpoint: {fb.endpoint}\n"
            f"  value: {fb.value}\n"
            f"  value_decimals: {fb.value_decimals}\n"
            f"  value_scale: {fb.value_scale}\n"
            f"  feedback_parsed: {fb.feedback_parsed}\n"
            f"  rule_category: {fb.rule_category}\n"
            f"  is_self_feedback: {fb.is_self_feedback}\n"
            f"--- Agent Metadata ---\n"
            f"  chain_id: {agent.chain_id}\n"
            f"  agent_id: {agent.agent_id}\n"
            f"  name: {agent.name}\n"
            f"  description: {agent.description}\n"
            f"  summary: {agent.summary}\n"
            f"  services: {agent.services}\n"
            f"  oasf_domains: {agent.oasf_domains}\n"
            f"  oasf_skills: {agent.oasf_skills}\n"
            f"  tags: {agent.tags}\n"
            f"--- LLM Output ---\n"
            f"  Raw Content: {raw}\n"
            f"  Final Classified Category: {final_cat}\n"
            f"=====================================================================\n"
        )
        with open(OUT_DIR / "llm_classification.log", "a", encoding="utf-8") as f:
            f.write(log_entry)

        _save_llm_cache(fb.id, final_cat)
        return final_cat
    except Exception as e:
        log_entry = (
            f"\n==================== LLM CLASSIFICATION FAILED ====================\n"
            f"  id: {fb.id}\n"
            f"  Error: {e}\n"
            f"===================================================================\n"
        )
        with open(OUT_DIR / "llm_classification.log", "a", encoding="utf-8") as f:
            f.write(log_entry)
        return "junk"


def _print_results(name: str, y_true: list, y_pred: list, sources: list[str]) -> dict:
    mf1 = f1_score(y_true, y_pred, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Macro F1: {mf1:.4f}")
    print(classification_report(y_true, y_pred, labels=LLM_OUTPUT_CATEGORIES, zero_division=0))
    stage_counts = {}
    for s in sources:
        stage_counts[s] = stage_counts.get(s, 0) + 1
    for s, n in sorted(stage_counts.items()):
        print(f"  {s}: {n} ({n/len(sources)*100:.1f}%)")
    return {"name": name, "macro_f1": mf1, "stage_counts": stage_counts}


def _sub_group_f1(y_true: list, y_pred: list, mask: list[bool], name: str) -> float:
    sub_true = [y for y, m in zip(y_true, mask) if m]
    sub_pred = [p for p, m in zip(y_pred, mask) if m]
    if not sub_true:
        return 0.0
    f1 = f1_score(sub_true, sub_pred, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
    print(f"  [{name} N={len(sub_true)}] Macro F1: {f1:.4f}")
    return f1


def run_ablation(gold: pd.DataFrame, run: int, skip_llm: bool, self_gate: bool = False) -> list[dict]:
    results = []
    y_true = gold["label"].tolist()
    rich_mask = gold["has_agent_metadata"].tolist()
    poor_mask = [not m for m in rich_mask]

    # ── Run 1: Rule only ──────────────────────────────────────────────────────
    if run in (0, 1):
        preds = []
        sources = []
        for _, row in gold.iterrows():
            cat = rule_classify(row)
            preds.append(cat if cat else "others")
            sources.append("rule" if cat else "default_others")
        results.append(_print_results("Run 1: Rule Only", y_true, preds, sources))
        results[-1]["f1_rich"] = _sub_group_f1(y_true, preds, rich_mask, "Gold-Rich")
        results[-1]["f1_poor"] = _sub_group_f1(y_true, preds, poor_mask, "Gold-Poor")

    # ── Run 2: Rule + SVM pair-tag (baseline) ─────────────────────────────────
    if run in (0, 2):
        train_df = pd.read_parquet(PAIR_TRAIN)
        X_tr = train_df.apply(build_text, axis=1).tolist()
        y_tr = train_df["label"].tolist()
        pair_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=8000, sublinear_tf=True)),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000), cv=3, method="sigmoid")),
        ])
        pair_pipe.fit(X_tr, y_tr)

        preds = []; sources = []
        for _, row in gold.iterrows():
            cat = rule_classify(row)
            if cat:
                preds.append(cat); sources.append("rule")
            else:
                text = build_text(row)
                preds.append(pair_pipe.predict([text])[0]); sources.append("pair_svm")
        results.append(_print_results("Run 2: Rule + SVM Pair (baseline)", y_true, preds, sources))
        results[-1]["f1_rich"] = _sub_group_f1(y_true, preds, rich_mask, "Gold-Rich")
        results[-1]["f1_poor"] = _sub_group_f1(y_true, preds, poor_mask, "Gold-Poor")

    # ── Runs 3–5: per-tag SVM + FAISS + LLM ──────────────────────────────────
    if run in (0, 3, 4, 5):
        per_tag_pipe = load_per_tag_svm()

    if run in (0, 4, 5):
        dc = DomainClassifier()

    for run_id in ([run] if run in (3, 4, 5) else [3, 4, 5]):
        use_faiss = run_id >= 4
        use_llm = run_id == 5 and not skip_llm

        preds = []; sources = []; llm_count = 0
        audit_rows = []
        llm_t0 = time.time()

        for _, row in gold.iterrows():
            tag1 = str(row.get("tag1","") or "").strip()
            tag2 = str(row.get("tag2","") or "").strip()
            scale = str(row.get("value_scale","") or "").strip()
            decimals = int(row.get("value_decimals", 0) or 0)
            agent_key = str(row.get("agent_key","") or "")
            true_label = row.get("label")
            has_meta = bool(row.get("has_agent_metadata"))

            def _record(pred: str, source: str, reason: str = "") -> None:
                preds.append(pred); sources.append(source)
                audit_rows.append({
                    "id": row.get("id"), "tag1": tag1, "tag2": tag2,
                    "value_scale": scale, "value_decimals": decimals,
                    "agent_key": agent_key, "has_agent_metadata": has_meta,
                    "true_label": true_label, "pred": pred, "stage": source,
                    "reason": reason, "correct": pred == true_label,
                })

            # Stage 0: self-feedback gate (mirrors Go SelfFeedbackResult override:
            # clientAddress == owner/agentWallet -> junk, before the tag cascade)
            if self_gate and bool(row.get("is_self", False)):
                _record("junk", "self_feedback", "clientAddress == owner/agentWallet"); continue

            # Stage 0.5: empty-tag rule (convention: with no tags, classify by scale —
            # unbounded -> junk, any bounded scale -> quality). Resolves these here
            # instead of sending them to the LLM, which tends to junk empty-tag records.
            if not tag1 and not tag2:
                lab = "junk" if scale.lower() == "unbounded" else "quality"
                _record(lab, "empty_tag_rule", f"empty tags, scale={scale or '?'}"); continue

            # Stage 1: rule
            cat = rule_classify(row)
            if cat:
                _record(cat, "rule"); continue

            # Stage 2: per-tag SVM voting (single source of truth: per_tag_svm.vote_per_tag)
            p1 = predict_quality_prob(per_tag_pipe, tag1, scale) if tag1 else 0.5
            p2 = predict_quality_prob(per_tag_pipe, tag2, scale) if tag2 else 0.5
            t2_empty = not bool(tag2)

            stage2_result = vote_per_tag(p1, p2, t2_empty=t2_empty, thresh=SVM_VOTE_THRESH)

            if stage2_result == "quality":
                _record("quality", "per_tag_svm", f"p1={p1:.2f},p2={p2:.2f}"); continue
            elif stage2_result == "non_quality":
                # SVM says non-quality but doesn't know if quantity or junk → Stage 3 resolves it
                if not use_faiss:
                    _record("quantity", "per_tag_svm_non_quality", f"p1={p1:.2f},p2={p2:.2f}"); continue

            # Stage 3: FAISS domain check
            if use_faiss:
                label3, reason = dc.classify(tag1, tag2, scale, decimals, agent_key)
                if label3 is not None:
                    _record(label3, "faiss", reason); continue

            # Stage 4: LLM
            if use_llm:
                llm_cat = llm_classify(row, LLM_MODEL)
                _record(llm_cat, "llm"); llm_count += 1
            else:
                # No LLM → use ML best guess
                guess = "quality" if p1 >= 0.50 else "quantity"
                _record(guess, "ml_default", f"p1={p1:.2f}")

        run_name = f"Run {run_id}: Rule + Per-Tag SVM" + (" + FAISS" if use_faiss else "") + (" + LLM" if use_llm else "")
        res = _print_results(run_name, y_true, preds, sources)
        if use_llm:
            elapsed = time.time() - llm_t0
            print(f"  LLM calls: {llm_count} ({llm_count/len(gold)*100:.1f}%)  avg latency: {elapsed/max(llm_count,1)*1000:.0f}ms")
        res["f1_rich"] = _sub_group_f1(y_true, preds, rich_mask, "Gold-Rich")
        res["f1_poor"] = _sub_group_f1(y_true, preds, poor_mask, "Gold-Poor")
        results.append(res)

        audit_path = OUT_DIR / f"audit_run{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
        res["audit_csv"] = str(audit_path)
        print(f"  Per-record audit saved to {audit_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=0, help="0=all, 1-5=specific run")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--self-gate", action="store_true",
                        help="enable Stage 0 self-feedback gate (clientAddress==owner/wallet -> junk)")
    parser.add_argument("--exclude-self", action="store_true",
                        help="drop self-feedback rows from the test set entirely, so the "
                             "ML/LLM stages are evaluated only on records that actually reach them")
    parser.add_argument("--gold", type=Path, default=GOLD_CSV,
                        help="test-set CSV (default gold_final.csv)")
    args = parser.parse_args()

    print(f"Loading gold test set from {args.gold} ...")
    gold = load_gold(args.gold)
    print(f"  Gold N={len(gold)}")

    print("Enriching gold with agent metadata (MongoDB lookup)...")
    gold = enrich_gold_with_agent_meta(gold)
    rich = gold["has_agent_metadata"].sum()
    print(f"  Gold-Rich: {rich} | Gold-Poor: {len(gold)-rich}")

    # Initialize/clear the LLM classification log file
    (OUT_DIR / "llm_classification.log").write_text("", encoding="utf-8")

    if args.exclude_self:
        n_self = int(gold["is_self"].sum())
        gold = gold[~gold["is_self"]].reset_index(drop=True)
        print(f"  --exclude-self: dropped {n_self} self-feedback rows -> N={len(gold)} "
              f"(evaluating ML/LLM only on records that reach them)")
    if args.self_gate:
        print(f"  Stage 0 self-feedback gate ENABLED ({int(gold['is_self'].sum())} self-feedback rows)")
    results = run_ablation(gold, args.run, args.skip_llm, args.self_gate)

    print("\n\n=== SUMMARY: MacroF1 by run (full / rich / poor) ===")
    for res in results:
        print(f"  {res['name']:55s} full={res['macro_f1']:.4f}  rich={res['f1_rich']:.4f}  poor={res['f1_poor']:.4f}")

    # Save results
    out_path = OUT_DIR / f"pipeline_3tier_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
