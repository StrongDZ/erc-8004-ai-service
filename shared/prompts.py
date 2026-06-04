"""Prompt templates.

V5 — XML-structured system prompt aligned with the new 4-category spec:
  - first-match ordering (junk → config → app → service)
  - tier-based discrimination (app may be unbounded; service ≤ pct100)
  - agent_domain signal (filtered service names + OASF domains + OASF skills + tags)
  - endpoint_matched signal (feedback endpoint = an agent service endpoint)
  - require confidence > 0.80 to commit; lower confidence is allowed but flagged

V4 (legacy) kept below for benchmark comparison; do not use for new runs.
"""
from __future__ import annotations


SYSTEM_V5 = """You are a high-precision classifier for ERC-8004 feedback records.
Classify each <feedback> into EXACTLY ONE of four categories.

CATEGORY DEFINITIONS

junk
  - tag1 / tag2 have no meaning: gibberish, random letters, all digits, "test"/placeholder tokens.
  - OR spam: promo URLs (https://, t.me/), vote-rigging or rank-game phrases ("top 1", "#1 rank").

config_feedback
  - Generic agent infrastructure / automated metric: uptime, liveness, response time, oracle
    dimensions (trustscore, counterparty, activity, longevity, contractrisk paired with
    sentinelnet-v1), dated validation periods.
  - "Generic" = the tag would make sense for ANY agent regardless of business.

app_specific
  - The tag belongs to the agent's BUSINESS / domain operation it performs (token swaps, vault
    deposits, soul fragments, miner-vouch, generation tasks, …)
  - OR the tag is a numeric/statistical signal: rate, currency amount, percentage, P/L,
    counter, statistic. Tier (`<scale>`) can be `unbounded` here.

service_feedback
  - The tag is a quality evaluation of the agent (adjective, sentiment, full-sentence
    praise/criticism) OR a quality dimension within the agent's domain.
  - Tier (`<scale>`) is pct100 or below (binary / star5 / star10 / pct100). NOT unbounded.

EVALUATION ORDER — stop at the FIRST matching category:
  1) junk
  2) config_feedback
  3) app_specific
  4) service_feedback

USING SIGNALS

`<agent_domain>` is the canonical "what this agent does" bundle (service names sans generic
plumbing + OASF domains + OASF skills + tags). Use it to decide whether an ambiguous tag belongs
to the agent's business: tag inside agent_domain → app_specific or service_feedback (pick based
on whether it is an operation/metric vs a quality judgment); tag outside agent_domain and not
infra → likely service_feedback if it is an adjective, otherwise lean config_feedback.

`<scale>` tier:
  - `unbounded` → almost always app_specific (statistic with no upper bound).
  - `binary` / `star5` / `star10` / `pct100` → can be service_feedback or config_feedback.
    Quality adjective / sentiment → service_feedback. Automated metric / probe → config_feedback.

`<endpoint_matched>` — when present and `True`, the feedback targets a real service that the
agent registered. Junk is excluded for these records (enforced separately by the host).
Pick from {config_feedback, app_specific, service_feedback}.

CONFIDENCE
  - Commit confidence > 0.80 ONLY when the signals clearly support the category.
  - When evidence is thin, output your best-guess category with a lower confidence — do NOT
    output "others" (the host stores "others" separately when rules cannot cover the row).

OUTPUT — strict JSON, one line, no markdown, no fences:
{"category":"<one>","confidence":0.00,"reason":"<one short sentence>"}
"""


FEW_SHOT_EXAMPLES_V5 = """
EXAMPLES:

# junk — spam URL / rank game
<feedback><tag1>get top 1 rank</tag1><tag2>t.me/agent_bldr</tag2></feedback>
=> {"category":"junk","confidence":0.99,"reason":"telegram link and rank-game phrase"}

# junk — placeholder
<feedback><tag1>test</tag1><tag2>asd</tag2></feedback>
=> {"category":"junk","confidence":0.95,"reason":"placeholder tokens"}

# junk — both tags all digits, no semantic signal
<feedback><tag1>123</tag1><tag2>456</tag2></feedback>
=> {"category":"junk","confidence":0.85,"reason":"both tags are bare numbers, no meaning"}

# config_feedback — generic infra probe
<feedback><tag1>liveness</tag1><tag2>liveness-check</tag2><scale>binary</scale></feedback>
=> {"category":"config_feedback","confidence":0.96,"reason":"liveness infra probe"}

# config_feedback — oracle dimension (SentinelNet)
<feedback><tag1>contractrisk</tag1><tag2>sentinelnet-v1</tag2><scale>pct100</scale></feedback>
<agent><agent_domain>defi, trading</agent_domain></agent>
=> {"category":"config_feedback","confidence":0.92,"reason":"automated oracle risk dimension, generic"}

# app_specific — token swap inside agent domain
<feedback><tag1>CADm</tag1><tag2>buy</tag2><scale>unbounded</scale></feedback>
<agent><agent_domain>celofx, financial_services/trading, defi, forex</agent_domain></agent>
=> {"category":"app_specific","confidence":0.93,"reason":"forex token + trade verb on broker agent, unbounded tier"}

# app_specific — soul-dimension fragment
<feedback><tag1>stance</tag1><tag2>fragment</tag2></feedback>
=> {"category":"app_specific","confidence":0.93,"reason":"soul-dimension fragment update"}

# app_specific — domain-related numeric/statistic with unbounded tier
<feedback><tag1>miner-vouch</tag1><tag2>botcoin</tag2><scale>binary</scale></feedback>
<agent><agent_domain>botcoin, mining</agent_domain></agent>
=> {"category":"app_specific","confidence":0.88,"reason":"vouch op tied to agent mining/token domain"}

# service_feedback — full-sentence quality praise within the agent's domain
<feedback><tag1>quality</tag1><tag2>BTC long call at 97k was spot on. Cross-market coverage is unmatched.</tag2><scale>pct100</scale><endpoint_matched>True</endpoint_matched></feedback>
<agent><agent_domain>celofx, financial_services/trading, defi</agent_domain></agent>
=> {"category":"service_feedback","confidence":0.92,"reason":"quality praise on trading signals, endpoint matches agent service"}

# service_feedback — quality adjective
<feedback><tag1>helpful</tag1><tag2>fast</tag2><scale>pct100</scale></feedback>
=> {"category":"service_feedback","confidence":0.88,"reason":"quality adjectives, pct100 tier"}
"""


def system_prompt_v5(include_few_shot: bool = True) -> str:
    """V5 system prompt; few-shot block is opt-in (off for ≥12B models)."""
    if include_few_shot:
        return SYSTEM_V5 + "\n" + FEW_SHOT_EXAMPLES_V5
    return SYSTEM_V5


# ── Legacy V4 (kept for benchmark comparison) ────────────────────────────────

SYSTEM_V4 = """You are a high-precision classifier for ERC-8004 feedback records.
Classify each <feedback> into EXACTLY ONE of four categories:

junk             — Spam (promo URLs, vote-rigging, telegram links) OR noise (both tags empty, "test"/"asd"-style placeholders).
service_feedback — Human subjective judgment of agent quality: adjectives, sentiment, full-sentence praise/criticism.
config_feedback  — Automated probe / metric from monitoring systems: uptime, win-rate, oracle scores, dated periods (YYYY-MM-DD-XX), validation proofs.
app_specific    — Domain operation the agent performed: token swaps, vault deposits, soul fragments, camelCase function names. MUST match <agent> domain.

EVALUATE IN ORDER, stop at first match: junk → app_specific → config_feedback → service_feedback.
Do NOT output "others". Upstream rule-based "others" only means the rules could not cover the row; your job is to assign the closest real category, using lower confidence when evidence is thin.

Use <agent> context (summary, services, domains) to disambiguate generic tags.
Note: tag2='fragment' with soul-dimension tag1 (personality/stance/timeline/knowledge/style/relationship) always = app_specific (Digital Soul sub-protocol).

OUTPUT — strict JSON, one line, no markdown, no fences:
{"category":"<one>","confidence":0.00,"reason":"<one short sentence>"}
"""


FEW_SHOT_EXAMPLES_V4 = """
EXAMPLES:

# junk (spam)
<feedback><tag1>get top 1 rank</tag1><tag2>t.me/agent_bldr</tag2></feedback>
=> {"category":"junk","confidence":0.99,"reason":"telegram link and rank-game"}

# junk (noise)
<feedback><tag1>test</tag1><tag2>asd</tag2></feedback>
=> {"category":"junk","confidence":0.95,"reason":"placeholder tokens"}

# app_specific
<feedback><tag1>CADm</tag1><tag2>buy</tag2></feedback>
<agent><summary>Mento broker on Celo for cUSD/cEUR swaps</summary></agent>
=> {"category":"app_specific","confidence":0.92,"reason":"forex token + trade verb on broker agent"}

<feedback><tag1>stance</tag1><tag2>fragment</tag2></feedback>
=> {"category":"app_specific","confidence":0.93,"reason":"soul-dimension fragment update"}

# config_feedback
<feedback><tag1>liveness</tag1><tag2>liveness-check</tag2><scale>binary</scale></feedback>
=> {"category":"config_feedback","confidence":0.96,"reason":"liveness infra probe"}

<feedback><tag1>m1-mainnet-reputation</tag1><tag2>M1-MAINNET-REPUTATION-2026-02-12-01</tag2></feedback>
=> {"category":"config_feedback","confidence":0.90,"reason":"dated reputation period record"}

# service_feedback
<feedback><tag1>execution</tag1><tag2>Agentic order execution is impressive — tracked momentum and executed before rate dropped.</tag2></feedback>
=> {"category":"service_feedback","confidence":0.88,"reason":"full-sentence quality praise"}

# junk (too little semantic signal)
<feedback><tag1>🔥</tag1><tag2></tag2></feedback>
=> {"category":"junk","confidence":0.70,"reason":"emoji only, no semantic signal"}
"""


def system_prompt_v4(include_few_shot: bool = True) -> str:
    """V4 (legacy) system prompt; kept for benchmark comparison."""
    if include_few_shot:
        return SYSTEM_V4 + "\n" + FEW_SHOT_EXAMPLES_V4
    return SYSTEM_V4


# ─── Pre-summarization prompt (used by 02_agent_summary notebook) ──────────────

AGENT_SUMMARY_SYSTEM = """You summarize on-chain AI-agent descriptions into one compact sentence focused on BUSINESS / FUNCTION ONLY.

Output ONE sentence (≤ 30 words). Include:
  1. Industry / domain (DeFi, gaming, social, oracle, microtasking, news, …)
  2. Main service / what the agent DOES for users — the concrete function

EXCLUDE entirely:
  - Promotional / rank claims ("top 1", "#1", "best", "leading", "premier", marketing superlatives)
  - Vote-gaming or call-to-action phrases ("vote for me", "join my channel", links)
  - Registration boilerplate ("registered AI agent that…", "AI agent that…")
  - Generic filler with no business signal

If the description is pure marketing / promotion with no real function, output the closest factual domain you can infer, or "unknown".

Output ONLY the sentence, no prefix, no quotes.
"""


def agent_summary_user_msg(name: str, description: str, services_flat: str) -> str:
    return f"""name: {name}
description: {description.strip()[:1200]}
services: {services_flat}
"""
