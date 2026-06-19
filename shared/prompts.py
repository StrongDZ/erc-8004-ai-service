"""Prompt templates.

V7 — split calls (current spec):
  - Call 1: category cascade junk -> quantity -> quality
  - Call 2: feature infrastructure | agent_domain | both (skipped for junk)
  - unbounded scale is NEVER quality; it is NOT automatically quantity

V6 — single-call two-axis (kept for benchmark comparison):
  - category + feature in one response
  - HARD RULE: unbounded scale is NEVER quality (legacy also mapped unbounded -> quantity)

V5/V4 (legacy 4-category junk/config/app/service) kept below for benchmark comparison;
do not use for new runs.
"""
from __future__ import annotations


SYSTEM_V6 = """You are a high-precision classifier for ERC-8004 feedback records.
For each <feedback> output TWO labels:
  1) category  — junk | quantity | quality      (decides whether/how it scores)
  2) feature   — infrastructure | agent_domain | both   (what it is ABOUT; null for junk)

Pick `category` with this CASCADE — stop at the FIRST matching layer:

LAYER 1 — junk
  - tag1/tag2 meaningless: gibberish, random letters, all-digits, "test"/placeholder, empty.
  - OR spam: promo URLs (https://, t.me/), vote-rigging / rank-game ("top 1", "#1 rank").
  - OR a garbage/anomalous value (un-normalized uint256-looking number).

LAYER 2 — quantity
  - The tag is a MEASURED value: a metric, rate, count, amount, P/L, volume, completion/success
    rate, or a domain OPERATION OUTCOME (what happened / how much) — EXCEPT trust/reputation
    indicators (those belong to quality).
  - HARD RULE: any `unbounded` scale is ALWAYS quantity, NEVER quality.
  - A "good-direction" metric (success rate, win rate, uptime, response time) is STILL quantity.

LAYER 3 — quality   (everything that is left)
  - A subjective judgment of how good the agent is: adjective (fast, reliable, excellent),
    sentiment, satisfaction score, full-sentence praise/criticism.
  - OR a TRUST / reputation / safety assessment (uy-tin): trust-score, reputation, safety-score,
    sybil-resistance, or an evaluation of the agent's security / verification SERVICE — provided
    the scale is bounded (unbounded already went to quantity at Layer 2).

ONLY `quality` feeds the agent's trust score. quantity and junk do not.

THEN assign `feature` (for quantity & quality only; use null for junk):
  - infrastructure — a GENERIC signal that would make sense for ANY agent regardless of its
    business: uptime, liveness, response time, reachability, oracle probe, A2A/MCP/Web health,
    a generic trust/reputation probe.
  - agent_domain — SPECIFIC to what THIS agent does (its business): trading, security audit,
    data labeling, soul fragments, forex swap, etc. Lean on the <agent_domain> signal.
  - both — a generic metric measured ON a domain-specific service (e.g. the response time of
    THIS agent's trading endpoint).

USING SIGNALS
`<agent_domain>` = what this agent does (service names minus generic plumbing + OASF domains/skills + tags).
  - Tag names a concept inside agent_domain -> feature = agent_domain.
  - Generic infra/trust metric unrelated to the domain -> feature = infrastructure.
  - Generic metric tied to a domain service -> feature = both.
`<endpoint_matched>` True -> feedback targets a registered service endpoint (lean agent_domain); junk excluded.

OUTPUT — strict JSON, one line, no markdown, no fences:
{"category":"<junk|quantity|quality>","feature":"<infrastructure|agent_domain|both|null>","confidence":0.00,"reason":"<one short sentence>"}
"""


FEW_SHOT_EXAMPLES_V6 = """
EXAMPLES:

# junk — spam (feature null)
<feedback><tag1>get top 1 rank</tag1><tag2>t.me/agent_bldr</tag2></feedback>
=> {"category":"junk","feature":null,"confidence":0.99,"reason":"telegram link and rank-game phrase"}

# quantity + infrastructure — generic uptime metric, applies to any agent
<feedback><tag1>uptime</tag1><tag2>liveness-check</tag2><scale>pct100</scale></feedback>
=> {"category":"quantity","feature":"infrastructure","confidence":0.95,"reason":"generic uptime metric, not tied to the agent's business"}

# quantity + agent_domain — domain operation outcome; unbounded forces quantity
<feedback><tag1>trade</tag1><tag2></tag2><scale>unbounded</scale></feedback>
<agent><agent_domain>celofx, financial_services/trading, defi</agent_domain></agent>
=> {"category":"quantity","feature":"agent_domain","confidence":0.92,"reason":"trade outcome on a trading agent; unbounded -> quantity"}

# quantity + both — generic latency metric measured on a domain service
<feedback><tag1>responsetime</tag1><tag2>trading-endpoint</tag2><scale>pct100</scale></feedback>
<agent><agent_domain>financial_services/trading</agent_domain></agent>
=> {"category":"quantity","feature":"both","confidence":0.88,"reason":"response time (generic metric) of the agent's trading service"}

# quantity — creditScore IS a trust indicator BUT unbounded -> quantity (hard rule)
<feedback><tag1>creditScore</tag1><tag2>credprotocol</tag2><scale>unbounded</scale></feedback>
=> {"category":"quantity","feature":"agent_domain","confidence":0.85,"reason":"unbounded forces quantity even though credit score is a trust indicator"}

# quality + agent_domain — evaluation of the agent's security-domain service (trust)
<feedback><tag1>Security Audit</tag1><tag2>Threat Detection</tag2><scale>pct100</scale></feedback>
<agent><agent_domain>DeFi security, analytics</agent_domain></agent>
=> {"category":"quality","feature":"agent_domain","confidence":0.9,"reason":"rates the agent's security-domain service — a trust assessment, bounded"}

# quality + agent_domain — subjective rating of a domain aspect
<feedback><tag1>signal-accuracy</tag1><tag2>spot on calls</tag2><scale>pct100</scale></feedback>
<agent><agent_domain>trading, signals</agent_domain></agent>
=> {"category":"quality","feature":"agent_domain","confidence":0.88,"reason":"rates how good the agent's signals are — subjective quality of a domain aspect"}

# quality + infrastructure — generic quality adjectives, not domain-specific
<feedback><tag1>helpful</tag1><tag2>fast</tag2><scale>pct100</scale></feedback>
=> {"category":"quality","feature":"infrastructure","confidence":0.85,"reason":"generic service-quality adjectives, not tied to a business domain"}

# quality + infrastructure — automated trust/reputation probe (uy-tin), bounded
<feedback><tag1>trust-score</tag1><tag2>reputation</tag2><scale>pct100</scale></feedback>
=> {"category":"quality","feature":"infrastructure","confidence":0.9,"reason":"automated trust/reputation assessment, generic across agents, bounded"}
"""


def system_prompt_v6(include_few_shot: bool = True) -> str:
    """V6 two-axis system prompt; few-shot block opt-in (off for >=12B models)."""
    if include_few_shot:
        return SYSTEM_V6 + "\n" + FEW_SHOT_EXAMPLES_V6
    return SYSTEM_V6


# ── V7 — split calls: category first, feature second ─────────────────────────

SYSTEM_V7_CATEGORY = """You are a high-precision classifier for ERC-8004 feedback records.
Output ONLY `category` — junk | quantity | quality (whether/how the feedback scores).

Pick `category` with this CASCADE — stop at the FIRST matching layer:

LAYER 1 — junk
  - tag1/tag2 meaningless: gibberish, random letters, all-digits, UUID placeholders, "test"/placeholder.
  - OR spam: promo URLs (https://, t.me/), vote-rigging / rank-game ("top 1", "#1 rank").
  - OR a garbage/anomalous value (un-normalized uint256-looking number).

LAYER 2 — quantity (must measure an OUTCOME — not a service evaluation)
  - Tag names a COUNT, RATE, AMOUNT, P/L, volume, completion count, win/loss tally,
    block height, timestamp delta, or a literal operation record (trade executed, swap done).
  - The value answers HOW MUCH / HOW MANY / DID IT HAPPEN — not HOW GOOD.
  - NOT quantity when tag names a SERVICE CAPABILITY the agent offers and the score rates
    how well that service performed: audit, review, verification, rating, analysis,
    labeling, intelligence, optimization (even on pct100/star5).

SCALE RULE
  - `unbounded` scale can NEVER be quality (no normalized satisfaction score).
  - `unbounded` does NOT automatically mean quantity — still apply Layer 1 or Layer 2
    by tag meaning (meaningless → junk; clear amount/outcome → quantity).

LAYER 3 — quality (everything else on bounded scales, plus service evaluations)
  - Subjective adjectives (fast, reliable, excellent, bad), sentiment, satisfaction score.
  - Trust/reputation/safety assessment (trust-score, sybil-resistance, uy-tin).
  - Evaluation of an agent SERVICE or DELIVERABLE on pct100/star5/star10/binary:
    *-audit, *-review, *-rating, *-verification, *-analysis, *-intelligence, *-labeling,
    agent_rating, content-intelligence, Security Audit, data-labeling, etc.

ONLY `quality` feeds the agent's trust score. quantity and junk do not.

DECISION TEST — your reason MUST state which question applied (min 8 words):
  - "HOW GOOD is the agent/service?" → quality
  - "HOW MUCH happened or raw outcome?" → quantity
  - meaningless/spam → junk
  - tag1 ends with -audit, -review, -rating, -verification, -analysis, -intelligence, -labeling
    → lean quality unless tag explicitly counts events (vouch-count, order-count, win-rate).

OUTPUT — strict JSON, one line, no markdown, no fences:
{"category":"<junk|quantity|quality>","confidence":0.00,"reason":"<one sentence citing HOW GOOD or HOW MUCH or junk signal>"}
"""


FEW_SHOT_EXAMPLES_V7_CATEGORY = """
EXAMPLES:

# junk — spam
<feedback><tag1>get top 1 rank</tag1><tag2>t.me/agent_bldr</tag2></feedback>
=> {"category":"junk","confidence":0.99,"reason":"spam rank-game phrase — meaningless promotion, not HOW GOOD or HOW MUCH"}

# junk — meaningless digits
<feedback><tag1>6</tag1><tag2>666</tag2><scale>pct100</scale></feedback>
=> {"category":"junk","confidence":0.98,"reason":"all-digit meaningless tags — junk signal, not a service rating"}

# junk — UUID placeholder
<feedback><tag1>02afee9d-f02c-4f46-a066-9d46c4d505a1</tag1><tag2></tag2><scale>pct100</scale></feedback>
=> {"category":"junk","confidence":0.95,"reason":"UUID placeholder tag — junk signal, no semantic HOW GOOD or HOW MUCH"}

# quantity — generic uptime metric
<feedback><tag1>uptime</tag1><tag2>liveness-check</tag2><scale>pct100</scale></feedback>
=> {"category":"quantity","confidence":0.95,"reason":"uptime completion rate — HOW MUCH availability, not HOW GOOD the service felt"}

# quantity — domain operation outcome (unbounded = not quality, still quantity by tag)
<feedback><tag1>trade</tag1><tag2></tag2><scale>unbounded</scale></feedback>
<agent><agent_domain>celofx, financial_services/trading, defi</agent_domain></agent>
=> {"category":"quantity","confidence":0.92,"reason":"trade outcome on trading agent — HOW MUCH happened; unbounded excludes quality only"}

# quantity — vouch completion rate (contrast with service rating)
<feedback><tag1>miner-vouch</tag1><tag2>botcoin</tag2><scale>pct100</scale></feedback>
<agent><agent_domain>botcoin, mining</agent_domain></agent>
=> {"category":"quantity","confidence":0.90,"reason":"vouch completion rate — HOW MUCH/miner outcome, not HOW GOOD the agent felt"}

# quantity — binary win/loss outcome
<feedback><tag1>claudelance</tag1><tag2>10</tag2><scale>binary</scale></feedback>
=> {"category":"quantity","confidence":0.93,"reason":"binary win/loss outcome — HOW MUCH happened, not a satisfaction score"}

# quality — domain SERVICE rating (NOT a raw metric)
<feedback><tag1>content-intelligence</tag1><tag2>human-verification</tag2><scale>pct100</scale></feedback>
=> {"category":"quality","confidence":0.90,"reason":"rates content-intelligence service — HOW GOOD the verification service performed"}

# quality — agent_rating service evaluation
<feedback><tag1>agent_rating</tag1><tag2>execution-market</tag2><scale>pct100</scale></feedback>
=> {"category":"quality","confidence":0.88,"reason":"agent_rating scores service satisfaction — HOW GOOD, not a raw event count"}

# quality — Security Audit trust assessment
<feedback><tag1>Security Audit</tag1><tag2>MEV Protection</tag2><scale>pct100</scale></feedback>
<agent><agent_domain>DeFi security, analytics</agent_domain></agent>
=> {"category":"quality","confidence":0.90,"reason":"security service trust assessment — HOW GOOD the audit capability is"}

# quality — generic adjectives
<feedback><tag1>helpful</tag1><tag2>fast</tag2><scale>pct100</scale></feedback>
=> {"category":"quality","confidence":0.85,"reason":"generic adjectives — HOW GOOD the service was, not HOW MUCH occurred"}

# quality — signal accuracy evaluation (contrast: signal-accuracy rates quality not count)
<feedback><tag1>signal-accuracy</tag1><tag2>spot on calls</tag2><scale>pct100</scale></feedback>
<agent><agent_domain>trading, signals</agent_domain></agent>
=> {"category":"quality","confidence":0.88,"reason":"rates signal accuracy — HOW GOOD the signals were, not a trade volume count"}
"""


SYSTEM_V7_FEATURE = """You are a feature-axis classifier for ERC-8004 feedback records.
The `category` is already assigned in <assigned_category> — do NOT re-classify it.
Output ONLY `feature` — what the feedback is ABOUT:

  - infrastructure — a GENERIC signal for ANY agent: uptime, liveness, response time,
    reachability, oracle probe, A2A/MCP/Web health, generic trust/reputation probe.
  - agent_domain — SPECIFIC to what THIS agent does (trading, security audit, data labeling, …).
    Lean on <agent_domain>.
  - both — a generic metric measured ON a domain-specific service (e.g. response time of
    THIS agent's trading endpoint).

If <assigned_category> is junk, output feature=null (should not happen — host skips feature for junk).

USING SIGNALS
`<agent_domain>` = what this agent does (service names minus generic plumbing + OASF + tags).
  - Tag inside agent_domain → lean agent_domain.
  - Generic infra unrelated to domain → infrastructure.
  - Generic metric on a domain service → both.
`<endpoint_matched>` True → feedback targets a registered service endpoint (lean agent_domain).

OUTPUT — strict JSON, one line, no markdown, no fences:
{"feature":"<infrastructure|agent_domain|both|null>","confidence":0.00,"reason":"<one short sentence>"}
"""


FEW_SHOT_EXAMPLES_V7_FEATURE = """
EXAMPLES:

<assigned_category>quantity</assigned_category>
<feedback><tag1>uptime</tag1><tag2>liveness-check</tag2><scale>pct100</scale></feedback>
=> {"feature":"infrastructure","confidence":0.95,"reason":"generic uptime probe for any agent"}

<assigned_category>quantity</assigned_category>
<feedback><tag1>trade</tag1><tag2></tag2><scale>unbounded</scale></feedback>
<agent><agent_domain>financial_services/trading, defi</agent_domain></agent>
=> {"feature":"agent_domain","confidence":0.92,"reason":"trade outcome specific to trading agent domain"}

<assigned_category>quantity</assigned_category>
<feedback><tag1>responsetime</tag1><tag2>trading-endpoint</tag2><scale>pct100</scale></feedback>
<agent><agent_domain>financial_services/trading</agent_domain></agent>
=> {"feature":"both","confidence":0.88,"reason":"generic latency metric on a domain trading service"}

<assigned_category>quality</assigned_category>
<feedback><tag1>Security Audit</tag1><tag2>Threat Detection</tag2><scale>pct100</scale></feedback>
<agent><agent_domain>DeFi security, analytics</agent_domain></agent>
=> {"feature":"agent_domain","confidence":0.90,"reason":"security-domain service evaluation"}

<assigned_category>quality</assigned_category>
<feedback><tag1>helpful</tag1><tag2>fast</tag2><scale>pct100</scale></feedback>
=> {"feature":"infrastructure","confidence":0.85,"reason":"generic service-quality adjectives"}
"""


def system_prompt_v7_category(include_few_shot: bool = True) -> str:
    if include_few_shot:
        return SYSTEM_V7_CATEGORY + "\n" + FEW_SHOT_EXAMPLES_V7_CATEGORY
    return SYSTEM_V7_CATEGORY


def system_prompt_v7_feature(include_few_shot: bool = True) -> str:
    if include_few_shot:
        return SYSTEM_V7_FEATURE + "\n" + FEW_SHOT_EXAMPLES_V7_FEATURE
    return SYSTEM_V7_FEATURE


SYSTEM_V5 = """You are a high-precision classifier for ERC-8004 feedback records.
Classify each <feedback> into EXACTLY ONE of four categories.

CATEGORY DEFINITIONS

junk
  - tag1 / tag2 have no meaning: gibberish, random letters, all digits, "test"/placeholder tokens.
  - OR spam: promo URLs (https://, t.me/), vote-rigging or rank-game phrases ("top 1", "#1 rank").

config_feedback
  - An automated, generic infrastructure metric: uptime, liveness, response time, oracle
    dimensions, dated validation periods, protocol-level probes.
  - ASK: would this tag make sense for ANY agent regardless of its business domain?
    If yes → config_feedback.

app_specific
  - The tag names a DOMAIN OPERATION (something the agent DID) or a DOMAIN OUTCOME METRIC
    (a measurable quantity of what was produced — completion count, trade volume, vouch count,
    order count, P/L amount, …).
  - The value records WHAT HAPPENED / HOW MUCH: an outcome, count, or rate of occurrence —
    NOT how good it was.
  - The agent_domain signal helps: a tag naming a concept inside agent_domain that reads as
    an action/event/outcome measurement → app_specific.
  - `<scale>` can be anything including pct100 — a 96% completion rate is app_specific.

service_feedback
  - The tag is a QUALITY JUDGMENT about the agent: an adjective (fast, reliable, excellent),
    sentiment, a satisfaction score, or full-sentence praise/criticism.
  - Tags MAY be domain-related — when the score rates HOW GOOD a domain aspect was
    (e.g., accuracy of trade calls, signal reliability, execution quality), it is
    service_feedback even if both tags name domain concepts.
  - The distinguishing question: does the value rate QUALITY (how good?) or measure OUTCOME
    (how much/many?)  → Quality rating = service_feedback; Outcome measurement = app_specific.

SCALE IS A SECONDARY SIGNAL — do NOT use it as the primary discriminator:
  - `unbounded` → almost always app_specific (counts/amounts have no natural ceiling).
  - `pct100` / `binary` → could be either; apply the quality-vs-outcome test above.

EVALUATION ORDER — stop at the FIRST matching category:
  1) junk
  2) config_feedback   (generic infra / automated probe)
  3) app_specific      (domain operation or domain metric)
  4) service_feedback  (quality judgment / sentiment)

USING SIGNALS

`<agent_domain>` — the canonical "what this agent does" bundle (service names excluding
generic plumbing like Web/Email/A2A + OASF domains + OASF skills + tags).
  - Tag inside agent_domain + reads as an action/event/outcome measurement → app_specific.
  - Tag inside agent_domain + rates quality of that domain aspect (how good?) → service_feedback.
  - Tag is a quality adjective/sentiment regardless of domain → service_feedback.
  - Tag is outside agent_domain and not generic infra → service_feedback if adjective,
    lean config_feedback otherwise.

`<endpoint_matched>` — when `True`, the feedback targets a registered service endpoint.
Junk is excluded for these records (enforced by the host). Pick from {config, app, service}.

CONFIDENCE
  - Confidence > 0.80 when signals clearly support the category.
  - When evidence is thin output your best-guess with lower confidence — do NOT output "others".

OUTPUT — strict JSON, one line, no markdown, no fences:
{"category":"<one>","confidence":0.00,"reason":"<one short sentence>"}
"""


FEW_SHOT_EXAMPLES_V5 = """
EXAMPLES:

# junk — spam URL / rank game
<feedback><tag1>get top 1 rank</tag1><tag2>t.me/agent_bldr</tag2></feedback>
=> {"category":"junk","confidence":0.99,"reason":"telegram link and rank-game phrase"}

# junk — placeholder tokens
<feedback><tag1>test</tag1><tag2>asd</tag2></feedback>
=> {"category":"junk","confidence":0.95,"reason":"placeholder tokens with no semantic signal"}

# config_feedback — generic infra probe (would apply to ANY agent)
<feedback><tag1>liveness</tag1><tag2>liveness-check</tag2><scale>binary</scale></feedback>
=> {"category":"config_feedback","confidence":0.96,"reason":"liveness infra probe, generic across all agents"}

# config_feedback — oracle dimension; tag2 names the probe protocol → generic infra regardless of agent domain
<feedback><tag1>trust</tag1><tag2>oracle-screening</tag2><scale>pct100</scale></feedback>
<agent><agent_domain>financial_services/trading, signals</agent_domain></agent>
=> {"category":"config_feedback","confidence":0.94,"reason":"oracle-screening is an automated infra probe, not a quality judgment about this agent"}

# app_specific — domain OPERATION (agent did a trade); scale=binary does not make this infra
<feedback><tag1>trade</tag1><tag2></tag2><scale>binay</scale></feedback>
<agent><agent_domain>celofx, financial_services/trading, defi, forex</agent_domain></agent>
=> {"category":"app_specific","confidence":0.92,"reason":"trade is a domain operation on a trading agent, not a generic infra probe"}

# app_specific — domain METRIC with pct100; pct100 alone does NOT imply service_feedback
<feedback><tag1>personality</tag1><tag2>fragment</tag2><scale>pct100</scale></feedback>
<agent><agent_domain>ensoul, digital_soul, nft</agent_domain></agent>
=> {"category":"app_specific","confidence":0.93,"reason":"soul-dimension fragment is a domain operation (Digital Soul protocol), not a quality adjective"}

# app_specific — domain metric; 96% completion rate is a measurable outcome, not a quality adjective
<feedback><tag1>miner-vouch</tag1><tag2>botcoin</tag2><scale>pct100</scale></feedback>
<agent><agent_domain>botcoin, mining</agent_domain></agent>
=> {"category":"app_specific","confidence":0.90,"reason":"vouch operation tied to agent's mining domain, pct100 is the completion rate not a satisfaction score"}

# service_feedback — quality ADJECTIVE; 'helpful' and 'fast' name how good the service was, not what it did
<feedback><tag1>helpful</tag1><tag2>fast</tag2><scale>pct100</scale></feedback>
=> {"category":"service_feedback","confidence":0.88,"reason":"quality adjectives describing service satisfaction, not a domain operation"}

# service_feedback — full-sentence quality praise; tag2 is sentiment, not an operation
<feedback><tag1>quality</tag1><tag2>BTC long call at 97k was spot on.</tag2><scale>pct100</scale><endpoint_matched>True</endpoint_matched></feedback>
<agent><agent_domain>celofx, financial_services/trading, defi</agent_domain></agent>
=> {"category":"service_feedback","confidence":0.93,"reason":"full-sentence quality praise about trading signals, sentiment not a domain operation"}

# service_feedback — both tags domain-related but score RATES quality of a domain aspect
<feedback><tag1>signal-accuracy</tag1><tag2>trade-calls</tag2><scale>pct100</scale></feedback>
<agent><agent_domain>financial_services/trading, defi, signals</agent_domain></agent>
=> {"category":"service_feedback","confidence":0.87,"reason":"rates how accurate the agent's signals were — quality evaluation of domain performance, not a completion count"}

# service_feedback vs app_specific — same domain (trading), different intent:
#   'reliability' rates quality → service_feedback
#   'trade' records an outcome → app_specific
<feedback><tag1>reliability</tag1><tag2>top-tier-logic</tag2><scale>pct100</scale></feedback>
<agent><agent_domain>financial_services/trading, defi</agent_domain></agent>
=> {"category":"service_feedback","confidence":0.87,"reason":"reliability rates the quality of the agent's logic, not a domain outcome measurement"}
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
