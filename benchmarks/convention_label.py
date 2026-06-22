#!/usr/bin/env python3
"""First-pass silver labeler encoding the gold labeling convention.

Convention (confirmed against the 320-row human gold set + analyst):

  junk      <- the tag pair is genuinely meaningless: UUID/hash, bare numbers,
               or known noise tokens (custom / vibez / asd / test).  NOTE: empty
               tags are NOT junk (gold labels empty-tag pct100 as quality).
  quantity  <- scale is UNBOUNDED, or a tag names an explicit throughput/rate/
               speed/score/freshness metric (success-rate, settlement-speed,
               creditScore, viral-score, blocktime-freshness, active/liveness ...).
               'trust*' tags are excluded (trust-score is a quality signal in gold).
               'fitness' is excluded (epoch-fitness is bounded -> quality).
  quality   <- DEFAULT.  Everything domain-relevant that is neither a named
               metric nor gibberish: sentiment, service judgments, completed
               domain actions/events (delivered, bounty-resolved, agentAction,
               match_completed, stake), ownerVerified, agentkarma_metadata,
               userRating, epoch-fitness, bare bounded scores.

This produces the first pass.  The minority quantity/junk picks are then
reviewed by hand (Opus) with agent-domain context to catch errors; the quality
default is the safe bucket.

Usage:
    .venv/bin/python3 -m benchmarks.convention_label
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ── junk detection ────────────────────────────────────────────────────────────
# Junk keys primarily off tag1 being gibberish (a UUID/emoji in tag2 alongside a
# real metric tag1 is just an entity reference -> NOT junk, e.g. gold labels
# attendance-rate/<uuid> as quantity).  Spam links/rank-manipulation are junk
# wherever they appear.  Mirrors the deployed Go rule patterns.
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
SPAM_URL_RE = re.compile(r"(?i)(t\.me/|telegram\.me|https?://)")
SPAM_RANK_RE = re.compile(r"(?i)(get\s+top|top\s*[0-9]|-{2,}>|#1\s+rank)")
EMOJI_ONLY_RE = re.compile(
    r"^[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    r"\U00002190-\U000021FF\U0000FE0F\U0001F3FB-\U0001F3FF\s\U0000200D]+$")
NOISE_TOKENS = {"custom", "vibez", "asd", "test"}
GIBBERISH_TOKENS = {"nsjak", "asdjck", "uh oh"}  # observed keyboard-mash / non-feedback
FARM_TOKENS = {"claudelance"}  # brand/farming tokens that are junk when paired with a bare index


def _is_num(s: str) -> bool:
    return bool(s) and re.fullmatch(r"-?\d+", s.strip()) is not None


def _is_gibberish_tag1(t1: str) -> bool:
    """tag1 itself carries no meaning (UUID / emoji-only / noise token / number)."""
    l1 = t1.lower()
    if UUID_RE.match(t1):
        return True
    if l1 in NOISE_TOKENS or l1 in GIBBERISH_TOKENS:
        return True
    if _is_num(l1):
        return True
    if t1 and EMOJI_ONLY_RE.match(t1):
        return True
    return False


def is_junk(tag1: str, tag2: str) -> bool:
    t1, t2 = tag1.strip(), tag2.strip()
    combined = f"{t1} {t2}"
    # spam links / rank-manipulation anywhere -> junk
    if SPAM_URL_RE.search(combined) or SPAM_RANK_RE.search(combined):
        return True
    # tag1 is gibberish, and tag2 is empty or also gibberish/numeric -> junk
    if _is_gibberish_tag1(t1) and (t2 == "" or _is_gibberish_tag1(t2) or _is_num(t2)):
        return True
    # farming token (e.g. claudelance) paired with a bare integer index -> junk
    if t1.lower() in FARM_TOKENS and _is_num(t2):
        return True
    return False


# ── quantity detection ────────────────────────────────────────────────────────
# Explicit named-metric vocabulary (substring match, word-ish).  Mirrors the Go
# quantityTag1Set/quantityTag2Set plus gold-observed metric names.
METRIC_SUBSTR = [
    "rate", "speed", "freshness", "finality", "uptime", "liveness", "latency",
    "throughput", "reachable", "attendance", "coverage", "scroll-stop",
    "response-time", "responsetime", "hybridsignal", "blocktime",
    "creditscore", "credit-score", "safety-score", "viral-score",
    "readability-score", "contractrisk", "longevity", "win-rate", "exit-rate",
    "completion-rate", "success-rate", "successrate", "execution-speed",
    "payment-speed", "settlement-speed", "coefficient", "sybilrisk",
    "identitycount", "credit_score", "credit-score",
]
METRIC_WHOLE = {"active", "activity", "counterparty"}
# tags that look metric-ish but are QUALITY by gold convention
QUALITY_OVERRIDE_SUBSTR = ["trust", "fitness"]


def _has_metric(tag: str) -> bool:
    t = tag.strip().lower()
    if not t:
        return False
    if any(ov in t for ov in QUALITY_OVERRIDE_SUBSTR):
        return False
    if t in METRIC_WHOLE:
        return True
    return any(m in t for m in METRIC_SUBSTR)


def _is_negative(value) -> bool:
    """True when the raw value is numeric and negative. Negative values are accepted
    but have no positive bounded scale (sign is decimals-invariant)."""
    try:
        return float(str(value).strip()) < 0
    except (ValueError, TypeError):
        return False


def classify(tag1: str, tag2: str, scale: str, is_self: bool = False,
             value=None) -> tuple[str, str]:
    t1, t2, sc = tag1.strip(), tag2.strip(), scale.strip().lower()
    # self-feedback (clientAddress == agent owner/wallet) is self-promotion, not a
    # credible third-party evaluation -> junk, regardless of how the content reads.
    if is_self:
        return "junk", "self-feedback: clientAddress == agent owner/wallet"
    if is_junk(t1, t2):
        return "junk", "gibberish/noise tag pair"
    # named metric on either tag -> quantity (even if bounded)
    if _has_metric(t1) or _has_metric(t2):
        return "quantity", "explicit named rate/speed/score/freshness metric"
    # negative value: accepted, but quality is only judged on a positive bounded scale,
    # so a negative is treated as unbounded -> quantity (never quality, never junk).
    if _is_negative(value):
        return "quantity", "negative value -> unbounded scale -> quantity (not a positive quality score)"
    # unbounded measurement -> quantity
    if sc == "unbounded":
        return "quantity", "unbounded-scale measurement"
    # default
    return "quality", "default: domain-relevant, non-metric, non-gibberish"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path,
                        default=ROOT / "data/labelled/others_to_label.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.records).fillna("")
    has_self = "is_self" in df.columns
    labels, reasons = [], []
    for _, r in df.iterrows():
        is_self = str(r["is_self"]).strip().lower() in ("true", "1", "1.0") if has_self else False
        lab, rsn = classify(str(r["tag1"]), str(r["tag2"]), str(r["scale"]), is_self,
                            value=r.get("value"))
        labels.append(lab)
        reasons.append(rsn)
    df["opus_label"] = labels
    df["opus_reason"] = reasons
    df.to_csv(args.records, index=False)

    print(f"First-pass labeled {len(df)} records")
    print(df["opus_label"].value_counts().to_string())
    print(f"\nquality {(df['opus_label']=='quality').mean()*100:.1f}% "
          f"(gold is ~90%)")


if __name__ == "__main__":
    main()
