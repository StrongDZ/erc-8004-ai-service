#!/usr/bin/env python3
"""Label "others" pool records using Claude Opus via the Anthropic API.

Reads the CSV prepared by `prepare_others_for_labeling.py`, sends each
unlabeled row to Opus with full agent + feedback context, writes the label
and brief reason back into the `opus_label` / `opus_reason` columns.

Run this script in a session where ANTHROPIC_API_KEY is set.

Usage:
    cd erc-8004-ai-service
    ANTHROPIC_API_KEY=sk-ant-... .venv/bin/python3 -m benchmarks.label_with_opus \\
        --input  data/labelled/others_to_label.csv \\
        --output data/labelled/others_to_label.csv \\
        --model  claude-opus-4-8 \\
        --batch  50

The script saves progress after every --batch rows so it is safe to interrupt
and resume (already-labeled rows are skipped).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

try:
    import anthropic
except ImportError:
    sys.exit("anthropic SDK not found — run: pip install anthropic")

VALID_LABELS = {"quality", "quantity", "junk"}

SYSTEM_PROMPT = """\
You are an expert labeler for an ERC-8004 on-chain agent reputation benchmark.

Your task: classify one piece of feedback left by a user about an AI agent.

## Taxonomy (choose exactly one)

quality   — A subjective judgment about the agent's service, skill, trustworthiness,
            reliability, professionalism, or experience quality.
            Examples: "helpful", "fast", "reliable", "great service", "transparency",
            "accurate", "excellent support"

quantity  — A measurable metric or performance outcome: a rate, score, count, speed,
            or binary status check. The value is objectively verifiable.
            Examples: "uptime", "successrate", "response-time", "win-rate", "liveness"

junk      — Meaningless, spam, test data, placeholder text, or feedback with no
            connection to the agent's services or business.
            Examples: gibberish tags, test records, unrelated keywords, competitor spam

## Decision guide

1. If the tag describes a MEASURED METRIC (rate / score / count / time / status) → quantity
2. If the tag expresses a SUBJECTIVE EVALUATION or SENTIMENT about the agent's work → quality
3. If the tag is UNRELATED to the agent's domain OR clearly spam/noise → junk
4. Use the agent domain context to disambiguate: a tag like "fast" for a DeFi settlement
   agent is quality (speed as a service attribute). "fast" as an unbounded numeric value
   might indicate quantity.

## Output format (JSON, nothing else)

{"label": "quality|quantity|junk", "reason": "one-sentence justification"}
"""


def _build_user_message(row: pd.Series) -> str:
    parts = ["<feedback>"]
    parts.append(f"  <tag1>{row.get('tag1', '') or '(empty)'}</tag1>")
    parts.append(f"  <tag2>{row.get('tag2', '') or '(empty)'}</tag2>")
    parts.append(f"  <scale>{row.get('scale', '') or '(unknown)'}</scale>")
    if row.get("value"):
        parts.append(f"  <value>{str(row['value'])[:40]}</value>")
    if row.get("offchain_note"):
        parts.append(f"  <offchain_note>{str(row['offchain_note'])[:300]}</offchain_note>")
    parts.append("</feedback>")
    parts.append("<agent>")
    if row.get("agent_name"):
        parts.append(f"  <name>{row['agent_name']}</name>")
    if row.get("agent_description"):
        parts.append(f"  <description>{str(row['agent_description'])[:500]}</description>")
    if row.get("agent_services"):
        parts.append(f"  <services>{row['agent_services']}</services>")
    if row.get("agent_oasf_domains_text"):
        parts.append(f"  <domains>{str(row['agent_oasf_domains_text'])[:300]}</domains>")
    if row.get("agent_oasf_skills_text"):
        parts.append(f"  <skills>{str(row['agent_oasf_skills_text'])[:400]}</skills>")
    if row.get("agent_tags"):
        parts.append(f"  <tags>{row['agent_tags']}</tags>")
    parts.append("</agent>")
    return "\n".join(parts)


def label_batch(
    client: "anthropic.Anthropic",
    df: pd.DataFrame,
    model: str,
    batch_size: int,
    out_path: Path,
) -> pd.DataFrame:
    import json, re

    total = len(df)
    unlabeled_mask = ~df["opus_label"].str.strip().str.lower().isin(VALID_LABELS)
    todo = df[unlabeled_mask].index.tolist()
    print(f"Total rows: {total} | Unlabeled: {len(todo)} | Will skip: {total - len(todo)}")

    for i, idx in enumerate(todo):
        row = df.loc[idx]
        user_msg = _build_user_message(row)

        try:
            resp = client.messages.create(
                model=model,
                max_tokens=80,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text.strip()
            # Parse JSON
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                # Try extracting JSON from surrounding text
                m = re.search(r'\{[^}]+\}', raw)
                parsed = json.loads(m.group()) if m else {}
            label = parsed.get("label", "").strip().lower()
            reason = parsed.get("reason", "").strip()
            if label not in VALID_LABELS:
                label = "junk"
                reason = f"parse_fallback: {raw[:100]}"
        except Exception as e:
            label = ""
            reason = f"error: {e}"

        df.at[idx, "opus_label"] = label
        df.at[idx, "opus_reason"] = reason

        if (i + 1) % batch_size == 0 or (i + 1) == len(todo):
            df.to_csv(out_path, index=False)
            done = (df["opus_label"].str.lower().isin(VALID_LABELS)).sum()
            print(f"  [{i+1}/{len(todo)}] saved — labeled so far: {done}/{total}")

        # Respect API rate limits
        time.sleep(0.1)

    return df


def main() -> None:
    ROOT = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data/labelled/others_to_label.csv")
    parser.add_argument("--output", type=Path, default=None,
                        help="defaults to same as --input (in-place)")
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--batch", type=int, default=50,
                        help="save progress every N rows")
    args = parser.parse_args()

    out_path = args.output or args.input

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    df = pd.read_csv(args.input).fillna("")
    print(f"Loaded {len(df)} rows from {args.input}")

    df = label_batch(client, df, args.model, args.batch, out_path)

    done = df["opus_label"].str.lower().isin(VALID_LABELS)
    print(f"\nDone. {done.sum()}/{len(df)} rows labeled.")
    print("Label distribution:")
    print(df.loc[done, "opus_label"].value_counts().to_string())
    print(f"\nOutput: {out_path}")
    print("Next step:")
    print(f"  .venv/bin/python3 -m benchmarks.merge_silver_gold")


if __name__ == "__main__":
    main()
