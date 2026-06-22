#!/usr/bin/env python3
"""Print a batch of still-unlabeled feedback records in compact, readable form.

Per-feedback labeling: EVERY record is labeled individually (no signature
collapsing), so value / value_decimals differences that push a record into a
different tier are never hidden.  The labeler (Claude Opus, in-conversation)
reads the output, decides a label per feedback_id, and writes a batch JSON
into data/labelled/label_batches/.  Records whose opus_label is already filled
are skipped, so the loop is resumable.

Usage:
    .venv/bin/python3 -m benchmarks.dump_batch --count 40
    .venv/bin/python3 -m benchmarks.dump_batch --count 40   # next 40, auto-skips done
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

VALID_LABELS = {"quality", "quantity", "junk"}


def _trunc(s, n):
    s = str(s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def dump(rec_path: Path, count: int, offset: int) -> None:
    df = pd.read_csv(rec_path).fillna("")
    if "opus_label" not in df.columns:
        df["opus_label"] = ""
    done_mask = df["opus_label"].astype(str).str.strip().str.lower().isin(VALID_LABELS)
    todo = df[~done_mask].iloc[offset:offset + count]
    total_done = done_mask.sum()
    print(f"### Progress: {total_done}/{len(df)} labeled | "
          f"{(~done_mask).sum()} remaining | showing {len(todo)} (offset {offset})")
    print("### Records grouped by agent (context shown once); LABEL EACH id individually.\n")

    # Preserve first-seen agent order within this batch
    for agent_key in dict.fromkeys(todo["agent_key"].tolist()):
        grp = todo[todo["agent_key"] == agent_key]
        r0 = grp.iloc[0]
        an = _trunc(r0["agent_name"], 70)
        print(f"████ AGENT {an or '(no name)'}  [{agent_key}]  ({len(grp)} feedback)")
        if str(r0.get("agent_description", "")).strip():
            print(f"   desc: {_trunc(r0['agent_description'], 320)}")
        if str(r0.get("agent_services", "")).strip():
            print(f"   services: {_trunc(r0['agent_services'], 180)}")
        if str(r0.get("agent_oasf_domains_text", "")).strip():
            print(f"   oasf_domains: {_trunc(r0['agent_oasf_domains_text'], 200)}")
        if str(r0.get("agent_oasf_skills_text", "")).strip():
            print(f"   oasf_skills: {_trunc(r0['agent_oasf_skills_text'], 220)}")
        if str(r0.get("agent_tags", "")).strip():
            print(f"   tags: {_trunc(r0['agent_tags'], 120)}")
        for _, r in grp.iterrows():
            extra = ""
            if str(r.get("endpoint", "")).strip():
                extra += f" | ep={_trunc(r['endpoint'], 50)}"
            if str(r.get("offchain_note", "")).strip():
                extra += f" | offchain={_trunc(r['offchain_note'], 160)}"
            print(f"   • id={r['feedback_id']}")
            print(f"       tag1={r['tag1'] or '∅'} | tag2={r['tag2'] or '∅'} | "
                  f"scale={r['scale'] or '?'} | value={_trunc(r['value'], 24)} | "
                  f"dec={r['value_decimals']}{extra}")
        print()


def main() -> None:
    ROOT = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path,
                        default=ROOT / "data/labelled/others_to_label.csv")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--offset", type=int, default=0,
                        help="skip this many unlabeled records before showing")
    args = parser.parse_args()
    dump(args.records, args.count, args.offset)


if __name__ == "__main__":
    main()
