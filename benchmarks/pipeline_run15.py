#!/usr/bin/env python3
"""Run 15: Unified fine-tuned classifier + Chow reject rule.

Implements the proposal from docs/unified_classifier_proposal.md:
  - Fine-tune encoder contrastively (BatchAllTripletLoss) on fused text:
      "{tag1} {tag2} {scale} [SEP] {agent_description}"
  - Train symmetric 3-class LogisticRegression head (class_weight='balanced').
  - Calibrate Chow reject threshold τ on held-out validation fold:
      choose smallest τ s.t. error on retained val records ≤ 10%.
  - Structural constraint: unbounded scale → zero P(quality), renormalise.
  - Evaluate on gold: if max(P) ≥ τ → argmax; else → LLM fallback (cache).

Two backbone variants:
  --backbone bge-small  → BAAI/bge-small-en-v1.5  (33M, 384-dim)   [Run 15a]
  --backbone bge-base   → BAAI/bge-base-en-v1.5   (109M, 768-dim)  [Run 15b]

Usage:
    cd erc-8004-ai-service
    .venv/bin/python3 -m benchmarks.pipeline_run15 \\
        --backbone bge-small --gold data/labelled/pure_others_stratified_dedup.csv
    .venv/bin/python3 -m benchmarks.pipeline_run15 \\
        --backbone bge-base  --gold data/labelled/pure_others_stratified_dedup.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.pipeline_3tier_v2 import LLM_MODEL, load_gold, llm_classify
from shared.types import LLM_OUTPUT_CATEGORIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits" / "agent_enriched"
OUT_DIR = ROOT / "data" / "benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BACKBONES = {
    "bge-small":  "BAAI/bge-small-en-v1.5",
    "bge-base":   "BAAI/bge-base-en-v1.5",
    "modernbert": "answerdotai/ModernBERT-base",
}
CLASSES = ["junk", "quality", "quantity"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
TAU_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


# ── Text builders ─────────────────────────────────────────────────────────────

def build_fused_text(tag1: str, tag2: str, scale: str, agent_text: str = "") -> str:
    tag_part = " ".join(p for p in [tag1.strip(), tag2.strip(), scale.strip()] if p)
    if (agent_text or "").strip():
        return f"{tag_part} [SEP] {agent_text.strip()[:400]}"
    return tag_part


def apply_unbounded_constraint(proba: np.ndarray, scales: list[str]) -> np.ndarray:
    """Zero P(quality) for unbounded scale records, renormalise."""
    out = proba.copy()
    q_idx = CLASS2IDX["quality"]
    for i, sc in enumerate(scales):
        if str(sc).strip().lower() == "unbounded":
            out[i, q_idx] = 0.0
            row_sum = out[i].sum()
            if row_sum > 0:
                out[i] /= row_sum
            else:
                out[i, CLASS2IDX["quantity"]] = 1.0
    return out


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def finetune_encoder(backbone_name: str, texts: list[str], labels: list[int],
                     epochs: int = 8, batch_size: int = 32, lr: float = 2e-5) -> "SentenceTransformer":
    """Manual contrastive fine-tuning loop using BatchAllTripletLoss.

    Avoids model.fit() (which requires the `datasets` package) by building
    batches directly from lists and calling the loss module as a plain nn.Module.
    """
    import random as _random

    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR

    from sentence_transformers import SentenceTransformer
    from sentence_transformers.losses import BatchAllTripletLoss

    log.info("Fine-tuning %s | %d examples | %d epochs | batch=%d | lr=%.0e",
             backbone_name, len(texts), epochs, batch_size, lr)

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    log.info("Device: %s", device)

    model = SentenceTransformer(backbone_name)
    model.to(device)
    loss_fn = BatchAllTripletLoss(model)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * max(1, len(texts) // batch_size)
    scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                         total_iters=max(1, total_steps // 10))

    label_tensor_all = torch.tensor(labels, dtype=torch.long)
    indices = list(range(len(texts)))

    for epoch in range(epochs):
        _random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            if len(batch_idx) < 4:
                continue  # too small for triplet mining

            batch_texts = [texts[i] for i in batch_idx]
            batch_labels = label_tensor_all[batch_idx].to(device)

            import torch as _torch
            features = model.tokenize(batch_texts)
            features = {
                k: v.to(device) if isinstance(v, _torch.Tensor) else v
                for k, v in features.items()
            }

            optimizer.zero_grad()
            loss_val = loss_fn([features], batch_labels)
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss_val.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        log.info("Epoch %d/%d  avg_loss=%.4f", epoch + 1, epochs, avg)

    # Save and reload to ensure clean state
    tmpdir = tempfile.mkdtemp(prefix="run15_")
    model.save(tmpdir)
    log.info("Fine-tuned model saved to %s", tmpdir)
    return SentenceTransformer(tmpdir)


# ── Chow threshold calibration ────────────────────────────────────────────────

def calibrate_tau(proba: np.ndarray, y_true: np.ndarray,
                  max_error: float = 0.10) -> tuple[float, dict]:
    """Return τ = smallest value s.t. error on retained records ≤ max_error."""
    max_probs = proba.max(axis=1)
    preds = proba.argmax(axis=1)
    stats = {}
    best_tau = TAU_GRID[-1]

    for tau in TAU_GRID:
        mask = max_probs >= tau
        n_kept = mask.sum()
        if n_kept == 0:
            continue
        err = 1.0 - (preds[mask] == y_true[mask]).mean()
        cov = n_kept / len(y_true)
        stats[tau] = {"coverage": round(cov, 4), "error": round(err, 4), "n_kept": int(n_kept)}
        if err <= max_error:
            best_tau = tau
            break

    log.info("Chow calibration → τ=%.2f", best_tau)
    for tau, s in stats.items():
        log.info("  τ=%.2f: coverage=%.2f  error=%.3f  n_kept=%d", tau, s["coverage"], s["error"], s["n_kept"])
    return best_tau, stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["bge-small", "bge-base", "modernbert"], default="bge-small")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-error", type=float, default=0.10,
                        help="Chow reject: maximum error rate on retained val records")
    args = parser.parse_args()

    backbone_id = BACKBONES[args.backbone]
    run_name = f"pipeline_run15_{args.backbone.replace('-', '_')}"
    log.info("=== %s ===  backbone=%s", run_name, backbone_id)

    # ── 1. Load training data ──────────────────────────────────────────────
    log.info("Loading training data from parquets...")
    ga = pd.read_parquet(SPLITS_DIR / "group_a.parquet")
    gb = pd.read_parquet(SPLITS_DIR / "group_b.parquet")
    df = pd.concat([ga, gb], ignore_index=True)
    df = df[df["label"].isin(CLASSES)].reset_index(drop=True)
    log.info("Train pool N=%d | %s", len(df), df["label"].value_counts().to_dict())

    # ── 2. Build fused texts ───────────────────────────────────────────────
    def make_fused(row: pd.Series) -> str:
        return build_fused_text(
            str(row.get("tag1") or ""),
            str(row.get("tag2") or ""),
            str(row.get("value_scale") or ""),
            str(row.get("agent_description") or ""),
        )

    df["fused_text"] = df.apply(make_fused, axis=1)
    df["label_idx"] = df["label"].map(CLASS2IDX)

    # Train/val split (80/20 stratified)
    train_df, val_df = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=42)
    log.info("Split → train=%d val=%d", len(train_df), len(val_df))

    # ── 3. Fine-tune encoder ───────────────────────────────────────────────
    t_finetune_start = time.monotonic()
    ft_model = finetune_encoder(
        backbone_id,
        train_df["fused_text"].tolist(),
        train_df["label_idx"].tolist(),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    finetune_secs = time.monotonic() - t_finetune_start
    log.info("Fine-tuning done in %.1fs", finetune_secs)

    # ── 4. Extract embeddings + train 3-class logreg ───────────────────────
    log.info("Encoding training set with fine-tuned encoder...")
    X_train = ft_model.encode(train_df["fused_text"].tolist(), normalize_embeddings=True,
                              batch_size=64, show_progress_bar=True)
    y_train = train_df["label_idx"].values

    log.info("Training 3-class LogisticRegression (class_weight=balanced)...")
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=42)
    clf.fit(X_train, y_train)

    # ── 5. Calibrate τ on val set ──────────────────────────────────────────
    log.info("Encoding validation set...")
    X_val = ft_model.encode(val_df["fused_text"].tolist(), normalize_embeddings=True,
                            batch_size=64, show_progress_bar=False)
    y_val = val_df["label_idx"].values
    val_scales = val_df["value_scale"].tolist()

    val_proba = clf.predict_proba(X_val)
    val_proba = apply_unbounded_constraint(val_proba, val_scales)
    tau, chow_stats = calibrate_tau(val_proba, y_val, max_error=args.max_error)

    val_acc = (val_proba.argmax(axis=1) == y_val).mean()
    log.info("Val accuracy (unconstrained): %.4f", val_acc)

    # ── 6. Load gold set ───────────────────────────────────────────────────
    log.info("Loading gold set from %s...", args.gold)
    gold = load_gold(args.gold)
    gold = gold[gold["label"].isin(LLM_OUTPUT_CATEGORIES)].reset_index(drop=True)
    log.info("Gold N=%d | %s", len(gold), gold["label"].value_counts().to_dict())

    def make_gold_fused(row: pd.Series) -> str:
        return build_fused_text(
            str(row.get("tag1") or ""),
            str(row.get("tag2") or ""),
            str(row.get("value_scale") or ""),
            str(row.get("agent_domain_text") or "") or str(row.get("agent_description") or ""),
        )

    gold_texts = [make_gold_fused(row) for _, row in gold.iterrows()]
    gold_scales = gold.get("value_scale", pd.Series([""] * len(gold))).tolist()

    log.info("Encoding gold set with fine-tuned encoder...")
    t_enc = time.monotonic()
    X_gold = ft_model.encode(gold_texts, normalize_embeddings=True,
                             batch_size=64, show_progress_bar=True)
    gold_proba = clf.predict_proba(X_gold)
    gold_proba = apply_unbounded_constraint(gold_proba, gold_scales)
    enc_secs = time.monotonic() - t_enc

    # ── 7. Evaluate across τ grid ──────────────────────────────────────────
    log.info("Evaluating across τ grid on gold set (LLM fills escalations)...")
    y_true = gold["label"].str.strip().str.lower().tolist()
    gold_max_probs = gold_proba.max(axis=1)

    results_by_tau: dict[float, dict] = {}
    for eval_tau in TAU_GRID:
        y_pred_list: list[str] = []
        n_model, n_llm, n_failed = 0, 0, 0
        for i, (_, row) in enumerate(gold.iterrows()):
            if gold_max_probs[i] >= eval_tau:
                cls_idx = gold_proba[i].argmax()
                y_pred_list.append(CLASSES[cls_idx])
                n_model += 1
            else:
                try:
                    llm_pred = llm_classify(row, LLM_MODEL)
                    llm_pred = llm_pred.strip().lower()
                    if llm_pred not in LLM_OUTPUT_CATEGORIES:
                        llm_pred = "quality"
                    y_pred_list.append(llm_pred)
                    n_llm += 1
                except Exception:
                    y_pred_list.append("quality")
                    n_failed += 1

        macro_f1 = f1_score(y_true, y_pred_list, labels=LLM_OUTPUT_CATEGORIES,
                            average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred_list, labels=LLM_OUTPUT_CATEGORIES,
                               average="weighted", zero_division=0)
        per_class = classification_report(y_true, y_pred_list, labels=LLM_OUTPUT_CATEGORIES,
                                          zero_division=0, output_dict=True)
        llm_rate = n_llm / len(y_true)

        results_by_tau[eval_tau] = {
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "quality_f1": round(per_class.get("quality", {}).get("f1-score", 0), 4),
            "quality_recall": round(per_class.get("quality", {}).get("recall", 0), 4),
            "quantity_f1": round(per_class.get("quantity", {}).get("f1-score", 0), 4),
            "quantity_recall": round(per_class.get("quantity", {}).get("recall", 0), 4),
            "junk_f1": round(per_class.get("junk", {}).get("f1-score", 0), 4),
            "llm_rate": round(llm_rate, 4),
            "n_model": n_model,
            "n_llm": n_llm,
            "n_failed": n_failed,
            "per_class_full": per_class,
        }

        log.info("τ=%.2f  Macro F1=%.4f  Weighted F1=%.4f  Qty Recall=%.3f  LLM=%.1f%%",
                 eval_tau, macro_f1, weighted_f1,
                 per_class.get("quantity", {}).get("recall", 0), llm_rate * 100)

    # ── 8. Print summary table ─────────────────────────────────────────────
    best_tau_result = results_by_tau.get(tau, results_by_tau[TAU_GRID[-1]])
    best_macro_tau = max(results_by_tau, key=lambda t: results_by_tau[t]["macro_f1"])

    print(f"\n=== {run_name}  backbone={args.backbone} ===")
    print(f"{'τ':>6} {'Macro F1':>9} {'W-F1':>9} {'Qty Recall':>11} {'LLM%':>7}")
    for tau_val in TAU_GRID:
        r = results_by_tau[tau_val]
        star = " ← Chow" if abs(tau_val - tau) < 0.001 else \
               " ← best Macro" if abs(tau_val - best_macro_tau) < 0.001 else ""
        print(f"{tau_val:>6.2f} {r['macro_f1']:>9.4f} {r['weighted_f1']:>9.4f} "
              f"{r['quantity_recall']:>11.3f} {r['llm_rate']*100:>6.1f}%{star}")

    print(f"\nChow τ={tau}  Macro F1={best_tau_result['macro_f1']}")
    print(f"Best Macro τ={best_macro_tau}  Macro F1={results_by_tau[best_macro_tau]['macro_f1']}")
    print(f"Fine-tune time: {finetune_secs:.0f}s  Gold encode: {enc_secs:.1f}s")

    # ── 9. Save results ────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "run": run_name,
        "backbone": backbone_id,
        "backbone_short": args.backbone,
        "gold_csv": str(args.gold),
        "n_gold": len(gold),
        "n_train": len(train_df),
        "epochs": args.epochs,
        "chow_tau": tau,
        "chow_stats": {str(k): v for k, v in chow_stats.items()},
        "finetune_secs": round(finetune_secs, 1),
        "timestamp": ts,
        "results_by_tau": {str(t): v for t, v in results_by_tau.items()},
        "chow_result": best_tau_result,
        "best_macro_tau": best_macro_tau,
        "best_macro_result": results_by_tau[best_macro_tau],
    }
    out_path = OUT_DIR / f"{run_name}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2))
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
