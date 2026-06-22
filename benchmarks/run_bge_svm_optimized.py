#!/usr/bin/env python3
"""Run 7 with BGE Embeddings for SVM (Optimized + Batched).

Trains a Calibrated SVM using BGE embeddings of '{tag} {scale}' features on the
agent-enriched training splits, then runs the cascading pipeline (Run 7) with a
sweep over SVM_VOTE_THRESH and Stage-3 tie-break thresholds.

This script uses batch encoding to complete the entire sweep in seconds.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report

# Add parent directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.types import LLM_OUTPUT_CATEGORIES
from benchmarks.pipeline_3tier import rule_classify
from benchmarks.pipeline_3tier_v2 import LLM_MODEL, enrich_gold_with_agent_meta, llm_classify, load_gold
from benchmarks.stage3_domain import _load_index, _load_model, scale_heuristic
from benchmarks.per_tag_svm import vote_per_tag

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits" / "agent_enriched"

def load_train_data() -> pd.DataFrame:
    group_a = pd.read_parquet(SPLITS_DIR / "group_a.parquet")
    group_b = pd.read_parquet(SPLITS_DIR / "group_b.parquet")
    df = pd.concat([group_a, group_b], ignore_index=True)
    # Exclude junk for Run 7 training
    df = df[df["label"] != "junk"].reset_index(drop=True)
    return df

def main():
    print("Loading datasets...")
    df_train = load_train_data()
    gold_path = DATA_DIR / "labelled" / "pure_others_to_label.csv"
    gold = load_gold(gold_path)
    gold = enrich_gold_with_agent_meta(gold)
    
    # Exclude self-feedback as per standard protocol
    gold = gold[~gold["is_self"]].reset_index(drop=True)
    print(f"Loaded train set (quality-vs-quantity): {len(df_train)} rows")
    print(f"Loaded gold set (excluding self): {len(gold)} rows")
    
    # Load BGE model
    print("Loading SentenceTransformer BGE model...")
    model = _load_model()
    
    # Collect all unique text strings to encode
    unique_texts = set()
    unique_tags = set()
    
    # Train texts
    train_rows = []
    for _, r in df_train.iterrows():
        binary = 1 if r["label"] == "quality" else 0
        scale = str(r.get("value_scale") or "").strip().lower()
        t1 = str(r.get("tag1") or "").strip().lower()
        t2 = str(r.get("tag2") or "").strip().lower()
        
        row_text_1, row_text_2 = None, None
        if t1:
            row_text_1 = f"{t1} {scale}"
            unique_texts.add(row_text_1)
        if t2:
            row_text_2 = f"{t2} {scale}"
            unique_texts.add(row_text_2)
            
        train_rows.append({
            "t1_text": row_text_1,
            "t2_text": row_text_2,
            "label_binary": binary
        })
        
    # Gold texts
    for _, r in gold.iterrows():
        tag1 = str(r.get("tag1", "") or "").strip().lower()
        tag2 = str(r.get("tag2", "") or "").strip().lower()
        scale = str(r.get("value_scale", "") or "").strip().lower()
        
        if tag1:
            unique_texts.add(f"{tag1} {scale}")
            unique_tags.add(tag1)
        if tag2:
            unique_texts.add(f"{tag2} {scale}")
            unique_tags.add(tag2)
            
    all_to_encode = list(unique_texts) + list(unique_tags)
    print(f"Batch encoding {len(all_to_encode)} unique texts/tags...")
    t_start = time.time()
    embeddings = model.encode(all_to_encode, normalize_embeddings=True, show_progress_bar=True)
    encoding_cache = {text: vec for text, vec in zip(all_to_encode, embeddings)}
    print(f"Encoding completed in {time.time() - t_start:.2f}s")
    
    # Build training features for SVM
    X_train = []
    y_train = []
    for row in train_rows:
        if row["t1_text"]:
            X_train.append(encoding_cache[row["t1_text"]])
            y_train.append(row["label_binary"])
        if row["t2_text"]:
            X_train.append(encoding_cache[row["t2_text"]])
            y_train.append(row["label_binary"])
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"Training SVM on {len(X_train)} single-tag samples...")
    clf = CalibratedClassifierCV(LinearSVC(C=0.3, max_iter=2000), cv=3, method="sigmoid")
    clf.fit(X_train, y_train)
    print("SVM training completed.")
    
    # Setup Stage 3 domain index
    index, key_to_pos = _load_index()
    
    # Predict function using cache
    def get_quality_prob(tag: str, scale: str) -> float:
        text = f"{tag.strip().lower()} {scale.strip().lower()}"
        vec = encoding_cache[text]
        proba = clf.predict_proba([vec])[0]
        # classes_[1] == 1 (quality)
        quality_idx = list(clf.classes_).index(1)
        return float(proba[quality_idx])
        
    def get_tag_cos(tag: str, agent_vec: np.ndarray) -> float:
        tag_vec = encoding_cache[tag.strip().lower()]
        return float(np.dot(tag_vec, agent_vec))
        
    # Evaluate gold set
    # Pre-calculate intermediate values for each row in gold to speed up sweep
    gold_eval_data = []
    for idx, row in gold.iterrows():
        tag1 = str(row.get("tag1", "") or "").strip()
        tag2 = str(row.get("tag2", "") or "").strip()
        scale = str(row.get("value_scale", "") or "").strip()
        agent_key = str(row.get("agent_key", "") or "")
        has_meta = bool(row.get("has_agent_metadata"))
        decimals = int(row.get("value_decimals") or 0)
        
        # Stage 0.5: empty-tag rule
        if not tag1 and not tag2:
            preds_fallback = "junk" if scale.lower() == "unbounded" else "quality"
            gold_eval_data.append({"stage": "empty_tag", "pred": preds_fallback, "row": row})
            continue
            
        # Stage 1: rule classify
        cat = rule_classify(row)
        if cat:
            gold_eval_data.append({"stage": "rule", "pred": cat, "row": row})
            continue
            
        # SVM probabilities
        p1 = get_quality_prob(tag1, scale) if tag1 else 0.5
        p2 = get_quality_prob(tag2, scale) if tag2 else 0.5
        t2_empty = not tag2
        
        # Domain cosine
        best_cos = None
        has_dom = False
        if has_meta:
            pos = key_to_pos.get(agent_key)
            if pos is not None:
                agent_vec = index.reconstruct(pos)
                tags = [t for t in (tag1, tag2) if t]
                if tags:
                    best_cos = max([get_tag_cos(t, agent_vec) for t in tags])
                    has_dom = True
                    
        gold_eval_data.append({
            "stage": "cascade",
            "tag1": tag1,
            "tag2": tag2,
            "scale": scale,
            "has_meta": has_meta,
            "decimals": decimals,
            "best_cos": best_cos,
            "has_dom": has_dom,
            "p1": p1,
            "p2": p2,
            "t2_empty": t2_empty,
            "row": row
        })
        
    y_true = gold["label"].tolist()
    
    # We sweep SVM_VOTE_THRESH and tb_thresh
    results = []
    print("\nSweeping threshold combinations...")
    print(f"{'SVM_THRESH':10} | {'TB_THRESH':10} | {'Macro F1':10} | {'F1 Ex Junk':10} | {'Quality Rec':11} | {'Qty Rec':9} | {'LLM Calls':9}")
    print("-" * 80)
    
    for svm_t in [0.60, 0.65, 0.70, 0.75, 0.80]:
        for tb_t in [0.20, 0.30, 0.40, 0.50, None]:
            preds = []
            llm_calls = 0
            for item in gold_eval_data:
                if item["stage"] == "empty_tag":
                    preds.append(item["pred"])
                    continue
                if item["stage"] == "rule":
                    preds.append(item["pred"])
                    continue
                    
                # SVM Vote
                vote = vote_per_tag(item["p1"], item["p2"], t2_empty=item["t2_empty"], thresh=svm_t)
                if vote == "quality":
                    preds.append("quality")
                    continue
                    
                # Stage 3
                if not item["has_meta"]:
                    preds.append(llm_classify(item["row"], LLM_MODEL))
                    llm_calls += 1
                    continue
                    
                if item["has_dom"] and item["best_cos"] is not None and item["best_cos"] > 0.55:
                    if item["scale"].lower() == "unbounded":
                        preds.append("quantity")
                    else:
                        if tb_t is None:
                            # Run 5 default: bounded -> quality
                            preds.append("quality")
                        else:
                            quality_lean = max(item["p1"], item["p2"]) if not item["t2_empty"] else item["p1"]
                            preds.append("quality" if quality_lean >= tb_t else "quantity")
                    continue
                    
                # Fallback to LLM
                preds.append(llm_classify(item["row"], LLM_MODEL))
                llm_calls += 1
                
            # Compute F1 scores
            f1_all = f1_score(y_true, preds, labels=LLM_OUTPUT_CATEGORIES, average="macro", zero_division=0)
            
            # Sub-score excluding 7 junk records
            gold_clean_indices = [i for i, y in enumerate(y_true) if y != "junk"]
            y_true_clean = [y_true[i] for i in gold_clean_indices]
            preds_clean = [preds[i] for i in gold_clean_indices]
            f1_clean = f1_score(y_true_clean, preds_clean, labels=["quality", "quantity"], average="macro", zero_division=0)
            
            # Recall for quality and quantity (on clean data for comparison)
            rep = classification_report(y_true_clean, preds_clean, labels=["quality", "quantity"], output_dict=True, zero_division=0)
            qual_rec = rep["quality"]["recall"]
            qty_rec = rep["quantity"]["recall"]
            
            tb_str = str(tb_t) if tb_t is not None else "None (R5)"
            print(f"{svm_t:<10.2f} | {tb_str:<10} | {f1_all:<10.4f} | {f1_clean:<10.4f} | {qual_rec:<11.4f} | {qty_rec:<9.4f} | {llm_calls:<9}")
            
            results.append({
                "svm_thresh": svm_t,
                "tb_thresh": tb_t,
                "f1_all": f1_all,
                "f1_clean": f1_clean,
                "qual_rec": qual_rec,
                "qty_rec": qty_rec,
                "llm_calls": llm_calls,
                "preds": preds
            })
            
    # Find best configuration by F1 Ex Junk
    best_res = max(results, key=lambda x: x["f1_clean"])
    print("\n" + "=" * 60)
    print(f"BEST CONFIGURATION (by F1 Ex Junk):")
    print(f"  SVM Thresh: {best_res['svm_thresh']}")
    print(f"  Tie-break Thresh: {best_res['tb_thresh'] if best_res['tb_thresh'] is not None else 'None (R5)'}")
    print(f"  Macro F1 (all 3 classes): {best_res['f1_all']:.4f}")
    print(f"  Macro F1 (ex junk):       {best_res['f1_clean']:.4f}")
    print(f"  Quality Recall:           {best_res['qual_rec']:.4f}")
    print(f"  Quantity Recall:          {best_res['qty_rec']:.4f}")
    print(f"  LLM Calls:                {best_res['llm_calls']} ({best_res['llm_calls']/len(gold)*100:.1f}%)")
    print("=" * 60)
    
    # Detailed report for the best config
    print("\nClassification Report for Best Config (All 3 classes):")
    print(classification_report(y_true, best_res["preds"], labels=LLM_OUTPUT_CATEGORIES, zero_division=0))
    
    # Let's save a summary json file
    output_summary = {
        "best_svm_thresh": best_res["svm_thresh"],
        "best_tb_thresh": best_res["tb_thresh"],
        "f1_all": best_res["f1_all"],
        "f1_clean": best_res["f1_clean"],
        "qual_rec": best_res["qual_rec"],
        "qty_rec": best_res["qty_rec"],
        "llm_calls": best_res["llm_calls"],
        "all_results": [
            {
                "svm_thresh": r["svm_thresh"],
                "tb_thresh": r["tb_thresh"],
                "f1_all": r["f1_all"],
                "f1_clean": r["f1_clean"],
                "qual_rec": r["qual_rec"],
                "qty_rec": r["qty_rec"],
                "llm_calls": r["llm_calls"]
            }
            for r in results
        ]
    }
    
    out_path = DATA_DIR / "benchmark_results" / "bge_svm_sweep_summary.json"
    out_path.write_text(json.dumps(output_summary, indent=2))
    print(f"Summary written to {out_path}")

if __name__ == "__main__":
    main()
