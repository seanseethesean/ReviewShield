#!/usr/bin/env python3
# File: src/models/hf_batch_infer.py

import os
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# ===== Fixed config (edit if needed) =====
INPUT_CSV = "data/clean/clean.csv"
TEXT_COL = "text"
LIMIT = 500
BATCH_SIZE = 16
OUTPUT_CSV = "artifacts/predictions_hf.csv"
# ========================================

# Candidate labels (NO 'none' here)
CANDIDATE_LABELS = [
    "advertisement",
    "irrelevant",
    "rant without visiting"
]
FALLBACK_LABEL = "genuine_review"
THRESHOLD = 0.55
HYPOTHESIS_TEMPLATE = "The review is {}."

MODEL_IDS = [
    "MoritzLaurer/deberta-v3-large-zeroshot-v2",
    "facebook/bart-large-mnli"
]

def get_pipeline():
    last_err = None
    for mid in MODEL_IDS:
        try:
            return pipeline(
                "zero-shot-classification",
                model=mid,
                device_map="auto"
            )
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to load any model. Last error: {last_err}")

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    if TEXT_COL not in df.columns:
        raise ValueError(f"Column '{TEXT_COL}' not found in {INPUT_CSV}. Columns: {list(df.columns)}")

    df = df.head(LIMIT).copy()
    texts = df[TEXT_COL].astype(str).fillna("").tolist()

    nli = get_pipeline()

    final_labels, top_labels, top_scores, rankings = [], [], [], []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Infer"):
        chunk = ["Review: " + t for t in texts[i:i+BATCH_SIZE]]  # small calibration prefix
        out = nli(
            chunk,
            CANDIDATE_LABELS,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=True
        )
        # Normalize list shape
        if isinstance(out, dict):
            out = [out]

        for o in out:
            scores = dict(zip(o["labels"], o["scores"]))
            top_label = max(scores, key=scores.get)
            top_score = scores[top_label]
            final = top_label if top_score >= THRESHOLD else FALLBACK_LABEL

            final_labels.append(final)
            top_labels.append(top_label)
            top_scores.append(top_score)
            rankings.append(";".join([f"{l}:{s:.3f}" for l, s in scores.items()]))

    df["pred_label_hf"] = final_labels          # calibrated final label
    df["pred_top_raw"] = top_labels             # raw top label from model
    df["pred_top_score"] = top_scores           # raw top score
    df["label_ranking_hf"] = rankings

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} predictions to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
