#!/usr/bin/env python3
# File: src/models/hf_predict_one.py

from transformers import pipeline

# ===== Try your text here =====
TEST_TEXT = "I visited yesterday, the staff were friendly"
# ==============================

# Candidate labels (NO 'none' here)
CANDIDATE_LABELS = [
    "advertisement",
    "irrelevant",
    "rant without visiting"
]
# Fallback if nothing is confident
FALLBACK_LABEL = "genuine_review"
THRESHOLD = 0.05  # raise to be stricter, lower to be looser

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
    nli = get_pipeline()
    out = nli(
        "Review: " + TEST_TEXT,            # small calibration prefix helps
        CANDIDATE_LABELS,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=True                   # independent scores per label
    )
    # Build score dict
    scores = dict(zip(out["labels"], out["scores"]))
    # Choose best label above threshold, else fallback
    top_label = max(scores, key=scores.get)
    top_score = scores[top_label]
    final_label = top_label if top_score >= THRESHOLD else FALLBACK_LABEL

    print(f"Text: {TEST_TEXT}")
    print(f"Final label: {final_label} (top={top_label} {top_score:.3f})")
    for lbl, sc in scores.items():
        print(f"{lbl}\t{sc:.3f}")

if __name__ == "__main__":
    main()
