# src/models/predict_one.py
import os, re, sys, pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

try:
    from joblib import load
except Exception:
    load = None  # fallback if joblib not installed

DATA_CSV = "data/clean/clean_pseudo_gpt4o.csv"
MODEL_PATH = "artifacts/baseline.joblib"

def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

def rule_based_label(text: str):
    t = text.lower()
    # obvious ad signals
    if re.search(r'(https?://|www\.)', t):
        return "advertisement"
    if re.search(r'\bpromo\b|\bdiscount\b|\buse code\b|\bvoucher\b', t):
        return "advertisement"
    # obvious no-visit rant signals
    if re.search(r"\bnever (been|visited)\b|\bdidn['’]?t even (go|visit)\b|\bnot even (go|visit)ed?\b", t):
        return "no_visit_rant"
    return None

def load_or_train_pipeline():
    # Try to load a saved model first (if you saved it in baseline.py)
    if load and os.path.exists(MODEL_PATH):
        try:
            return load(MODEL_PATH)
        except Exception:
            pass  # fall through to train

    # Train quickly on the labeled CSV as fallback
    df = pd.read_csv(DATA_CSV)
    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    pipe = build_pipeline()
    pipe.fit(X, y)
    return pipe

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/models/predict_one.py 'your review text here'")
        sys.exit(1)

    # Join all CLI args into the review text (handles missing quotes or ! issues if user escapes)
    text = " ".join(sys.argv[1:]).strip()

    # 1) Rule-based fast path
    rb = rule_based_label(text)
    if rb:
        print(rb)
        return

    # 2) ML model path
    pipe = load_or_train_pipeline()
    pred = pipe.predict([text])[0]
    print(pred)

if __name__ == "__main__":
    main()
