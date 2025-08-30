# src/models/baseline.py
import os
import json
from pathlib import Path
from joblib import dump

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ---------------------------
# Config
# ---------------------------
INPUT_CSV = os.getenv("INPUT_CSV", "data/clean/clean_pseudo_gpt4o.csv")
TEXT_COL = os.getenv("TEXT_COL", "text")
LABEL_COL = os.getenv("LABEL_COL", "label")
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
PRED_OUT = ARTIFACTS_DIR / "predictions.csv"
METRICS_OUT = ARTIFACTS_DIR / "metrics.json"

TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# Only allow these canonical labels
VALID_LABELS = {"advertisement", "irrelevant", "no_visit_rant", "none"}


def load_data(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Expected columns '{text_col}' and '{label_col}' in {path}")
    # Make sure labels are in the allowed set
    df[label_col] = df[label_col].astype(str).where(df[label_col].isin(VALID_LABELS), other="none")
    # Clean text minimalistically
    df[text_col] = df[text_col].astype(str).fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    # Drop empty text rows
    df = df[df[text_col].str.len() > 0].copy()
    return df


def collapse_tiny_classes(y: pd.Series, test_size: float) -> pd.Series:
    """
    Keep any class with >=2 samples (enough for stratify).
    Only collapse classes with <2 samples into 'none'.
    """
    counts = y.value_counts()
    tiny = set(counts[counts < 2].index)
    if tiny:
        y = y.where(~y.isin(tiny), other="none")
    return y

def ensure_stratifiable(y: pd.Series) -> pd.Series:
    """
    Final guard: if any class still has <2 after collapsing, collapse again.
    """
    counts = y.value_counts()
    tiny = set(counts[counts < 2].index)
    if tiny:
        y = y.where(~y.isin(tiny), other="none")
    return y


def build_pipeline() -> Pipeline:
    """
    A very basic baseline: TF-IDF + Logistic Regression.
    class_weight='balanced' helps a bit with imbalance.
    """
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            n_jobs=None,
            random_state=RANDOM_STATE,
            solver="lbfgs",
            multi_class="auto",
        )),
    ])
    return pipe

# Downsample majority "none" so the classifier actually sees minorities
def rebalance_for_training(texts, labels, max_ratio=5, seed=42):
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({"text": texts, "label": labels})
    maj = df[df.label == "none"]
    mins = df[df.label != "none"]
    if mins.empty:
        return texts, labels  # nothing to balance
    cap = min(len(maj), max_ratio * len(mins))
    maj_ds = maj.sample(n=cap, random_state=seed) if len(maj) > cap else maj
    out = pd.concat([maj_ds, mins], ignore_index=True).sample(frac=1, random_state=seed)
    return out["text"].values, out["label"].values

def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load
    df = load_data(INPUT_CSV, TEXT_COL, LABEL_COL)
    print(f"[baseline] Loaded {len(df)} rows from {INPUT_CSV}")

    # 2) X, y
    X = df[TEXT_COL].values
    y = df[LABEL_COL].astype(str)

    # 3) Guard for stratified split
    #    Collapse tiny classes into "none" so stratify won't crash.
    y = collapse_tiny_classes(pd.Series(y), TEST_SIZE)
    y = ensure_stratifiable(y)
    y = y.values

    # 4) Split (now safe to stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("[baseline] Split sizes:",
          f"train={len(X_train)}, test={len(X_test)}")
    print("[baseline] Train label counts:\n", pd.Series(y_train).value_counts())
    print("[baseline] Test  label counts:\n", pd.Series(y_test).value_counts())

    # 5) Train
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # 6) Evaluate
    y_pred = pipe.predict(X_test)
    labels_sorted = sorted(list(set(y_test) | set(y_pred)))
    report = classification_report(y_test, y_pred, labels=labels_sorted, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    micro_f1 = f1_score(y_test, y_pred, average="micro")
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    # 7) Save predictions
    pred_df = pd.DataFrame({
        "text": X_test,
        "y_true": y_test,
        "y_pred": y_pred,
    })
    pred_df.to_csv(PRED_OUT, index=False)

    # 8) Save metrics
    metrics = {
        "labels": labels_sorted,
        "classification_report": report,  # dict form
        "confusion_matrix": cm.tolist(),
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    # 9) Console summary
    print("\n[baseline] ===== Metrics (summary) =====")
    print(f"micro_f1   : {micro_f1:.4f}")
    print(f"macro_f1   : {macro_f1:.4f}")
    print(f"weighted_f1: {weighted_f1:.4f}")
    print("\nPer-class precision/recall/F1:")
    for lab in labels_sorted:
        pr = report.get(lab, {})
        p = pr.get("precision", 0.0)
        r = pr.get("recall", 0.0)
        f1 = pr.get("f1-score", 0.0)
        s = pr.get("support", 0)
        print(f"  {lab:15s} p={p:.3f} r={r:.3f} f1={f1:.3f} n={s}")

    print(f"\n[baseline] Wrote predictions -> {PRED_OUT}")
    print(f"[baseline] Wrote metrics     -> {METRICS_OUT}")
    # === Finalize: train on ALL data (balanced) and save model for inference ===
    X_all = df[TEXT_COL].astype(str).values
    y_all = df[LABEL_COL].astype(str).values

    # Rebalance to avoid "always none"
    X_all_bal, y_all_bal = rebalance_for_training(X_all, y_all, max_ratio=5, seed=RANDOM_STATE)

    final_pipe = build_pipeline()
    final_pipe.fit(X_all_bal, y_all_bal)

    model_path = ARTIFACTS_DIR / "baseline.joblib"
    dump(final_pipe, model_path)
    print(f"[baseline] Saved trained pipeline -> {model_path}")

if __name__ == "__main__":
    main()
