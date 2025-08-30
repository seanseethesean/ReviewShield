import os, json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

DATA = "data/clean/clean_pseudo.csv"
MODEL_OUT = "artifacts/models/baseline.joblib"
PREDS_OUT = "artifacts/preds/baseline.csv"
METRICS_OUT = "artifacts/eval/baseline_metrics.json"
CM_OUT = "artifacts/eval/baseline_confusion.csv"
os.makedirs("artifacts/models", exist_ok=True)
os.makedirs("artifacts/preds", exist_ok=True)
os.makedirs("artifacts/eval", exist_ok=True)

df = pd.read_csv(DATA)
X = df["text"].astype(str).values
y = df["label"].astype(str).values

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_dev, X_test, y_dev, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

vec = TfidfVectorizer(max_features=40000, ngram_range=(1,2))
Xtr = vec.fit_transform(X_train)
Xde = vec.transform(X_dev)
Xte = vec.transform(X_test)

le = LabelEncoder()
ytr = le.fit_transform(y_train)
yde = le.transform(y_dev)
yte = le.transform(y_test)

clf = LogisticRegression(max_iter=200, n_jobs=None, class_weight="balanced")
clf.fit(Xtr, ytr)

# dev tuning: nothing fancy, just sanity-check performance
yhat_dev = clf.predict(Xde)

# test
yhat = clf.predict(Xte)
report = classification_report(yte, yhat, target_names=le.classes_, output_dict=True)
cm = confusion_matrix(yte, yhat)

# save artifacts
joblib.dump({"vec": vec, "clf": clf, "le": le}, MODEL_OUT)
pd.DataFrame({"text": X_test, "gold": le.inverse_transform(yte), "pred": le.inverse_transform(yhat)}).to_csv(PREDS_OUT, index=False)
pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(CM_OUT)
with open(METRICS_OUT, "w") as f: json.dump(report, f, indent=2)
print("Baseline done.")