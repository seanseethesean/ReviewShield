import os, pandas as pd
from transformers import pipeline

INP = "data/clean/clean.csv"             # uses your 265,389 rows
OUT = "data/clean/clean_pseudo.csv"
LABELS = ["advertisement","irrelevant","no_visit_rant","none"]
TEMPLATE = "This review is an example of {}."
BATCH = 256                               # lower if RAM is tight
THRESH = 0.55                             # raise for fewer false positives

print("[ZSL] loading data…")
df = pd.read_csv(INP)
# keep only real strings
df = df.copy()
df["text"] = df["text"].astype(str)          # coerce to str (NaN -> "nan")
df = df[df["text"].str.strip().str.lower() != "nan"]  # drop former NaNs
df = df[df["text"].str.strip().str.len() > 0]         # drop empty
df.reset_index(drop=True, inplace=True)

# faster CPU model (good quality): distilbert-mnli
# model_id = "typeform/distilbert-base-uncased-mnli"
# higher-quality but slower/larger: bart-large-mnli
model_id = "typeform/distilbert-base-uncased-mnli"

print(f"[ZSL] loading model: {model_id}")
clf = pipeline("zero-shot-classification", model=model_id, device_map="auto", truncation=True)

preds = []
n = len(df)
for i in range(0, n, BATCH):
    raw_chunk = df["text"].iloc[i:i + BATCH]
    chunk = [t.strip() for t in raw_chunk if isinstance(t, str) and len(t.strip()) > 0]
    if not chunk:
        print(f"[ZSL] skipped empty batch at rows {i}-{i + BATCH}")
        continue
    try:
        out = clf(chunk, LABELS, hypothesis_template=TEMPLATE, multi_label=False)
    except Exception as e:
        print(f"[ZSL] batch {i}-{i + BATCH} failed: {e}")
        continue
    if isinstance(out, dict): out = [out]
    for o in out:
        lab, score = o["labels"][0], float(o["scores"][0])
        preds.append(lab if score >= THRESH else "none")
    print(f"[ZSL] {min(i+BATCH, n)}/{n}")

res = df.copy()
res["label"] = preds
os.makedirs(os.path.dirname(OUT), exist_ok=True)
res.to_csv(OUT, index=False)
print(f"[ZSL] wrote {len(res)} rows -> {OUT}")
