import pandas as pd, re, os
INP = "data/raw/reviews.csv"
OUT = "data/clean/clean.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

df = pd.read_csv(INP)  # expect a text column like 'review' or 'text'
text_col = "review" if "review" in df.columns else "text"
if text_col not in df.columns:
    raise ValueError("Expected a 'review' or 'text' column.")

# minimal clean
def norm(x):
    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

df["text"] = df[text_col].map(norm)
# If you have ground-truth labels already, ensure it's in 'label'. Otherwise set to 'none'.
if "label" not in df.columns:
    df["label"] = "none"

# trim to a manageable subset for speed (optional)
df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text"]).head(50000)
df = df[["text", "label"]]
df.to_csv(OUT, index=False)
print(f"wrote {len(df)} rows -> {OUT}")
