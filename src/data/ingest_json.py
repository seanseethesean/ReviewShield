import os, re, json, glob
import pandas as pd

RAW_DIR = "data/raw"
OUT = "data/clean/clean.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

TEXT_CANDIDATES = ["review", "text", "content", "body", "comment", "reviewText"]

def norm(x: str) -> str:
    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def extract_text_from_obj(o):
    if not isinstance(o, dict):
        return None
    for k in TEXT_CANDIDATES:
        if k in o and isinstance(o[k], (str, int, float)):
            return norm(o[k])
    return None

rows = []

def handle_json_array(path):
    # For JSON arrays: load incrementally if possible; otherwise read and iterate
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)  # assumes it's a list of objects
    for o in data:
        t = extract_text_from_obj(o)
        if t:
            rows.append(t)

def handle_jsonl(path, max_lines=None):
    total, parsed, skipped = 0, 0, 0
    sample_path = "data/clean/skipped_sample.jsonl"
    # clear previous sample
    if os.path.exists(sample_path):
        os.remove(sample_path)

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            total += 1
            if max_lines and i > max_lines:
                break
            line = line.strip()
            if not line:
                skipped += 1
                continue
            try:
                o = json.loads(line)
            except Exception:
                skipped += 1
                # save first 20 bad lines
                if skipped <= 20:
                    with open(sample_path, "a", encoding="utf-8") as fout:
                        fout.write(line + "\n")
                continue

            t = extract_text_from_obj(o)
            if t:
                rows.append(t)
                parsed += 1
            else:
                skipped += 1
                if skipped <= 20:
                    with open(sample_path, "a", encoding="utf-8") as fout:
                        fout.write(line + "\n")

    print(f"[INGEST] {path}: total={total}, parsed={parsed}, skipped={skipped}")
    if os.path.exists(sample_path):
        print(f"[INGEST] wrote skipped sample -> {sample_path}")

json_paths = glob.glob(os.path.join(RAW_DIR, "*.json")) + glob.glob(os.path.join(RAW_DIR, "*.jsonl")) + glob.glob(os.path.join(RAW_DIR, "*.ndjson"))

for p in json_paths:
    print(f"[INGEST] Streaming JSONL -> {p}")
    handle_jsonl(p)  # force all through JSONL parser

print(f"[INGEST] collected raw rows: {len(rows)}")

df = pd.DataFrame({"text": rows})
df = df[df["text"].str.len() > 0].drop_duplicates()
df["label"] = "none"

df.to_csv(OUT, index=False)
print(f"[INGEST] wrote {len(df)} rows -> {OUT}")
