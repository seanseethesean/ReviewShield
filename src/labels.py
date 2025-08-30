import os, json, time, pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PROMPT = open("src/prompts/policy.md").read().strip()

INP = "data/clean/clean.csv"
OUT = "data/clean/clean_pseudo_gpt4o.csv"

# choose size: start small, then scale if needed
N = 1000

df = pd.read_csv(INP).head(N).copy()
labels, reasons = [], []

BATCH = 50
for i in range(0, len(df), BATCH):
    batch = df["text"].iloc[i:i+BATCH].astype(str).tolist()
    user_prompt = "\n".join([f"INPUT: {t}\nOUTPUT:" for t in batch])

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    out_text = resp.choices[0].message.content

    # parse line-by-line JSON objects
    batch_labels, batch_reasons = [], []
    for line in out_text.splitlines():
        lab, rea = "none", ""
        if "{" in line and "}" in line:
            try:
                js = json.loads(line[line.find("{"): line.rfind("}")+1])
                lab = js.get("label", "none")
                rea = js.get("reason", "")
            except:
                pass
        if lab not in {"advertisement", "irrelevant", "no_visit_rant", "none"}:
            lab = "none"
        batch_labels.append(lab); batch_reasons.append(rea)

    # pad if model returned fewer lines than inputs
    while len(batch_labels) < len(batch):
        batch_labels.append("none"); batch_reasons.append("")

    labels.extend(batch_labels[:len(batch)])
    reasons.extend(batch_reasons[:len(batch)])

    print(f"[GPT4o] {i+len(batch)}/{len(df)} labeled")
    time.sleep(0.5)  # polite pacing, as before

df["label"] = labels[:len(df)]
df["reason"] = reasons[:len(df)]
df[["text","label","reason"]].to_csv(OUT, index=False)
print(f"[GPT4o] wrote {len(df)} rows -> {OUT}")
