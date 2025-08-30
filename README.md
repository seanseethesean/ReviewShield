# Review Moderation (Hackathon Project)

## Project Overview
This project builds a moderation pipeline for Google-style reviews.
Goal: automatically flag reviews that violate policy categories:
- **advertisement**: promotional content, referral links, sales pitches.
- **irrelevant**: unrelated to the business (random stories, off-topic text).
- **no_visit_rant**: rants/complaints where reviewer admits no visit.
- **none**: all other legitimate, on-topic reviews.

We compare multiple approaches:
1. **Baseline** (TF-IDF + Logistic Regression)
2. **Zero-Shot Classification** (Hugging Face NLI models)
3. **Few-Shot Prompting** (LLMs guided by `src/prompts/policy.md`)

The pipeline produces labeled datasets, model predictions, evaluation metrics, and prompt artifacts for analysis.

---

## Setup Instructions

### 1. Clone the repo
git clone <your-repo-url>
cd review-moderation-ml

### 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

### 3. Install dependencies
pip install -U pip
pip install pandas scikit-learn transformers torch accelerate datasets evaluate matplotlib rich joblib python-dotenv jupyter

---

## How to reproduce results

### 1. Data Preparation
Place raw Google Reviews dataset into data/raw/
For JSON data:
python src/data/ingest_json.py

### 2. Zero-Shot Labeling (pseudo-labels)
python src/models/zero_shot_pseudo.py

### 3. Baseline Model (TF-IDF + Logistic Regression)
python src/models/baseline.py

### 4. Evaluation
python src/eval/metrics.py

### 5. Prompt Engineering
Instructions and few-shot examples are in src/prompts/policy.md.
Optionally, run a few-shot demo with a small instruction-tuned LLM.

--- 

## Team Contributions

- **Sean See** — Data ingestion, JSON/CSV cleaning pipeline, repository setup.  
- **Cyril Pedrina** — Zero-shot classification pipeline (Hugging Face), pseudo-label generation.  
- **Jade Ng** — Baseline model (TF-IDF + Logistic Regression), training, and evaluation metrics.  
- **Eleanor Cheak** — Prompt engineering (`policy.md`), few-shot examples, and LLM experimentation.  
- **Zelda Seow** — Documentation, and final presentation delivery.
