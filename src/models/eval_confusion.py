from transformers import pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# ==== Your candidate labels ====
CANDIDATE_LABELS = [
    "advertisement",
    "irrelevant",
    "rant without visiting"
]
FALLBACK_LABEL = "genuine_review"
THRESHOLD = 0.05
HYPOTHESIS_TEMPLATE = "The review is {}."

MODEL_ID = "MoritzLaurer/deberta-v3-large-zeroshot-v2"

# ==== Example data ====
# Replace with your dataset
texts = [
    "Best pizza! Visit www.pizzapromo.com for discounts!",
    "I visited yesterday, the staff were friendly",
    "This app is terrible and I never used it"
]
true_labels = [
    "advertisement",
    "genuine_review",
    "rant without visiting"
]

# ==== Build pipeline ====
nli = pipeline(
    "zero-shot-classification",
    model=MODEL_ID,
    device_map="auto"
)

# ==== Predict labels ====
pred_labels = []
for txt in texts:
    out = nli("Review: " + txt,
              CANDIDATE_LABELS,
              hypothesis_template=HYPOTHESIS_TEMPLATE,
              multi_label=True)
    scores = dict(zip(out["labels"], out["scores"]))
    top_label = max(scores, key=scores.get)
    top_score = scores[top_label]
    final = top_label if top_score >= THRESHOLD else FALLBACK_LABEL
    pred_labels.append(final)

# ==== Evaluation ====
print("Classification report:\n")
print(classification_report(true_labels, pred_labels))

cm = confusion_matrix(true_labels, pred_labels, labels=CANDIDATE_LABELS+[FALLBACK_LABEL])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CANDIDATE_LABELS+[FALLBACK_LABEL])
disp.plot(cmap="Blues", xticks_rotation=45)
plt.show()
