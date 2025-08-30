# src/eval/visualise.py
import os, json
import pandas as pd

# Use a non-interactive backend to avoid macOS GUI issues
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS_PATH = "artifacts/metrics.json"
OUT_DIR = "artifacts/eval"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(f"Missing {METRICS_PATH}. Run baseline.py first.")

    m = json.load(open(METRICS_PATH))
    labels = m["labels"]
    cm = np.array(m["confusion_matrix"])
    report = pd.DataFrame(m["classification_report"]).T

    # 1) Save a tidy CSV of precision/recall/F1/support
    cols = ["precision", "recall", "f1-score", "support"]
    report_out = report[cols].round(4)
    report_out.to_csv(f"{OUT_DIR}/baseline_report.csv")
    print(f"Wrote {OUT_DIR}/baseline_report.csv")

    # 2) Confusion matrix heatmap (matplotlib only)
    fig, ax = plt.subplots(figsize=(4.5, 3.8), dpi=150)
    im = ax.imshow(cm, aspect="auto")
    ax.set_title("Confusion Matrix — Baseline")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    cm_path = f"{OUT_DIR}/baseline_confusion.png"
    plt.savefig(cm_path)
    plt.close(fig)
    print(f"Wrote {cm_path}")

    # 3) Per-class F1 bar chart (excluding summary rows)
    per_class = report_out.loc[
        [k for k in report_out.index if k not in ("accuracy", "macro avg", "weighted avg")]
    ]
    fig2, ax2 = plt.subplots(figsize=(5.0, 3.2), dpi=150)
    ax2.bar(per_class.index, per_class["f1-score"].values)
    ax2.set_title("Per-class F1 — Baseline")
    ax2.set_ylabel("F1")
    ax2.set_ylim(0, 1.0)
    for tick in ax2.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")
    plt.tight_layout()
    f1_path = f"{OUT_DIR}/baseline_f1.png"
    plt.savefig(f1_path)
    plt.close(fig2)
    print(f"Wrote {f1_path}")

if __name__ == "__main__":
    main()
