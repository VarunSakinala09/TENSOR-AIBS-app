
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]

FILE = (
    ROOT / "results/week8/adaptive_group_cv"
    / "out_of_fold_predictions.csv"
)

df = pd.read_csv(FILE)

print("\nMETRICS WITHIN EACH SELECTED CONFIGURATION")
print("=" * 65)

for config, group in df.groupby("selected_configuration"):
    y = group["true_label"].astype(int)

    print(f"\n{config}")
    print("-" * 45)

    for method, col in [
        ("Fixed 40 Hz", "fixed40_prediction"),
        ("Fixed 50 Hz", "fixed50_prediction"),
        ("Adaptive", "adaptive_prediction"),
    ]:
        pred = group[col].astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y, pred, labels=[0, 1]
        ).ravel()

        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        accuracy = (tp + tn) / len(group)

        print(
            f"{method:12s} | "
            f"Accuracy={accuracy:.4f} | "
            f"Sensitivity={sensitivity:.4f} | "
            f"Specificity={specificity:.4f} | "
            f"TP={tp}, FN={fn}, FP={fp}, TN={tn}"
        )

print("\nAnalysis complete. No models retrained.")
