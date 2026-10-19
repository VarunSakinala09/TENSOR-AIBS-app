
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

characteristics = pd.read_csv(
    ROOT / "results/week5/signal_characteristics.csv"
)

predictions = pd.read_csv(
    ROOT / "results/week8/adaptive_group_cv"
    / "out_of_fold_predictions.csv"
)

data = characteristics.merge(
    predictions[
        ["record_id", "true_label", "adaptive_prediction"]
    ],
    on="record_id",
    how="inner",
    validate="one_to_one",
)

# Check the original signal-characteristics values.
data["flat_candidate"] = data["mean_std"].abs() < 1e-12

print("\nFLAT-CANDIDATE RECORDS BY TRUE LABEL")
print("=" * 60)

print(
    data.groupby("label")["flat_candidate"]
    .agg(["count", "sum"])
    .to_string()
)

print("\nFLAT-CANDIDATE RECORDS BY ADAPTIVE PREDICTION")
print("=" * 60)

print(
    data.groupby("adaptive_prediction")["flat_candidate"]
    .agg(["count", "sum"])
    .to_string()
)

print("\nFLAT-CANDIDATE RECORDS BY LABEL AND PREDICTION")
print("=" * 60)

print(
    data.groupby(["label", "adaptive_prediction"])
    ["flat_candidate"]
    .agg(["count", "sum"])
    .to_string()
)

print("\nAudit complete. No models retrained.")



# --------------------------------------------------
# NON-FLAT ECG PERFORMANCE ANALYSIS
# --------------------------------------------------

from sklearn.metrics import confusion_matrix

print("\n\nNON-FLAT ECG PERFORMANCE")
print("=" * 65)

# Load predictions separately to avoid merged-column name conflicts.
predictions = pd.read_csv(
    ROOT / "results/week8/adaptive_group_cv"
    / "out_of_fold_predictions.csv"
)

# Merge only the required prediction columns.
analysis = characteristics.merge(
    predictions[
        [
            "record_id",
            "true_label",
            "fixed40_prediction",
            "fixed50_prediction",
            "adaptive_prediction",
        ]
    ],
    on="record_id",
    how="inner",
    validate="one_to_one",
)

nonflat = analysis[
    analysis["mean_std"].abs() >= 1e-12
].copy()

print("Total records:", len(analysis))
print("Non-flat records:", len(nonflat))
print(
    "Non-flat acceptable:",
    (nonflat["true_label"].astype(int) == 0).sum()
)
print(
    "Non-flat unacceptable:",
    (nonflat["true_label"].astype(int) == 1).sum()
)

for name, column in [
    ("Fixed 40 Hz", "fixed40_prediction"),
    ("Fixed 50 Hz", "fixed50_prediction"),
    ("Adaptive", "adaptive_prediction"),
]:
    y = nonflat["true_label"].astype(int)
    pred = nonflat[column].astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0
    precision = tp / (tp + fp) if tp + fp else 0

    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity else 0
    )

    accuracy = (tp + tn) / len(nonflat)

    print(
        f"\n{name}\n"
        f"Accuracy:    {accuracy:.4f}\n"
        f"Sensitivity: {sensitivity:.4f}\n"
        f"Specificity: {specificity:.4f}\n"
        f"F1-score:    {f1:.4f}\n"
        f"TP={tp}, FN={fn}, FP={fp}, TN={tn}"
    )

print("\nNon-flat analysis complete. No models retrained.")
