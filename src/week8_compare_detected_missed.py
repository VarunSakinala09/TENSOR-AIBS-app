
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

characteristics_path = (
    ROOT / "results/week5/signal_characteristics.csv"
)
predictions_path = (
    ROOT / "results/week8/adaptive_group_cv"
    / "out_of_fold_predictions.csv"
)

characteristics = pd.read_csv(characteristics_path)
predictions = pd.read_csv(predictions_path)

# Keep only the columns needed for this analysis.
prediction_columns = [
    "record_id",
    "outer_fold",
    "true_label",
    "fixed40_prediction",
    "fixed50_prediction",
    "adaptive_prediction",
]

predictions = predictions[prediction_columns].copy()

# Confirm each record appears only once in each file.
if characteristics["record_id"].duplicated().any():
    raise ValueError("Duplicate record IDs in characteristics file.")

if predictions["record_id"].duplicated().any():
    raise ValueError("Duplicate record IDs in predictions file.")

data = characteristics.merge(
    predictions,
    on="record_id",
    how="inner",
    validate="one_to_one",
)

if len(data) != len(characteristics) or len(data) != len(predictions):
    raise ValueError("The two files do not contain matching record IDs.")

# Check that labels agree between the original characteristics and
# the labels used for out-of-fold evaluation.
label_map = {"Acceptable": 0, "Unacceptable": 1}
original_labels = data["label"].map(label_map)

if original_labels.isna().any():
    raise ValueError("Unexpected labels in characteristics file.")

if not (original_labels.astype(int) == data["true_label"].astype(int)).all():
    raise ValueError("Label mismatch between the two files.")

# Analyse unacceptable ECGs only.
bad = data[data["true_label"].astype(int) == 1].copy()

bad["adaptive_outcome"] = bad["adaptive_prediction"].astype(int).map(
    {1: "Detected", 0: "Missed"}
)

features = [
    "mean_baseline_variation",
    "max_baseline_variation",
    "mean_high_frequency_ratio",
    "max_high_frequency_ratio",
    "mean_powerline_ratio",
    "max_powerline_ratio",
    "mean_rms",
    "mean_std",
]

print("\nDATA CHECK")
print("=" * 65)
print("Matched records:", len(data))
print("Unacceptable records:", len(bad))
print("Unique signal groups are not recalculated in this script.")

print("\nADAPTIVE DETECTION COUNTS")
print("=" * 65)
print(bad["adaptive_outcome"].value_counts().to_string())

print("\nCHARACTERISTICS: DETECTED VS MISSED")
print("=" * 90)

summary = bad.groupby("adaptive_outcome")[features].agg(
    ["median", "mean"]
).T

print(summary.round(6).to_string())

print("\nFIXED MODEL DETECTION COUNTS")
print("=" * 65)

for method in [
    "fixed40_prediction",
    "fixed50_prediction",
    "adaptive_prediction",
]:
    detected = int((bad[method].astype(int) == 1).sum())
    missed = int((bad[method].astype(int) == 0).sum())
    print(f"{method}: detected={detected}, missed={missed}")

print("\nAnalysis complete. No models retrained.")
