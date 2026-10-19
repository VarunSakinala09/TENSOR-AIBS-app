
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PREDICTION_FILE = (
    ROOT / "results/week8/frequency_comparison"
    / "out_of_fold_predictions.csv"
)

GROUP_FILE = (
    ROOT / "results/week4/group_aware_cv"
    / "record_signal_groups.csv"
)

OUTPUT_DIR = ROOT / "results/week8/frequency_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pred = pd.read_csv(
    PREDICTION_FILE,
    dtype={"record_id": str}
)

groups = pd.read_csv(
    GROUP_FILE,
    dtype={"record_id": str}
)

required = {
    "configuration", "fold", "record_id",
    "true_label", "predicted_label"
}

if not required.issubset(pred.columns):
    raise ValueError(f"Prediction file must contain: {required}")

# Verify one out-of-fold prediction per record per configuration.
counts = pred.groupby(
    ["configuration", "record_id"]
).size()

print("PREDICTION AUDIT")
print("Rows:", len(pred))
print("Configurations:", sorted(pred["configuration"].unique()))
print("Records per configuration:")
print(pred.groupby("configuration")["record_id"].nunique())

if not (counts == 1).all():
    raise ValueError(
        "Some records have missing or repeated predictions per configuration"
    )

# Convert each configuration into one row per record.
wide = pred.pivot(
    index="record_id",
    columns="configuration",
    values="predicted_label"
).reset_index()

truth = pred[
    ["record_id", "true_label"]
].drop_duplicates()

if truth["record_id"].duplicated().any():
    raise ValueError("Conflicting true labels found")

wide = wide.merge(
    truth, on="record_id", how="inner", validate="one_to_one"
)

if not {"40Hz", "50Hz"}.issubset(wide.columns):
    raise ValueError("Both 40Hz and 50Hz predictions are required")

wide["correct_40"] = wide["40Hz"] == wide["true_label"]
wide["correct_50"] = wide["50Hz"] == wide["true_label"]

# Determine which configuration correctly handles each record.
wide["comparison"] = "Both correct"
wide.loc[
    ~wide["correct_40"] & ~wide["correct_50"], "comparison"
] = "Both wrong"
wide.loc[
    wide["correct_40"] & ~wide["correct_50"], "comparison"
] = "40Hz only correct"
wide.loc[
    ~wide["correct_40"] & wide["correct_50"], "comparison"
] = "50Hz only correct"

print("\nOVERALL ERROR COMPARISON")
print(wide["comparison"].value_counts().to_string())

for label in ["Unacceptable", "Acceptable"]:
    subset = wide[wide["true_label"] == label]

    print(f"\nTRUE CLASS: {label}")
    print("Records:", len(subset))
    print(subset["comparison"].value_counts().to_string())

    print(
        "40 Hz correct:",
        f"{subset['correct_40'].mean():.4f}"
    )
    print(
        "50 Hz correct:",
        f"{subset['correct_50'].mean():.4f}"
    )

# Attach signal groups for further investigation.
wide = wide.merge(
    groups[["record_id", "signal_hash"]],
    on="record_id",
    how="left",
    validate="one_to_one"
)

if wide["signal_hash"].isna().any():
    raise ValueError("Some predictions have no signal-group mapping")

# Save every record where the two configurations disagree.
disagreements = wide[wide["40Hz"] != wide["50Hz"]].copy()

disagreements.to_csv(
    OUTPUT_DIR / "frequency_disagreement_records.csv",
    index=False
)

wide.to_csv(
    OUTPUT_DIR / "frequency_error_analysis.csv",
    index=False
)

print("\nRecords where predictions differ:", len(disagreements))
print(
    "Saved:",
    OUTPUT_DIR / "frequency_error_analysis.csv"
)
print(
    "Saved:",
    OUTPUT_DIR / "frequency_disagreement_records.csv"
)
