
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DISAGREEMENT_FILE = (
    ROOT / "results/week8/frequency_comparison"
    / "frequency_disagreement_records.csv"
)

CHARACTERISTICS_FILE = (
    ROOT / "results/week5/signal_characteristics.csv"
)

OUTPUT_DIR = ROOT / "results/week8/frequency_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

disagreements = pd.read_csv(
    DISAGREEMENT_FILE, dtype={"record_id": str}
)

characteristics = pd.read_csv(
    CHARACTERISTICS_FILE, dtype={"record_id": str}
)

if "record_id" not in disagreements.columns:
    raise ValueError("Disagreement file has no record_id column")

if "record_id" not in characteristics.columns:
    raise ValueError("Characteristics file has no record_id column")

if characteristics["record_id"].duplicated().any():
    raise ValueError(
        "Signal-characteristics file has duplicate record IDs. "
        "Resolve them before merging."
    )

# Join only the records where the two frequency models disagree.
merged = disagreements.merge(
    characteristics,
    on="record_id",
    how="left",
    suffixes=("", "_characteristic"),
    validate="one_to_one",
    indicator=True,
)

matched = merged["_merge"] == "both"

print("DISAGREEMENT CHARACTERISTICS ANALYSIS")
print("Disagreement records:", len(disagreements))
print("Matched characteristics:", int(matched.sum()))
print("Missing characteristics:", int((~matched).sum()))

if not matched.all():
    print("\nIDs without characteristics:")
    print(
        merged.loc[~matched, "record_id"].head(20).to_list()
    )

merged = merged[matched].drop(columns="_merge")

# Assign a simple explanation group to each disagreement.
def classify_row(row):
    true_label = row["true_label"]
    pred40 = row["40Hz"]
    pred50 = row["50Hz"]

    correct40 = pred40 == true_label
    correct50 = pred50 == true_label

    if correct40 and not correct50:
        return "40Hz_only_correct"
    if correct50 and not correct40:
        return "50Hz_only_correct"
    if not correct40 and not correct50:
        return "both_wrong"
    return "both_correct"


merged["error_group"] = merged.apply(classify_row, axis=1)

print("\nDisagreement groups:")
print(merged["error_group"].value_counts().to_string())

# Summarize existing numeric signal characteristics.
excluded = {
    "fold",
    "record_id",
}

numeric_columns = [
    c for c in characteristics.select_dtypes(
        include=np.number
    ).columns
    if c not in excluded
]

# Use columns from the merged table, avoiding duplicate ID fields.
numeric_columns = [
    c for c in numeric_columns if c in merged.columns
]

if numeric_columns:
    summary = merged.groupby("error_group")[numeric_columns].agg(
        ["count", "median", "mean"]
    )

    summary.to_csv(
        OUTPUT_DIR / "disagreement_characteristics_summary.csv"
    )

    print("\nMedian characteristics by disagreement group:")
    print(
        merged.groupby("error_group")[numeric_columns]
        .median()
        .round(4)
        .to_string()
    )
else:
    print("\nNo numeric signal-characteristic columns found.")

merged.to_csv(
    OUTPUT_DIR / "disagreement_characteristics.csv",
    index=False,
)

print("\nSaved enriched records to:")
print(OUTPUT_DIR / "disagreement_characteristics.csv")

print("\nSaved group summary to:")
print(OUTPUT_DIR / "disagreement_characteristics_summary.csv")
