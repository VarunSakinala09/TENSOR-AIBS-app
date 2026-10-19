
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ERROR_FILE = (
    ROOT / "results/week8/frequency_comparison"
    / "frequency_error_analysis.csv"
)

CHAR_FILE = (
    ROOT / "results/week5/signal_characteristics.csv"
)

OUT_DIR = ROOT / "results/week8/frequency_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

errors = pd.read_csv(ERROR_FILE, dtype={"record_id": str})
chars = pd.read_csv(CHAR_FILE, dtype={"record_id": str})

if errors["record_id"].duplicated().any():
    raise ValueError("Error-analysis file has duplicate record IDs")

if chars["record_id"].duplicated().any():
    raise ValueError("Characteristics file has duplicate record IDs")

data = errors.merge(
    chars,
    on="record_id",
    how="left",
    suffixes=("", "_char"),
    validate="one_to_one",
    indicator=True,
)

print("FULL-DATASET HYPOTHESIS CHECK")
print("Records:", len(data))
print("Matched characteristics:", (data["_merge"] == "both").sum())

data = data[data["_merge"] == "both"].drop(columns="_merge")

numeric_candidates = [
    "mean_baseline_variation",
    "max_baseline_variation",
    "mean_high_frequency_ratio",
    "mean_rms",
    "mean_std",
    "mean_powerline_ratio",
]

numeric_cols = [
    c for c in numeric_candidates if c in data.columns
]

if not numeric_cols:
    raise ValueError("No expected numeric characteristics found")

print("\nGROUP COUNTS")
print(data["comparison"].value_counts().to_string())

print("\nMEDIAN CHARACTERISTICS BY GROUP")
print(
    data.groupby("comparison")[numeric_cols]
    .median()
    .round(4)
    .to_string()
)

print("\nMEAN CHARACTERISTICS BY GROUP")
print(
    data.groupby("comparison")[numeric_cols]
    .mean()
    .round(4)
    .to_string()
)

# Save a reproducible summary for later analysis.
summary = data.groupby("comparison")[numeric_cols].agg(
    ["count", "median", "mean", "std"]
)

summary.to_csv(
    OUT_DIR / "full_dataset_hypothesis_summary.csv"
)

print("\nSaved summary:")
print(OUT_DIR / "full_dataset_hypothesis_summary.csv")
print("\nNo models retrained; no signal processing rerun.")
