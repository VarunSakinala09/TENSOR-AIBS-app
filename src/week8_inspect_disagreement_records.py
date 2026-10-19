
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILE = (
    ROOT / "results/week8/frequency_comparison"
    / "disagreement_characteristics.csv"
)

df = pd.read_csv(FILE, dtype={"record_id": str})

print("DISAGREEMENT RECORD INSPECTION")
print("=" * 65)
print("Total records:", len(df))

# Show available columns to verify the saved prediction schema.
print("\nAvailable columns:")
print(df.columns.tolist())

# Identify likely prediction and label columns.
for col in df.columns:
    if any(word in col.lower() for word in
           ["label", "true", "actual", "40hz", "50hz", "predict"]):
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).to_string())

# Inspect characteristics for the records where only one model was correct.
cols = [
    "record_id",
    "error_group",
    "mean_baseline_variation",
    "max_baseline_variation",
    "mean_high_frequency_ratio",
    "mean_rms",
    "mean_std",
]

cols = [c for c in cols if c in df.columns]

print("\nAll disagreement records:")
print(
    df[cols]
    .sort_values(
        "mean_baseline_variation",
        ascending=False
    )
    .to_string(index=False)
)

print("\nFinished. No models retrained.")
