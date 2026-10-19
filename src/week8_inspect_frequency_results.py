
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILE = (
    ROOT / "results" / "week8" / "frequency_screening"
    / "corrected_spectral_summary.csv"
)

df = pd.read_csv(FILE, dtype={"record_id": str})

limits = [30, 40, 50, 70, 100]
power_cols = [f"power_above_{limit}hz" for limit in limits]

print("FREQUENCY SCREENING AUDIT")
print("Rows:", len(df))
print("Unique record IDs:", df["record_id"].nunique())
print("\nLabel counts:")
print(df["label"].value_counts().to_string())

# Identify records with zero measured power above every limit.
# These need special attention because their spectral information
# may be uninformative.
all_zero = df[power_cols].fillna(0).eq(0).all(axis=1)

print("\nRecords with zero power above every candidate limit:", all_zero.sum())
print("Labels among these records:")
print(df.loc[all_zero, "label"].value_counts().to_string())

print("\nMean power fractions across all records:")
print(df[power_cols].mean().round(5).to_string())

print("\nMean power fractions by label:")
print(df.groupby("label")[power_cols].mean().round(5).to_string())

# Collapse identical signal groups so duplicated recordings
# do not receive extra weight in the exploratory summary.
if "signal_hash" in df.columns:
    print("\nUnique signal groups:", df["signal_hash"].nunique())

    # Check whether any identical signal group has conflicting labels.
    label_counts = df.groupby("signal_hash")["label"].nunique()
    print("Signal groups with conflicting labels:", (label_counts > 1).sum())

    unique_signals = (
        df.sort_values("record_id")
        .drop_duplicates("signal_hash")
    )

    print("\nMean power fractions after deduplicating signals:")
    print(unique_signals[power_cols].mean().round(5).to_string())

print("\nAudit complete. No files modified.")
