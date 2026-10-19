
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SCREENING_FILE = (
    ROOT / "results" / "week8" / "frequency_screening"
    / "spectral_summary.csv"
)

GROUP_FILE = (
    ROOT / "results" / "week4" / "group_aware_cv"
    / "record_signal_groups.csv"
)

if not SCREENING_FILE.exists():
    raise FileNotFoundError(f"Missing screening file: {SCREENING_FILE}")

if not GROUP_FILE.exists():
    raise FileNotFoundError(f"Missing record mapping: {GROUP_FILE}")

screening = pd.read_csv(SCREENING_FILE, dtype={"record_id": str})
groups = pd.read_csv(GROUP_FILE, dtype={"record_id": str})

required = {"record_id", "signal_hash", "label"}
if not required.issubset(groups.columns):
    raise ValueError(f"Mapping must contain columns: {required}")

# Keep only records present in our established mapping.
# Do not count multiple copies of the same signal as independent records.

# Check duplicate IDs before merging
print("Screening rows:", len(screening))
print("Unique screening IDs:", screening["record_id"].nunique())
print(
    "Repeated screening IDs:",
    screening["record_id"].duplicated().sum()
)

# Keep one screening row per record ID for this preliminary check.
# We will verify the repeated measurements before final reporting.
screening = screening.drop_duplicates(
    subset="record_id",
    keep="first"
)

merged = screening.merge(
    groups[["record_id", "signal_hash", "label"]].drop_duplicates(
        subset="record_id"
    ),
    on="record_id",
    how="inner",
    validate="one_to_one"
)


# Choose one representative record per signal hash.
# Sorting makes the selection deterministic.
unique_signals = (
    merged.sort_values("record_id")
    .drop_duplicates(subset="signal_hash", keep="first")
    .copy()
)

limits = [30, 40, 50, 70, 100]
columns = [f"power_above_{limit}hz" for limit in limits]

print("SCREENING DATA CHECK")
print("Raw screening rows:", len(screening))
print("Matched mapped records:", len(merged))
print("Unique signal groups:", merged["signal_hash"].nunique())
print("Unique representatives:", len(unique_signals))
print("Label counts among representatives:")
print(unique_signals["label"].value_counts().to_string())

print("\nMean power fraction above each limit:")
print(unique_signals[columns].mean().round(5).to_string())

output = (
    ROOT / "results" / "week8" / "frequency_screening"
    / "unique_signal_summary.csv"
)

unique_signals.to_csv(output, index=False)
print("\nSaved:", output)
