
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "results" / "week8" / "dataset_wide_lead_audit"
    / "all_record_lead_features.csv"
)

OUTPUT = (
    ROOT / "results" / "week8" / "dataset_wide_lead_audit"
    / "record_pattern_summary.csv"
)

df = pd.read_csv(INPUT)

# Build record-level flags from the per-lead audit.
records = (
    df.groupby(["record_id", "label"])
    .agg(
        constant_leads=("constant", "sum"),
        leads_with_minus80=("fraction_minus80", lambda x: int((x > 0).sum())),
        leads_entirely_minus80=(
            "fraction_minus80",
            lambda x: int((x == 1).sum())
        ),
        leads_with_nonfinite=("nonfinite_count", lambda x: int((x > 0).sum())),
    )
    .reset_index()
)

records["has_constant_lead"] = records["constant_leads"] > 0
records["has_minus80_sample"] = records["leads_with_minus80"] > 0
records["has_entire_lead_minus80"] = records["leads_entirely_minus80"] > 0

records.to_csv(OUTPUT, index=False)

print("\nRECORD-LEVEL PATTERN COUNTS")
print("=" * 75)

for feature in [
    "has_constant_lead",
    "has_minus80_sample",
    "has_entire_lead_minus80",
]:
    print(f"\n{feature}")
    print(
        pd.crosstab(
            records["label"],
            records[feature],
            margins=True
        ).to_string()
    )

print("\nPATTERN OVERLAP")
print("=" * 75)
print(
    pd.crosstab(
        records["has_constant_lead"],
        records["has_entire_lead_minus80"],
        margins=True
    ).to_string()
)

print(f"\nSaved: {OUTPUT}")
