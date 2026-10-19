
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset" / "raw" / "set-a"

MAPPING_FILE = (
    ROOT / "results" / "week4" / "group_aware_cv"
    / "record_signal_groups.csv"
)

OUTPUT_DIR = ROOT / "results" / "week8" / "dataset_wide_lead_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_record(record_id):
    candidates = [
        DATASET_DIR / record_id,
        DATASET_DIR / "set-a" / record_id,
    ]

    for path in candidates:
        if Path(str(path) + ".hea").exists():
            return path

    return None


mapping = pd.read_csv(MAPPING_FILE)
rows = []

for index, row in mapping.iterrows():
    record_id = str(row["record_id"])
    record_path = find_record(record_id)

    if record_path is None:
        print(f"Missing header: {record_id}")
        continue

    try:
        signal, fields = wfdb.rdsamp(str(record_path))
    except Exception as error:
        print(f"Read error for {record_id}: {error}")
        continue

    lead_names = fields.get("sig_name", [])

    for lead_index in range(signal.shape[1]):
        x = signal[:, lead_index]
        finite = np.isfinite(x)
        values = x[finite]

        if len(values) == 0:
            rows.append({
                "record_id": record_id,
                "label": row["label"],
                "lead": lead_names[lead_index],
                "std": np.nan,
                "range": np.nan,
                "unique_values": 0,
                "constant": True,
                "fraction_zero": np.nan,
                "fraction_minus80": np.nan,
                "fraction_near_minus80": np.nan,
                "nonfinite_count": int(np.sum(~finite)),
            })
            continue

        rows.append({
            "record_id": record_id,
            "label": row["label"],
            "lead": (
                lead_names[lead_index]
                if lead_index < len(lead_names)
                else f"Lead_{lead_index + 1}"
            ),
            "std": float(np.std(values)),
            "range": float(np.max(values) - np.min(values)),
            "unique_values": int(np.unique(values).size),
            "constant": bool(np.all(values == values[0])),
            "fraction_zero": float(np.mean(values == 0)),
            "fraction_minus80": float(np.mean(values == -80)),
            "fraction_near_minus80": float(
                np.mean(np.isclose(values, -80, atol=1e-8))
            ),
            "nonfinite_count": int(np.sum(~finite)),
        })

    if (index + 1) % 100 == 0:
        print(f"Processed {index + 1}/{len(mapping)} records")


results = pd.DataFrame(rows)

results.to_csv(
    OUTPUT_DIR / "all_record_lead_features.csv",
    index=False
)

# Summarize constant leads by label and lead.
summary = (
    results.groupby(["label", "lead"])
    .agg(
        records=("record_id", "nunique"),
        constant_leads=("constant", "sum"),
        median_std=("std", "median"),
        median_range=("range", "median"),
        median_unique_values=("unique_values", "median"),
        records_with_minus80=("fraction_minus80", lambda x: int((x > 0).sum())),
        records_with_nonfinite=("nonfinite_count", lambda x: int((x > 0).sum())),
    )
    .reset_index()
)

summary.to_csv(
    OUTPUT_DIR / "lead_quality_summary.csv",
    index=False
)

print("\nLEAD-WISE SUMMARY")
print("=" * 90)
print(summary.to_string(index=False))

# Record-level summary: how many leads are constant in each record?
record_summary = (
    results.groupby(["record_id", "label"])
    .agg(
        total_leads=("lead", "size"),
        constant_leads=("constant", "sum"),
        leads_with_minus80=(
            "fraction_minus80",
            lambda x: int((x > 0).sum())
        ),
        leads_with_nonfinite=(
            "nonfinite_count",
            lambda x: int((x > 0).sum())
        ),
    )
    .reset_index()
)

record_summary.to_csv(
    OUTPUT_DIR / "record_quality_summary.csv",
    index=False
)

print("\nRECORD-LEVEL SUMMARY")
print("=" * 90)
print(
    record_summary.groupby("label")
    .agg(
        records=("record_id", "nunique"),
        records_with_constant_leads=("constant_leads", lambda x: int((x > 0).sum())),
        total_constant_leads=("constant_leads", "sum"),
        records_with_minus80=("leads_with_minus80", lambda x: int((x > 0).sum())),
        records_with_nonfinite=("leads_with_nonfinite", lambda x: int((x > 0).sum())),
    )
    .to_string()
)

print("\nSaved results to:")
print(OUTPUT_DIR)
