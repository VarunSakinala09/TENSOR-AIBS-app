
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

OUTPUT_DIR = ROOT / "results" / "week8" / "lead_quality_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_record(record_id):
    paths = [
        DATASET_DIR / record_id,
        DATASET_DIR / "set-a" / record_id,
    ]

    for path in paths:
        if Path(str(path) + ".hea").exists():
            return path

    return None


def longest_constant_run(signal):
    if len(signal) == 0:
        return 0

    longest = 1
    current = 1

    for i in range(1, len(signal)):
        if signal[i] == signal[i - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    return longest


def extract_features(signal):
    signal = np.asarray(signal, dtype=float)

    stds = []
    ranges = []
    unique_counts = []
    constant_flags = []
    minus80_flags = []
    minus80_fractions = []
    zero_fractions = []
    constant_run_fractions = []
    nonfinite_counts = []

    for lead_index in range(signal.shape[1]):
        raw = signal[:, lead_index]
        finite = np.isfinite(raw)
        x = raw[finite]

        nonfinite_counts.append(int(np.sum(~finite)))

        if len(x) == 0:
            stds.append(np.nan)
            ranges.append(np.nan)
            unique_counts.append(0)
            constant_flags.append(0)
            minus80_flags.append(0)
            minus80_fractions.append(np.nan)
            zero_fractions.append(np.nan)
            constant_run_fractions.append(np.nan)
            continue

        stds.append(float(np.std(x)))
        ranges.append(float(np.max(x) - np.min(x)))
        unique_counts.append(int(np.unique(x).size))

        constant_flags.append(int(np.all(x == x[0])))
        minus80_flags.append(int(np.all(x == -80.0)))

        minus80_fractions.append(float(np.mean(x == -80.0)))
        zero_fractions.append(float(np.mean(x == 0.0)))

        longest_run = longest_constant_run(x)
        constant_run_fractions.append(longest_run / len(x))

    stds = np.asarray(stds, dtype=float)
    ranges = np.asarray(ranges, dtype=float)
    unique_counts = np.asarray(unique_counts, dtype=float)
    minus80_fractions = np.asarray(minus80_fractions, dtype=float)
    zero_fractions = np.asarray(zero_fractions, dtype=float)
    constant_run_fractions = np.asarray(
        constant_run_fractions, dtype=float
    )

    return {
        "lead_count": signal.shape[1],
        "constant_lead_count": int(sum(constant_flags)),
        "entire_lead_minus80_count": int(sum(minus80_flags)),
        "leads_with_minus80_count": int(
            np.sum(minus80_fractions > 0)
        ),
        "leads_with_zero_count": int(
            np.sum(zero_fractions > 0)
        ),
        "total_nonfinite_samples": int(sum(nonfinite_counts)),
        "min_lead_std": float(np.nanmin(stds)),
        "median_lead_std": float(np.nanmedian(stds)),
        "mean_lead_std": float(np.nanmean(stds)),
        "min_lead_range": float(np.nanmin(ranges)),
        "median_lead_range": float(np.nanmedian(ranges)),
        "mean_lead_range": float(np.nanmean(ranges)),
        "min_unique_values": float(np.nanmin(unique_counts)),
        "median_unique_values": float(np.nanmedian(unique_counts)),
        "max_zero_fraction": float(np.nanmax(zero_fractions)),
        "max_minus80_fraction": float(
            np.nanmax(minus80_fractions)
        ),
        "max_constant_run_fraction": float(
            np.nanmax(constant_run_fractions)
        ),
    }


mapping = pd.read_csv(MAPPING_FILE)

feature_rows = []
failed_records = []

for index, row in mapping.iterrows():
    record_id = str(row["record_id"])
    record_path = find_record(record_id)

    if record_path is None:
        failed_records.append({
            "record_id": record_id,
            "error": "Header not found",
        })
        continue

    try:
        signal, fields = wfdb.rdsamp(str(record_path))

        features = extract_features(signal)

        feature_rows.append({
            "record_id": record_id,
            "label": row["label"],
            **features,
        })

    except Exception as error:
        failed_records.append({
            "record_id": record_id,
            "error": str(error),
        })

    if (index + 1) % 100 == 0:
        print(f"Processed {index + 1}/{len(mapping)} records")


features_df = pd.DataFrame(feature_rows)

features_path = OUTPUT_DIR / "lead_quality_features.csv"
features_df.to_csv(features_path, index=False)

failed_df = pd.DataFrame(
    failed_records,
    columns=["record_id", "error"],
)

failed_df.to_csv(
    OUTPUT_DIR / "failed_records.csv",
    index=False,
)

print("\nFEATURE EXTRACTION SUMMARY")
print("=" * 65)
print(f"Expected records: {len(mapping)}")
print(f"Successfully processed: {len(features_df)}")
print(f"Failed records: {len(failed_df)}")
print(f"Unique record IDs: {features_df['record_id'].nunique()}")
print(
    f"Duplicate record IDs: "
    f"{features_df['record_id'].duplicated().sum()}"
)

if not features_df.empty:
    feature_columns = [
        column for column in features_df.columns
        if column not in ["record_id", "label"]
    ]

    print(
        "Missing feature values: "
        f"{features_df[feature_columns].isna().sum().sum()}"
    )

    print("\nClass-wise feature medians:")
    print(
        features_df.groupby("label")[
            [
                "constant_lead_count",
                "entire_lead_minus80_count",
                "leads_with_minus80_count",
                "median_lead_std",
                "median_lead_range",
            ]
        ].median().to_string()
    )

print(f"\nFeatures saved to: {features_path}")
print(f"Failures saved to: {OUTPUT_DIR / 'failed_records.csv'}")
