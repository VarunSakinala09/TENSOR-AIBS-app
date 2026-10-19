
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "results" / "adaptive_preprocessing_v2"
    / "oof_predictions.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "adaptive_preprocessing_v2"
    / "error_analysis"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

required = [
    "record_id",
    "label",
    "baseline_prediction",
    "highpass_prediction",
    "adaptive_prediction",
    "mean_baseline_variation"
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

if df["record_id"].duplicated().any():
    raise ValueError("Duplicate record IDs found.")

if df[required].isna().any().any():
    raise ValueError("Missing values found in required columns.")

# Identify correct and incorrect predictions
df["baseline_correct"] = (
    df["baseline_prediction"] == df["label"]
)
df["highpass_correct"] = (
    df["highpass_prediction"] == df["label"]
)
df["adaptive_correct"] = (
    df["adaptive_prediction"] == df["label"]
)

# Compare baseline and adaptive outcomes
def compare_predictions(row):
    base = row["baseline_correct"]
    adaptive = row["adaptive_correct"]

    if base and adaptive:
        return "Both correct"
    elif base and not adaptive:
        return "Baseline correct, adaptive wrong"
    elif not base and adaptive:
        return "Baseline wrong, adaptive correct"
    return "Both wrong"

df["baseline_vs_adaptive"] = df.apply(
    compare_predictions, axis=1
)

# Summarize prediction changes
print("\n1. BASELINE VS ADAPTIVE")
print(df["baseline_vs_adaptive"].value_counts())

# Inspect records whose outcomes changed
changed = df[
    df["baseline_correct"] != df["adaptive_correct"]
].copy()

changed.to_csv(
    OUTPUT_DIR / "records_with_changed_correctness.csv",
    index=False
)

print("\n2. RECORDS WITH CHANGED CORRECTNESS")
print("Total:", len(changed))
print(
    changed[
        [
            "record_id", "label",
            "baseline_prediction", "adaptive_prediction",
            "mean_baseline_variation",
            "baseline_correct", "adaptive_correct"
        ]
    ].to_string(index=False)
)

# Separate unacceptable ECGs that each model misses
unacceptable = df[df["label"] == "Unacceptable"].copy()

baseline_missed = unacceptable[
    unacceptable["baseline_prediction"] == "Acceptable"
]
adaptive_missed = unacceptable[
    unacceptable["adaptive_prediction"] == "Acceptable"
]

print("\n3. UNACCEPTABLE ECGs MISSED")
print("Baseline missed:", len(baseline_missed))
print("Adaptive missed:", len(adaptive_missed))

newly_missed = baseline_missed.merge(
    adaptive_missed[["record_id"]],
    on="record_id",
    how="inner"
)

baseline_only_missed = baseline_missed[
    ~baseline_missed["record_id"].isin(
        adaptive_missed["record_id"]
    )
]

adaptive_only_missed = adaptive_missed[
    ~adaptive_missed["record_id"].isin(
        baseline_missed["record_id"]
    )
]

print("Missed by both:", len(newly_missed))
print("Missed only by baseline:", len(baseline_only_missed))
print("Missed only by adaptive:", len(adaptive_only_missed))

adaptive_only_missed.to_csv(
    OUTPUT_DIR / "unacceptable_missed_only_by_adaptive.csv",
    index=False
)

baseline_only_missed.to_csv(
    OUTPUT_DIR / "unacceptable_missed_only_by_baseline.csv",
    index=False
)

# Descriptive analysis by baseline-variation terciles
# This is for diagnosis, not for selecting a model using test labels.
df["variation_group"] = pd.qcut(
    df["mean_baseline_variation"],
    q=3,
    labels=["Low", "Medium", "High"],
    duplicates="drop"
)

group_rows = []

for group, part in df.groupby(
    "variation_group", observed=True
):
    for method, column in [
        ("Baseline", "baseline_correct"),
        ("Fixed_HighPass", "highpass_correct"),
        ("Adaptive", "adaptive_correct")
    ]:
        unacceptable_part = part[
            part["label"] == "Unacceptable"
        ]
        acceptable_part = part[
            part["label"] == "Acceptable"
        ]

        sensitivity = (
            (unacceptable_part["adaptive_prediction"] == "Unacceptable").mean()
            if method == "Adaptive"
            else
            (unacceptable_part["baseline_prediction"] == "Unacceptable").mean()
            if method == "Baseline"
            else
            (unacceptable_part["highpass_prediction"] == "Unacceptable").mean()
        )

        specificity = (
            (acceptable_part[column.replace("_correct", "_prediction")] == "Acceptable").mean()
            if False else np.nan
        )

        prediction_column = {
            "Baseline": "baseline_prediction",
            "Fixed_HighPass": "highpass_prediction",
            "Adaptive": "adaptive_prediction"
        }[method]

        specificity = (
            (acceptable_part[prediction_column] == "Acceptable").mean()
            if len(acceptable_part) else np.nan
        )

        group_rows.append({
            "variation_group": group,
            "method": method,
            "records": len(part),
            "unacceptable_records": len(unacceptable_part),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "accuracy": part[column].mean()
        })

group_summary = pd.DataFrame(group_rows)
group_summary.to_csv(
    OUTPUT_DIR / "variation_group_summary.csv",
    index=False
)

print("\n4. RESULTS BY BASELINE-VARIATION GROUP")
print(group_summary.round(4).to_string(index=False))

print("\nSaved analysis files to:")
print(OUTPUT_DIR)


# 5. Compare signal characteristics by prediction outcome

characteristics_path = (
    ROOT / "results" / "week5" / "signal_characteristics.csv"
)

characteristics = pd.read_csv(characteristics_path)

df = df.merge(
    characteristics,
    on="record_id",
    how="left",
    validate="one_to_one",
    suffixes=("", "_characteristics")
)

candidate_columns = [
    "mean_rms",
    "max_rms",
    "mean_std",
    "max_std",
    "mean_abs_amplitude",
    "mean_baseline_variation",
    "max_baseline_variation",
    "mean_high_frequency_ratio",
    "max_high_frequency_ratio",
    "mean_powerline_ratio",
    "max_powerline_ratio"
]

available = [
    col for col in candidate_columns
    if col in df.columns
]

print("\n5. MEDIAN SIGNAL CHARACTERISTICS BY OUTCOME")

summary = df.groupby("baseline_vs_adaptive")[available].median()
print(summary.round(5).to_string())

summary.to_csv(
    OUTPUT_DIR / "outcome_characteristic_medians.csv"
)

print("\nDiagnostic summary saved.")

