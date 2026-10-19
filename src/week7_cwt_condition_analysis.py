import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CHARACTERISTICS_FILE = os.path.join(
    BASE_DIR,
    "results",
    "week5",
    "signal_characteristics.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "week7",
    "condition_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD SIGNAL CHARACTERISTICS
# ============================================================

print("=" * 70)
print("EXPERIMENT 7C")
print("CWT CONDITION ANALYSIS")
print("=" * 70)

print("\nLoading signal characteristics...")

df = pd.read_csv(
    CHARACTERISTICS_FILE
)

print(
    "Dataset shape:",
    df.shape
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "record_id",
    "label",
    "mean_rms",
    "max_rms",
    "mean_abs_amplitude",
    "mean_baseline_variation",
    "max_baseline_variation",
    "mean_high_frequency_ratio",
    "max_high_frequency_ratio",
    "mean_powerline_ratio",
    "max_powerline_ratio"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing columns: "
        + str(missing_columns)
    )


# ============================================================
# LOAD CWT CLASSIFICATION RESULTS
# ============================================================

CWT_RESULTS_FILE = os.path.join(
    BASE_DIR,
    "results",
    "week7",
    "cwt_d",
    "cwt_d_vs_fixed_fold_results.csv"
)

print("\nLoading CWT classification results...")

results = pd.read_csv(
    CWT_RESULTS_FILE
)

print(
    "Fold result shape:",
    results.shape
)


# ============================================================
# IMPORTANT
# ============================================================

print(
    "\nThe classification result file contains "
    "fold-level performance, not per-record predictions."
)

print(
    "Therefore, this experiment will first analyze "
    "signal-condition groups using the available "
    "characteristics and overall CWT comparison."
)


# ============================================================
# CREATE CONDITION GROUPS
# ============================================================

# We use baseline variation because it was the
# main condition parameter investigated earlier.

df["baseline_group"] = pd.qcut(
    df["mean_baseline_variation"],
    q=3,
    labels=[
        "Low",
        "Medium",
        "High"
    ],
    duplicates="drop"
)


# ============================================================
# GROUP SUMMARY
# ============================================================

group_summary = (
    df
    .groupby(
        "baseline_group",
        observed=True
    )
    .agg(
        record_count=(
            "record_id",
            "count"
        ),

        mean_baseline_variation=(
            "mean_baseline_variation",
            "mean"
        ),

        median_baseline_variation=(
            "mean_baseline_variation",
            "median"
        ),

        mean_rms=(
            "mean_rms",
            "mean"
        ),

        mean_high_frequency_ratio=(
            "mean_high_frequency_ratio",
            "mean"
        ),

        acceptable_count=(
            "label",
            lambda x: (
                x == "Acceptable"
            ).sum()
        ),

        unacceptable_count=(
            "label",
            lambda x: (
                x == "Unacceptable"
            ).sum()
        )
    )
    .reset_index()
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("BASELINE-VARIATION CONDITION GROUPS")
print("=" * 70)

print(
    group_summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE GROUP SUMMARY
# ============================================================

group_file = os.path.join(
    OUTPUT_DIR,
    "cwt_condition_groups.csv"
)

group_summary.to_csv(
    group_file,
    index=False
)


# ============================================================
# OVERALL CWT COMPARISON
# ============================================================

comparison_file = os.path.join(
    BASE_DIR,
    "results",
    "week7",
    "cwt_d",
    "cwt_d_vs_fixed_comparison.csv"
)

comparison = pd.read_csv(
    comparison_file
)

print("\n")
print("=" * 70)
print("OVERALL CWT COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# CALCULATE PERCENTAGE IMPROVEMENT
# ============================================================

fixed_sensitivity = comparison.loc[
    comparison["Metric"] == "Sensitivity",
    "Fixed_Baseline"
].iloc[0]

cwt_d_sensitivity = comparison.loc[
    comparison["Metric"] == "Sensitivity",
    "CWT_D"
].iloc[0]

sensitivity_difference = (
    cwt_d_sensitivity
    -
    fixed_sensitivity
)

relative_sensitivity_change = (
    sensitivity_difference
    /
    fixed_sensitivity
) * 100


print("\nSensitivity:")
print(
    f"Fixed CWT : {fixed_sensitivity:.4f}"
)

print(
    f"CWT-D     : {cwt_d_sensitivity:.4f}"
)

print(
    f"Absolute change: "
    f"{sensitivity_difference:+.4f}"
)

print(
    f"Relative change: "
    f"{relative_sensitivity_change:+.2f}%"
)


# ============================================================
# CONDITION VISUALIZATION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.boxplot(
    [
        df.loc[
            df["baseline_group"] == group,
            "mean_baseline_variation"
        ]
        for group in [
            "Low",
            "Medium",
            "High"
        ]
    ],
    tick_labels=[
        "Low",
        "Medium",
        "High"
    ]
)

plt.xlabel(
    "Baseline Variation Condition"
)

plt.ylabel(
    "Mean Baseline Variation"
)

plt.title(
    "ECG Signal Conditions Based on Baseline Variation"
)

plt.tight_layout()

plot_file = os.path.join(
    OUTPUT_DIR,
    "baseline_condition_groups.png"
)

plt.savefig(
    plot_file,
    dpi=300
)

plt.close()


# ============================================================
# HIGH-FREQUENCY CHARACTERISTIC BY CONDITION
# ============================================================

condition_stats = (
    df
    .groupby(
        "baseline_group",
        observed=True
    )[
        [
            "mean_baseline_variation",
            "mean_high_frequency_ratio",
            "mean_rms",
            "mean_abs_amplitude"
        ]
    ]
    .mean()
    .reset_index()
)

condition_stats_file = os.path.join(
    OUTPUT_DIR,
    "condition_characteristics_summary.csv"
)

condition_stats.to_csv(
    condition_stats_file,
    index=False
)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n")
print("=" * 70)
print("EXPERIMENT 7C SUMMARY")
print("=" * 70)

print(
    "\nThree signal-condition groups were created "
    "using baseline variation."
)

print(
    "The groups are Low, Medium, and High."
)

print(
    "\nThe full-dataset CWT comparison showed:"
)

if sensitivity_difference > 0:

    print(
        "CWT-D has higher overall sensitivity "
        "than the fixed CWT."
    )

elif sensitivity_difference < 0:

    print(
        "CWT-D has lower overall sensitivity "
        "than the fixed CWT."
    )

else:

    print(
        "Both configurations have the same "
        "overall sensitivity."
    )

print(
    "\nIMPORTANT:"
)

print(
    "This experiment does NOT yet prove that "
    "different conditions require different CWT settings."
)

print(
    "It only establishes the signal-condition "
    "groups and records the overall CWT result."
)

print(
    "\nA true adaptive CWT rule requires "
    "per-record validation or training-fold "
    "selection between configurations."
)

print("\nSaved files:")

print(
    group_file
)

print(
    condition_stats_file
)

print(
    plot_file
)

print("\n")
print("=" * 70)
print("EXPERIMENT 7C COMPLETE")
print("=" * 70)