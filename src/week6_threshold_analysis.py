import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PATH
# =========================================================

output_path = os.path.join(
    "results",
    "week6",
    "preprocessing"
)


# =========================================================
# LOAD EXPERIMENT 2 RESULTS
# =========================================================

input_file = os.path.join(
    output_path,
    "highpass_all_records_analysis.csv"
)

df = pd.read_csv(input_file)


print("=" * 60)
print("EXPERIMENT 5: THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 60)


# =========================================================
# DATA-DRIVEN PERCENTILE THRESHOLDS
# =========================================================

candidate_percentiles = [
    50,
    60,
    70,
    75,
    80,
    85,
    90
]


results = []


for percentile in candidate_percentiles:

    threshold = np.percentile(
        df["raw_baseline_variation"],
        percentile
    )

    high_condition = (
        df["raw_baseline_variation"] > threshold
    )

    high_count = high_condition.sum()
    low_count = len(df) - high_count

    high_percentage = (
        high_count / len(df)
    ) * 100

    low_percentage = (
        low_count / len(df)
    ) * 100

    high_effect = df.loc[
        high_condition,
        "baseline_reduction_percent"
    ].median()

    low_effect = df.loc[
        ~high_condition,
        "baseline_reduction_percent"
    ].median()

    results.append({

        "percentile": percentile,

        "threshold": threshold,

        "low_count": low_count,

        "high_count": high_count,

        "low_percentage": low_percentage,

        "high_percentage": high_percentage,

        "low_median_reduction":
            low_effect,

        "high_median_reduction":
            high_effect
    })


results_df = pd.DataFrame(results)


# =========================================================
# DISPLAY RESULTS
# =========================================================

print()
print("CANDIDATE THRESHOLDS")
print("=" * 60)

print(
    results_df.to_string(index=False)
)


# =========================================================
# SAVE RESULTS
# =========================================================

output_file = os.path.join(
    output_path,
    "threshold_sensitivity_analysis.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print()
print("Results saved to:")
print(output_file)


# =========================================================
# PLOT 1
# PREPROCESSING EFFECT
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    results_df["percentile"],
    results_df["high_median_reduction"],
    marker="o",
    label="High-condition ECGs"
)

plt.plot(
    results_df["percentile"],
    results_df["low_median_reduction"],
    marker="o",
    label="Low-condition ECGs"
)

plt.xlabel(
    "Baseline variation percentile"
)

plt.ylabel(
    "Median baseline variation reduction (%)"
)

plt.title(
    "Preprocessing Effect at Different Thresholds"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plot_file = os.path.join(
    output_path,
    "threshold_preprocessing_effect.png"
)

plt.savefig(
    plot_file,
    dpi=200
)

plt.close()


# =========================================================
# PLOT 2
# NUMBER OF ECGs SELECTED
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    results_df["percentile"],
    results_df["high_percentage"],
    marker="o"
)

plt.xlabel(
    "Baseline variation percentile"
)

plt.ylabel(
    "ECGs receiving high-pass (%)"
)

plt.title(
    "Adaptive Preprocessing Selection"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plot_file_2 = os.path.join(
    output_path,
    "adaptive_preprocessing_selection.png"
)

plt.savefig(
    plot_file_2,
    dpi=200
)

plt.close()


print()
print("Plots saved:")
print(plot_file)
print(plot_file_2)


print()
print("=" * 60)
print("EXPERIMENT 5 COMPLETED")
print("=" * 60)