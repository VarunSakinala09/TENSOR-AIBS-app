
import os
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch


# --------------------------------------------------
# PATHS
# --------------------------------------------------

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

dataset_path = os.path.join(
    project_root, "dataset", "raw", "set-a"
)

output_path = os.path.join(
    project_root, "results", "week6", "preprocessing"
)

os.makedirs(output_path, exist_ok=True)


# --------------------------------------------------
# REPRESENTATIVE ECG RECORDS FROM WEEK 5
# --------------------------------------------------

record_names = ["2357247", "1192637", "2989538"]


# --------------------------------------------------
# PREPROCESSING FUNCTIONS
# --------------------------------------------------

def high_pass_filter(signal, fs, cutoff):
    sos = butter(
        4, cutoff, btype="highpass",
        fs=fs, output="sos"
    )
    return sosfiltfilt(sos, signal, axis=0)


def low_pass_filter(signal, fs, cutoff):
    sos = butter(
        4, cutoff, btype="lowpass",
        fs=fs, output="sos"
    )
    return sosfiltfilt(sos, signal, axis=0)


def band_pass_filter(signal, fs, low, high):
    sos = butter(
        4, [low, high], btype="bandpass",
        fs=fs, output="sos"
    )
    return sosfiltfilt(sos, signal, axis=0)


def notch_filter(signal, fs, frequency):
    b, a = iirnotch(frequency, Q=30, fs=fs)
    return filtfilt(b, a, signal, axis=0)


# --------------------------------------------------
# SIGNAL CHARACTERISTICS
# --------------------------------------------------

def calculate_characteristics(signal, fs):

    rms = np.sqrt(np.mean(signal ** 2, axis=0))
    std = np.std(signal, axis=0)
    abs_amplitude = np.mean(np.abs(signal), axis=0)

    window_size = max(1, int(0.2 * fs))
    kernel = np.ones(window_size) / window_size

    baseline = np.zeros_like(signal)

    for lead in range(signal.shape[1]):
        baseline[:, lead] = np.convolve(
            signal[:, lead], kernel, mode="same"
        )

    baseline_variation = np.std(baseline, axis=0)

    high_ratios = []
    powerline_ratios = []

    for lead in range(signal.shape[1]):

        frequencies, power = welch(
            signal[:, lead], fs=fs
        )

        total_power = np.sum(power)

        if total_power <= 0:
            high_ratio = 0
            powerline_ratio = 0
        else:
            high_ratio = (
                np.sum(power[frequencies >= 20])
                / total_power
            )

            powerline_ratio = (
                np.sum(
                    power[
                        (frequencies >= 49)
                        & (frequencies <= 51)
                    ]
                )
                / total_power
            )

        high_ratios.append(high_ratio)
        powerline_ratios.append(powerline_ratio)

    return {
        "mean_rms": np.mean(rms),
        "max_rms": np.max(rms),
        "mean_std": np.mean(std),
        "max_std": np.max(std),
        "mean_abs_amplitude": np.mean(abs_amplitude),
        "mean_baseline_variation": np.mean(baseline_variation),
        "max_baseline_variation": np.max(baseline_variation),
        "mean_high_frequency_ratio": np.mean(high_ratios),
        "max_high_frequency_ratio": np.max(high_ratios),
        "mean_powerline_ratio": np.mean(powerline_ratios),
        "max_powerline_ratio": np.max(powerline_ratios)
    }


# --------------------------------------------------
# EXPERIMENTAL FILTER SETTINGS
# --------------------------------------------------
# These are initial settings, not final adaptive rules.

configurations = {
    "raw": None,
    "highpass": {"type": "highpass", "cutoff": 0.5},
    "lowpass": {"type": "lowpass", "cutoff": 40},
    "bandpass": {
        "type": "bandpass",
        "low": 0.5,
        "high": 40
    },
    "notch": {"type": "notch", "frequency": 50}
}


# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

all_results = []

for record_name in record_names:

    print("\n" + "=" * 60)
    print("Processing record:", record_name)
    print("=" * 60)

    record = wfdb.rdrecord(
        os.path.join(dataset_path, record_name)
    )

    signal = record.p_signal
    fs = record.fs
    lead_names = record.sig_name

    print("Signal shape:", signal.shape)
    print("Sampling frequency:", fs)

    processed_signals = {}

    for config_name, config in configurations.items():

        print("Testing preprocessing:", config_name)

        if config_name == "raw":
            processed_signal = signal.copy()

        elif config["type"] == "highpass":
            processed_signal = high_pass_filter(
                signal, fs, config["cutoff"]
            )

        elif config["type"] == "lowpass":
            processed_signal = low_pass_filter(
                signal, fs, config["cutoff"]
            )

        elif config["type"] == "bandpass":
            processed_signal = band_pass_filter(
                signal, fs, config["low"], config["high"]
            )

        elif config["type"] == "notch":
            processed_signal = notch_filter(
                signal, fs, config["frequency"]
            )

        processed_signals[config_name] = processed_signal

        characteristics = calculate_characteristics(
            processed_signal, fs
        )

        characteristics["record_id"] = record_name
        characteristics["preprocessing"] = config_name

        all_results.append(characteristics)

    # --------------------------------------------------
    # PLOT ALL PREPROCESSING METHODS
    # --------------------------------------------------

    time = np.arange(signal.shape[0]) / fs

    # Use Lead II if available.
    lead_index = (
        lead_names.index("II")
        if "II" in lead_names else 0
    )

    lead_name = lead_names[lead_index]

    fig, axes = plt.subplots(
        len(processed_signals),
        1,
        figsize=(13, 11),
        sharex=True
    )

    for ax, (name, processed_signal) in zip(
        axes, processed_signals.items()
    ):
        ax.plot(
            time,
            processed_signal[:, lead_index],
            linewidth=0.8
        )

        ax.set_title(name.capitalize())
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)")

    fig.suptitle(
        f"ECG Preprocessing Comparison\n"
        f"Record: {record_name} | Lead: {lead_name}",
        fontsize=14
    )

    plt.tight_layout()

    plot_path = os.path.join(
        output_path,
        f"record_{record_name}_preprocessing.png"
    )

    plt.savefig(plot_path, dpi=200)
    plt.close()

    print("Graph saved:", plot_path)


# --------------------------------------------------
# SAVE CSV RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(all_results)

columns = [
    "record_id",
    "preprocessing",
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

results_df = results_df[columns]

csv_path = os.path.join(
    output_path,
    "preprocessing_characteristics.csv"
)

results_df.to_csv(csv_path, index=False)

print("\n" + "=" * 60)
print("PREPROCESSING ANALYSIS COMPLETED")
print("=" * 60)

print("CSV saved:", csv_path)
print("Graphs saved in:", output_path)
print("\n", results_df.to_string(index=False))
# =========================================================
# EXPERIMENT 2
# HIGH-PASS EFFECT ON ALL LABELLED ECG RECORDINGS
# =========================================================

print("\n")
print("=" * 60)
print("EXPERIMENT 2: ALL 998 ECG RECORDINGS")
print("=" * 60)

# ---------------------------------------------------------
# LOAD DATASET LABEL LISTS
# ---------------------------------------------------------

acceptable_file = os.path.join(
    dataset_path,
    "RECORDS-acceptable"
)

unacceptable_file = os.path.join(
    dataset_path,
    "RECORDS-unacceptable"
)

with open(acceptable_file, "r") as f:
    acceptable_records = [
        line.strip()
        for line in f
        if line.strip()
    ]

with open(unacceptable_file, "r") as f:
    unacceptable_records = [
        line.strip()
        for line in f
        if line.strip()
    ]

labelled_records = []

for record_name in acceptable_records:
    labelled_records.append(
        (record_name, "Acceptable")
    )

for record_name in unacceptable_records:
    labelled_records.append(
        (record_name, "Unacceptable")
    )

labelled_records.sort()


print("Acceptable records:", len(acceptable_records))
print("Unacceptable records:", len(unacceptable_records))
print("Total labelled records:", len(labelled_records))


# ---------------------------------------------------------
# ANALYSE RAW VS HIGH-PASS
# ---------------------------------------------------------

all_highpass_results = []

for count, (record_name, label) in enumerate(
    labelled_records,
    start=1
):

    print(
        f"Processing {count}/{len(labelled_records)}: "
        f"{record_name}"
    )

    record = wfdb.rdrecord(
        os.path.join(
            dataset_path,
            record_name
        )
    )

    signal = record.p_signal
    fs = record.fs

    # ---------------------------------------------
    # RAW CHARACTERISTICS
    # ---------------------------------------------

    raw_characteristics = calculate_characteristics(
        signal,
        fs
    )

    # ---------------------------------------------
    # HIGH-PASS FILTER
    # ---------------------------------------------

    highpass_signal = high_pass_filter(
        signal,
        fs,
        0.5
    )

    # ---------------------------------------------
    # HIGH-PASS CHARACTERISTICS
    # ---------------------------------------------

    highpass_characteristics = calculate_characteristics(
        highpass_signal,
        fs
    )

    raw_baseline = (
        raw_characteristics[
            "mean_baseline_variation"
        ]
    )

    filtered_baseline = (
        highpass_characteristics[
            "mean_baseline_variation"
        ]
    )

    # ---------------------------------------------
    # PERCENTAGE REDUCTION
    # ---------------------------------------------

    if raw_baseline != 0:

        reduction = (
            (raw_baseline - filtered_baseline)
            / raw_baseline
        ) * 100

    else:
        reduction = 0

    all_highpass_results.append({

        "record_id": record_name,

        "label": label,

        "raw_baseline_variation":
            raw_baseline,

        "highpass_baseline_variation":
            filtered_baseline,

        "baseline_reduction_percent":
            reduction,

        "raw_mean_rms":
            raw_characteristics["mean_rms"],

        "highpass_mean_rms":
            highpass_characteristics["mean_rms"]

    })


# ---------------------------------------------------------
# CREATE DATAFRAME
# ---------------------------------------------------------

highpass_df = pd.DataFrame(
    all_highpass_results
)


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

highpass_csv = os.path.join(
    output_path,
    "highpass_all_records_analysis.csv"
)

highpass_df.to_csv(
    highpass_csv,
    index=False
)


print()
print("Results saved to:")
print(highpass_csv)


# =========================================================
# SUMMARY BY QUALITY LABEL
# =========================================================

summary = (
    highpass_df
    .groupby("label")
    [
        [
            "raw_baseline_variation",
            "highpass_baseline_variation",
            "baseline_reduction_percent"
        ]
    ]
    .agg(["mean", "median", "std"])
)

print()
print("=" * 60)
print("BASELINE VARIATION SUMMARY")
print("=" * 60)

print(summary)


# =========================================================
# PLOT 1
# RAW VS HIGH-PASS BASELINE VARIATION
# =========================================================

plt.figure(figsize=(9, 7))

plt.scatter(
    highpass_df["raw_baseline_variation"],
    highpass_df["highpass_baseline_variation"],
    alpha=0.5
)

maximum = max(
    highpass_df["raw_baseline_variation"].max(),
    highpass_df["highpass_baseline_variation"].max()
)

plt.plot(
    [0, maximum],
    [0, maximum],
    linestyle="--"
)

plt.xlabel(
    "Raw baseline variation"
)

plt.ylabel(
    "High-pass baseline variation"
)

plt.title(
    "Baseline Variation Before vs After High-pass Filtering"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

scatter_path = os.path.join(
    output_path,
    "baseline_variation_raw_vs_highpass.png"
)

plt.savefig(
    scatter_path,
    dpi=200
)

plt.close()

print()
print("Scatter plot saved:")
print(scatter_path)


# =========================================================
# PLOT 2
# BASELINE REDUCTION BY QUALITY GROUP
# =========================================================

plt.figure(figsize=(8, 6))

acceptable_reduction = highpass_df[
    highpass_df["label"] == "Acceptable"
]["baseline_reduction_percent"]

unacceptable_reduction = highpass_df[
    highpass_df["label"] == "Unacceptable"
]["baseline_reduction_percent"]

plt.boxplot(
    [
        acceptable_reduction,
        unacceptable_reduction
    ],
    labels=[
        "Acceptable",
        "Unacceptable"
    ]
)

plt.ylabel(
    "Baseline variation reduction (%)"
)

plt.title(
    "Effect of High-pass Filtering on Baseline Variation"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

boxplot_path = os.path.join(
    output_path,
    "baseline_reduction_by_quality.png"
)

plt.savefig(
    boxplot_path,
    dpi=200
)

plt.close()

print()
print("Boxplot saved:")
print(boxplot_path)


print()
print("=" * 60)
print("EXPERIMENT 2 COMPLETED")
print("=" * 60)

# =========================================================
# EXPERIMENT 3
# DATA-DRIVEN BASELINE VARIATION ANALYSIS
# =========================================================

print("\n")
print("=" * 60)
print("EXPERIMENT 3: DATA-DRIVEN BASELINE ANALYSIS")
print("=" * 60)


# ---------------------------------------------------------
# LOAD EXPERIMENT 2 RESULTS
# ---------------------------------------------------------

highpass_csv = os.path.join(
    output_path,
    "highpass_all_records_analysis.csv"
)

analysis_df = pd.read_csv(highpass_csv)


# ---------------------------------------------------------
# CREATE DATA-DRIVEN BASELINE GROUPS
# ---------------------------------------------------------
# qcut creates groups based on the actual distribution
# of the dataset instead of using manually chosen limits.

analysis_df["baseline_group"] = pd.qcut(
    analysis_df["raw_baseline_variation"],
    q=5,
    labels=[
        "Very Low",
        "Low",
        "Medium",
        "High",
        "Very High"
    ],
    duplicates="drop"
)


# ---------------------------------------------------------
# SUMMARY BY BASELINE GROUP
# ---------------------------------------------------------

group_summary = (
    analysis_df
    .groupby("baseline_group", observed=False)
    .agg(
        record_count=("record_id", "count"),

        median_raw_baseline=(
            "raw_baseline_variation",
            "median"
        ),

        median_highpass_baseline=(
            "highpass_baseline_variation",
            "median"
        ),

        median_reduction=(
            "baseline_reduction_percent",
            "median"
        ),

        q25_reduction=(
            "baseline_reduction_percent",
            lambda x: x.quantile(0.25)
        ),

        q75_reduction=(
            "baseline_reduction_percent",
            lambda x: x.quantile(0.75)
        )
    )
    .reset_index()
)


print()
print("=" * 60)
print("BASELINE VARIATION GROUP SUMMARY")
print("=" * 60)

print(
    group_summary.to_string(index=False)
)


# ---------------------------------------------------------
# SAVE GROUP SUMMARY
# ---------------------------------------------------------

group_summary_path = os.path.join(
    output_path,
    "baseline_variation_group_analysis.csv"
)

group_summary.to_csv(
    group_summary_path,
    index=False
)

print()
print("Group analysis saved:")
print(group_summary_path)


# =========================================================
# ANALYSIS BY QUALITY LABEL
# =========================================================

label_group_summary = (
    analysis_df
    .groupby(
        ["baseline_group", "label"],
        observed=False
    )
    .agg(
        record_count=("record_id", "count"),

        median_raw_baseline=(
            "raw_baseline_variation",
            "median"
        ),

        median_reduction=(
            "baseline_reduction_percent",
            "median"
        )
    )
    .reset_index()
)


print()
print("=" * 60)
print("BASELINE GROUP + QUALITY SUMMARY")
print("=" * 60)

print(
    label_group_summary.to_string(index=False)
)


label_group_path = os.path.join(
    output_path,
    "baseline_group_quality_analysis.csv"
)

label_group_summary.to_csv(
    label_group_path,
    index=False
)


# =========================================================
# PLOT: BASELINE GROUP VS HIGH-PASS EFFECT
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    group_summary["baseline_group"].astype(str),
    group_summary["median_reduction"],
    marker="o",
    linewidth=2
)

plt.xlabel(
    "Raw baseline variation group"
)

plt.ylabel(
    "Median baseline variation reduction (%)"
)

plt.title(
    "High-pass Effect Across Baseline Variation Groups"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

group_plot_path = os.path.join(
    output_path,
    "baseline_group_vs_highpass_effect.png"
)

plt.savefig(
    group_plot_path,
    dpi=200
)

plt.close()

print()
print("Group effect plot saved:")
print(group_plot_path)


# =========================================================
# SPEARMAN CORRELATION
# =========================================================

from scipy.stats import spearmanr

correlation, p_value = spearmanr(
    analysis_df["raw_baseline_variation"],
    analysis_df["baseline_reduction_percent"]
)

print()
print("=" * 60)
print("BASELINE VARIATION vs HIGH-PASS EFFECT")
print("=" * 60)

print(
    "Spearman correlation:",
    round(correlation, 4)
)

print(
    "p-value:",
    p_value
)


# =========================================================
# SAVE CORRELATION RESULT
# =========================================================

correlation_df = pd.DataFrame({
    "variable_x": [
        "raw_baseline_variation"
    ],

    "variable_y": [
        "baseline_reduction_percent"
    ],

    "spearman_correlation": [
        correlation
    ],

    "p_value": [
        p_value
    ]
})

correlation_path = os.path.join(
    output_path,
    "baseline_highpass_correlation.csv"
)

correlation_df.to_csv(
    correlation_path,
    index=False
)


print()
print("Correlation result saved:")
print(correlation_path)


print()
print("=" * 60)
print("EXPERIMENT 3 COMPLETED")
print("=" * 60)

# =========================================================
# EXPERIMENT 4
# FINE-GRAINED DATA-DRIVEN THRESHOLD ANALYSIS
# =========================================================

print("\n")
print("=" * 60)
print("EXPERIMENT 4: FINE-GRAINED BASELINE ANALYSIS")
print("=" * 60)


# ---------------------------------------------------------
# LOAD RESULTS FROM EXPERIMENT 2
# ---------------------------------------------------------

highpass_csv = os.path.join(
    output_path,
    "highpass_all_records_analysis.csv"
)

fine_df = pd.read_csv(highpass_csv)


# ---------------------------------------------------------
# RAW BASELINE VARIATION QUANTILES
# ---------------------------------------------------------

quantiles = [
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.99
]

print()
print("=" * 60)
print("RAW BASELINE VARIATION QUANTILES")
print("=" * 60)

for q in quantiles:

    value = fine_df[
        "raw_baseline_variation"
    ].quantile(q)

    print(
        f"{int(q * 100)}th percentile: "
        f"{value:.6f}"
    )


# ---------------------------------------------------------
# CREATE 10 DATA-DRIVEN GROUPS
# ---------------------------------------------------------

fine_df["decile"] = pd.qcut(
    fine_df["raw_baseline_variation"],
    q=10,
    labels=[
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
        "D9",
        "D10"
    ],
    duplicates="drop"
)


# ---------------------------------------------------------
# DECILE SUMMARY
# ---------------------------------------------------------

decile_summary = (
    fine_df
    .groupby(
        "decile",
        observed=False
    )
    .agg(

        record_count=(
            "record_id",
            "count"
        ),

        minimum_baseline=(
            "raw_baseline_variation",
            "min"
        ),

        median_baseline=(
            "raw_baseline_variation",
            "median"
        ),

        maximum_baseline=(
            "raw_baseline_variation",
            "max"
        ),

        median_reduction=(
            "baseline_reduction_percent",
            "median"
        ),

        q25_reduction=(
            "baseline_reduction_percent",
            lambda x: x.quantile(0.25)
        ),

        q75_reduction=(
            "baseline_reduction_percent",
            lambda x: x.quantile(0.75)
        )
    )
    .reset_index()
)


print()
print("=" * 60)
print("DECILE ANALYSIS")
print("=" * 60)

print(
    decile_summary.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# SAVE DECILE RESULTS
# ---------------------------------------------------------

decile_path = os.path.join(
    output_path,
    "baseline_variation_decile_analysis.csv"
)

decile_summary.to_csv(
    decile_path,
    index=False
)

print()
print("Decile analysis saved:")
print(decile_path)


# =========================================================
# PLOT DECILE VS HIGH-PASS EFFECT
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    decile_summary["decile"].astype(str),
    decile_summary["median_reduction"],
    marker="o",
    linewidth=2
)

plt.xlabel(
    "Baseline variation decile"
)

plt.ylabel(
    "Median baseline variation reduction (%)"
)

plt.title(
    "High-pass Effect Across Baseline Variation Deciles"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

decile_plot_path = os.path.join(
    output_path,
    "baseline_decile_highpass_effect.png"
)

plt.savefig(
    decile_plot_path,
    dpi=200
)

plt.close()

print()
print("Decile plot saved:")
print(decile_plot_path)


# =========================================================
# FIND LARGEST CHANGE BETWEEN ADJACENT DECILES
# =========================================================

median_values = (
    decile_summary[
        "median_reduction"
    ].to_numpy()
)

decile_changes = np.diff(
    median_values
)

largest_change_index = np.argmax(
    decile_changes
)


print()
print("=" * 60)
print("LARGEST CHANGE BETWEEN ADJACENT DECILES")
print("=" * 60)

print(
    "From:",
    decile_summary.loc[
        largest_change_index,
        "decile"
    ]
)

print(
    "To:",
    decile_summary.loc[
        largest_change_index + 1,
        "decile"
    ]
)

print(
    "Change in median reduction:",
    round(
        decile_changes[
            largest_change_index
        ],
        4
    ),
    "%"
)


# ---------------------------------------------------------
# SHOW TRANSITION REGION
# ---------------------------------------------------------

candidate_lower = decile_summary.loc[
    largest_change_index + 1,
    "minimum_baseline"
]

candidate_upper = decile_summary.loc[
    largest_change_index + 1,
    "maximum_baseline"
]

print()
print(
    "Candidate transition region:"
)

print(
    f"{candidate_lower:.6f}"
    f" to "
    f"{candidate_upper:.6f}"
)


print()
print("=" * 60)
print("EXPERIMENT 4 COMPLETED")
print("=" * 60)