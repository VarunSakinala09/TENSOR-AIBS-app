import os
import numpy as np
import pandas as pd
import wfdb
import pywt

from scipy.signal import welch


# =========================================================
# PATHS
# =========================================================

DATA_PATH = os.path.join(
    "dataset",
    "raw",
    "set-a"
)

BASELINE_FILE = os.path.join(
    "results",
    "week6",
    "preprocessing",
    "highpass_all_records_analysis.csv"
)

OUTPUT_PATH = os.path.join(
    "results",
    "week7"
)

os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)


# =========================================================
# PARAMETERS
# =========================================================

FS = 500

WAVELET = "morl"

CONFIGURATIONS = {

    "Baseline": {
        "low_frequency": 1,
        "high_frequency": 40,
        "levels": 100
    },

    "A": {
        "low_frequency": 1,
        "high_frequency": 20,
        "levels": 100
    },

    "B": {
        "low_frequency": 1,
        "high_frequency": 40,
        "levels": 50
    },

    "C": {
        "low_frequency": 1,
        "high_frequency": 40,
        "levels": 150
    },

    "D": {
        "low_frequency": 1,
        "high_frequency": 50,
        "levels": 100
    }
}


# =========================================================
# LOAD RECORD IDS
# =========================================================

acceptable_file = os.path.join(
    DATA_PATH,
    "RECORDS-acceptable"
)

unacceptable_file = os.path.join(
    DATA_PATH,
    "RECORDS-unacceptable"
)


acceptable_records = pd.read_csv(
    acceptable_file,
    header=None
)[0].astype(str).tolist()


unacceptable_records = pd.read_csv(
    unacceptable_file,
    header=None
)[0].astype(str).tolist()


record_labels = {}

for record_id in acceptable_records:
    record_labels[record_id] = 0

for record_id in unacceptable_records:
    record_labels[record_id] = 1


# =========================================================
# LOAD BASELINE CHARACTERISTICS
# =========================================================

baseline_df = pd.read_csv(
    BASELINE_FILE
)

baseline_df["record_id"] = (
    baseline_df["record_id"]
    .astype(str)
)


# =========================================================
# SELECT SIGNAL CONDITION FEATURES
# =========================================================

condition_columns = [
    "raw_baseline_variation"
]


condition_df = baseline_df[
    [
        "record_id",
        "raw_baseline_variation"
    ]
].copy()


# =========================================================
# DATA-DRIVEN SIGNAL CONDITION GROUPS
# =========================================================

condition_df["condition_group"] = pd.qcut(
    condition_df[
        "raw_baseline_variation"
    ],
    q=3,
    labels=[
        "Low",
        "Medium",
        "High"
    ],
    duplicates="drop"
)


print("=" * 70)
print("EXPERIMENT 7: ADAPTIVE CWT CONFIGURATION ANALYSIS")
print("=" * 70)

print()
print("Signal-condition groups:")

print(
    condition_df[
        "condition_group"
    ].value_counts(
        sort=False
    )
)


# =========================================================
# CWT FUNCTION
# =========================================================

def calculate_cwt(
    signal,
    low_frequency,
    high_frequency,
    levels
):

    frequencies = np.linspace(
        low_frequency,
        high_frequency,
        levels
    )

    central_frequency = (
        pywt.central_frequency(
            WAVELET
        )
    )

    scales = (
        central_frequency *
        FS /
        frequencies
    )

    coefficients, _ = pywt.cwt(
        signal,
        scales,
        WAVELET,
        sampling_period=1 / FS
    )

    return np.abs(
        coefficients
    )


# =========================================================
# REPRESENTATIVE RECORDS
# =========================================================

representatives = []

for group in [
    "Low",
    "Medium",
    "High"
]:

    group_records = condition_df[
        condition_df[
            "condition_group"
        ] == group
    ]

    if len(group_records) > 0:

        representative = (
            group_records
            .sort_values(
                "raw_baseline_variation"
            )
            .iloc[
                len(group_records) // 2
            ]
        )

        representatives.append(
            (
                representative[
                    "record_id"
                ],
                group
            )
        )


print()
print("=" * 70)
print("REPRESENTATIVE RECORDS")
print("=" * 70)

for record_id, group in representatives:

    value = condition_df.loc[
        condition_df["record_id"]
        == record_id,
        "raw_baseline_variation"
    ].iloc[0]

    print(
        f"{group}: "
        f"{record_id} "
        f"(baseline={value:.6f})"
    )


# =========================================================
# RUN CWT CONFIGURATION ANALYSIS
# =========================================================

results = []


for record_id, condition_group in representatives:

    print()
    print("=" * 70)

    print(
        f"Processing {condition_group}: "
        f"{record_id}"
    )

    print("=" * 70)


    record_path = os.path.join(
        DATA_PATH,
        record_id
    )

    record = wfdb.rdrecord(
        record_path
    )

    signal = record.p_signal


    # -----------------------------------------------------
    # USE ALL 12 LEADS
    # -----------------------------------------------------

    for config_name, config in (
        CONFIGURATIONS.items()
    ):

        print(
            f"Testing configuration: "
            f"{config_name}"
        )


        reconstruction_errors = []


        for lead in range(
            signal.shape[1]
        ):

            lead_signal = (
                signal[:, lead]
            )


            scalogram = calculate_cwt(
                lead_signal,
                config[
                    "low_frequency"
                ],
                config[
                    "high_frequency"
                ],
                config[
                    "levels"
                ]
            )


            # -------------------------------------------------
            # SIMPLE CWT ENERGY SUMMARY
            # -------------------------------------------------

            original_energy = np.mean(
                np.abs(
                    lead_signal
                ) ** 2
            )

            scalogram_energy = np.mean(
                scalogram ** 2
            )


            reconstruction_errors.append(
                abs(
                    scalogram_energy -
                    original_energy
                )
            )


        mean_error = np.mean(
            reconstruction_errors
        )


        results.append({

            "record_id": record_id,

            "condition_group":
                condition_group,

            "baseline_variation":
                condition_df.loc[
                    condition_df[
                        "record_id"
                    ] == record_id,
                    "raw_baseline_variation"
                ].iloc[0],

            "configuration":
                config_name,

            "low_frequency":
                config[
                    "low_frequency"
                ],

            "high_frequency":
                config[
                    "high_frequency"
                ],

            "levels":
                config[
                    "levels"
                ],

            "mean_energy_difference":
                mean_error
        })


# =========================================================
# SAVE RESULTS
# =========================================================

results_df = pd.DataFrame(
    results
)


output_file = os.path.join(
    OUTPUT_PATH,
    "adaptive_cwt_configuration_analysis.csv"
)


results_df.to_csv(
    output_file,
    index=False
)


print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


print()
print(
    "Results saved to:"
)

print(
    output_file
)


# =========================================================
# FIND BEST CONFIGURATION PER CONDITION
# =========================================================

best_configs = (
    results_df
    .sort_values(
        "mean_energy_difference"
    )
    .groupby(
        "condition_group",
        as_index=False
    )
    .first()
)


print()
print("=" * 70)
print("BEST CONFIGURATION PER SIGNAL CONDITION")
print("=" * 70)

print(
    best_configs[
        [
            "condition_group",
            "configuration",
            "mean_energy_difference"
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 70)
print("EXPERIMENT 7 COMPLETED")
print("=" * 70)