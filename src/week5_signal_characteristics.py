import os
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import welch


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = "dataset/raw/set-a"
OUTPUT_PATH = "results/week5/signal_characteristics.csv"

os.makedirs("results/week5", exist_ok=True)


# ============================================================
# LOAD LABELS
# ============================================================

acceptable_file = os.path.join(
    DATASET_PATH,
    "RECORDS-acceptable"
)

unacceptable_file = os.path.join(
    DATASET_PATH,
    "RECORDS-unacceptable"
)


def read_record_list(path):
    with open(path, "r") as file:
        return {
            line.strip()
            for line in file
            if line.strip()
        }


acceptable_records = read_record_list(acceptable_file)
unacceptable_records = read_record_list(unacceptable_file)


# ============================================================
# COMBINE LABELS
# ============================================================

record_labels = {}

for record_id in acceptable_records:
    record_labels[record_id] = "Acceptable"

for record_id in unacceptable_records:
    record_labels[record_id] = "Unacceptable"


print("\nLABELLED RECORDS")
print("----------------")
print("Acceptable:", len(acceptable_records))
print("Unacceptable:", len(unacceptable_records))
print("Total labelled:", len(record_labels))


# ============================================================
# HELPER FUNCTION
# ============================================================

def calculate_characteristics(signal, sampling_frequency):

    # --------------------------------------------------------
    # Overall signal statistics
    # --------------------------------------------------------

    rms_per_lead = np.sqrt(
        np.mean(signal ** 2, axis=0)
    )

    std_per_lead = np.std(
        signal,
        axis=0
    )

    mean_abs_per_lead = np.mean(
        np.abs(signal),
        axis=0
    )


    # --------------------------------------------------------
    # Baseline variation
    #
    # Approximate slow variation using a moving average.
    # Window is derived from sampling frequency.
    # --------------------------------------------------------

    baseline_window = max(
        int(sampling_frequency * 0.2),
        1
    )

    kernel = np.ones(
        baseline_window
    ) / baseline_window

    baseline_variations = []

    for lead_index in range(signal.shape[1]):

        lead = signal[:, lead_index]

        baseline = np.convolve(
            lead,
            kernel,
            mode="same"
        )

        variation = np.std(baseline)

        baseline_variations.append(variation)


    # --------------------------------------------------------
    # Frequency-domain characteristics
    # --------------------------------------------------------

    high_frequency_ratios = []
    powerline_ratios = []

    for lead_index in range(signal.shape[1]):

        lead = signal[:, lead_index]

        frequencies, power = welch(
            lead,
            fs=sampling_frequency,
            nperseg=min(
                len(lead),
                sampling_frequency * 2
            )
        )

        total_power = np.trapezoid(
            power,
            frequencies
        )

        if total_power > 0:

            # High-frequency region above 20 Hz
            high_frequency_mask = frequencies >= 20

            high_frequency_power = np.trapezoid(
                power[high_frequency_mask],
                frequencies[high_frequency_mask]
            )

            high_frequency_ratio = (
                high_frequency_power /
                total_power
            )

            # Narrow region around 50 Hz
            powerline_mask = (
                (frequencies >= 49) &
                (frequencies <= 51)
            )

            if np.any(powerline_mask):

                powerline_power = np.trapezoid(
                    power[powerline_mask],
                    frequencies[powerline_mask]
                )

                powerline_ratio = (
                    powerline_power /
                    total_power
                )

            else:
                powerline_ratio = 0.0

        else:
            high_frequency_ratio = 0.0
            powerline_ratio = 0.0


        high_frequency_ratios.append(
            high_frequency_ratio
        )

        powerline_ratios.append(
            powerline_ratio
        )


    # --------------------------------------------------------
    # Aggregate across 12 leads
    # --------------------------------------------------------

    return {
        "mean_rms": np.mean(rms_per_lead),
        "max_rms": np.max(rms_per_lead),

        "mean_std": np.mean(std_per_lead),
        "max_std": np.max(std_per_lead),

        "mean_abs_amplitude": np.mean(
            mean_abs_per_lead
        ),

        "mean_baseline_variation": np.mean(
            baseline_variations
        ),

        "max_baseline_variation": np.max(
            baseline_variations
        ),

        "mean_high_frequency_ratio": np.mean(
            high_frequency_ratios
        ),

        "max_high_frequency_ratio": np.max(
            high_frequency_ratios
        ),

        "mean_powerline_ratio": np.mean(
            powerline_ratios
        ),

        "max_powerline_ratio": np.max(
            powerline_ratios
        )
    }


# ============================================================
# PROCESS RECORDS
# ============================================================

results = []

sorted_records = sorted(record_labels.keys())

print("\nPROCESSING RECORDS")
print("------------------")

for index, record_id in enumerate(
    sorted_records,
    start=1
):

    record_path = os.path.join(
        DATASET_PATH,
        record_id
    )

    signal, fields = wfdb.rdsamp(record_path)

    sampling_frequency = fields["fs"]

    characteristics = calculate_characteristics(
        signal,
        sampling_frequency
    )

    row = {
        "record_id": record_id,
        "label": record_labels[record_id],
        "sampling_frequency": sampling_frequency,
        "number_of_leads": signal.shape[1],
        "number_of_samples": signal.shape[0]
    }

    row.update(characteristics)

    results.append(row)

    if index % 50 == 0 or index == 1:
        print(
            f"Processed {index}/{len(sorted_records)}"
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

data = pd.DataFrame(results)


# ============================================================
# SAVE
# ============================================================

data.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("SIGNAL CHARACTERISTICS COMPLETE")
print("=" * 60)

print("\nDataset shape:")
print(data.shape)

print("\nColumns:")
print(list(data.columns))

print("\nLabel distribution:")
print(data["label"].value_counts())

print("\nFirst five records:")
print(data.head())

print("\nSaved to:")
print(OUTPUT_PATH)