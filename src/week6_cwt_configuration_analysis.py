import os
import time
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pywt


# ============================================================
# PROJECT PATHS
# ============================================================

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

dataset_path = os.path.join(
    project_root,
    "dataset",
    "raw",
    "set-a"
)

output_path = os.path.join(
    project_root,
    "results",
    "week6",
    "cwt_configuration_analysis"
)

os.makedirs(output_path, exist_ok=True)


# ============================================================
# EXPERIMENTAL RECORDS
# ============================================================

records = {
    "cluster_0": "2357247",
    "cluster_1": "1192637",
    "cluster_2": "2989538"
}


# ============================================================
# CWT CONFIGURATIONS
# ============================================================

configurations = {
    "baseline": {
        "min_frequency": 1,
        "max_frequency": 40,
        "num_frequencies": 100
    },

    "config_A": {
        "min_frequency": 1,
        "max_frequency": 20,
        "num_frequencies": 100
    },

    "config_B": {
        "min_frequency": 1,
        "max_frequency": 40,
        "num_frequencies": 50
    },

    "config_C": {
        "min_frequency": 1,
        "max_frequency": 40,
        "num_frequencies": 150
    },

    "config_D": {
        "min_frequency": 1,
        "max_frequency": 50,
        "num_frequencies": 100
    }
}


# ============================================================
# WAVELET
# ============================================================

wavelet = "morl"


# ============================================================
# PROCESS EACH RECORD
# ============================================================

results = []


for cluster_name, record_name in records.items():

    print("\n" + "=" * 70)
    print("Processing:", cluster_name)
    print("Record:", record_name)
    print("=" * 70)

    record = wfdb.rdrecord(
        os.path.join(dataset_path, record_name)
    )

    signal = record.p_signal
    fs = record.fs
    lead_names = record.sig_name

    # --------------------------------------------------------
    # Use Lead II
    # --------------------------------------------------------

    lead_index = list(lead_names).index("II")

    ecg_signal = signal[:, lead_index]

    print("Sampling frequency:", fs)
    print("Lead:", lead_names[lead_index])
    print("Signal length:", len(ecg_signal))


    # --------------------------------------------------------
    # Test every CWT configuration
    # --------------------------------------------------------

    for config_name, config in configurations.items():

        min_frequency = config["min_frequency"]
        max_frequency = config["max_frequency"]
        num_frequencies = config["num_frequencies"]

        frequencies = np.linspace(
            min_frequency,
            max_frequency,
            num_frequencies
        )

        central_frequency = pywt.central_frequency(
            wavelet
        )

        scales = (
            central_frequency * fs / frequencies
        )

        start_time = time.perf_counter()

        coefficients, actual_frequencies = pywt.cwt(
            ecg_signal,
            scales,
            wavelet,
            sampling_period=1 / fs
        )

        power = np.abs(coefficients)

        elapsed_time = (
            time.perf_counter() - start_time
        )

        # ----------------------------------------------------
        # Save scalogram
        # ----------------------------------------------------

        figure_path = os.path.join(
            output_path,
            f"{cluster_name}_{record_name}_{config_name}.png"
        )

        plt.figure(figsize=(10, 4))

        plt.imshow(
            power,
            extent=[
                0,
                len(ecg_signal) / fs,
                max_frequency,
                min_frequency
            ],
            aspect="auto"
        )

        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency (Hz)")

        plt.title(
            f"{cluster_name} | Record {record_name} | "
            f"{config_name} | Lead II"
        )

        plt.colorbar(
            label="Magnitude"
        )

        plt.savefig(
            figure_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        # ----------------------------------------------------
        # Store experiment result
        # ----------------------------------------------------

        results.append({
            "cluster": cluster_name,
            "record_id": record_name,
            "configuration": config_name,
            "min_frequency": min_frequency,
            "max_frequency": max_frequency,
            "num_frequencies": num_frequencies,
            "scalogram_rows": power.shape[0],
            "scalogram_columns": power.shape[1],
            "computation_time_seconds": elapsed_time
        })

        print(
            f"{config_name}: "
            f"shape={power.shape}, "
            f"time={elapsed_time:.4f}s"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

import pandas as pd

results_df = pd.DataFrame(results)

csv_path = os.path.join(
    output_path,
    "cwt_configuration_results.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("WEEK 6 - CWT CONFIGURATION ANALYSIS COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(output_path)

print("\nCSV file:")
print(csv_path)