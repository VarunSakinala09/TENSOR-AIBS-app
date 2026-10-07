import os
import time

import numpy as np
import pandas as pd
import wfdb
import pywt
import tensorly as tl

from tensorly.decomposition import tucker


# ============================================================
# PROJECT PATHS
# ============================================================

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
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
    "cwt_tucker_analysis"
)

os.makedirs(output_path, exist_ok=True)


# ============================================================
# REPRESENTATIVE RECORDS
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
        "frequency_min": 1,
        "frequency_max": 40,
        "number_of_frequencies": 100
    },

    "config_A": {
        "frequency_min": 1,
        "frequency_max": 20,
        "number_of_frequencies": 100
    },

    "config_B": {
        "frequency_min": 1,
        "frequency_max": 40,
        "number_of_frequencies": 50
    },

    "config_C": {
        "frequency_min": 1,
        "frequency_max": 40,
        "number_of_frequencies": 150
    },

    "config_D": {
        "frequency_min": 1,
        "frequency_max": 50,
        "number_of_frequencies": 100
    }
}


# ============================================================
# BASELINE TUCKER COMPRESSION RATIOS
# ============================================================

time_compression_ratio = 50 / 5000
frequency_compression_ratio = 20 / 100
lead_compression_ratio = 6 / 12


# ============================================================
# WAVELET
# ============================================================

wavelet = "morl"


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# PROCESS RECORDS
# ============================================================

for cluster_name, record_name in records.items():

    print("\n" + "=" * 70)
    print("Cluster:", cluster_name)
    print("Record:", record_name)
    print("=" * 70)

    record_path = os.path.join(
        dataset_path,
        record_name
    )

    signal, fields = wfdb.rdsamp(record_path)

    fs = fields["fs"]
    lead_names = fields["sig_name"]

    print("Signal shape:", signal.shape)
    print("Sampling frequency:", fs)
    print("Number of leads:", len(lead_names))


    # ========================================================
    # TEST EACH CWT CONFIGURATION
    # ========================================================

    for config_name, config in configurations.items():

        print("\n" + "-" * 60)
        print("Configuration:", config_name)
        print("-" * 60)

        frequency_min = config["frequency_min"]
        frequency_max = config["frequency_max"]
        number_of_frequencies = (
            config["number_of_frequencies"]
        )

        # ----------------------------------------------------
        # FREQUENCIES
        # ----------------------------------------------------

        frequencies = np.linspace(
            frequency_min,
            frequency_max,
            number_of_frequencies
        )

        central_frequency = pywt.central_frequency(
            wavelet
        )

        scales = (
            central_frequency
            * fs
            / frequencies
        )


        # ----------------------------------------------------
        # CREATE SCALOGRAMS
        # ----------------------------------------------------

        scalograms = []

        start_cwt = time.perf_counter()

        for lead_index, lead_name in enumerate(
            lead_names
        ):

            lead_signal = signal[:, lead_index]

            coefficients, actual_frequencies = pywt.cwt(
                lead_signal,
                scales,
                wavelet,
                sampling_period=1 / fs
            )

            power = np.abs(coefficients)

            scalograms.append(power)


        scalograms = np.array(scalograms)

        cwt_time = (
            time.perf_counter() - start_cwt
        )


        # ----------------------------------------------------
        # CREATE TENSOR
        # ----------------------------------------------------

        tensor = np.transpose(
            scalograms,
            (2, 1, 0)
        )

        print(
            "Tensor shape:",
            tensor.shape
        )


        # ----------------------------------------------------
        # ADAPT TUCKER RANK TO TENSOR SIZE
        # ----------------------------------------------------

        time_rank = max(
            1,
            round(
                tensor.shape[0]
                * time_compression_ratio
            )
        )

        frequency_rank = max(
            1,
            round(
                tensor.shape[1]
                * frequency_compression_ratio
            )
        )

        lead_rank = max(
            1,
            round(
                tensor.shape[2]
                * lead_compression_ratio
            )
        )

        rank = [
            time_rank,
            frequency_rank,
            lead_rank
        ]

        print(
            "Tucker rank:",
            rank
        )


        # ----------------------------------------------------
        # TUCKER DECOMPOSITION
        # ----------------------------------------------------

        start_tucker = time.perf_counter()

        core, factors = tucker(
            tensor,
            rank=rank,
            init="svd"
        )

        tucker_time = (
            time.perf_counter()
            - start_tucker
        )


        print(
            "Core shape:",
            core.shape
        )


        # ----------------------------------------------------
        # RECONSTRUCTION
        # ----------------------------------------------------

        reconstructed_tensor = tl.tucker_to_tensor(
            (core, factors)
        )


        # ----------------------------------------------------
        # RECONSTRUCTION ERROR
        # ----------------------------------------------------

        numerator = np.linalg.norm(
            tensor - reconstructed_tensor
        )

        denominator = np.linalg.norm(
            tensor
        )

        reconstruction_error = (
            numerator / denominator
        )


        # ----------------------------------------------------
        # CORE FEATURES
        # ----------------------------------------------------

        number_of_features = (
            core.size
        )


        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        results.append({

            "cluster": cluster_name,

            "record_id": record_name,

            "configuration": config_name,

            "frequency_min": frequency_min,

            "frequency_max": frequency_max,

            "number_of_frequencies":
                number_of_frequencies,

            "tensor_time_dimension":
                tensor.shape[0],

            "tensor_frequency_dimension":
                tensor.shape[1],

            "tensor_lead_dimension":
                tensor.shape[2],

            "time_rank":
                time_rank,

            "frequency_rank":
                frequency_rank,

            "lead_rank":
                lead_rank,

            "core_features":
                number_of_features,

            "reconstruction_error":
                reconstruction_error,

            "cwt_time_seconds":
                cwt_time,

            "tucker_time_seconds":
                tucker_time,

            "total_processing_time_seconds":
                cwt_time + tucker_time
        })


        print(
            "Reconstruction error:",
            reconstruction_error
        )

        print(
            "Number of core features:",
            number_of_features
        )

        print(
            "CWT time:",
            round(cwt_time, 4),
            "seconds"
        )

        print(
            "Tucker time:",
            round(tucker_time, 4),
            "seconds"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_dataframe = pd.DataFrame(
    results
)

output_file = os.path.join(
    output_path,
    "cwt_tucker_results.csv"
)

results_dataframe.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("WEEK 6 - CWT + TUCKER ANALYSIS COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(output_file)

print("\nRESULT SUMMARY:")
print(
    results_dataframe[
        [
            "cluster",
            "configuration",
            "tensor_frequency_dimension",
            "frequency_rank",
            "core_features",
            "reconstruction_error"
        ]
    ].to_string(index=False)
)