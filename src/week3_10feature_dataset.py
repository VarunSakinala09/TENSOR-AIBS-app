import os
import numpy as np
import pandas as pd
import wfdb
import pywt

from tensorly.decomposition import tucker


# ==================================================
# PROJECT PATHS
# ==================================================

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

dataset_path = os.path.join(
    project_root,
    "dataset",
    "raw",
    "set-a"
)

results_path = os.path.join(
    project_root,
    "results",
    "week3"
)

os.makedirs(results_path, exist_ok=True)


# ==================================================
# READ DATASET LABELS
# ==================================================

acceptable_file = os.path.join(
    dataset_path,
    "RECORDS-acceptable"
)

unacceptable_file = os.path.join(
    dataset_path,
    "RECORDS-unacceptable"
)


with open(acceptable_file, "r") as file:
    acceptable_records = [
        line.strip()
        for line in file
        if line.strip()
    ]


with open(unacceptable_file, "r") as file:
    unacceptable_records = [
        line.strip()
        for line in file
        if line.strip()
    ]


records = []

for record_name in acceptable_records:
    records.append((record_name, "Acceptable"))

for record_name in unacceptable_records:
    records.append((record_name, "Unacceptable"))


records = sorted(records)


print("\nDATASET")
print("-------")
print("Total records:", len(records))
print("Acceptable:", len(acceptable_records))
print("Unacceptable:", len(unacceptable_records))


# ==================================================
# CWT SETTINGS
# ==================================================

frequency_min = 1
frequency_max = 40
number_of_frequencies = 100

wavelet = "morl"


# ==================================================
# BASELINE TUCKER RANK
# ==================================================

# Initial experimental rank.
# This is NOT the final adaptive rank.

rank = [50, 20, 6]


# ==================================================
# STORAGE
# ==================================================

all_features = []
all_record_names = []
all_labels = []


# ==================================================
# PROCESS RECORDS
# ==================================================

for record_index, (record_name, label) in enumerate(
    records,
    start=1
):

    print("\n" + "=" * 60)
    print(
        f"Processing {record_index}/{len(records)}: "
        f"{record_name}"
    )
    print("Label:", label)
    print("=" * 60)


    # --------------------------------------------------
    # LOAD ECG
    # --------------------------------------------------

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


    # --------------------------------------------------
    # CWT FREQUENCIES
    # --------------------------------------------------

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


    # --------------------------------------------------
    # CREATE SCALOGRAMS
    # --------------------------------------------------

    scalograms = []


    for lead_index, lead_name in enumerate(lead_names):

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


    # --------------------------------------------------
    # CREATE 3D TENSOR
    # --------------------------------------------------

    tensor = np.transpose(
        scalograms,
        (2, 1, 0)
    )

    print("Tensor shape:", tensor.shape)


    # --------------------------------------------------
    # TUCKER DECOMPOSITION
    # --------------------------------------------------

    core, factors = tucker(
        tensor,
        rank=rank,
        init="svd"
    )

    print("Core tensor shape:", core.shape)


    # --------------------------------------------------
    # CORE → FEATURE VECTOR
    # --------------------------------------------------

    features = core.reshape(-1)

    print("Feature vector shape:", features.shape)


    # --------------------------------------------------
    # STORE
    # --------------------------------------------------

    all_features.append(features)
    all_record_names.append(record_name)
    all_labels.append(label)


# ==================================================
# CREATE FEATURE MATRIX
# ==================================================

feature_array = np.array(all_features)

print("\n" + "=" * 60)
print("COMPLETE FEATURE DATASET")
print("=" * 60)

print("Feature matrix shape:", feature_array.shape)
print("Number of ECG records:", feature_array.shape[0])
print("Number of features per ECG:", feature_array.shape[1])


# ==================================================
# CREATE FEATURE COLUMN NAMES
# ==================================================

feature_columns = [
    f"feature_{index + 1}"
    for index in range(feature_array.shape[1])
]


feature_dataframe = pd.DataFrame(
    feature_array,
    columns=feature_columns
)


# ==================================================
# ADD RECORD IDs AND LABELS
# ==================================================

feature_dataframe.insert(
    0,
    "record_id",
    all_record_names
)

feature_dataframe["label"] = all_labels


# ==================================================
# SAVE
# ==================================================

output_file = os.path.join(
    results_path,
    "features_full_dataset.csv"
)

feature_dataframe.to_csv(
    output_file,
    index=False
)


print("\nFeature dataset saved to:")
print(output_file)

print("\nFinal dataframe shape:")
print(feature_dataframe.shape)