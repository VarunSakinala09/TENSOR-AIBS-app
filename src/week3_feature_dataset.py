import os
import numpy as np
import pandas as pd
import wfdb
import pywt

import tensorly as tl
from tensorly.decomposition import tucker


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


# --------------------------------------------------
# READ DATASET LABELS
# --------------------------------------------------

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


# --------------------------------------------------
# SELECT 10 RECORDS FOR TESTING
# --------------------------------------------------

records = sorted(records)[:10]

print("\nSELECTED RECORDS")
print("----------------")

for record_name, label in records:
    print(record_name, "->", label)


# --------------------------------------------------
# CWT SETTINGS
# --------------------------------------------------

frequency_min = 1
frequency_max = 40
number_of_frequencies = 100

wavelet = "morl"


# --------------------------------------------------
# TUCKER SETTINGS
# --------------------------------------------------

rank = [50, 20, 6]


# --------------------------------------------------
# PROCESS ECG RECORDS
# --------------------------------------------------

all_features = []
all_labels = []
all_record_names = []


for record_number, (record_name, label) in enumerate(records, start=1):

    print("\n" + "=" * 50)
    print("Processing:", record_name)
    print("Label:", label)
    print("=" * 50)

    record_path = os.path.join(
        dataset_path,
        record_name
    )

    # Load ECG
    signal, fields = wfdb.rdsamp(record_path)

    fs = fields["fs"]
    lead_names = fields["sig_name"]

    print("Signal shape:", signal.shape)
    print("Sampling frequency:", fs)
    print("Number of leads:", len(lead_names))


    # --------------------------------------------------
    # CREATE CWT SCAlograms
    # --------------------------------------------------

    frequencies = np.linspace(
        frequency_min,
        frequency_max,
        number_of_frequencies
    )

    central_frequency = pywt.central_frequency(wavelet)

    scales = (
        central_frequency
        * fs
        / frequencies
    )


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
    # STORE RESULTS
    # --------------------------------------------------

    all_features.append(features)
    all_labels.append(label)
    all_record_names.append(record_name)


# --------------------------------------------------
# CREATE FEATURE DATAFRAME
# --------------------------------------------------

feature_array = np.array(all_features)

print("\n" + "=" * 50)
print("FINAL FEATURE DATASET")
print("=" * 50)

print("Feature array shape:", feature_array.shape)
print("Number of records:", feature_array.shape[0])
print("Number of features:", feature_array.shape[1])


# Create feature column names dynamically

feature_columns = [
    f"feature_{index + 1}"
    for index in range(feature_array.shape[1])
]


feature_dataframe = pd.DataFrame(
    feature_array,
    columns=feature_columns
)


# Add record ID and label

feature_dataframe.insert(
    0,
    "record_id",
    all_record_names
)

feature_dataframe["label"] = all_labels


# --------------------------------------------------
# SAVE CSV
# --------------------------------------------------

output_file = os.path.join(
    results_path,
    "features_10_records.csv"
)

feature_dataframe.to_csv(
    output_file,
    index=False
)


print("\nFeature dataset saved to:")

print(output_file)

print("\nDataset shape including record_id and label:")

print(feature_dataframe.shape)

print("\nFirst few rows:")

print(feature_dataframe.iloc[:, :8])