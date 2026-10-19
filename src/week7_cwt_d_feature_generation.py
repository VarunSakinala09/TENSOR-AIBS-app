import os
import wfdb
import numpy as np
import pandas as pd
import pywt
import tensorly as tl

from tensorly.decomposition import tucker


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "raw",
    "set-a"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "week7",
    "cwt_d"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD LABELLED RECORDS
# ============================================================

acceptable_file = os.path.join(
    DATASET_DIR,
    "RECORDS-acceptable"
)

unacceptable_file = os.path.join(
    DATASET_DIR,
    "RECORDS-unacceptable"
)


def read_record_list(file_path):
    with open(file_path, "r") as f:
        return [
            line.strip()
            for line in f
            if line.strip()
        ]


acceptable_records = read_record_list(acceptable_file)
unacceptable_records = read_record_list(unacceptable_file)

records = acceptable_records + unacceptable_records

print("=" * 70)
print("EXPERIMENT 7B-1")
print("CWT-D FEATURE GENERATION")
print("=" * 70)

print(f"Acceptable records   : {len(acceptable_records)}")
print(f"Unacceptable records : {len(unacceptable_records)}")
print(f"Total labelled       : {len(records)}")


# ============================================================
# CWT CONFIGURATION D
# ============================================================

FS = 500

MIN_FREQUENCY = 1
MAX_FREQUENCY = 50
NUMBER_OF_LEVELS = 100

frequencies = np.linspace(
    MIN_FREQUENCY,
    MAX_FREQUENCY,
    NUMBER_OF_LEVELS
)

wavelet = "morl"

central_frequency = pywt.central_frequency(wavelet)

scales = (
    central_frequency * FS
) / frequencies


# ============================================================
# TUCKER RANK
# ============================================================

# Same experimental rank used in the fixed baseline
TUCKER_RANK = [
    50,
    20,
    6
]


# ============================================================
# FUNCTION: LOAD ECG
# ============================================================

def load_ecg(record_id):

    record_path = os.path.join(
        DATASET_DIR,
        record_id
    )

    record = wfdb.rdrecord(record_path)

    signal = record.p_signal

    return signal


# ============================================================
# FUNCTION: CREATE CWT TENSOR
# ============================================================

def create_cwt_tensor(signal):

    lead_scalograms = []

    for lead_index in range(signal.shape[1]):

        lead_signal = signal[:, lead_index]

        coefficients, _ = pywt.cwt(
            lead_signal,
            scales,
            wavelet,
            sampling_period=1 / FS
        )

        scalogram = np.abs(coefficients)

        lead_scalograms.append(scalogram)

    # Current structure:
    # Lead × Frequency × Time
    tensor = np.stack(
        lead_scalograms,
        axis=0
    )

    # Convert to:
    # Time × Frequency × Lead

    tensor = np.transpose(
        tensor,
        (2, 1, 0)
    )

    return tensor


# ============================================================
# FUNCTION: TUCKER FEATURES
# ============================================================

def create_tucker_features(tensor):

    tensor = tl.tensor(
        tensor,
        dtype=tl.float32
    )

    core, factors = tucker(
        tensor,
        rank=TUCKER_RANK,
        init="svd"
    )

    features = np.asarray(
        core
    ).flatten()

    return features


# ============================================================
# PROCESS RECORDS
# ============================================================

all_features = []
processed_records = []

total_records = len(records)

for index, record_id in enumerate(records, start=1):

    print(
        f"[{index:4d}/{total_records}] "
        f"Processing {record_id}...",
        end=" ",
        flush=True
    )

    try:

        # ----------------------------------------------------
        # Load ECG
        # ----------------------------------------------------

        signal = load_ecg(record_id)

        # ----------------------------------------------------
        # CWT
        # ----------------------------------------------------

        tensor = create_cwt_tensor(
            signal
        )

        # ----------------------------------------------------
        # Tucker decomposition
        # ----------------------------------------------------

        features = create_tucker_features(
            tensor
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        all_features.append(
            features
        )

        processed_records.append(
            record_id
        )

        print(
            f"OK | Tensor {tensor.shape} | "
            f"Features {features.shape}"
        )

    except Exception as e:

        print(
            f"FAILED | {e}"
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

feature_matrix = np.vstack(
    all_features
)

print()
print("=" * 70)
print("FEATURE GENERATION COMPLETE")
print("=" * 70)

print(
    f"Feature matrix shape: {feature_matrix.shape}"
)


# ============================================================
# CREATE COLUMN NAMES
# ============================================================

feature_columns = [
    f"feature_{i}"
    for i in range(
        feature_matrix.shape[1]
    )
]


features_df = pd.DataFrame(
    feature_matrix,
    columns=feature_columns
)

features_df.insert(
    0,
    "record_id",
    processed_records
)


# ============================================================
# ADD LABELS
# ============================================================

label_dictionary = {}

for record_id in acceptable_records:
    label_dictionary[record_id] = "Acceptable"

for record_id in unacceptable_records:
    label_dictionary[record_id] = "Unacceptable"


features_df["label"] = (
    features_df["record_id"]
    .map(label_dictionary)
)


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "cwt_d_features.csv"
)

features_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print(f"Saved to:")
print(output_file)

print()
print("Final dataset shape:")
print(features_df.shape)

print()
print("Label distribution:")
print(
    features_df["label"].value_counts()
)

print()
print("Configuration:")
print(
    f"Frequency range : "
    f"{MIN_FREQUENCY}-{MAX_FREQUENCY} Hz"
)

print(
    f"CWT levels      : "
    f"{NUMBER_OF_LEVELS}"
)

print(
    f"Tucker rank     : "
    f"{TUCKER_RANK}"
)

print(
    f"Feature count   : "
    f"{feature_matrix.shape[1]}"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)