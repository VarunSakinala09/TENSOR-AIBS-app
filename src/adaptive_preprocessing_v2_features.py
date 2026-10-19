
import os
import numpy as np
import pandas as pd
import wfdb
import pywt
import tensorly as tl

from pathlib import Path
from scipy.signal import butter, sosfiltfilt
from tensorly.decomposition import tucker

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "raw" / "set-a"
OUTPUT_DIR = ROOT / "results" / "adaptive_preprocessing_v2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep the existing baseline configuration
FS_EXPECTED = 500
FREQUENCIES = np.linspace(1, 40, 100)
TUCKER_RANK = [50, 20, 6]

# Load labelled records
acceptable_file = DATA_DIR / "RECORDS-acceptable"
unacceptable_file = DATA_DIR / "RECORDS-unacceptable"

def read_record_list(path):
    with open(path, "r") as file:
        return [line.strip() for line in file if line.strip()]

records = (
    [(name, "Acceptable") for name in read_record_list(acceptable_file)]
    + [(name, "Unacceptable") for name in read_record_list(unacceptable_file)]
)
records.sort()

def extract_features(record_name, apply_filter):
    record_path = DATA_DIR / record_name
    signal, fields = wfdb.rdsamp(str(record_path))

    fs = fields["fs"]

    if fs != FS_EXPECTED:
        raise ValueError(
            f"{record_name}: expected {FS_EXPECTED} Hz, got {fs} Hz"
        )

    if signal.shape != (5000, 12):
        raise ValueError(
            f"{record_name}: unexpected ECG shape {signal.shape}"
        )

    # Optional high-pass filtering before CWT
    if apply_filter:
        sos = butter(
            4, 0.5, btype="highpass", fs=fs, output="sos"
        )
        signal = sosfiltfilt(sos, signal, axis=0)

    # Generate CWT scalograms for all 12 leads
    scales = pywt.central_frequency("morl") * fs / FREQUENCIES
    scalograms = []

    for lead in range(signal.shape[1]):
        coefficients, _ = pywt.cwt(
            signal[:, lead],
            scales,
            "morl",
            sampling_period=1 / fs
        )
        scalograms.append(np.abs(coefficients))

    # Leads x Frequency x Time -> Time x Frequency x Leads
    tensor = np.transpose(np.asarray(scalograms), (2, 1, 0))

    # Same experimental Tucker rank as the existing baseline
    core, _ = tucker(
        tensor,
        rank=TUCKER_RANK,
        init="svd"
    )

    return core.reshape(-1)

def generate_dataset(apply_filter, output_name):
    rows = []

    for index, (record_name, label) in enumerate(records, start=1):
        features = extract_features(record_name, apply_filter)

        if len(features) != 6000:
            raise ValueError(
                f"{record_name}: expected 6000 features, got {len(features)}"
            )

        rows.append([record_name, *features, label])

        if index % 50 == 0 or index == len(records):
            print(f"Processed {index}/{len(records)} records")

    columns = (
        ["record_id"]
        + [f"feature_{i}" for i in range(1, 6001)]
        + ["label"]
    )

    df = pd.DataFrame(rows, columns=columns)
    output_path = OUTPUT_DIR / output_name
    df.to_csv(output_path, index=False)

    print("\nSaved:", output_path)
    print("Dataset shape:", df.shape)
    print(df["label"].value_counts())
    print("Filtering applied:", apply_filter)

if __name__ == "__main__":
    print("Generating HIGH-PASS-FILTERED features...")
    generate_dataset(
        apply_filter=True,
        output_name="highpass_features_100levels.csv"
    )
