
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import wfdb
from tensorly.decomposition import tucker


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "raw" / "set-a"

RANK_RESULTS = (
    ROOT / "results" / "week2" /
    "rank_budget_all_records" / "multi_record_summary.csv"
)

OUT_DIR = ROOT / "results" / "week2" / "adaptive_feature_pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUT_DIR / "pilot_feature_checks.csv"

FIXED_RANK = (50, 20, 6)

# Common shape must contain all supported adaptive ranks.
# These limits match the rank configurations explored so far.
COMMON_SHAPE = (50, 36, 12)

FREQUENCIES = np.linspace(1, 40, 100)
WAVELET = "morl"


# --------------------------------------------------
# LOAD LABELS AND SAVED RANKS
# --------------------------------------------------

def read_record_list(filename):
    path = DATA_DIR / filename
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


acceptable = set(read_record_list("RECORDS-acceptable"))
unacceptable = set(read_record_list("RECORDS-unacceptable"))

rank_results = pd.read_csv(RANK_RESULTS)

# Select up to 3 records from each class that have saved results.
selected = []

for label, label_records in [
    ("Acceptable", acceptable),
    ("Unacceptable", unacceptable)
]:
    candidates = rank_results[
        (rank_results["label"] == label)
        & rank_results["record_id"].astype(str).isin(label_records)
    ]

    selected.extend(candidates.head(3).to_dict("records"))

if not selected:
    raise RuntimeError(
        "No matching records found in the saved rank results."
    )

print("Pilot records:", len(selected))
for row in selected:
    print(row["record_id"], row["label"], row["selected_rank"])


# --------------------------------------------------
# BUILD ECG TENSOR
# --------------------------------------------------

def make_tensor(record_id):
    signal, fields = wfdb.rdsamp(str(DATA_DIR / record_id))
    fs = fields["fs"]

    scales = pywt.central_frequency(WAVELET) * fs / FREQUENCIES

    scalograms = []

    for lead_index in range(signal.shape[1]):
        coefficients, _ = pywt.cwt(
            signal[:, lead_index],
            scales,
            WAVELET,
            sampling_period=1 / fs
        )

        scalograms.append(np.abs(coefficients).T)

    tensor = np.stack(scalograms, axis=2)
    return tensor.astype(np.float32)


# --------------------------------------------------
# PAD CORE WITHOUT TRUNCATING
# --------------------------------------------------

def pad_core(core, common_shape):
    if any(core.shape[i] > common_shape[i] for i in range(3)):
        raise ValueError(
            f"Core shape {core.shape} exceeds {common_shape}; "
            "refusing to truncate."
        )

    padded = np.zeros(common_shape, dtype=np.float32)
    slices = tuple(slice(0, size) for size in core.shape)
    padded[slices] = core
    return padded


# --------------------------------------------------
# PILOT CHECKS
# --------------------------------------------------

results = []

for row in selected:
    record_id = str(row["record_id"])
    label = row["label"]
    adaptive_rank = tuple(ast.literal_eval(row["selected_rank"]))

    print(f"\nProcessing {record_id} ({label})")

    tensor = make_tensor(record_id)

    fixed_core, _ = tucker(
        tensor,
        rank=list(FIXED_RANK),
        init="svd"
    )

    adaptive_core, _ = tucker(
        tensor,
        rank=list(adaptive_rank),
        init="svd"
    )

    fixed_vector = pad_core(fixed_core, COMMON_SHAPE).reshape(-1)
    adaptive_vector = pad_core(adaptive_core, COMMON_SHAPE).reshape(-1)

    if fixed_vector.shape != adaptive_vector.shape:
        raise RuntimeError("Feature lengths do not match.")

    if not (
        np.isfinite(fixed_vector).all()
        and np.isfinite(adaptive_vector).all()
    ):
        raise RuntimeError("Non-finite feature values detected.")

    results.append({
        "record_id": record_id,
        "label": label,
        "fixed_rank": str(FIXED_RANK),
        "adaptive_rank": str(adaptive_rank),
        "fixed_core_shape": str(fixed_core.shape),
        "adaptive_core_shape": str(adaptive_core.shape),
        "fixed_feature_count": len(fixed_vector),
        "adaptive_feature_count": len(adaptive_vector),
        "finite_features": True
    })

    print("Fixed core:", fixed_core.shape)
    print("Adaptive core:", adaptive_core.shape)
    print("Both vector lengths:", len(fixed_vector))
    print("Finite values: OK")


pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

print("\nPILOT COMPLETED")
print("Saved:", OUTPUT_FILE)
print(pd.DataFrame(results).to_string(index=False))
