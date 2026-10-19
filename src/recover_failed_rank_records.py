
import csv
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import wfdb
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "raw" / "set-a"
OUT_DIR = ROOT / "results" / "week2" / "rank_budget_recovery"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUT_DIR / "recovered_records.csv"
ERROR_FILE = OUT_DIR / "recovery_errors.csv"

RECORDS = [
    ("2930238", "Unacceptable"),
    ("2966470", "Unacceptable"),
    ("2984959", "Unacceptable"),
]

FIXED_RANK = (50, 20, 6)
MAX_RANK = (50, 36, 12)
FREQUENCIES = np.linspace(1, 40, 100)
WAVELET = "morl"
TOLERANCE = 1e-8

SUMMARY_COLUMNS = [
    "record_id", "label", "fixed_rank", "selected_rank",
    "fixed_error", "selected_error",
    "fixed_representation_size", "selected_representation_size",
    "representation_reduction_percent",
    "error_reduction_percent", "improved_both", "elapsed_seconds"
]

ERROR_COLUMNS = ["record_id", "label", "error_message"]

def load_tensor(record_id):
    signal, fields = wfdb.rdsamp(str(DATA_DIR / record_id))
    scales = pywt.central_frequency(WAVELET) * fields["fs"] / FREQUENCIES

    leads = []
    for i in range(signal.shape[1]):
        coefficients, _ = pywt.cwt(
            signal[:, i], scales, WAVELET,
            sampling_period=1 / fields["fs"]
        )
        leads.append(np.abs(coefficients).T)

    return np.stack(leads, axis=2).astype(np.float32)

def representation_size(shape, rank):
    return int(np.prod(rank) + sum(shape[i] * rank[i] for i in range(3)))

def evaluate(tensor, rank):
    core, factors = tucker(
        tensor, rank=list(rank), init="svd", tol=1e-5
    )
    reconstructed = tucker_to_tensor((core, factors))

    error = np.linalg.norm(tensor - reconstructed) / (
        np.linalg.norm(tensor) + 1e-12
    )

    if tuple(core.shape) != tuple(rank):
        raise RuntimeError(
            f"Requested rank {rank}, actual core shape {core.shape}"
        )

    return {
        "rank": tuple(rank),
        "error": float(error),
        "representation_size": representation_size(tensor.shape, rank)
    }

def neighbours(rank, shape, baseline_size, evaluated):
    candidates = set()

    for decrease_mode in range(3):
        for increase_mode in range(3):
            if decrease_mode == increase_mode:
                continue

            candidate = list(rank)
            candidate[decrease_mode] -= 1
            candidate[increase_mode] += 1

            if any(
                candidate[i] < 1
                or candidate[i] > min(shape[i], MAX_RANK[i])
                for i in range(3)
            ):
                continue

            candidate = tuple(candidate)

            if candidate in evaluated:
                continue

            if representation_size(shape, candidate) >= baseline_size:
                continue

            candidates.add(candidate)

    return sorted(candidates)

def search_ranks(tensor):
    shape = tensor.shape
    baseline = evaluate(tensor, FIXED_RANK)
    evaluated = {FIXED_RANK: baseline}

    current_rank = FIXED_RANK
    current_result = baseline

    while True:
        candidates = neighbours(
            current_rank, shape, baseline["representation_size"], evaluated
        )

        if not candidates:
            break

        improving = []

        for candidate in candidates:
            print("    Testing rank:", candidate, flush=True)
            result = evaluate(tensor, candidate)
            evaluated[candidate] = result

            if result["error"] < current_result["error"] - TOLERANCE:
                improving.append(result)

        if not improving:
            break

        current_result = min(
            improving,
            key=lambda item: (item["error"], item["representation_size"])
        )
        current_rank = current_result["rank"]

    qualifying = [
        result for result in evaluated.values()
        if result["representation_size"] < baseline["representation_size"]
        and result["error"] < baseline["error"] - TOLERANCE
    ]

    selected = min(
        qualifying,
        key=lambda item: (item["error"], item["representation_size"])
    ) if qualifying else baseline

    return baseline, selected

def append_row(path, columns, row):
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def completed_ids():
    if not SUMMARY_FILE.exists():
        return set()
    return set(pd.read_csv(SUMMARY_FILE)["record_id"].astype(str))

done = completed_ids()

for record_id, label in RECORDS:
    if record_id in done:
        print(f"\nAlready recovered: {record_id}")
        continue

    print(f"\nRecovering {record_id} ({label})", flush=True)
    start = time.time()

    try:
        tensor = load_tensor(record_id)
        baseline, selected = search_ranks(tensor)

        row = {
            "record_id": record_id,
            "label": label,
            "fixed_rank": str(baseline["rank"]),
            "selected_rank": str(selected["rank"]),
            "fixed_error": baseline["error"],
            "selected_error": selected["error"],
            "fixed_representation_size": baseline["representation_size"],
            "selected_representation_size": selected["representation_size"],
            "representation_reduction_percent": 100 * (
                baseline["representation_size"] -
                selected["representation_size"]
            ) / baseline["representation_size"],
            "error_reduction_percent": 100 * (
                baseline["error"] - selected["error"]
            ) / (baseline["error"] + 1e-12),
            "improved_both": (
                selected["error"] < baseline["error"] - TOLERANCE
                and selected["representation_size"] <
                baseline["representation_size"]
            ),
            "elapsed_seconds": time.time() - start,
        }

        append_row(SUMMARY_FILE, SUMMARY_COLUMNS, row)
        print("Recovered:", record_id, row["selected_rank"], flush=True)

    except Exception as exc:
        traceback.print_exc()
        append_row(ERROR_FILE, ERROR_COLUMNS, {
            "record_id": record_id,
            "label": label,
            "error_message": repr(exc),
        })
        print("Failed:", record_id, exc, flush=True)

print("\nRecovery finished.")
print("Recovered results:", SUMMARY_FILE)
print("Recovery errors:", ERROR_FILE)
