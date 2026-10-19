
from pathlib import Path
import csv
import time
import random

import numpy as np
import pandas as pd
import pywt
import wfdb
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor
import traceback


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "raw" / "set-a"

OUT_DIR = ROOT / "results" / "week2" / "rank_budget_all_records"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUT_DIR / "multi_record_summary.csv"
HISTORY_FILE = OUT_DIR / "search_history.csv"
ERROR_FILE = OUT_DIR / "failed_records.csv"

FIXED_RANK = (50, 20, 6)
FREQUENCIES = np.linspace(1, 40, 100)
WAVELET = "morl"
# Maximum ranks compatible with the planned common feature shape.
MAX_ADAPTIVE_RANK = (50, 36, 12)

# Numerical comparison tolerance
TOLERANCE = 1e-8


# --------------------------------------------------
# READ LABELLED RECORD LISTS
# --------------------------------------------------

def read_records(filename):
    path = DATA_DIR / filename

    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


acceptable = read_records("RECORDS-acceptable")
unacceptable = read_records("RECORDS-unacceptable")

records = [(r, "Acceptable") for r in acceptable]
records += [(r, "Unacceptable") for r in unacceptable]

# Remove duplicate IDs without changing their first occurrence.
unique_records = {}
for record_id, label in records:
    unique_records.setdefault(record_id, label)

records = list(unique_records.items())

print("=" * 65)
print("ALL-RECORD ADAPTIVE TUCKER-RANK VALIDATION")
print("=" * 65)
print(f"Unique labelled records: {len(records)}")
print(f"Acceptable: {sum(label == 'Acceptable' for _, label in records)}")
print(f"Unacceptable: {sum(label == 'Unacceptable' for _, label in records)}")
print(f"Output folder: {OUT_DIR}")


# --------------------------------------------------
# BUILD ECG TENSOR: TIME x FREQUENCY x LEAD
# --------------------------------------------------

def load_tensor(record_id):
    record_path = DATA_DIR / record_id
    signal, fields = wfdb.rdsamp(str(record_path))

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

        # Each lead: time x frequency
        scalograms.append(np.abs(coefficients).T)

    tensor = np.stack(scalograms, axis=2)
    return tensor.astype(np.float32)


# --------------------------------------------------
# TUCKER METRICS
# --------------------------------------------------

def representation_size(shape, rank):
    core_size = int(np.prod(rank))
    factor_size = sum(
        int(shape[i]) * int(rank[i])
        for i in range(len(shape))
    )
    return core_size + factor_size


def evaluate(tensor, rank):
    core, factors = tucker(
        tensor,
        rank=list(rank),
        init="svd",
        tol=1e-5
    )

    reconstructed = tucker_to_tensor((core, factors))

    error = (
        np.linalg.norm(tensor - reconstructed)
        / (np.linalg.norm(tensor) + 1e-12)
    )

    size = representation_size(tensor.shape, rank)

    return {
        "rank": tuple(int(x) for x in rank),
        "error": float(error),
        "representation_size": int(size)
    }


# --------------------------------------------------
# GREEDY RANK-BUDGET SEARCH
# --------------------------------------------------

def neighbours(rank, shape, baseline_size, evaluated):
    candidates = set()
    rank = list(rank)

    for decrease_mode in range(3):
        for increase_mode in range(3):
            if decrease_mode == increase_mode:
                continue

            candidate = rank.copy()
            candidate[decrease_mode] -= 1
            candidate[increase_mode] += 1

            if any(
                candidate[i] < 1 or candidate[i] > min(shape[i], MAX_ADAPTIVE_RANK[i])
                for i in range(3)
            ):
                continue
            

            
                

            candidate = tuple(candidate)

            if candidate == tuple(rank) or candidate in evaluated:
                continue

            if representation_size(shape, candidate) >= baseline_size:
                continue

            candidates.add(candidate)

    return sorted(candidates)


def search_ranks(tensor):
    shape = tensor.shape

    baseline = evaluate(tensor, FIXED_RANK)
    baseline_size = baseline["representation_size"]

    evaluated = {FIXED_RANK: baseline}
    history = [{
        "rank": str(FIXED_RANK),
        "error": baseline["error"],
        "representation_size": baseline_size,
        "stage": "fixed_baseline"
    }]

    current_rank = FIXED_RANK
    current_result = baseline
    iteration = 0

    while True:
        candidate_ranks = neighbours(
            current_rank, shape, baseline_size, evaluated
        )

        if not candidate_ranks:
            break

        iteration += 1
        candidate_results = []

        for candidate_rank in candidate_ranks:
            result = evaluate(tensor, candidate_rank)
            evaluated[candidate_rank] = result

            history.append({
                "rank": str(candidate_rank),
                "error": result["error"],
                "representation_size": result["representation_size"],
                "stage": f"iteration_{iteration}"
            })

            candidate_results.append(result)

        # Move only when reconstruction error improves.
        improving = [
            result for result in candidate_results
            if result["error"] < current_result["error"] - TOLERANCE
        ]

        if not improving:
            break

        current_result = min(
            improving,
            key=lambda item: (
                item["error"],
                item["representation_size"]
            )
        )
        current_rank = current_result["rank"]

    # Only retain a result that improves error and size
    # relative to the fixed baseline.
    qualifying = [
        result for result in evaluated.values()
        if result["representation_size"] < baseline_size
        and result["error"] < baseline["error"] - TOLERANCE
    ]

    selected = min(
        qualifying,
        key=lambda item: (
            item["error"],
            item["representation_size"]
        )
    ) if qualifying else baseline

    return baseline, selected, history


# --------------------------------------------------
# RESUMABLE OUTPUT FILES
# --------------------------------------------------

SUMMARY_COLUMNS = [
    "record_id", "label", "fixed_rank", "selected_rank",
    "fixed_error", "selected_error",
    "fixed_representation_size", "selected_representation_size",
    "representation_reduction_percent",
    "error_reduction_percent", "improved_both", "elapsed_seconds"
]

HISTORY_COLUMNS = [
    "record_id", "rank", "error",
    "representation_size", "stage"
]

ERROR_COLUMNS = ["record_id", "label", "error_message"]


def completed_ids():
    if not SUMMARY_FILE.exists():
        return set()

    try:
        df = pd.read_csv(SUMMARY_FILE)
        return set(df["record_id"].astype(str))
    except (pd.errors.EmptyDataError, KeyError):
        return set()


def append_row(path, columns, row):
    write_header = not path.exists() or path.stat().st_size == 0

    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# --------------------------------------------------
# PROCESS ALL RECORDS
# --------------------------------------------------
# TEMPORARY: diagnose one failed record only

done = completed_ids()

# Load previously failed records
failed_ids = set()

if ERROR_FILE.exists():
    failed_df = pd.read_csv(ERROR_FILE)
    if "record_id" in failed_df.columns:
        failed_ids = set(failed_df["record_id"].astype(str))

# Retry only failed records not already completed successfully
records = [
    (record_id, label)
    for record_id, label in records
    if record_id in failed_ids and record_id not in done
]

total = len(records)
start_all = time.time()

print(f"Records to retry: {total}")

for index, (record_id, label) in enumerate(records, start=1):
    if record_id in done:
        continue

    start_record = time.time()

    print(f"\n[{index}/{total}] Record: {record_id} ({label})")

    try:
        tensor = load_tensor(record_id)
        print(f"Tensor shape: {tensor.shape}")

        baseline, selected, history = search_ranks(tensor)

        for item in history:
            append_row(
                HISTORY_FILE,
                HISTORY_COLUMNS,
                {"record_id": record_id, **item}
            )

        size_reduction = (
            100
            * (baseline["representation_size"]
               - selected["representation_size"])
            / baseline["representation_size"]
        )

        error_reduction = (
            100
            * (baseline["error"] - selected["error"])
            / (baseline["error"] + 1e-12)
        )

        improved_both = (
            selected["error"] < baseline["error"] - TOLERANCE
            and selected["representation_size"]
            < baseline["representation_size"]
        )

        row = {
            "record_id": record_id,
            "label": label,
            "fixed_rank": str(baseline["rank"]),
            "selected_rank": str(selected["rank"]),
            "fixed_error": baseline["error"],
            "selected_error": selected["error"],
            "fixed_representation_size": baseline["representation_size"],
            "selected_representation_size": selected["representation_size"],
            "representation_reduction_percent": size_reduction,
            "error_reduction_percent": error_reduction,
            "improved_both": improved_both,
            "elapsed_seconds": time.time() - start_record
        }

        append_row(SUMMARY_FILE, SUMMARY_COLUMNS, row)

        print(f"Fixed error:    {baseline['error']:.6f}")
        print(f"Selected rank:  {selected['rank']}")
        print(f"Selected error: {selected['error']:.6f}")
        print(f"Size reduction: {size_reduction:.2f}%")
        print(f"Improved both:  {improved_both}")
        print(f"Saved progress: {record_id}")

    except Exception as exc:
        traceback.print_exc()
        if ERROR_FILE.exists():
            failed_df = pd.read_csv(ERROR_FILE)
            if "record_id" in failed_df.columns:
                failed_df = failed_df[failed_df["record_id"].astype(str) != str(record_id)]
            else:
                 failed_df = pd.DataFrame(columns=ERROR_COLUMNS)
                 failed_df = pd.concat(
        [
            failed_df,
            pd.DataFrame([{
                "record_id": record_id,
                "label": label,
                "error_message": repr(exc)
            }])
        ],
        ignore_index=True
    )
    
        print(f"ERROR for {record_id}: {exc}")
        print("Failure recorded; continuing to next record.")


    if index % 10 == 0:
        print(
            f"\nProgress: {index}/{total} records visited; "
            f"elapsed {((time.time() - start_all) / 60):.1f} minutes"
        )


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

if SUMMARY_FILE.exists():
    summary = pd.read_csv(SUMMARY_FILE)

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)
    print(f"Successfully processed: {len(summary)}")
    print(f"Improved both error and size: {summary['improved_both'].sum()}")
    print(
        "Mean representation reduction: "
        f"{summary['representation_reduction_percent'].mean():.2f}%"
    )
    print(
        "Median representation reduction: "
        f"{summary['representation_reduction_percent'].median():.2f}%"
    )
    print(
        "Mean error reduction: "
        f"{summary['error_reduction_percent'].mean():.2f}%"
    )
    print(f"\nSummary saved to: {SUMMARY_FILE}")

if ERROR_FILE.exists():
    failures = pd.read_csv(ERROR_FILE)
    print(f"Recorded failures: {len(failures)}")
    print(f"Failure details: {ERROR_FILE}")

print("\nRun completed.")
