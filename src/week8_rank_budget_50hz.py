
import csv
import hashlib
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import wfdb
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor


# ==================================================
# 1. SETTINGS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "raw" / "set-a"

GROUP_FILE = (
    ROOT / "results" / "week4" / "group_aware_cv"
    / "record_signal_groups.csv"
)

OUT_DIR = ROOT / "results" / "week8" / "rank_budget_50hz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUT_DIR / "rank_summary_50hz.csv"
HISTORY_FILE = OUT_DIR / "rank_history_50hz.csv"
FAILED_FILE = OUT_DIR / "failed_records_50hz.csv"

FIXED_RANK = (50, 20, 6)
MAX_RANK = (50, 36, 12)
FREQUENCIES = np.linspace(1, 50, 100)
WAVELET = "morl"
TOLERANCE = 1e-8

SUMMARY_COLUMNS = [
    "record_id", "label", "signal_hash",
    "fixed_rank", "selected_rank",
    "fixed_error", "selected_error",
    "fixed_representation_size", "selected_representation_size",
    "representation_reduction_percent",
    "error_reduction_percent", "improved_both",
    "elapsed_seconds"
]

HISTORY_COLUMNS = [
    "record_id", "rank", "error", "representation_size", "stage"
]

FAILED_COLUMNS = ["record_id", "label", "error_message"]


# ==================================================
# 2. LOAD RECORDS AND GROUPS
# ==================================================

def load_record_list():
    if not GROUP_FILE.exists():
        raise FileNotFoundError(f"Missing group file: {GROUP_FILE}")

    groups = pd.read_csv(GROUP_FILE, dtype={"record_id": str})
    groups["record_id"] = groups["record_id"].str.strip()

    if groups["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs in group mapping.")

    required = {"record_id", "label", "signal_hash"}
    if not required.issubset(groups.columns):
        raise ValueError(f"Group file must contain: {sorted(required)}")

    return groups


def find_record_path(record_id):
    headers = list(DATA_DIR.rglob(f"{record_id}.hea"))

    for header in headers:
        if header.with_suffix(".dat").exists():
            return header.with_suffix("")

    raise FileNotFoundError(f"No complete WFDB record found: {record_id}")


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# ==================================================
# 3. BUILD THE 1–50 HZ TENSOR
# ==================================================

def load_tensor(record_id, expected_hash):
    record_path = find_record_path(record_id)

    # Confirm that the selected raw file matches the audited group.
    actual_hash = sha256_file(record_path.with_suffix(".dat"))

    if actual_hash != expected_hash:
        raise ValueError(
            f"{record_id}: raw signal hash does not match group mapping."
        )

    signal, fields = wfdb.rdsamp(str(record_path))
    fs = fields["fs"]

    if signal.shape != (5000, 12):
        raise ValueError(
            f"{record_id}: unexpected signal shape {signal.shape}"
        )

    if not np.isfinite(signal).all():
        raise ValueError(f"{record_id}: ECG contains non-finite values")

    scales = pywt.central_frequency(WAVELET) * fs / FREQUENCIES
    scalograms = []

    for lead in range(signal.shape[1]):
        coefficients, _ = pywt.cwt(
            signal[:, lead],
            scales,
            WAVELET,
            sampling_period=1 / fs
        )
        scalograms.append(np.abs(coefficients).T)

    tensor = np.stack(scalograms, axis=2).astype(np.float32)

    if tensor.shape != (5000, 100, 12):
        raise ValueError(f"Unexpected CWT tensor shape: {tensor.shape}")

    if not np.isfinite(tensor).all():
        raise ValueError("CWT tensor contains non-finite values")

    return tensor


# ==================================================
# 4. TUCKER METRICS AND SEARCH
# ==================================================

def representation_size(shape, rank):
    return (
        int(np.prod(rank))
        + sum(int(dim) * int(r) for dim, r in zip(shape, rank))
    )


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
        / max(np.linalg.norm(tensor), 1e-12)
    )

    return {
        "rank": tuple(int(x) for x in rank),
        "error": float(error),
        "size": representation_size(tensor.shape, rank)
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
    baseline = evaluate(tensor, FIXED_RANK)
    baseline_size = baseline["size"]

    evaluated = {FIXED_RANK: baseline}
    history = [{
        "rank": str(FIXED_RANK),
        "error": baseline["error"],
        "representation_size": baseline_size,
        "stage": "fixed_baseline"
    }]

    current = baseline
    iteration = 0

    while True:
        candidate_ranks = neighbours(
            current["rank"], tensor.shape, baseline_size, evaluated
        )

        if not candidate_ranks:
            break

        iteration += 1
        candidate_results = []

        for candidate_rank in candidate_ranks:
            result = evaluate(tensor, candidate_rank)
            evaluated[candidate_rank] = result
            candidate_results.append(result)

            history.append({
                "rank": str(candidate_rank),
                "error": result["error"],
                "representation_size": result["size"],
                "stage": f"iteration_{iteration}"
            })

        improving = [
            result for result in candidate_results
            if result["error"] < current["error"] - TOLERANCE
        ]

        if not improving:
            break

        current = min(
            improving,
            key=lambda item: (item["error"], item["size"])
        )

    qualifying = [
        item for item in evaluated.values()
        if item["size"] < baseline_size
        and item["error"] < baseline["error"] - TOLERANCE
    ]

    selected = (
        min(qualifying, key=lambda item: (item["error"], item["size"]))
        if qualifying else baseline
    )

    return baseline, selected, history


# ==================================================
# 5. RESUMABLE CSV HELPERS
# ==================================================

def read_csv_or_empty(path, columns):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)

    return pd.read_csv(path, dtype={"record_id": str})


def append_row(path, columns, row):
    write_header = not path.exists() or path.stat().st_size == 0

    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


# ==================================================
# 6. PROCESS RECORDS
# ==================================================

def main():
    records = load_record_list()

    completed_df = read_csv_or_empty(SUMMARY_FILE, SUMMARY_COLUMNS)
    completed = set(completed_df["record_id"].astype(str))

    failed_df = read_csv_or_empty(FAILED_FILE, FAILED_COLUMNS)
    failed_ids = set(failed_df["record_id"].astype(str))

    # Resume unfinished records; do not repeat successful records.
    todo = records[~records["record_id"].isin(completed)].copy()

    print("=" * 65)
    print("WEEK 8: 1–50 HZ ADAPTIVE TUCKER RANK SEARCH")
    print("=" * 65)
    print("Labelled records:", len(records))
    print("Already completed:", len(completed))
    print("Records remaining:", len(todo))
    print("Output folder:", OUT_DIR)

    for index, row in enumerate(todo.itertuples(index=False), start=1):
        record_id = str(row.record_id)
        label = str(row.label)
        expected_hash = str(row.signal_hash)
        started = time.perf_counter()

        print(f"\n[{index}/{len(todo)}] {record_id} ({label})")

        try:
            tensor = load_tensor(record_id, expected_hash)
            print("Tensor shape:", tensor.shape)

            baseline, selected, history = search_ranks(tensor)

            for item in history:
                append_row(
                    HISTORY_FILE,
                    HISTORY_COLUMNS,
                    {"record_id": record_id, **item}
                )

            size_reduction = (
                100 * (baseline["size"] - selected["size"])
                / baseline["size"]
            )

            error_reduction = (
                100 * (baseline["error"] - selected["error"])
                / max(baseline["error"], 1e-12)
            )

            improved_both = (
                selected["error"] < baseline["error"] - TOLERANCE
                and selected["size"] < baseline["size"]
            )

            result = {
                "record_id": record_id,
                "label": label,
                "signal_hash": expected_hash,
                "fixed_rank": str(baseline["rank"]),
                "selected_rank": str(selected["rank"]),
                "fixed_error": baseline["error"],
                "selected_error": selected["error"],
                "fixed_representation_size": baseline["size"],
                "selected_representation_size": selected["size"],
                "representation_reduction_percent": size_reduction,
                "error_reduction_percent": error_reduction,
                "improved_both": improved_both,
                "elapsed_seconds": time.perf_counter() - started
            }

            append_row(SUMMARY_FILE, SUMMARY_COLUMNS, result)

            # Remove this record from the failure log after success.
            failed_df = read_csv_or_empty(FAILED_FILE, FAILED_COLUMNS)
            failed_df = failed_df[
                failed_df["record_id"].astype(str) != record_id
            ]
            failed_df.to_csv(FAILED_FILE, index=False)

            print("Selected rank:", selected["rank"])
            print(f"Fixed error: {baseline['error']:.6f}")
            print(f"Selected error: {selected['error']:.6f}")
            print(f"Representation reduction: {size_reduction:.2f}%")
            print("Improved both:", improved_both)

        except Exception as exc:
            traceback.print_exc()

            failed_df = read_csv_or_empty(FAILED_FILE, FAILED_COLUMNS)
            failed_df = failed_df[
                failed_df["record_id"].astype(str) != record_id
            ]

            failed_df = pd.concat([
                failed_df,
                pd.DataFrame([{
                    "record_id": record_id,
                    "label": label,
                    "error_message": repr(exc)
                }])
            ], ignore_index=True)

            failed_df.to_csv(FAILED_FILE, index=False)
            print("Failed; error saved. Continuing.")

    final = read_csv_or_empty(SUMMARY_FILE, SUMMARY_COLUMNS)

    print("\n" + "=" * 65)
    print("1–50 HZ RANK SEARCH SUMMARY")
    print("=" * 65)
    print("Successfully processed:", final["record_id"].nunique())

    if not final.empty:
        print("Duplicate result IDs:", final["record_id"].duplicated().sum())
        print("Improved both:", final["improved_both"].astype(bool).sum())
        print(
            "Mean representation reduction:",
            f"{final['representation_reduction_percent'].mean():.2f}%"
        )
        print(
            "Mean reconstruction-error reduction:",
            f"{final['error_reduction_percent'].mean():.2f}%"
        )

    failures = read_csv_or_empty(FAILED_FILE, FAILED_COLUMNS)
    print("Failed records:", failures["record_id"].nunique())
    print("Summary:", SUMMARY_FILE)
    print("History:", HISTORY_FILE)
    print("Failures:", FAILED_FILE)


if __name__ == "__main__":
    main()
