
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import pywt
import tensorly as tl

from tensorly.decomposition import tucker


# ==================================================
# 1. PATHS AND EXPERIMENT SETTINGS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "raw" / "set-a"

OUTPUT = ROOT / "results" / "week2" / "rank_budget_validation"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Representative records selected for validation
RECORDS = ["1002867", "2357247", "1192637", "2989538"]

# Existing fixed-rank baseline
FIXED_RANK = (50, 20, 6)

ERROR_TOLERANCE = 1e-8


# ==================================================
# 2. CREATE STANDARD CWT TENSOR
# ==================================================

def create_tensor(signal, fs):

    frequencies = np.linspace(1, 40, 100)
    scales = pywt.central_frequency("morl") * fs / frequencies

    scalograms = []

    for lead in range(signal.shape[1]):
        coefficients, _ = pywt.cwt(
            signal[:, lead],
            scales,
            "morl",
            sampling_period=1 / fs
        )

        # Convert frequency x time to time x frequency
        scalograms.append(np.abs(coefficients).T)

    return np.stack(scalograms, axis=2)


# ==================================================
# 3. REPRESENTATION SIZE
# ==================================================

def get_representation_size(shape, rank):

    core_elements = int(np.prod(rank))

    factor_elements = sum(
        dimension * component_rank
        for dimension, component_rank in zip(shape, rank)
    )

    return core_elements + factor_elements


# ==================================================
# 4. EVALUATE ONE RANK CONFIGURATION
# ==================================================

def evaluate(tensor, rank):

    start = time.perf_counter()

    core, factors = tucker(
        tensor,
        rank=tuple(rank),
        init="svd"
    )

    reconstructed = tl.tucker_to_tensor((core, factors))

    denominator = np.linalg.norm(tensor)

    error = (
        np.linalg.norm(tensor - reconstructed)
        / max(denominator, 1e-12)
    )

    return {
        "rank": tuple(rank),
        "error": float(error),
        "elements": get_representation_size(tensor.shape, rank),
        "seconds": time.perf_counter() - start
    }


# ==================================================
# 5. GREEDY RANK-REDISTRIBUTION SEARCH
# ==================================================

def search_ranks(tensor, record_id):

    baseline = evaluate(tensor, FIXED_RANK)
    baseline_error = baseline["error"]
    baseline_size = baseline["elements"]

    current = baseline
    current_rank = FIXED_RANK

    evaluated = {FIXED_RANK: baseline}
    history = []

    iteration = 0

    while True:

        neighbours = set()

        # Dynamically generate rank transfers between modes.
        for decrease_mode, increase_mode in itertools.permutations(
            range(tensor.ndim), 2
        ):

            candidate = list(current_rank)

            if candidate[decrease_mode] <= 1:
                continue

            if candidate[increase_mode] >= tensor.shape[increase_mode]:
                continue

            candidate[decrease_mode] -= 1
            candidate[increase_mode] += 1
            candidate = tuple(candidate)

            # Only investigate representations smaller than baseline.
            candidate_size = get_representation_size(
                tensor.shape, candidate
            )

            if candidate_size >= baseline_size:
                continue

            if candidate not in evaluated:
                neighbours.add(candidate)

        if not neighbours:
            break

        improved = []

        for candidate in sorted(neighbours):

            result = evaluate(tensor, candidate)
            evaluated[candidate] = result

            history.append({
                "record_id": record_id,
                "iteration": iteration + 1,
                "rank": str(candidate),
                "error": result["error"],
                "representation_elements": result["elements"],
                "decomposition_seconds": result["seconds"]
            })

            if result["error"] < current["error"] - ERROR_TOLERANCE:
                improved.append(result)

        if not improved:
            break

        # Greedy choice: lowest measured error among improving neighbours.
        current = min(
            improved,
            key=lambda item: (item["error"], item["elements"])
        )

        current_rank = current["rank"]
        iteration += 1

        print(
            f"  Iteration {iteration}: rank={current_rank}, "
            f"error={current['error']:.6f}, "
            f"elements={current['elements']}"
        )

    # Report the best configuration actually found.
    qualifies = [
        result for result in evaluated.values()
        if result["elements"] < baseline_size
        and result["error"] < baseline_error - ERROR_TOLERANCE
    ]

    if qualifies:
        best = min(
            qualifies,
            key=lambda item: (item["error"], item["elements"])
        )
    else:
        best = baseline

    summary = {
        "record_id": record_id,
        "fixed_rank": str(FIXED_RANK),
        "fixed_error": baseline_error,
        "fixed_elements": baseline_size,
        "selected_rank": str(best["rank"]),
        "selected_error": best["error"],
        "selected_elements": best["elements"],
        "error_change": best["error"] - baseline_error,
        "representation_reduction_percent": (
            100 * (baseline_size - best["elements"]) / baseline_size
        ),
        "lower_error_and_smaller": (
            best["error"] < baseline_error - ERROR_TOLERANCE
            and best["elements"] < baseline_size
        ),
        "unique_configurations_evaluated": len(evaluated)
    }

    return summary, history, list(evaluated.values())


# ==================================================
# 6. RUN VALIDATION
# ==================================================

summaries = []
all_history = []
all_configurations = []

for record_id in RECORDS:

    print("\n" + "=" * 60)
    print("RECORD:", record_id)
    print("=" * 60)

    record_path = DATASET / record_id
    signal, fields = wfdb.rdsamp(str(record_path))

    tensor = create_tensor(signal, fields["fs"])

    if tensor.shape[1] != 100 or tensor.shape[2] != 12:
        raise ValueError(
            f"Unexpected tensor shape for {record_id}: {tensor.shape}"
        )

    if any(
        rank > dimension
        for rank, dimension in zip(FIXED_RANK, tensor.shape)
    ):
        raise ValueError(
            f"Fixed rank is incompatible with record {record_id}"
        )

    print("Tensor shape:", tensor.shape)
    print("Fixed rank:", FIXED_RANK)

    summary, history, configurations = search_ranks(
        tensor, record_id
    )

    summaries.append(summary)
    all_history.extend(history)

    for result in configurations:
        all_configurations.append({
            "record_id": record_id,
            "rank": str(result["rank"]),
            "error": result["error"],
            "representation_elements": result["elements"],
            "decomposition_seconds": result["seconds"]
        })

    print("\nSelected rank:", summary["selected_rank"])
    print("Fixed error:", round(summary["fixed_error"], 6))
    print("Selected error:", round(summary["selected_error"], 6))
    print(
        "Representation reduction (%):",
        round(summary["representation_reduction_percent"], 3)
    )
    print(
        "Lower error and smaller representation:",
        summary["lower_error_and_smaller"]
    )


# ==================================================
# 7. SAVE RESULTS WITHOUT OVERWRITING OLD EXPERIMENTS
# ==================================================

pd.DataFrame(summaries).to_csv(
    OUTPUT / "multi_record_summary.csv",
    index=False
)

pd.DataFrame(all_history).to_csv(
    OUTPUT / "search_history.csv",
    index=False
)

pd.DataFrame(all_configurations).to_csv(
    OUTPUT / "evaluated_configurations.csv",
    index=False
)

print("\nVALIDATION COMPLETED")
print("Summary:", OUTPUT / "multi_record_summary.csv")
print("Search history:", OUTPUT / "search_history.csv")
print("All configurations:", OUTPUT / "evaluated_configurations.csv")
