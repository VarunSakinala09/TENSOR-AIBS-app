
import time
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import pywt
import tensorly as tl

from tensorly.decomposition import tucker


# ==================================================
# 1. SETTINGS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "raw" / "set-a"

OUTPUT = ROOT / "results" / "week2" / "rank_budget_search"
OUTPUT.mkdir(parents=True, exist_ok=True)

RECORD_ID = "1002867"

# Existing fixed baseline
FIXED_RANK = (50, 20, 6)

# Numerical tolerance only; not an error-improvement allowance
TOLERANCE = 1e-8

# ==================================================
# 2. BUILD THE STANDARD CWT TENSOR
# ==================================================

record, fields = wfdb.rdsamp(str(DATASET / RECORD_ID))
fs = fields["fs"]

frequencies = np.linspace(1, 40, 100)
scales = pywt.central_frequency("morl") * fs / frequencies

scalograms = []

for lead in range(record.shape[1]):
    coefficients, _ = pywt.cwt(
        record[:, lead],
        scales,
        "morl",
        sampling_period=1 / fs
    )
    scalograms.append(np.abs(coefficients).T)

tensor = np.stack(scalograms, axis=2)

print("Record:", RECORD_ID)
print("Tensor shape:", tensor.shape)
print("Axis order: Time x Frequency x Leads")


# ==================================================
# 3. REPRESENTATION SIZE AND EVALUATION
# ==================================================

def representation_size(shape, rank):
    core_elements = int(np.prod(rank))
    factor_elements = sum(
        dimension * component_rank
        for dimension, component_rank in zip(shape, rank)
    )
    return core_elements + factor_elements


def evaluate(rank):
    start = time.perf_counter()

    core, factors = tucker(
        tensor,
        rank=tuple(rank),
        init="svd"
    )

    reconstructed = tl.tucker_to_tensor((core, factors))

    error = (
        np.linalg.norm(tensor - reconstructed)
        / max(np.linalg.norm(tensor), 1e-12)
    )

    return {
        "rank": tuple(rank),
        "error": float(error),
        "elements": representation_size(tensor.shape, rank),
        "seconds": time.perf_counter() - start
    }


# ==================================================
# 4. BASELINE
# ==================================================

baseline = evaluate(FIXED_RANK)
baseline_error = baseline["error"]
baseline_size = baseline["elements"]

print("\nFIXED BASELINE")
print("Rank:", FIXED_RANK)
print("Error:", baseline_error)
print("Representation elements:", baseline_size)


# ==================================================
# 5. DYNAMIC LOCAL RANK SEARCH
# ==================================================

# Each iteration generates rank changes automatically:
# one mode decreases by one while another increases by one.
# No manually specified candidate-rank list is used.

current_rank = tuple(FIXED_RANK)
current_result = baseline

evaluated = {current_rank: baseline}
history = []
iteration = 0

while True:
    neighbours = set()

    for decrease_mode, increase_mode in itertools.permutations(range(3), 2):
        candidate = list(current_rank)

        if candidate[decrease_mode] <= 1:
            continue

        if candidate[increase_mode] >= tensor.shape[increase_mode]:
            continue

        candidate[decrease_mode] -= 1
        candidate[increase_mode] += 1
        candidate = tuple(candidate)

        # Enforce a smaller representation than the baseline.
        if representation_size(tensor.shape, candidate) >= baseline_size:
            continue

        if candidate not in evaluated:
            neighbours.add(candidate)

    if not neighbours:
        print("\nNo unevaluated, smaller neighbouring ranks remain.")
        break

    results = []

    for candidate in sorted(neighbours):
        result = evaluate(candidate)
        evaluated[candidate] = result
        results.append(result)

        history.append({
            "iteration": iteration + 1,
            "rank": str(candidate),
            "error": result["error"],
            "elements": result["elements"],
            "seconds": result["seconds"]
        })

    # Move only to a candidate that improves reconstruction error.
    better = [
        result for result in results
        if result["error"] < current_result["error"] - TOLERANCE
    ]

    if not better:
        print("\nNo neighbouring configuration improved the current error.")
        break

    current_result = min(
        better,
        key=lambda result: (result["error"], result["elements"])
    )

    current_rank = current_result["rank"]
    iteration += 1

    print(
        f"Iteration {iteration}: "
        f"rank={current_rank}, "
        f"error={current_result['error']:.6f}, "
        f"elements={current_result['elements']}"
    )


# ==================================================
# 6. SAVE ALL EVALUATED CONFIGURATIONS
# ==================================================

results_df = pd.DataFrame([
    {
        "rank": str(rank),
        "error": result["error"],
        "representation_elements": result["elements"],
        "decomposition_seconds": result["seconds"],
        "smaller_than_baseline": result["elements"] < baseline_size,
        "lower_error_than_baseline": result["error"] < baseline_error
    }
    for rank, result in evaluated.items()
])

results_df.to_csv(
    OUTPUT / "evaluated_rank_configurations.csv",
    index=False
)

pd.DataFrame(history).to_csv(
    OUTPUT / "search_history.csv",
    index=False
)


# ==================================================
# 7. REPORT
# ==================================================

valid = results_df[
    results_df["smaller_than_baseline"]
    & results_df["lower_error_than_baseline"]
]

print("\nFINAL REPORT")
print("Baseline rank:", FIXED_RANK)
print("Baseline error:", baseline_error)
print("Baseline elements:", baseline_size)
print("Unique configurations evaluated:", len(evaluated))

if not valid.empty:
    best = valid.sort_values(
        ["error", "representation_elements"]
    ).iloc[0]

    print("\nA configuration satisfying both targets was found:")
    print("Rank:", best["rank"])
    print("Error:", best["error"])
    print("Representation elements:", best["representation_elements"])
else:
    print(
        "\nNo evaluated configuration satisfied both targets. "
        "This local search does not prove that no such configuration exists."
    )

print("\nResults saved to:", OUTPUT)
