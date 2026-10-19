
import os
import time
import numpy as np
import pandas as pd
import wfdb
import pywt
import tensorly as tl

from pathlib import Path
from tensorly.decomposition import tucker


# ==================================================
# 1. PROJECT PATHS AND BASELINE
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "raw" / "set-a"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "week2"
    / "error_constrained_tucker"
)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Fixed rank is the reference baseline, not an adaptive candidate.
FIXED_RANK = (50, 20, 6)

ENERGY_THRESHOLD = 0.95
ERROR_TOLERANCE = 1e-5

# Validate on one record first. Expand only after validation.
RECORD_ID = "1002867"


# ==================================================
# 2. CREATE THE STANDARD CWT TENSOR
# ==================================================

def create_tensor(signal, fs):

    frequencies = np.linspace(1, 40, 100)

    scales = (
        pywt.central_frequency("morl")
        * fs
        / frequencies
    )

    scalograms = []

    for lead_index in range(signal.shape[1]):

        coefficients, _ = pywt.cwt(
            signal[:, lead_index],
            scales,
            "morl",
            sampling_period=1 / fs
        )

        # Convert (frequency, time) to (time, frequency)
        scalograms.append(np.abs(coefficients).T)

    # Standard order: Time x Frequency x Leads
    return np.stack(scalograms, axis=2)


# ==================================================
# 3. MODE UNFOLDINGS AND SINGULAR VALUES
# ==================================================

def calculate_mode_singular_values(tensor):

    time_dim, frequency_dim, lead_dim = tensor.shape

    unfoldings = [
        tensor.reshape(time_dim, -1),
        np.moveaxis(tensor, 1, 0).reshape(frequency_dim, -1),
        np.moveaxis(tensor, 2, 0).reshape(lead_dim, -1)
    ]

    singular_values = []

    for matrix in unfoldings:
        values = np.linalg.svd(
            matrix,
            compute_uv=False
        )
        singular_values.append(values)

    return singular_values


def rank_from_energy(singular_values, threshold):

    energy = singular_values ** 2
    total_energy = energy.sum()

    if total_energy <= 0:
        return 1

    cumulative_energy = np.cumsum(energy) / total_energy

    return int(
        np.searchsorted(cumulative_energy, threshold) + 1
    )


def omitted_energy(singular_values, rank):

    energy = singular_values ** 2
    total_energy = energy.sum()

    if total_energy <= 0:
        return 0.0

    retained = energy[:rank].sum()

    return float(max(0.0, 1.0 - retained / total_energy))


# ==================================================
# 4. TUCKER DECOMPOSITION AND METRICS
# ==================================================

def evaluate_rank(tensor, rank):

    start = time.perf_counter()

    core, factors = tucker(
        tensor,
        rank=tuple(rank),
        init="svd"
    )

    reconstructed = tl.tucker_to_tensor((core, factors))

    elapsed = time.perf_counter() - start

    tensor_norm = np.linalg.norm(tensor)

    error = (
        np.linalg.norm(tensor - reconstructed)
        / max(tensor_norm, 1e-12)
    )

    representation_size = (
        core.size
        + sum(factor.size for factor in factors)
    )

    return {
        "rank": tuple(rank),
        "error": float(error),
        "representation_size": int(representation_size),
        "compression_ratio": float(
            tensor.size / representation_size
        ),
        "decomposition_seconds": float(elapsed)
    }


# ==================================================
# 5. LOAD RECORD AND CHECK DIMENSIONS
# ==================================================

record_path = DATASET_PATH / RECORD_ID

signal, fields = wfdb.rdsamp(str(record_path))

tensor = create_tensor(signal, fields["fs"])

if tensor.ndim != 3:
    raise ValueError("Expected a 3D ECG tensor.")

if any(
    rank > dimension
    for rank, dimension in zip(FIXED_RANK, tensor.shape)
):
    raise ValueError(
        "Fixed baseline rank exceeds a tensor dimension."
    )

print("\nECG INFORMATION")
print("----------------")
print("Record:", RECORD_ID)
print("Original ECG shape:", signal.shape)
print("Tensor shape:", tensor.shape)
print("Expected axis order: Time x Frequency x Leads")
print("Original elements:", tensor.size)


# ==================================================
# 6. CALCULATE DATA-DRIVEN INITIAL RANKS
# ==================================================

mode_singular_values = calculate_mode_singular_values(tensor)

initial_ranks = [
    rank_from_energy(values, ENERGY_THRESHOLD)
    for values in mode_singular_values
]

# Keep the initial ranks within the fixed baseline limits.
# This allows a valid comparison against the baseline.
current_rank = [
    min(initial, maximum)
    for initial, maximum in zip(initial_ranks, FIXED_RANK)
]

print("\nINITIAL RANK SELECTION")
print("----------------------")
print("Energy threshold:", ENERGY_THRESHOLD)
print("Energy-derived ranks:", initial_ranks)
print("Starting ranks:", current_rank)
print("Fixed reference rank:", FIXED_RANK)


# ==================================================
# 7. MEASURE THE FIXED BASELINE
# ==================================================

fixed_result = evaluate_rank(tensor, FIXED_RANK)
fixed_error = fixed_result["error"]

print("\nFIXED BASELINE")
print("--------------")
print("Rank:", FIXED_RANK)
print("Relative reconstruction error:", fixed_error)
print("Representation elements:",
      fixed_result["representation_size"])


# ==================================================
# 8. ERROR-CONSTRAINED ADAPTIVE SEARCH
# ==================================================

history = []
iteration = 0

current_result = evaluate_rank(tensor, current_rank)


def save_iteration(rank, result, mode_energies):

    history.append({
        "iteration": iteration,
        "rank": str(tuple(rank)),
        "relative_error": result["error"],
        "baseline_error": fixed_error,
        "error_limit": fixed_error + ERROR_TOLERANCE,
        "representation_elements": result["representation_size"],
        "compression_ratio": result["compression_ratio"],
        "decomposition_seconds": result["decomposition_seconds"],
        "time_omitted_energy": mode_energies[0],
        "frequency_omitted_energy": mode_energies[1],
        "lead_omitted_energy": mode_energies[2]
    })


mode_energies = [
    omitted_energy(values, rank)
    for values, rank in zip(mode_singular_values, current_rank)
]

save_iteration(current_rank, current_result, mode_energies)

print("\nADAPTIVE RANK SEARCH")
print("--------------------")

while current_result["error"] > fixed_error + ERROR_TOLERANCE:

    # Identify modes that can still grow toward the fixed baseline.
    eligible_modes = [
        mode
        for mode in range(3)
        if current_rank[mode] < FIXED_RANK[mode]
    ]

    if not eligible_modes:
        break

    # Increase the rank of the mode with the greatest omitted
    # singular-value energy. No predefined candidate list.
    selected_mode = max(
        eligible_modes,
        key=lambda mode: mode_energies[mode]
    )

    current_rank[selected_mode] += 1
    iteration += 1

    # At the fixed baseline rank, reuse the measured baseline.
    if tuple(current_rank) == FIXED_RANK:
        current_result = fixed_result
    else:
        current_result = evaluate_rank(tensor, current_rank)

    mode_energies = [
        omitted_energy(values, rank)
        for values, rank in zip(mode_singular_values, current_rank)
    ]

    save_iteration(current_rank, current_result, mode_energies)

    print(
        f"Iteration {iteration}: "
        f"rank={tuple(current_rank)}, "
        f"error={current_result['error']:.6f}, "
        f"elements={current_result['representation_size']}"
    )


# ==================================================
# 9. SAVE AND REPORT RESULTS
# ==================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    OUTPUT_PATH / "adaptive_rank_search_history.csv",
    index=False
)

qualifies = (
    current_result["error"]
    <= fixed_error + ERROR_TOLERANCE
)

comparison = pd.DataFrame([
    {
        "method": "Fixed",
        **fixed_result
    },
    {
        "method": "Adaptive",
        **current_result
    }
])

comparison.to_csv(
    OUTPUT_PATH / "fixed_vs_adaptive.csv",
    index=False
)

print("\nFINAL COMPARISON")
print("----------------")
print("Fixed rank:", FIXED_RANK)
print("Fixed error:", fixed_error)
print("Adaptive rank:", tuple(current_rank))
print("Adaptive error:", current_result["error"])
print("Error constraint satisfied:", qualifies)
print("Adaptive representation elements:",
      current_result["representation_size"])
print("Fixed representation elements:",
      fixed_result["representation_size"])
print("Adaptive compression ratio:",
      current_result["compression_ratio"])

if qualifies:
    reduction = 100 * (
        1 - current_result["representation_size"]
        / fixed_result["representation_size"]
    )

    print("Representation reduction (%):", reduction)

    if reduction > 0:
        print("Result: error constraint satisfied with smaller representation.")
    else:
        print("Result: error constraint satisfied, but no compression advantage.")
else:
    print("Result: error constraint was not satisfied.")

print("\nResults saved in:", OUTPUT_PATH)
print("This experiment evaluates one ECG record only.")
