import os
import numpy as np
import wfdb
import pywt
import tensorly as tl
from tensorly.decomposition import tucker


# --------------------------------------------------
# 1. PROJECT PATH
# --------------------------------------------------

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

record_path = os.path.join(
    project_root,
    "dataset",
    "raw",
    "set-a",
    "1002867"
)


# --------------------------------------------------
# 2. LOAD ECG
# --------------------------------------------------

signal, fields = wfdb.rdsamp(record_path)

fs = fields["fs"]
lead_names = fields["sig_name"]

print("\nECG INFORMATION")
print("----------------")
print("Original ECG shape:", signal.shape)
print("Sampling frequency:", fs)
print("Number of leads:", len(lead_names))
print("Leads:", lead_names)


# --------------------------------------------------
# 3. CWT PARAMETERS
# --------------------------------------------------

frequency_min = 1
frequency_max = 40
number_of_frequencies = 100

frequencies = np.linspace(
    frequency_min,
    frequency_max,
    number_of_frequencies
)

central_frequency = pywt.central_frequency("morl")

scales = (
    central_frequency * fs / frequencies
)


# --------------------------------------------------
# 4. CREATE SCALOGRAMS
# --------------------------------------------------

scalograms = []

for lead_index, lead_name in enumerate(lead_names):

    lead_signal = signal[:, lead_index]

    coefficients, actual_frequencies = pywt.cwt(
        lead_signal,
        scales,
        "morl",
        sampling_period=1 / fs
    )

    power = np.abs(coefficients)

    scalograms.append(power)

    print(
        f"{lead_name}:",
        power.shape
    )


scalograms = np.array(scalograms)


# --------------------------------------------------
# 5. CREATE 3D TENSOR
# --------------------------------------------------

tensor = np.transpose(
    scalograms,
    (2, 1, 0)
)

print("\n3D TENSOR")
print("---------")
print("Tensor shape:", tensor.shape)


# --------------------------------------------------
# 6. ENERGY RETENTION
# --------------------------------------------------

energy_threshold = 0.95


# --------------------------------------------------
# 7. FIND RANK FROM MODE-WISE SVD
# --------------------------------------------------

def find_rank_from_energy(matrix, threshold):

    singular_values = np.linalg.svd(
        matrix,
        compute_uv=False
    )

    energy = singular_values ** 2

    cumulative_energy = np.cumsum(energy)

    total_energy = cumulative_energy[-1]

    retained_energy = (
        cumulative_energy / total_energy
    )

    rank = np.searchsorted(
        retained_energy,
        threshold
    ) + 1

    return rank, retained_energy


# --------------------------------------------------
# 8. MODE 1: TIME
# --------------------------------------------------

time_mode = tensor.reshape(
    tensor.shape[0],
    -1
)

time_rank, time_energy = find_rank_from_energy(
    time_mode,
    energy_threshold
)


# --------------------------------------------------
# 9. MODE 2: FREQUENCY
# --------------------------------------------------

frequency_mode = np.moveaxis(
    tensor,
    1,
    0
)

frequency_mode = frequency_mode.reshape(
    tensor.shape[1],
    -1
)

frequency_rank, frequency_energy = find_rank_from_energy(
    frequency_mode,
    energy_threshold
)


# --------------------------------------------------
# 10. MODE 3: LEADS
# --------------------------------------------------

lead_mode = np.moveaxis(
    tensor,
    2,
    0
)

lead_mode = lead_mode.reshape(
    tensor.shape[2],
    -1
)

lead_rank, lead_energy = find_rank_from_energy(
    lead_mode,
    energy_threshold
)


# --------------------------------------------------
# 11. AUTOMATICALLY SELECTED RANK
# --------------------------------------------------

rank = [
    time_rank,
    frequency_rank,
    lead_rank
]

print("\nADAPTIVE RANK SELECTION")
print("-----------------------")
print("Energy threshold:", energy_threshold)
print("Automatically selected rank:", rank)


# --------------------------------------------------
# 12. TUCKER DECOMPOSITION
# --------------------------------------------------

core, factors = tucker(
    tensor,
    rank=rank,
    init="svd"
)


# --------------------------------------------------
# 13. TUCKER RESULTS
# --------------------------------------------------

print("\nTUCKER RESULTS")
print("--------------")

print(
    "Core tensor shape:",
    core.shape
)

print(
    "Number of factor matrices:",
    len(factors)
)

for factor_index, factor in enumerate(
    factors,
    start=1
):

    print(
        f"Factor matrix {factor_index} shape:",
        factor.shape
    )


# --------------------------------------------------
# 14. RECONSTRUCT TENSOR
# --------------------------------------------------

reconstructed_tensor = tl.tucker_to_tensor(
    (core, factors)
)

print("\nRECONSTRUCTION")
print("--------------")

print(
    "Reconstructed tensor shape:",
    reconstructed_tensor.shape
)


# --------------------------------------------------
# 15. RECONSTRUCTION ERROR
# --------------------------------------------------

difference = (
    tensor - reconstructed_tensor
)

reconstruction_error = (
    np.linalg.norm(difference)
    /
    np.linalg.norm(tensor)
)

print(
    "Reconstruction error:",
    reconstruction_error
)

print(
    "Retained information estimate:",
    1 - reconstruction_error
)