
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

scales = central_frequency * fs / frequencies


# --------------------------------------------------
# 4. CREATE SCALOGRAMS
# --------------------------------------------------

scalograms = []

for lead_index, lead_name in enumerate(lead_names):

    coefficients, actual_frequencies = pywt.cwt(
        signal[:, lead_index],
        scales,
        "morl",
        sampling_period=1 / fs
    )

    scalograms.append(np.abs(coefficients))

scalograms = np.asarray(scalograms)


# --------------------------------------------------
# 5. CREATE 3D TENSOR
# --------------------------------------------------

tensor = np.transpose(scalograms, (2, 1, 0))

time_dimension, frequency_dimension, lead_dimension = (
    tensor.shape
)

print("\n3D TENSOR")
print("---------")
print("Tensor shape:", tensor.shape)
print("Time:", time_dimension)
print("Frequency:", frequency_dimension)
print("Leads:", lead_dimension)
print("Original elements:", tensor.size)


# --------------------------------------------------
# 6. MODE-WISE ENERGY-BASED RANK SELECTION
# --------------------------------------------------

energy_threshold = 0.95


def find_rank_from_energy(matrix, threshold):

    singular_values = np.linalg.svd(
        matrix,
        compute_uv=False
    )

    energy = singular_values ** 2
    total_energy = energy.sum()

    if total_energy == 0:
        return 1, 1.0

    cumulative_energy = np.cumsum(energy) / total_energy

    rank = np.searchsorted(
        cumulative_energy,
        threshold
    ) + 1

    rank = min(rank, len(singular_values))

    return int(rank), float(cumulative_energy[rank - 1])


# Mode 1: time
time_mode = tensor.reshape(time_dimension, -1)

time_rank, time_retained = find_rank_from_energy(
    time_mode,
    energy_threshold
)


# Mode 2: frequency
frequency_mode = np.moveaxis(tensor, 1, 0)
frequency_mode = frequency_mode.reshape(frequency_dimension, -1)

frequency_rank, frequency_retained = find_rank_from_energy(
    frequency_mode,
    energy_threshold
)


# Mode 3: leads
lead_mode = np.moveaxis(tensor, 2, 0)
lead_mode = lead_mode.reshape(lead_dimension, -1)

lead_rank, lead_retained = find_rank_from_energy(
    lead_mode,
    energy_threshold
)


rank = [
    time_rank,
    frequency_rank,
    lead_rank
]

print("\nADAPTIVE RANK SELECTION")
print("-----------------------")
print("Energy threshold:", energy_threshold)
print("Selected rank:", rank)
print("Time-mode retained energy:", time_retained)
print("Frequency-mode retained energy:", frequency_retained)
print("Lead-mode retained energy:", lead_retained)


# --------------------------------------------------
# 7. TUCKER DECOMPOSITION
# --------------------------------------------------

core, factors = tucker(
    tensor,
    rank=rank,
    init="svd"
)


# --------------------------------------------------
# 8. RECONSTRUCT TENSOR
# --------------------------------------------------

reconstructed_tensor = tl.tucker_to_tensor(
    (core, factors)
)

difference = tensor - reconstructed_tensor

reconstruction_error = (
    np.linalg.norm(difference)
    / max(np.linalg.norm(tensor), 1e-12)
)


# --------------------------------------------------
# 9. REPRESENTATION SIZE AND COMPRESSION
# --------------------------------------------------

core_size = core.size

factor_sizes = [
    factor.size for factor in factors
]

representation_size = core_size + sum(factor_sizes)

compression_ratio = (
    tensor.size / representation_size
    if representation_size else 0
)

complexity = representation_size / tensor.size


# --------------------------------------------------
# 10. RESULTS
# --------------------------------------------------

print("\nTUCKER RESULTS")
print("--------------")
print("Core shape:", core.shape)

for index, factor in enumerate(factors, start=1):
    print(f"Factor {index} shape:", factor.shape)

print("Reconstructed shape:", reconstructed_tensor.shape)
print("Relative reconstruction error:", reconstruction_error)
print("Representation elements:", representation_size)
print("Compression ratio:", compression_ratio)
print("Normalized representation size:", complexity)

print(
    "\nNote: Mode-wise 95% energy retention does not guarantee "
    "95% total Tucker reconstruction energy."
)
