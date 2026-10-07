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


# --------------------------------------------------
# 3. CWT
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

    scalograms.append(
        np.abs(coefficients)
    )


scalograms = np.array(scalograms)


# --------------------------------------------------
# 5. CREATE TENSOR
# --------------------------------------------------

tensor = np.transpose(
    scalograms,
    (2, 1, 0)
)

print("\nTENSOR")
print("------")
print("Tensor shape:", tensor.shape)


# --------------------------------------------------
# 6. GET ACTUAL DIMENSIONS
# --------------------------------------------------

dimensions = tensor.shape

print("Actual dimensions:", dimensions)


# --------------------------------------------------
# 7. GENERATE RANK CANDIDATES
# --------------------------------------------------

def generate_candidates(dimension):

    fractions = np.array([
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50
    ])

    candidates = np.ceil(
        dimension * fractions
    ).astype(int)

    candidates = np.clip(
        candidates,
        1,
        dimension
    )

    return sorted(
        set(candidates.tolist())
    )


time_candidates = generate_candidates(
    dimensions[0]
)

frequency_candidates = generate_candidates(
    dimensions[1]
)

lead_candidates = generate_candidates(
    dimensions[2]
)


print("\nRANK CANDIDATES")
print("---------------")
print("Time:", time_candidates)
print("Frequency:", frequency_candidates)
print("Leads:", lead_candidates)


# --------------------------------------------------
# 8. TEST RANK COMBINATIONS
# --------------------------------------------------

results = []


# To keep computation reasonable,
# use corresponding candidate positions.

number_of_tests = min(
    len(time_candidates),
    len(frequency_candidates),
    len(lead_candidates)
)


for index in range(number_of_tests):

    current_rank = [
        time_candidates[index],
        frequency_candidates[index],
        lead_candidates[index]
    ]

    print(
        "\nTesting rank:",
        current_rank
    )

    core, factors = tucker(
        tensor,
        rank=current_rank,
        init="svd"
    )

    reconstructed_tensor = tl.tucker_to_tensor(
        (core, factors)
    )

    difference = (
        tensor - reconstructed_tensor
    )

    reconstruction_error = (
        np.linalg.norm(difference)
        /
        np.linalg.norm(tensor)
    )

    results.append(
        (
            current_rank,
            reconstruction_error
        )
    )

    print(
        "Reconstruction error:",
        reconstruction_error
    )


# --------------------------------------------------
# 9. FIND BEST RESULT
# --------------------------------------------------

best_result = min(
    results,
    key=lambda item: item[1]
)

best_rank = best_result[0]
best_error = best_result[1]


# --------------------------------------------------
# 10. FINAL RESULT
# --------------------------------------------------

print("\nRANK ANALYSIS")
print("-------------")

for rank, error in results:

    print(
        "Rank:",
        rank,
        "| Error:",
        error
    )


print("\nLOWEST RECONSTRUCTION ERROR")
print("----------------------------")
print("Rank:", best_rank)
print("Error:", best_error)