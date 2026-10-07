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
# 6. ACTUAL DIMENSIONS
# --------------------------------------------------

dimensions = tensor.shape

original_size = tensor.size

print("Original tensor elements:", original_size)


# --------------------------------------------------
# 7. GENERATE CANDIDATE RANKS
# --------------------------------------------------

def generate_candidates(dimension):

    fractions = np.array([
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
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


candidate_lists = []

for dimension in dimensions:

    candidate_lists.append(
        generate_candidates(dimension)
    )


print("\nCANDIDATE RANKS")
print("---------------")

for index, candidates in enumerate(
    candidate_lists,
    start=1
):

    print(
        f"Mode {index}:",
        candidates
    )


# --------------------------------------------------
# 8. TEST RANK COMBINATIONS
# --------------------------------------------------

number_of_tests = min(
    len(candidates)
    for candidates in candidate_lists
)

results = []


print("\nRANK ANALYSIS")
print("-------------")


for index in range(number_of_tests):

    rank = [
        int(candidate_lists[mode][index])
        for mode in range(len(dimensions))
    ]

    print(
        "\nTesting rank:",
        rank
    )


    # Tucker decomposition
    core, factors = tucker(
        tensor,
        rank=rank,
        init="svd"
    )


    # Reconstruction
    reconstructed_tensor = tl.tucker_to_tensor(
        (core, factors)
    )


    # Reconstruction error
    difference = (
        tensor - reconstructed_tensor
    )

    reconstruction_error = (
        np.linalg.norm(difference)
        /
        np.linalg.norm(tensor)
    )


    # Tucker representation size
    representation_size = core.size

    for factor in factors:

        representation_size += factor.size


    # Compression ratio
    compression_ratio = (
        original_size /
        representation_size
    )


    results.append({
        "rank": rank,
        "error": reconstruction_error,
        "size": representation_size,
        "compression": compression_ratio
    })


    print(
        "Error:",
        reconstruction_error
    )

    print(
        "Representation size:",
        representation_size
    )

    print(
        "Compression ratio:",
        compression_ratio
    )


# --------------------------------------------------
# 9. FIND PARETO-OPTIMAL RESULTS
# --------------------------------------------------

pareto_results = []

for current in results:

    dominated = False

    for other in results:

        if current is other:
            continue

        better_or_equal_error = (
            other["error"] <= current["error"]
        )

        better_or_equal_size = (
            other["size"] <= current["size"]
        )

        strictly_better = (
            other["error"] < current["error"]
            or
            other["size"] < current["size"]
        )

        if (
            better_or_equal_error
            and
            better_or_equal_size
            and
            strictly_better
        ):

            dominated = True
            break


    if not dominated:

        pareto_results.append(current)


# --------------------------------------------------
# 10. DISPLAY PARETO RESULTS
# --------------------------------------------------

print("\nPARETO-OPTIMAL RESULTS")
print("----------------------")

for result in pareto_results:

    print(
        "Rank:",
        result["rank"]
    )

    print(
        "Reconstruction error:",
        result["error"]
    )

    print(
        "Representation size:",
        result["size"]
    )

    print(
        "Compression ratio:",
        result["compression"]
    )

    print()