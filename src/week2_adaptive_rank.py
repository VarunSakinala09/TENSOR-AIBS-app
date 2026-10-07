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
# 6. GET ACTUAL DIMENSIONS
# --------------------------------------------------

time_dimension = tensor.shape[0]
frequency_dimension = tensor.shape[1]
lead_dimension = tensor.shape[2]

original_size = tensor.size

print("\nACTUAL DIMENSIONS")
print("-----------------")
print("Time:", time_dimension)
print("Frequency:", frequency_dimension)
print("Leads:", lead_dimension)
print("Original tensor elements:", original_size)


# --------------------------------------------------
# 7. GENERATE CANDIDATE RANKS
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
    time_dimension
)

frequency_candidates = generate_candidates(
    frequency_dimension
)

lead_candidates = generate_candidates(
    lead_dimension
)


# --------------------------------------------------
# 8. TEST SAME-POSITION CANDIDATES
# --------------------------------------------------

number_of_tests = min(
    len(time_candidates),
    len(frequency_candidates),
    len(lead_candidates)
)


results = []


print("\nADAPTIVE RANK ANALYSIS")
print("----------------------")


for index in range(number_of_tests):

    rank = [
        int(time_candidates[index]),
        int(frequency_candidates[index]),
        int(lead_candidates[index])
    ]

    print("\nTesting rank:", rank)


    # --------------------------------------------------
    # Tucker decomposition
    # --------------------------------------------------

    core, factors = tucker(
        tensor,
        rank=rank,
        init="svd"
    )


    # --------------------------------------------------
    # Reconstruction
    # --------------------------------------------------

    reconstructed_tensor = tl.tucker_to_tensor(
        (core, factors)
    )


    # --------------------------------------------------
    # Reconstruction error
    # --------------------------------------------------

    difference = (
        tensor - reconstructed_tensor
    )

    reconstruction_error = (
        np.linalg.norm(difference)
        /
        np.linalg.norm(tensor)
    )


    # --------------------------------------------------
    # Tucker representation size
    # --------------------------------------------------

    representation_size = core.size

    for factor in factors:

        representation_size += factor.size


    # --------------------------------------------------
    # Compression ratio
    # --------------------------------------------------

    compression_ratio = (
        original_size /
        representation_size
    )


    # --------------------------------------------------
    # Normalized complexity
    # --------------------------------------------------

    complexity = (
        representation_size /
        original_size
    )


    # --------------------------------------------------
    # Combined score
    # --------------------------------------------------

    score = (
        reconstruction_error +
        complexity
    )


    results.append({
        "rank": rank,
        "error": reconstruction_error,
        "representation_size": representation_size,
        "compression_ratio": compression_ratio,
        "complexity": complexity,
        "score": score
    })


    print(
        "Reconstruction error:",
        reconstruction_error
    )

    print(
        "Tucker representation size:",
        representation_size
    )

    print(
        "Compression ratio:",
        compression_ratio
    )

    print(
        "Combined score:",
        score
    )


# --------------------------------------------------
# 9. SELECT DATA-DRIVEN RANK
# --------------------------------------------------

best_result = min(
    results,
    key=lambda result: result["score"]
)


# --------------------------------------------------
# 10. FINAL RESULT
# --------------------------------------------------

print("\nFINAL ADAPTIVE RESULT")
print("---------------------")

print(
    "Selected rank:",
    best_result["rank"]
)

print(
    "Reconstruction error:",
    best_result["error"]
)

print(
    "Tucker representation size:",
    best_result["representation_size"]
)

print(
    "Compression ratio:",
    best_result["compression_ratio"]
)

print(
    "Combined score:",
    best_result["score"]
)