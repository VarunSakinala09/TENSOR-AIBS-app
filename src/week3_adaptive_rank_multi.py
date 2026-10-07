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

dataset_path = os.path.join(
    project_root,
    "dataset",
    "raw",
    "set-a"
)


# --------------------------------------------------
# 2. FIND AVAILABLE ECG RECORDS
# --------------------------------------------------

record_files = []

for filename in os.listdir(dataset_path):

    if filename.endswith(".hea"):

        record_name = os.path.splitext(filename)[0]

        record_files.append(record_name)


record_files = sorted(record_files)


print("\nDATASET INFORMATION")
print("-------------------")
print("Available records:", len(record_files))


# --------------------------------------------------
# 3. SELECT A SMALL DATA-DRIVEN SAMPLE
# --------------------------------------------------

number_of_records = min(
    10,
    len(record_files)
)

selected_records = record_files[:number_of_records]

print(
    "Records selected for experiment:",
    len(selected_records)
)


# --------------------------------------------------
# 4. CWT SETTINGS
# --------------------------------------------------

frequency_min = 1
frequency_max = 40
number_of_frequencies = 100


# --------------------------------------------------
# 5. RANK CANDIDATE FUNCTION
# --------------------------------------------------

def generate_candidates(dimension):

    fractions = np.array([
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30
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


# --------------------------------------------------
# 6. PROCESS EACH RECORD
# --------------------------------------------------

all_results = []


for record_index, record_name in enumerate(
    selected_records,
    start=1
):

    print("\n====================================")
    print(
        "Processing record:",
        record_name
    )
    print(
        f"Progress: {record_index}/{len(selected_records)}"
    )
    print("====================================")


    # --------------------------------------------------
    # Load ECG
    # --------------------------------------------------

    record_path = os.path.join(
        dataset_path,
        record_name
    )

    signal, fields = wfdb.rdsamp(
        record_path
    )

    fs = fields["fs"]
    lead_names = fields["sig_name"]


    print(
        "ECG shape:",
        signal.shape
    )

    print(
        "Sampling frequency:",
        fs
    )

    print(
        "Number of leads:",
        len(lead_names)
    )


    # --------------------------------------------------
    # Create CWT scales
    # --------------------------------------------------

    frequencies = np.linspace(
        frequency_min,
        frequency_max,
        number_of_frequencies
    )

    central_frequency = pywt.central_frequency(
        "morl"
    )

    scales = (
        central_frequency
        * fs
        / frequencies
    )


    # --------------------------------------------------
    # Create scalograms
    # --------------------------------------------------

    scalograms = []


    for lead_index in range(
        len(lead_names)
    ):

        lead_signal = signal[
            :,
            lead_index
        ]

        coefficients, actual_frequencies = pywt.cwt(
            lead_signal,
            scales,
            "morl",
            sampling_period=1 / fs
        )

        scalograms.append(
            np.abs(coefficients)
        )


    scalograms = np.array(
        scalograms
    )


    # --------------------------------------------------
    # Create tensor
    # --------------------------------------------------

    tensor = np.transpose(
        scalograms,
        (2, 1, 0)
    )


    print(
        "Tensor shape:",
        tensor.shape
    )


    # --------------------------------------------------
    # Generate candidates from actual dimensions
    # --------------------------------------------------

    dimensions = tensor.shape

    candidate_lists = []

    for dimension in dimensions:

        candidate_lists.append(
            generate_candidates(
                dimension
            )
        )


    number_of_tests = min(
        len(candidates)
        for candidates in candidate_lists
    )


    # --------------------------------------------------
    # Rank analysis
    # --------------------------------------------------

    record_results = []


    for candidate_index in range(
        number_of_tests
    ):

        rank = [
            int(
                candidate_lists[mode][
                    candidate_index
                ]
            )
            for mode in range(
                len(dimensions)
            )
        ]


        # Tucker decomposition

        core, factors = tucker(
            tensor,
            rank=rank,
            init="svd"
        )


        # Reconstruction

        reconstructed_tensor = (
            tl.tucker_to_tensor(
                (core, factors)
            )
        )


        # Reconstruction error

        difference = (
            tensor
            - reconstructed_tensor
        )

        reconstruction_error = (
            np.linalg.norm(
                difference
            )
            /
            np.linalg.norm(
                tensor
            )
        )


        # Tucker representation size

        representation_size = core.size

        for factor in factors:

            representation_size += factor.size


        # Compression ratio

        compression_ratio = (
            tensor.size
            /
            representation_size
        )


        record_results.append({
            "rank": rank,
            "error": reconstruction_error,
            "size": representation_size,
            "compression": compression_ratio
        })


    # --------------------------------------------------
    # Find best reconstruction candidate
    # --------------------------------------------------

    best_result = min(
        record_results,
        key=lambda result: result["error"]
    )


    # --------------------------------------------------
    # Store result
    # --------------------------------------------------

    all_results.append({
        "record": record_name,
        "tensor_shape": tensor.shape,
        "rank": best_result["rank"],
        "error": best_result["error"],
        "compression": best_result["compression"]
    })


    print(
        "Selected rank based on lowest tested error:",
        best_result["rank"]
    )

    print(
        "Reconstruction error:",
        best_result["error"]
    )

    print(
        "Compression ratio:",
        best_result["compression"]
    )


# --------------------------------------------------
# 7. FINAL MULTI-RECORD RESULTS
# --------------------------------------------------

print("\n\nMULTI-RECORD ADAPTIVE ANALYSIS")
print("===============================")

for result in all_results:

    print(
        "\nRecord:",
        result["record"]
    )

    print(
        "Tensor shape:",
        result["tensor_shape"]
    )

    print(
        "Selected rank:",
        result["rank"]
    )

    print(
        "Reconstruction error:",
        result["error"]
    )

    print(
        "Compression ratio:",
        result["compression"]
    )