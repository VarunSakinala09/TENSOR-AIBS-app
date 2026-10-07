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

print("\nECG")
print("---")
print("Signal shape:", signal.shape)
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

central_frequency = pywt.central_frequency(
    "morl"
)

scales = (
    central_frequency
    * fs
    / frequencies
)


# --------------------------------------------------
# 4. CREATE SCALOGRAMS
# --------------------------------------------------

scalograms = []

for lead_index, lead_name in enumerate(
    lead_names
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

    power = np.abs(
        coefficients
    )

    scalograms.append(
        power
    )


scalograms = np.array(
    scalograms
)


# --------------------------------------------------
# 5. CREATE 3D TENSOR
# --------------------------------------------------

tensor = np.transpose(
    scalograms,
    (2, 1, 0)
)

print("\nTENSOR")
print("------")
print("Tensor shape:", tensor.shape)


# --------------------------------------------------
# 6. TUCKER RANK
# --------------------------------------------------

# Experimental rank for this first feature test
rank = [
    50,
    20,
    6
]


# --------------------------------------------------
# 7. TUCKER DECOMPOSITION
# --------------------------------------------------

core, factors = tucker(
    tensor,
    rank=rank,
    init="svd"
)


print("\nTUCKER")
print("------")
print("Core tensor shape:", core.shape)


# --------------------------------------------------
# 8. CONVERT CORE TO FEATURE VECTOR
# --------------------------------------------------

features = core.reshape(
    -1
)


# --------------------------------------------------
# 9. DISPLAY ACTUAL FEATURE INFORMATION
# --------------------------------------------------

print("\nFEATURE VECTOR")
print("--------------")

print(
    "Feature vector shape:",
    features.shape
)

print(
    "Number of features:",
    features.size
)

print(
    "First few feature values:",
    features[:10]
)