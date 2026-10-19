import os
import numpy as np
import wfdb
import pywt
import tensorly as tl
from tensorly.decomposition import tucker


# --------------------------------------------------
# 1. PROJECT PATH
# --------------------------------------------------

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
# 4. CREATE SCALOGRAMS FOR ALL LEADS
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


# Convert list to NumPy array
scalograms = np.array(scalograms)

print("\nSCALOGRAMS")
print("----------")
print("Scalograms array shape:", scalograms.shape)


# --------------------------------------------------
# 5. CONVERT TO 3D TENSOR
# --------------------------------------------------

tensor = np.transpose(
    scalograms,
    (2, 1, 0)
)

print("\n3D TENSOR")
print("---------")
print("Tensor shape:", tensor.shape)

print("Tensor dimensions:")
print("Time:", tensor.shape[0])
print("Frequency:", tensor.shape[1])
print("Leads:", tensor.shape[2])


# --------------------------------------------------
# 6. TUCKER DECOMPOSITION
# --------------------------------------------------

# Experimental Tucker ranks
time_rank = 50
frequency_rank = 20
lead_rank = 6

rank = [
    time_rank,
    frequency_rank,
    lead_rank
]

print("\nTUCKER DECOMPOSITION")
print("--------------------")
print("Experimental rank:", rank)


# Perform Tucker decomposition
core, factors = tucker(
    tensor,
    rank=rank,
    init="svd"
)


# --------------------------------------------------
# 7. DISPLAY ACTUAL TUCKER RESULTS
# --------------------------------------------------

print("\nTUCKER RESULTS")
print("--------------")

print("Core tensor shape:", core.shape)

print("Number of factor matrices:", len(factors))

for factor_index, factor in enumerate(factors, start=1):

    print(
        f"Factor matrix {factor_index} shape:",
        factor.shape
    )


# --------------------------------------------------
# 8. RECONSTRUCT TENSOR
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


