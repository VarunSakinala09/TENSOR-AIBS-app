
import time
import numpy as np
import wfdb
import pywt
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor

# --------------------------------------------------
# 1. Load the same ECG record
# --------------------------------------------------

record_id = "1002867"
record_path = f"dataset/raw/set-a/{record_id}"

record = wfdb.rdrecord(record_path)
ecg = record.p_signal
fs = record.fs

print("\nECG INFORMATION")
print("----------------")
print("Record ID:", record_id)
print("ECG shape:", ecg.shape)
print("Sampling frequency:", fs)

# --------------------------------------------------
# 2. Convert ECG into a CWT tensor
# --------------------------------------------------

frequencies = np.linspace(1, 40, 100)
scales = pywt.central_frequency("morl") * fs / frequencies

scalograms = []

for lead in range(ecg.shape[1]):
    coefficients, _ = pywt.cwt(
        ecg[:, lead],
        scales,
        "morl",
        sampling_period=1 / fs
    )
    scalograms.append(np.abs(coefficients))

# Shape: Time x Frequency x Leads
tensor = np.stack(scalograms, axis=-1)
original_elements = tensor.size
original_norm = np.linalg.norm(tensor)

print("\nTENSOR INFORMATION")
print("------------------")
print("Tensor shape:", tensor.shape)
print("Original elements:", original_elements)

# --------------------------------------------------
# 3. Compare two Tucker rank configurations
# --------------------------------------------------

rank_configs = {
    "Fixed rank": [50, 20, 6],
    "Adaptive rank": [24, 8, 4]
}

results = []

for method, rank in rank_configs.items():

    start_time = time.perf_counter()

    core, factors = tucker(
        tensor,
        rank=rank,
        init="svd"
    )

    reconstructed = tucker_to_tensor((core, factors))

    elapsed_time = time.perf_counter() - start_time

    reconstruction_error = (
        np.linalg.norm(tensor - reconstructed) / original_norm
    )

    core_elements = core.size
    factor_elements = sum(factor.size for factor in factors)
    representation_elements = core_elements + factor_elements

    compression_ratio = original_elements / representation_elements

    results.append({
        "method": method,
        "rank": rank,
        "core_shape": core.shape,
        "core_elements": core_elements,
        "factor_elements": factor_elements,
        "representation_elements": representation_elements,
        "compression_ratio": compression_ratio,
        "reconstruction_error": reconstruction_error,
        "time_seconds": elapsed_time
    })

# --------------------------------------------------
# 4. Print comparison
# --------------------------------------------------

print("\nTUCKER RANK COMPARISON")
print("----------------------")

for result in results:
    print(f"\nMethod: {result['method']}")
    print("Rank:", result["rank"])
    print("Core shape:", result["core_shape"])
    print("Core elements:", result["core_elements"])
    print("Factor elements:", result["factor_elements"])
    print("Total representation elements:",
          result["representation_elements"])
    print("Compression ratio:",
          round(result["compression_ratio"], 2))
    print("Relative reconstruction error:",
          round(result["reconstruction_error"], 6))
    print("Decomposition time (seconds):",
          round(result["time_seconds"], 4))

fixed = results[0]
adaptive = results[1]

print("\nFINAL COMPARISON")
print("----------------")
print("Adaptive representation reduction (%):",
      round(
          (1 - adaptive["representation_elements"]
           / fixed["representation_elements"]) * 100,
          2
      ))

print("Adaptive reconstruction error change:",
      round(
          adaptive["reconstruction_error"]
          - fixed["reconstruction_error"],
          6
      ))

print("\nNote: Timing varies with system load.")
print("This experiment uses one ECG record only.")
