
import numpy as np
import pywt
import wfdb
import tensorly as tl

from tensorly.decomposition import tucker

# Load one ECG record from Set A
record_path = "dataset/raw/set-a/1002867"
signal, fields = wfdb.rdsamp(record_path)

fs = fields["fs"]
print("Original ECG shape:", signal.shape)
print("Sampling frequency:", fs)

# Generate Morlet scalograms for all 12 leads
frequencies = np.linspace(1, 40, 95)
scalograms = []

for lead in range(signal.shape[1]):
    scales = pywt.central_frequency("morl") * fs / frequencies

    coefficients, actual_frequencies = pywt.cwt(
        signal[:, lead],
        scales,
        "morl",
        sampling_period=1 / fs
    )

    scalograms.append(np.abs(coefficients))

# Shape before arranging dimensions: Leads x Frequency x Time
scalograms = np.array(scalograms)

# Arrange as Frequency x Time x Leads
tensor = np.transpose(scalograms, (1, 2, 0))

print("Tensor shape:", tensor.shape)

# Reference paper's best reported Tucker rank
rank = [5, 5, 12]

core, factors = tucker(tensor, rank=rank, init="svd")

features = core.reshape(-1)

print("Tucker core shape:", core.shape)
print("Number of features:", len(features))
print("Factor shapes:", [factor.shape for factor in factors])

assert tensor.shape == (95, 5000, 12)
assert core.shape == (5, 5, 12)
assert len(features) == 300

print("First reproduction test completed.")
