import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Load ECG record
record = wfdb.rdrecord("dataset/raw/set-a/1002867")

# Sampling frequency
fs = record.fs

# Lead names
lead_names = record.sig_name

# Frequencies for CWT
frequencies = np.linspace(1, 40, 100)

# Convert frequency to scale
central_frequency = pywt.central_frequency("morl")
scales = central_frequency * fs / frequencies

# Store all scalograms
scalograms = []

# Process all 12 leads
for i in range(12):

    # Select one lead
    signal = record.p_signal[:, i]

    # CWT
    coefficients, freqs = pywt.cwt(
        signal,
        scales,
        "morl",
        sampling_period=1 / fs
    )

    # Magnitude
    power = np.abs(coefficients)

    # Store scalogram
    scalograms.append(power)

    # Create time axis
    time = np.arange(len(signal)) / fs

    # Plot scalogram
    plt.figure(figsize=(10, 5))

    plt.pcolormesh(
        time,
        freqs,
        power,
        shading="auto"
    )

    plt.colorbar(label="CWT Magnitude")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")

    plt.title(f"CWT Scalogram - Lead {lead_names[i]}")

    plt.tight_layout()
    plt.show()

# Convert list into NumPy array
scalograms = np.array(scalograms)

print("\nScalograms shape:", scalograms.shape)