import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pywt

# -----------------------------
# Project paths
# -----------------------------
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

dataset_path = os.path.join(
    project_root, "dataset", "raw", "set-a"
)

output_path = os.path.join(
    project_root, "results", "week2", "scalograms"
)

os.makedirs(output_path, exist_ok=True)


# -----------------------------
# Load ECG record
# -----------------------------
record_name = "1002867"

record = wfdb.rdrecord(
    os.path.join(dataset_path, record_name)
)

signal = record.p_signal
fs = record.fs
lead_names = record.sig_name

print("Record:", record_name)
print("Signal shape:", signal.shape)
print("Sampling frequency:", fs)
print("Number of leads:", len(lead_names))
print("Leads:", lead_names)


# -----------------------------
# CWT settings
# -----------------------------
frequencies = np.linspace(1, 40, 100)

central_frequency = pywt.central_frequency("morl")

scales = (
    central_frequency * fs / frequencies
)


# -----------------------------
# Generate scalograms
# -----------------------------
for i, lead in enumerate(lead_names):

    ecg_signal = signal[:, i]

    coefficients, actual_frequencies = pywt.cwt(
        ecg_signal,
        scales,
        "morl",
        sampling_period=1 / fs
    )

    power = np.abs(coefficients)

    print(
        f"{lead}: scalogram shape = {power.shape}"
    )

    # Save scalogram
    plt.figure(figsize=(10, 4))

    plt.imshow(
        power,
        extent=[
            0,
            len(ecg_signal) / fs,
            frequencies[-1],
            frequencies[0]
        ],
        aspect="auto"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title(
        f"Scalogram - Record {record_name} - Lead {lead}"
    )

    plt.colorbar(label="Magnitude")

    filename = f"{record_name}_{lead}_scalogram.png"

    save_path = os.path.join(
        output_path,
        filename
    )

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


print("=" * 60)
print("WEEK 2 - 12 SCALOGRAMS COMPLETE")
print("=" * 60)
print("Saved in:")
print(output_path) #we applied same CWT settings to all 12 leads so we got the same scalogram shape.


#“It doesn't necessarily have to, sir. We are currently using 1–40 Hz as
#  our initial baseline setting. One part of our proposed Adaptive STP is to 
# investigate whether the relevant frequency range should change according to the 
# quality or characteristics of the recording.”


#1–40 Hz  → frequency range we selected
#100      → number of frequency levels we selected   “100 is the initial number of frequency
#  levels we selected to get sufficient time-frequency resolution. It is a 
# processing parameter, so we can experiment with it rather than treating 
# it as a fixed biological property.”
#5000     → comes from the actual ECG: 500 Hz × 10 sec
#12       → comes from the 12 ECG leads