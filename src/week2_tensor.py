import os
import wfdb
import numpy as np
import pywt


# ---------------------------------------
# Project paths
# ---------------------------------------

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

dataset_path = os.path.join(
    project_root,
    "dataset",
    "raw",
    "set-a"
)


# ---------------------------------------
# Record selected for Week 2 demonstration
# ---------------------------------------

record_name = "1002867"


# ---------------------------------------
# Load ECG record
# ---------------------------------------

record = wfdb.rdrecord(
    os.path.join(dataset_path, record_name)
)

signal = record.p_signal
fs = record.fs
lead_names = record.sig_name


print("Record:", record_name)
print("Original ECG shape:", signal.shape)
print("Sampling frequency:", fs)
print("Number of leads:", len(lead_names))
print("Leads:", lead_names)


# ---------------------------------------
# CWT settings
# ---------------------------------------

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


# ---------------------------------------
# Generate scalograms
# ---------------------------------------

scalograms = []

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

    scalograms.append(power)


# ---------------------------------------
# Convert scalograms into NumPy array
# ---------------------------------------

scalograms = np.array(scalograms)

print()
print("Scalograms array shape:", scalograms.shape)


# ---------------------------------------
# Rearrange dimensions
#
# Current:
# Lead × Frequency × Time
#
# Final:
# Time × Frequency × Lead
# ---------------------------------------

tensor = np.transpose(
    scalograms,
    (2, 1, 0)
)


# ---------------------------------------
# Display actual tensor dimensions
# ---------------------------------------

print()
print("=" * 60)
print("3D TENSOR CREATED")
print("=" * 60)

print("Tensor shape:", tensor.shape)

print()
print("Tensor dimensions:")
print("Time      :", tensor.shape[0])
print("Frequency :", tensor.shape[1])
print("Leads     :", tensor.shape[2])