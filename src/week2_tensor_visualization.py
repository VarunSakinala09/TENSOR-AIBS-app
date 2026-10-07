import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
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

output_path = os.path.join(
    project_root,
    "results",
    "week2"
)

os.makedirs(output_path, exist_ok=True)


# ---------------------------------------
# Load ECG
# ---------------------------------------

record_name = "1002867"

record = wfdb.rdrecord(
    os.path.join(dataset_path, record_name)
)

signal = record.p_signal
fs = record.fs
lead_names = record.sig_name


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

    scalograms.append(power)


# ---------------------------------------
# Create tensor
# ---------------------------------------

scalograms = np.array(scalograms)

tensor = np.transpose(
    scalograms,
    (2, 1, 0)
)


# ---------------------------------------
# Read dimensions from actual tensor
# ---------------------------------------

number_of_time_points = tensor.shape[0]
number_of_frequency_points = tensor.shape[1]
number_of_leads = tensor.shape[2]


print("Tensor shape:", tensor.shape)

print()
print("Time points:", number_of_time_points)
print("Frequency points:", number_of_frequency_points)
print("Number of leads:", number_of_leads)


# ---------------------------------------
# Create 3D visualization
# ---------------------------------------

fig = plt.figure(figsize=(12, 9))

ax = fig.add_subplot(111, projection="3d")


# Use a small number of time samples
# only for making the visualization easier to see.

plot_step = max(
    1,
    number_of_time_points // 300
)

time_indices = np.arange(
    0,
    number_of_time_points,
    plot_step
)

frequency_indices = np.arange(
    number_of_frequency_points
)

time_values = time_indices / fs

frequency_values = frequencies


# ---------------------------------------
# Plot one surface for each lead
# ---------------------------------------

for lead_index in range(number_of_leads):

    scalogram = tensor[
        time_indices,
        :,
        lead_index
    ]

    T, F = np.meshgrid(
        time_values,
        frequency_values,
        indexing="ij"
    )

    Z = np.full_like(
        T,
        lead_index
    )

    # Normalize the scalogram
    maximum = np.max(scalogram)

    if maximum > 0:
        surface_value = scalogram / maximum
    else:
        surface_value = scalogram

    ax.plot_surface(
        T,
        F,
        Z,
        facecolors=plt.cm.viridis(surface_value),
        linewidth=0,
        antialiased=True,
        shade=False
    )


# ---------------------------------------
# Axis labels
# ---------------------------------------

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Frequency (Hz)")
ax.set_zlabel("Lead")


# ---------------------------------------
# Lead labels
# ---------------------------------------

ax.set_zticks(
    np.arange(number_of_leads)
)

ax.set_zticklabels(
    lead_names
)


ax.set_title(
    f"3D ECG Scalogram Tensor - Record {record_name}"
)


plt.tight_layout()


# ---------------------------------------
# Save figure
# ---------------------------------------

save_path = os.path.join(
    output_path,
    "tensor_3d_visualization.png"
)

plt.savefig(
    save_path,
    dpi=250,
    bbox_inches="tight"
)

plt.show()

print()
print("3D tensor visualization saved at:")
print(save_path)