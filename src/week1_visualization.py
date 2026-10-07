import os
import wfdb
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# Project paths
# --------------------------------------------------

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
    "week1",
    "ecg_plots"
)

# Create output folder
os.makedirs(output_path, exist_ok=True)


# --------------------------------------------------
# Function to save ECG
# --------------------------------------------------

def save_ecg(record_name, label):

    record_path = os.path.join(
        dataset_path,
        record_name
    )

    # Load ECG
    record = wfdb.rdrecord(record_path)

    signal = record.p_signal
    fs = record.fs
    leads = record.sig_name

    # Time axis
    time = np.arange(signal.shape[0]) / fs

    # Create figure
    fig, axes = plt.subplots(
        12,
        1,
        figsize=(14, 12),
        sharex=True
    )

    # Plot each lead
    for i in range(12):

        axes[i].plot(
            time,
            signal[:, i],
            linewidth=0.8
        )

        axes[i].set_ylabel(
            leads[i],
            rotation=0,
            labelpad=25
        )

        axes[i].grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(
        f"12-Lead ECG - {record_name} ({label})",
        fontsize=14
    )

    axes[-1].set_xlabel("Time (seconds)")

    # Adjust layout
    plt.tight_layout()

    # Save file
    filename = f"{label.lower()}_{record_name}.png"

    save_path = os.path.join(
        output_path,
        filename
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    # Close figure
    plt.close(fig)

    print("Saved:", save_path)


# --------------------------------------------------
# Generate Week 1 graphs
# --------------------------------------------------

print("Generating Week 1 ECG graphs...")

save_ecg(
    "1002867",
    "Acceptable"
)

save_ecg(
    "1002603",
    "Unacceptable"
)

print()
print("=" * 60)
print("WEEK 1 ECG VISUALIZATION COMPLETE")
print("=" * 60)
print("Graphs saved in:")
print(output_path)