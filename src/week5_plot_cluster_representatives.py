import os
import pandas as pd
import matplotlib.pyplot as plt
import wfdb


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

representative_file = (
    "results/week5/clustering/representatives/"
    "cluster_representative_records.csv"
)

dataset_folder = "dataset/raw/set-a"

output_folder = (
    "results/week5/clustering/representatives/"
    "ecg_plots"
)

os.makedirs(output_folder, exist_ok=True)


# ---------------------------------------------------------
# Load representative records
# ---------------------------------------------------------

representatives = pd.read_csv(representative_file)

print("Representative records:")
print(representatives.to_string(index=False))


# ---------------------------------------------------------
# Process each representative
# ---------------------------------------------------------

for _, row in representatives.iterrows():

    cluster = int(row["cluster"])
    record_id = str(row["record_id"])

    record_path = os.path.join(
        dataset_folder,
        record_id
    )

    print("\nProcessing:")
    print("Cluster:", cluster)
    print("Record:", record_id)

    # -----------------------------------------------------
    # Read ECG
    # -----------------------------------------------------

    signal, fields = wfdb.rdsamp(record_path)

    sampling_frequency = fields["fs"]
    lead_names = fields["sig_name"]

    number_of_samples = signal.shape[0]

    time = (
        range(number_of_samples)
    )

    # Convert samples to seconds
    time = [
        sample / sampling_frequency
        for sample in time
    ]

    # -----------------------------------------------------
    # Create figure
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        len(lead_names),
        1,
        figsize=(14, 20),
        sharex=True
    )

    # Handle case of one lead
    if len(lead_names) == 1:
        axes = [axes]

    # -----------------------------------------------------
    # Plot all leads
    # -----------------------------------------------------

    for lead_index, lead_name in enumerate(lead_names):

        axes[lead_index].plot(
            time,
            signal[:, lead_index]
        )

        axes[lead_index].set_ylabel(
            lead_name
        )

        axes[lead_index].grid(
            True,
            alpha=0.3
        )

    axes[-1].set_xlabel(
        "Time (seconds)"
    )

    fig.suptitle(
        f"Cluster {cluster} - ECG {record_id}",
        fontsize=16
    )

    plt.tight_layout()

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_file = os.path.join(
        output_folder,
        f"cluster_{cluster}_record_{record_id}.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print("Saved:", output_file)


print("\n----------------------------------------")
print("Representative ECG plotting completed!")
print("----------------------------------------")

print("Output folder:")
print(output_folder)