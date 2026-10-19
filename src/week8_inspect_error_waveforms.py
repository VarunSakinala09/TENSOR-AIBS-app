
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb


# Project paths
ROOT = Path(__file__).resolve().parents[1]

PREDICTION_FILE = (
    ROOT / "results" / "week8" / "adaptive_group_cv"
    / "out_of_fold_predictions.csv"
)

CHARACTERISTICS_FILE = (
    ROOT / "results" / "week5" / "signal_characteristics.csv"
)

DATASET_DIR = ROOT / "dataset" / "raw" / "set-a"

OUTPUT_DIR = (
    ROOT / "results" / "week8" / "waveform_inspection"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load predictions and signal characteristics
predictions = pd.read_csv(PREDICTION_FILE)
characteristics = pd.read_csv(CHARACTERISTICS_FILE)

df = predictions.merge(
    characteristics.drop(columns=["label"], errors="ignore"),
    on="record_id",
    how="left",
    validate="one_to_one"
)

# Label convention: 1 = Unacceptable, 0 = Acceptable
df["result"] = "Other"

df.loc[
    (df["true_label"] == 1) & (df["adaptive_prediction"] == 0),
    "result"
] = "Missed_Unacceptable"

df.loc[
    (df["true_label"] == 1) & (df["adaptive_prediction"] == 1),
    "result"
] = "Detected_Unacceptable"

df.loc[
    (df["true_label"] == 0) & (df["adaptive_prediction"] == 0),
    "result"
] = "Correct_Acceptable"


# Prefer records with non-flat signals for the first comparison
df["mean_std"] = pd.to_numeric(df["mean_std"], errors="coerce")

nonflat = df[df["mean_std"].abs() > 1e-8]

groups = {
    "Missed_Unacceptable": nonflat[
        (nonflat["true_label"] == 1)
        & (nonflat["adaptive_prediction"] == 0)
    ],
    "Detected_Unacceptable": nonflat[
        (nonflat["true_label"] == 1)
        & (nonflat["adaptive_prediction"] == 1)
    ],
    "Correct_Acceptable": nonflat[
        (nonflat["true_label"] == 0)
        & (nonflat["adaptive_prediction"] == 0)
    ],
}

# Select representative records near the group's median
# mean RMS, rather than choosing only extreme examples.
selected = []

for group_name, group_df in groups.items():
    if group_df.empty:
        print(f"No records available for {group_name}")
        continue

    group_df = group_df.dropna(subset=["mean_rms"])

    if group_df.empty:
        print(f"No valid RMS values for {group_name}")
        continue

    target_rms = group_df["mean_rms"].median()

    group_df = group_df.copy()
    group_df["rms_distance"] = (
        group_df["mean_rms"] - target_rms
    ).abs()

    # Up to three representative records per group
    chosen = group_df.nsmallest(3, "rms_distance")

    for _, row in chosen.iterrows():
        selected.append({
            "record_id": str(row["record_id"]),
            "group": group_name,
            "mean_rms": row["mean_rms"],
            "mean_std": row["mean_std"],
        })


# Save the selected record list
selected_df = pd.DataFrame(selected)
selected_df.to_csv(
    OUTPUT_DIR / "selected_records.csv",
    index=False
)

print("\nSELECTED ECG RECORDS")
print("=" * 70)
print(selected_df.to_string(index=False))


# Find a record header in either supported directory layout
def find_record_header(record_id):
    possible_paths = [
        DATASET_DIR / record_id,
        DATASET_DIR / "set-a" / record_id,
    ]

    for record_path in possible_paths:
        if Path(str(record_path) + ".hea").exists():
            return record_path

    return None


# Plot the 12 leads for each selected record
for item in selected:
    record_id = item["record_id"]
    group_name = item["group"]

    record_path = find_record_header(record_id)

    if record_path is None:
        print(f"Header not found for {record_id}")
        continue

    try:
        signal, fields = wfdb.rdsamp(str(record_path))
    except Exception as error:
        print(f"Could not read {record_id}: {error}")
        continue

    sampling_frequency = fields["fs"]
    lead_names = fields.get("sig_name")

    if not lead_names or len(lead_names) != signal.shape[1]:
        lead_names = [
            f"Lead {i + 1}" for i in range(signal.shape[1])
        ]

    time = np.arange(signal.shape[0]) / sampling_frequency

    fig, axes = plt.subplots(
        signal.shape[1], 1,
        figsize=(14, 18),
        sharex=True
    )

    for lead_index, ax in enumerate(axes):
        ax.plot(time, signal[:, lead_index], linewidth=0.7)
        ax.set_ylabel(lead_names[lead_index], rotation=0, labelpad=30)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{group_name} | Record {record_id}\n"
        f"Mean RMS: {item['mean_rms']:.4f}, "
        f"Mean STD: {item['mean_std']:.4f}",
        fontsize=13
    )

    axes[-1].set_xlabel("Time (seconds)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_file = OUTPUT_DIR / (
        f"{group_name}_{record_id}.png"
    )

    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_file}")


print("\nInspection complete.")
print(f"Results folder: {OUTPUT_DIR}")
