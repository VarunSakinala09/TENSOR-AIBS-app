import os
import numpy as np
import pandas as pd
import wfdb


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

dataset_folder = "dataset/raw/set-a"

output_folder = "results/week5/lead_level"

os.makedirs(output_folder, exist_ok=True)


# ---------------------------------------------------------
# Load labelled records
# ---------------------------------------------------------

characteristics_file = "results/week5/signal_characteristics.csv"

characteristics = pd.read_csv(characteristics_file)

record_ids = characteristics["record_id"].astype(str).tolist()


# ---------------------------------------------------------
# Store lead-level results
# ---------------------------------------------------------

all_results = []


# ---------------------------------------------------------
# Process every ECG
# ---------------------------------------------------------

for record_number, record_id in enumerate(record_ids, start=1):

    record_path = os.path.join(
        dataset_folder,
        record_id
    )

    signal, fields = wfdb.rdsamp(record_path)

    fs = fields["fs"]
    lead_names = fields["sig_name"]

    number_of_samples = signal.shape[0]

    # -----------------------------------------------------
    # Dynamic baseline window
    # -----------------------------------------------------

    baseline_window = max(
        1,
        int(fs * 0.2)
    )

    # -----------------------------------------------------
    # Analyze every lead
    # -----------------------------------------------------

    for lead_index, lead_name in enumerate(lead_names):

        lead = signal[:, lead_index]

        # -------------------------------------------------
        # Raw RMS
        # -------------------------------------------------

        raw_rms = np.sqrt(
            np.mean(lead ** 2)
        )

        # -------------------------------------------------
        # Remove DC offset
        # -------------------------------------------------

        centered_lead = (
            lead - np.mean(lead)
        )

        centered_rms = np.sqrt(
            np.mean(centered_lead ** 2)
        )

        # -------------------------------------------------
        # Standard deviation
        # -------------------------------------------------

        standard_deviation = np.std(
            lead
        )

        # -------------------------------------------------
        # Amplitude range
        # -------------------------------------------------

        amplitude_range = (
            np.max(lead) -
            np.min(lead)
        )

        # -------------------------------------------------
        # Baseline estimation
        # -------------------------------------------------

        baseline = (
            pd.Series(lead)
            .rolling(
                window=baseline_window,
                center=True,
                min_periods=1
            )
            .mean()
            .to_numpy()
        )

        baseline_variation = np.std(
            baseline
        )

        # -------------------------------------------------
        # Save result
        # -------------------------------------------------

        all_results.append({
            "record_id": record_id,
            "label": characteristics.loc[
                characteristics["record_id"].astype(str)
                == record_id,
                "label"
            ].iloc[0],

            "lead": lead_name,

            "raw_rms": raw_rms,

            "centered_rms": centered_rms,

            "standard_deviation":
                standard_deviation,

            "baseline_variation":
                baseline_variation,

            "amplitude_range":
                amplitude_range
        })

    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    if record_number == 1 or record_number % 50 == 0:
        print(
            f"Processed {record_number}/"
            f"{len(record_ids)} records"
        )


# ---------------------------------------------------------
# Create dataframe
# ---------------------------------------------------------

lead_df = pd.DataFrame(
    all_results
)


# ---------------------------------------------------------
# Save detailed lead-level dataset
# ---------------------------------------------------------

output_file = os.path.join(
    output_folder,
    "lead_level_characteristics.csv"
)

lead_df.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# Create record-level summary
# ---------------------------------------------------------

summary_rows = []


for record_id, group in lead_df.groupby(
    "record_id"
):

    summary_rows.append({

        "record_id":
            record_id,

        "label":
            group["label"].iloc[0],

        "mean_raw_rms":
            group["raw_rms"].mean(),

        "max_raw_rms":
            group["raw_rms"].max(),

        "mean_centered_rms":
            group["centered_rms"].mean(),

        "max_centered_rms":
            group["centered_rms"].max(),

        "mean_standard_deviation":
            group["standard_deviation"].mean(),

        "max_standard_deviation":
            group["standard_deviation"].max(),

        "mean_baseline_variation":
            group["baseline_variation"].mean(),

        "max_baseline_variation":
            group["baseline_variation"].max(),

        "mean_amplitude_range":
            group["amplitude_range"].mean(),

        "max_amplitude_range":
            group["amplitude_range"].max()
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ---------------------------------------------------------
# Save summary
# ---------------------------------------------------------

summary_file = os.path.join(
    output_folder,
    "record_level_lead_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ---------------------------------------------------------
# Final output
# ---------------------------------------------------------

print("\n----------------------------------------")
print("Lead-level analysis completed!")
print("----------------------------------------")

print(
    "Lead-level dataset shape:",
    lead_df.shape
)

print(
    "Record-level summary shape:",
    summary_df.shape
)

print("\nLead-level dataset saved to:")
print(output_file)

print("\nRecord-level summary saved to:")
print(summary_file)