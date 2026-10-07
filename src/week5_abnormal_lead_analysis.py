import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

input_file = (
    "results/week5/lead_level/"
    "lead_level_characteristics.csv"
)

output_folder = "results/week5/lead_level"

os.makedirs(output_folder, exist_ok=True)


# ---------------------------------------------------------
# Load lead-level dataset
# ---------------------------------------------------------

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)


# ---------------------------------------------------------
# Features used for abnormal-lead analysis
# ---------------------------------------------------------

features = [
    "raw_rms",
    "centered_rms",
    "baseline_variation",
    "amplitude_range"
]


# ---------------------------------------------------------
# Calculate thresholds from Acceptable records
# ---------------------------------------------------------

acceptable_df = df[
    df["label"] == "Acceptable"
].copy()


threshold_rows = []


for lead in df["lead"].unique():

    lead_data = acceptable_df[
        acceptable_df["lead"] == lead
    ]

    for feature in features:

        threshold = np.percentile(
            lead_data[feature].dropna(),
            95
        )

        threshold_rows.append({
            "lead": lead,
            "feature": feature,
            "percentile": 95,
            "threshold": threshold
        })


threshold_df = pd.DataFrame(
    threshold_rows
)


# ---------------------------------------------------------
# Save thresholds
# ---------------------------------------------------------

threshold_file = os.path.join(
    output_folder,
    "acceptable_lead_thresholds.csv"
)

threshold_df.to_csv(
    threshold_file,
    index=False
)


print("\n----------------------------------------")
print("Data-driven thresholds")
print("----------------------------------------")

print(
    threshold_df.to_string(index=False)
)


# ---------------------------------------------------------
# Create lookup dictionary
# ---------------------------------------------------------

threshold_lookup = {}

for _, row in threshold_df.iterrows():

    key = (
        row["lead"],
        row["feature"]
    )

    threshold_lookup[key] = row["threshold"]


# ---------------------------------------------------------
# Detect abnormal features for every lead
# ---------------------------------------------------------

result_rows = []


for (record_id, lead), group in df.groupby(
    ["record_id", "lead"]
):

    row = group.iloc[0]

    abnormal_features = []


    for feature in features:

        value = row[feature]

        threshold = threshold_lookup[
            (lead, feature)
        ]

        if value > threshold:

            abnormal_features.append(
                feature
            )


    result_rows.append({

        "record_id":
            record_id,

        "label":
            row["label"],

        "lead":
            lead,

        "abnormal_feature_count":
            len(abnormal_features),

        "abnormal_features":
            ", ".join(abnormal_features),

        "is_abnormal_lead":
            len(abnormal_features) > 0
    })


lead_results = pd.DataFrame(
    result_rows
)


# ---------------------------------------------------------
# Count abnormal leads per ECG
# ---------------------------------------------------------

record_rows = []


for record_id, group in lead_results.groupby(
    "record_id"
):

    label = group["label"].iloc[0]

    abnormal_lead_count = int(
        group["is_abnormal_lead"].sum()
    )

    total_leads = len(group)

    abnormal_lead_ratio = (
        abnormal_lead_count /
        total_leads
    )

    record_rows.append({

        "record_id":
            record_id,

        "label":
            label,

        "total_leads":
            total_leads,

        "abnormal_lead_count":
            abnormal_lead_count,

        "abnormal_lead_ratio":
            abnormal_lead_ratio
    })


record_results = pd.DataFrame(
    record_rows
)


# ---------------------------------------------------------
# Save lead-level abnormality results
# ---------------------------------------------------------

lead_output = os.path.join(
    output_folder,
    "lead_abnormality_details.csv"
)

lead_results.to_csv(
    lead_output,
    index=False
)


# ---------------------------------------------------------
# Save record-level abnormality results
# ---------------------------------------------------------

record_output = os.path.join(
    output_folder,
    "record_abnormal_lead_counts.csv"
)

record_results.to_csv(
    record_output,
    index=False
)


# ---------------------------------------------------------
# Compare abnormal-lead counts by class
# ---------------------------------------------------------

print("\n----------------------------------------")
print("Abnormal lead count by ECG quality")
print("----------------------------------------")

class_summary = (
    record_results
    .groupby("label")["abnormal_lead_count"]
    .describe()
)

print(
    class_summary.round(2)
)


# ---------------------------------------------------------
# Distribution of abnormal-lead counts
# ---------------------------------------------------------

print("\n----------------------------------------")
print("Abnormal lead count distribution")
print("----------------------------------------")

distribution = pd.crosstab(
    record_results["abnormal_lead_count"],
    record_results["label"]
)

print(distribution)


# ---------------------------------------------------------
# Final information
# ---------------------------------------------------------

print("\n----------------------------------------")
print("Abnormal-lead analysis completed!")
print("----------------------------------------")

print(
    "Lead-level result shape:",
    lead_results.shape
)

print(
    "Record-level result shape:",
    record_results.shape
)

print("\nThresholds saved to:")
print(threshold_file)

print("\nLead-level abnormality details saved to:")
print(lead_output)

print("\nRecord-level abnormal lead counts saved to:")
print(record_output)