import pandas as pd
import os


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

cluster_file = (
    "results/week5/clustering/"
    "signal_condition_clusters.csv"
)

abnormal_file = (
    "results/week5/lead_level/"
    "record_abnormal_lead_counts.csv"
)

output_folder = "results/week5/clustering"

os.makedirs(output_folder, exist_ok=True)


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

clusters = pd.read_csv(
    cluster_file
)

abnormal = pd.read_csv(
    abnormal_file
)

print("Cluster dataset shape:", clusters.shape)
print("Abnormal-lead dataset shape:", abnormal.shape)


# ---------------------------------------------------------
# Keep cluster information needed for merging
# ---------------------------------------------------------

cluster_info = clusters[
    [
        "record_id",
        "signal_condition_cluster"
    ]
].copy()


# ---------------------------------------------------------
# Merge
# ---------------------------------------------------------

merged = pd.merge(
    abnormal,
    cluster_info,
    on="record_id",
    how="inner"
)


print("\nMerged dataset shape:", merged.shape)


# ---------------------------------------------------------
# Summary by cluster
# ---------------------------------------------------------

print("\n----------------------------------------")
print("Abnormal leads by signal-condition cluster")
print("----------------------------------------")

cluster_summary = (
    merged
    .groupby("signal_condition_cluster")
    ["abnormal_lead_count"]
    .describe()
)

print(
    cluster_summary.round(2)
)


# ---------------------------------------------------------
# Class composition inside each cluster
# ---------------------------------------------------------

print("\n----------------------------------------")
print("ECG quality within each cluster")
print("----------------------------------------")

cluster_quality = pd.crosstab(
    merged["signal_condition_cluster"],
    merged["label"]
)

print(cluster_quality)


cluster_quality_percent = (
    pd.crosstab(
        merged["signal_condition_cluster"],
        merged["label"],
        normalize="index"
    )
    * 100
)

print("\nPercentages:")
print(
    cluster_quality_percent.round(2)
)


# ---------------------------------------------------------
# Mean abnormal leads by cluster and label
# ---------------------------------------------------------

print("\n----------------------------------------")
print("Mean abnormal leads by cluster and label")
print("----------------------------------------")

cluster_label_summary = (
    merged
    .groupby(
        [
            "signal_condition_cluster",
            "label"
        ]
    )["abnormal_lead_count"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max"
        ]
    )
)

print(
    cluster_label_summary.round(2)
)


# ---------------------------------------------------------
# Save merged dataset
# ---------------------------------------------------------

merged_file = os.path.join(
    output_folder,
    "cluster_abnormal_lead_combined.csv"
)

merged.to_csv(
    merged_file,
    index=False
)


# ---------------------------------------------------------
# Save summaries
# ---------------------------------------------------------

summary_file = os.path.join(
    output_folder,
    "cluster_abnormal_lead_summary.csv"
)

cluster_summary.to_csv(
    summary_file
)


print("\n----------------------------------------")
print("Cluster + abnormal-lead analysis completed!")
print("----------------------------------------")

print("\nCombined dataset saved to:")
print(merged_file)

print("\nSummary saved to:")
print(summary_file)