import pandas as pd
import os

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

input_file = "results/week5/clustering/signal_condition_clusters.csv"
output_folder = "results/week5/clustering"

os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------
# Load clustered dataset
# ---------------------------------------------------------

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)

# ---------------------------------------------------------
# Cluster sizes
# ---------------------------------------------------------

print("\nCluster sizes:")
print(
    df["signal_condition_cluster"]
    .value_counts()
    .sort_index()
)

# ---------------------------------------------------------
# Cluster vs actual label
# ---------------------------------------------------------

cross_tab = pd.crosstab(
    df["signal_condition_cluster"],
    df["label"]
)

print("\nCluster vs ECG quality:")
print(cross_tab)

# ---------------------------------------------------------
# Percentage distribution inside each cluster
# ---------------------------------------------------------

cluster_percentages = pd.crosstab(
    df["signal_condition_cluster"],
    df["label"],
    normalize="index"
) * 100

print("\nPercentage of labels within each cluster:")
print(
    cluster_percentages.round(2)
)

# ---------------------------------------------------------
# Percentage of clusters within each class
# ---------------------------------------------------------

label_percentages = pd.crosstab(
    df["label"],
    df["signal_condition_cluster"],
    normalize="index"
) * 100

print("\nPercentage of clusters within each ECG quality class:")
print(
    label_percentages.round(2)
)

# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

cross_tab_file = os.path.join(
    output_folder,
    "cluster_label_crosstab.csv"
)

percentage_file = os.path.join(
    output_folder,
    "cluster_label_percentages.csv"
)

cross_tab.to_csv(cross_tab_file)
cluster_percentages.to_csv(percentage_file)

print("\n----------------------------------------")
print("Cluster-label analysis completed!")
print("----------------------------------------")

print("Cross-tab saved to:")
print(cross_tab_file)

print("\nCluster percentage analysis saved to:")
print(percentage_file)