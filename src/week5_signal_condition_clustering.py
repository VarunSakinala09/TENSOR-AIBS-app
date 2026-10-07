import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

input_file = "results/week5/signal_characteristics.csv"

output_folder = "results/week5/clustering"

os.makedirs(output_folder, exist_ok=True)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)


# ---------------------------------------------------------
# Selected signal characteristics
# ---------------------------------------------------------

features = [
    "mean_rms",
    "max_baseline_variation",
    "mean_high_frequency_ratio"
]

X = df[features].copy()


# ---------------------------------------------------------
# Check for missing values
# ---------------------------------------------------------

print("\nMissing values:")

print(X.isnull().sum())


# ---------------------------------------------------------
# Standardize features
# ---------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


print("\nFeatures standardized.")


# ---------------------------------------------------------
# Test different numbers of clusters
# ---------------------------------------------------------

cluster_results = []

print("\nTesting cluster numbers:")

for k in range(2, 6):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        cluster_labels
    )

    cluster_results.append({
        "number_of_clusters": k,
        "silhouette_score": score
    })

    print(
        f"k = {k} | "
        f"Silhouette score = {score:.4f}"
    )


# ---------------------------------------------------------
# Convert results to dataframe
# ---------------------------------------------------------

cluster_results_df = pd.DataFrame(cluster_results)


# ---------------------------------------------------------
# Select best number of clusters
# ---------------------------------------------------------

best_row = cluster_results_df.loc[
    cluster_results_df["silhouette_score"].idxmax()
]

best_k = int(best_row["number_of_clusters"])

best_score = best_row["silhouette_score"]


print("\n----------------------------------------")
print("Best number of clusters:", best_k)
print("Best silhouette score:", round(best_score, 4))
print("----------------------------------------")


# ---------------------------------------------------------
# Save silhouette results
# ---------------------------------------------------------

cluster_results_file = os.path.join(
    output_folder,
    "cluster_selection_results.csv"
)

cluster_results_df.to_csv(
    cluster_results_file,
    index=False
)


# ---------------------------------------------------------
# Train final clustering model
# ---------------------------------------------------------

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

final_labels = final_model.fit_predict(X_scaled)


# ---------------------------------------------------------
# Add cluster labels to dataset
# ---------------------------------------------------------

clustered_df = df[
    ["record_id", "label"] + features
].copy()

clustered_df["signal_condition_cluster"] = final_labels


# ---------------------------------------------------------
# Cluster summary
# ---------------------------------------------------------

print("\nCluster sizes:")

cluster_counts = (
    clustered_df["signal_condition_cluster"]
    .value_counts()
    .sort_index()
)

print(cluster_counts)


print("\nCluster characteristics:")

cluster_summary = (
    clustered_df
    .groupby("signal_condition_cluster")[features]
    .mean()
)

print(cluster_summary)


# ---------------------------------------------------------
# Save clustered dataset
# ---------------------------------------------------------

clustered_file = os.path.join(
    output_folder,
    "signal_condition_clusters.csv"
)

clustered_df.to_csv(
    clustered_file,
    index=False
)


# ---------------------------------------------------------
# PCA for visualization
# ---------------------------------------------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 7))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=final_labels
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "Data-Driven Signal Condition Clusters"
)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.tight_layout()


pca_file = os.path.join(
    output_folder,
    "signal_condition_clusters_pca.png"
)

plt.savefig(
    pca_file,
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# Silhouette score plot
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    cluster_results_df["number_of_clusters"],
    cluster_results_df["silhouette_score"],
    marker="o"
)

plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")

plt.title(
    "Cluster Selection Using Silhouette Score"
)

plt.xticks(
    cluster_results_df["number_of_clusters"]
)

plt.tight_layout()


silhouette_file = os.path.join(
    output_folder,
    "silhouette_scores.png"
)

plt.savefig(
    silhouette_file,
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# Final output
# ---------------------------------------------------------

print("\n----------------------------------------")
print("Clustering completed!")
print("----------------------------------------")

print("Cluster selection results:")
print(cluster_results_file)

print("\nClustered dataset:")
print(clustered_file)

print("\nPCA visualization:")
print(pca_file)

print("\nSilhouette plot:")
print(silhouette_file)