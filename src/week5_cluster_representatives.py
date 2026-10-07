import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

input_file = "results/week5/signal_characteristics.csv"

output_folder = "results/week5/clustering/representatives"

os.makedirs(output_folder, exist_ok=True)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)


# ---------------------------------------------------------
# Characteristics used for clustering
# ---------------------------------------------------------

features = [
    "mean_rms",
    "max_baseline_variation",
    "mean_high_frequency_ratio"
]

X = df[features].copy()


# ---------------------------------------------------------
# Standardize
# ---------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ---------------------------------------------------------
# K-Means with selected number of clusters
# ---------------------------------------------------------

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_labels = model.fit_predict(X_scaled)


# ---------------------------------------------------------
# Find representative record for each cluster
# ---------------------------------------------------------

df["signal_condition_cluster"] = cluster_labels

representatives = []


for cluster_id in sorted(df["signal_condition_cluster"].unique()):

    cluster_indices = np.where(
        cluster_labels == cluster_id
    )[0]

    cluster_points = X_scaled[cluster_indices]

    center = model.cluster_centers_[cluster_id]

    distances = np.linalg.norm(
        cluster_points - center,
        axis=1
    )

    closest_position = np.argmin(distances)

    representative_index = cluster_indices[
        closest_position
    ]

    record = df.iloc[representative_index]

    representatives.append({
        "cluster": cluster_id,
        "record_id": record["record_id"],
        "label": record["label"],
        "mean_rms": record["mean_rms"],
        "max_baseline_variation":
            record["max_baseline_variation"],
        "mean_high_frequency_ratio":
            record["mean_high_frequency_ratio"]
    })


# ---------------------------------------------------------
# Save representatives
# ---------------------------------------------------------

representatives_df = pd.DataFrame(
    representatives
)

output_file = os.path.join(
    output_folder,
    "cluster_representative_records.csv"
)

representatives_df.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("\n----------------------------------------")
print("Representative ECG records")
print("----------------------------------------")

print(
    representatives_df.to_string(index=False)
)

print("\nSaved to:")
print(output_file)