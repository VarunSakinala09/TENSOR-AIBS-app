import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

input_file = "results/week5/signal_characteristics.csv"
output_folder = "results/week5"

os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)

# ---------------------------------------------------------
# Select actual signal characteristics
# ---------------------------------------------------------

features = [
    "mean_rms",
    "max_rms",
    "mean_std",
    "max_std",
    "mean_abs_amplitude",
    "mean_baseline_variation",
    "max_baseline_variation",
    "mean_high_frequency_ratio",
    "max_high_frequency_ratio",
    "mean_powerline_ratio",
    "max_powerline_ratio"
]

feature_data = df[features]

# ---------------------------------------------------------
# Calculate correlation
# ---------------------------------------------------------

correlation_matrix = feature_data.corr(method="pearson")

print("\nCorrelation matrix:")
print(correlation_matrix.to_string())

# ---------------------------------------------------------
# Save correlation matrix
# ---------------------------------------------------------

output_csv = os.path.join(
    output_folder,
    "feature_correlation_matrix.csv"
)

correlation_matrix.to_csv(output_csv)

# ---------------------------------------------------------
# Find highly correlated pairs
# ---------------------------------------------------------

print("\nHighly correlated feature pairs:")

for i in range(len(features)):

    for j in range(i + 1, len(features)):

        correlation = correlation_matrix.iloc[i, j]

        if abs(correlation) >= 0.80:

            print(
                f"{features[i]} <-> {features[j]} : "
                f"{correlation:.4f}"
            )

# ---------------------------------------------------------
# Create correlation plot
# ---------------------------------------------------------

plt.figure(figsize=(12, 10))

plt.imshow(
    correlation_matrix,
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(features)),
    features,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(features)),
    features
)

plt.title("Correlation Between Signal Characteristics")

plt.tight_layout()

output_image = os.path.join(
    output_folder,
    "feature_correlation_matrix.png"
)

plt.savefig(
    output_image,
    dpi=300
)

plt.close()

print("\n----------------------------------------")
print("Correlation analysis completed!")
print("----------------------------------------")

print("Correlation matrix saved to:")
print(output_csv)

print("\nCorrelation plot saved to:")
print(output_image)