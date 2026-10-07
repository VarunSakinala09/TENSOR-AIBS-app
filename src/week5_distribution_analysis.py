import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

input_file = "results/week5/signal_characteristics.csv"
output_folder = "results/week5/distributions"

os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)

# ---------------------------------------------------------
# Characteristics selected from previous analysis
# ---------------------------------------------------------

features = [
    "max_rms",
    "mean_rms",
    "mean_abs_amplitude",
    "max_baseline_variation",
    "mean_baseline_variation"
]

# ---------------------------------------------------------
# Create distribution plots
# ---------------------------------------------------------

for feature in features:

    print("Creating plot:", feature)

    acceptable = df.loc[
        df["label"] == "Acceptable",
        feature
    ].dropna()

    unacceptable = df.loc[
        df["label"] == "Unacceptable",
        feature
    ].dropna()

    plt.figure(figsize=(9, 6))

    plt.boxplot(
        [acceptable, unacceptable],
        labels=["Acceptable", "Unacceptable"],
        showfliers=False
    )

    plt.ylabel(feature)
    plt.title("Distribution of " + feature)

    plt.tight_layout()

    output_file = os.path.join(
        output_folder,
        feature + "_boxplot.png"
    )

    plt.savefig(output_file, dpi=300)
    plt.close()

# ---------------------------------------------------------
# Combined boxplot
# ---------------------------------------------------------

plt.figure(figsize=(12, 7))

data = []

labels = []

for feature in features:

    acceptable = df.loc[
        df["label"] == "Acceptable",
        feature
    ].dropna()

    unacceptable = df.loc[
        df["label"] == "Unacceptable",
        feature
    ].dropna()

    data.append(acceptable)
    data.append(unacceptable)

    labels.append(feature + "\nAcceptable")
    labels.append(feature + "\nUnacceptable")

plt.boxplot(
    data,
    labels=labels,
    showfliers=False
)

plt.ylabel("Feature value")
plt.title("Signal Characteristic Distributions")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

combined_file = os.path.join(
    output_folder,
    "all_selected_features_boxplot.png"
)

plt.savefig(combined_file, dpi=300)
plt.close()

print("\n----------------------------------------")
print("Distribution analysis completed!")
print("----------------------------------------")

print("Plots saved to:")
print(output_folder)