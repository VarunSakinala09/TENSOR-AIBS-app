import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

input_file = "results/week5/signal_characteristics.csv"
output_file = "results/week5/percentile_analysis.csv"

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)

# ---------------------------------------------------------
# Automatically identify signal characteristics
# ---------------------------------------------------------

metadata_columns = ["record_id", "label"]

feature_columns = [
    column for column in df.columns
    if column not in metadata_columns
]

print("Number of characteristics:", len(feature_columns))
print("Characteristics:", feature_columns)

# ---------------------------------------------------------
# Percentiles to calculate
# ---------------------------------------------------------

percentiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# ---------------------------------------------------------
# Calculate percentiles separately for each class
# ---------------------------------------------------------

results = []

for label in sorted(df["label"].dropna().unique()):

    class_data = df[df["label"] == label]

    print("\nProcessing:", label)
    print("Number of records:", len(class_data))

    for feature in feature_columns:

        values = class_data[feature].dropna()

        percentile_values = np.percentile(
            values,
            [p * 100 for p in percentiles]
        )

        row = {
            "label": label,
            "feature": feature
        }

        for percentile, value in zip(percentiles, percentile_values):
            row[f"p{int(percentile * 100)}"] = value

        results.append(row)

# ---------------------------------------------------------
# Create output dataframe
# ---------------------------------------------------------

percentile_df = pd.DataFrame(results)

# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

os.makedirs(os.path.dirname(output_file), exist_ok=True)

percentile_df.to_csv(output_file, index=False)

print("\n----------------------------------------")
print("Percentile analysis completed!")
print("----------------------------------------")

print("Output shape:", percentile_df.shape)
print("Saved to:", output_file)

# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\nPercentile results:")
print(percentile_df.to_string(index=False))