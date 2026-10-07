import pandas as pd
import numpy as np
import os


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
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)

print("\nLabels:")
print(df["label"].value_counts())


# ---------------------------------------------------------
# Features to analyze
# ---------------------------------------------------------

features = [
    "raw_rms",
    "centered_rms",
    "standard_deviation",
    "baseline_variation",
    "amplitude_range"
]


# ---------------------------------------------------------
# Analyze every lead and every feature
# ---------------------------------------------------------

results = []


for lead in df["lead"].unique():

    lead_data = df[
        df["lead"] == lead
    ]

    acceptable = lead_data[
        lead_data["label"] == "Acceptable"
    ]

    unacceptable = lead_data[
        lead_data["label"] == "Unacceptable"
    ]

    for feature in features:

        acceptable_values = (
            acceptable[feature]
            .dropna()
            .to_numpy()
        )

        unacceptable_values = (
            unacceptable[feature]
            .dropna()
            .to_numpy()
        )

        acceptable_mean = (
            np.mean(acceptable_values)
        )

        unacceptable_mean = (
            np.mean(unacceptable_values)
        )

        acceptable_median = (
            np.median(acceptable_values)
        )

        unacceptable_median = (
            np.median(unacceptable_values)
        )

        acceptable_std = (
            np.std(
                acceptable_values,
                ddof=1
            )
        )

        unacceptable_std = (
            np.std(
                unacceptable_values,
                ddof=1
            )
        )

        # -------------------------------------------------
        # Pooled standard deviation
        # -------------------------------------------------

        n1 = len(acceptable_values)
        n2 = len(unacceptable_values)

        pooled_std = np.sqrt(
            (
                (n1 - 1) * acceptable_std ** 2
                +
                (n2 - 1) * unacceptable_std ** 2
            )
            /
            (n1 + n2 - 2)
        )

        if pooled_std > 0:

            effect_size = abs(
                unacceptable_mean -
                acceptable_mean
            ) / pooled_std

        else:

            effect_size = 0.0

        results.append({

            "lead":
                lead,

            "feature":
                feature,

            "acceptable_mean":
                acceptable_mean,

            "unacceptable_mean":
                unacceptable_mean,

            "acceptable_median":
                acceptable_median,

            "unacceptable_median":
                unacceptable_median,

            "acceptable_std":
                acceptable_std,

            "unacceptable_std":
                unacceptable_std,

            "absolute_effect_size":
                effect_size
        })


# ---------------------------------------------------------
# Create results dataframe
# ---------------------------------------------------------

results_df = pd.DataFrame(results)


# ---------------------------------------------------------
# Sort by effect size
# ---------------------------------------------------------

results_df = results_df.sort_values(
    "absolute_effect_size",
    ascending=False
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output_file = os.path.join(
    output_folder,
    "lead_level_feature_comparison.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

print("\n----------------------------------------")
print("Top lead-level characteristics")
print("----------------------------------------")

print(
    results_df.head(20).to_string(
        index=False
    )
)

print("\n----------------------------------------")
print("Analysis completed!")
print("----------------------------------------")

print("Output shape:", results_df.shape)

print("\nSaved to:")
print(output_file)