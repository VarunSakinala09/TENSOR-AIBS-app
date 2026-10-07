import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD SIGNAL CHARACTERISTICS
# ============================================================

input_path = "results/week5/signal_characteristics.csv"

data = pd.read_csv(input_path)

print("\nDATASET")
print("-------")
print("Shape:", data.shape)


# ============================================================
# 2. SELECT NUMERICAL SIGNAL CHARACTERISTICS
# ============================================================

excluded_columns = [
    "record_id",
    "label",
    "sampling_frequency",
    "number_of_leads",
    "number_of_samples"
]

feature_columns = [
    column
    for column in data.columns
    if column not in excluded_columns
]


print("\nSIGNAL CHARACTERISTICS")
print("----------------------")
print("Number of characteristics:", len(feature_columns))

print("\nCharacteristics:")
for column in feature_columns:
    print("-", column)


# ============================================================
# 3. CLASS-WISE STATISTICS
# ============================================================

acceptable = data[
    data["label"] == "Acceptable"
]

unacceptable = data[
    data["label"] == "Unacceptable"
]


print("\nCLASS COUNTS")
print("------------")
print("Acceptable:", len(acceptable))
print("Unacceptable:", len(unacceptable))


# ============================================================
# 4. CALCULATE STATISTICS
# ============================================================

results = []


for feature in feature_columns:

    acceptable_values = acceptable[feature].values
    unacceptable_values = unacceptable[feature].values

    acceptable_mean = np.mean(
        acceptable_values
    )

    unacceptable_mean = np.mean(
        unacceptable_values
    )

    acceptable_median = np.median(
        acceptable_values
    )

    unacceptable_median = np.median(
        unacceptable_values
    )

    acceptable_std = np.std(
        acceptable_values,
        ddof=1
    )

    unacceptable_std = np.std(
        unacceptable_values,
        ddof=1
    )

    acceptable_min = np.min(
        acceptable_values
    )

    unacceptable_min = np.min(
        unacceptable_values
    )

    acceptable_max = np.max(
        acceptable_values
    )

    unacceptable_max = np.max(
        unacceptable_values
    )


    # --------------------------------------------------------
    # Pooled standard deviation
    # --------------------------------------------------------

    pooled_std = np.sqrt(
        (
            acceptable_std ** 2
            +
            unacceptable_std ** 2
        ) / 2
    )


    # --------------------------------------------------------
    # Standardized mean difference
    # --------------------------------------------------------

    if pooled_std > 0:

        effect_size = (
            unacceptable_mean
            -
            acceptable_mean
        ) / pooled_std

    else:

        effect_size = 0.0


    results.append({
        "feature": feature,

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

        "acceptable_min":
            acceptable_min,

        "unacceptable_min":
            unacceptable_min,

        "acceptable_max":
            acceptable_max,

        "unacceptable_max":
            unacceptable_max,

        "absolute_effect_size":
            abs(effect_size)
    })


# ============================================================
# 5. CREATE RESULTS TABLE
# ============================================================

analysis = pd.DataFrame(results)


# ============================================================
# 6. SORT BY CLASS SEPARATION
# ============================================================

analysis = analysis.sort_values(
    "absolute_effect_size",
    ascending=False
)


# ============================================================
# 7. PRINT RESULTS
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    200
)

pd.set_option(
    "display.float_format",
    lambda value: f"{value:.6f}"
)


print("\n")
print("=" * 120)
print("CLASS-WISE SIGNAL CHARACTERISTIC ANALYSIS")
print("=" * 120)

print(
    analysis.to_string(index=False)
)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

output_path = (
    "results/week5/"
    "feature_distribution_analysis.csv"
)

analysis.to_csv(
    output_path,
    index=False
)


print("\n")
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print("\nResults saved to:")
print(output_path)


# ============================================================
# 9. TOP CHARACTERISTICS
# ============================================================

print("\nTOP CHARACTERISTICS BY CLASS SEPARATION")
print("----------------------------------------")

top_count = min(
    5,
    len(analysis)
)

for index in range(top_count):

    row = analysis.iloc[index]

    print(
        f"{index + 1}. "
        f"{row['feature']} "
        f"(effect size = "
        f"{row['absolute_effect_size']:.4f})"
    )

#The important thing is that Adaptive STP isn't just "more noise → more filtering."

#It becomes:

#Measure the characteristics of the current ECG → determine what kind of processing is 
#appropriate → configure the subsequent scalogram/tensor pipeline accordingly.

##Number of characteristics: 11

#Characteristics:
#- mean_rms
#- max_rms
#- mean_std
#- max_std
#- mean_abs_amplitude
#- mean_baseline_variation
#- max_baseline_variation
#- mean_high_frequency_ratio
#- max_high_frequency_ratio
#- mean_powerline_ratio
#- max_powerline_ratio