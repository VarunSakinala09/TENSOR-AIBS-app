
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILE = (
    ROOT / "results/week8/adaptive_group_cv"
    / "out_of_fold_predictions.csv"
)

OUT_DIR = ROOT / "results/week8/adaptive_group_cv"
df = pd.read_csv(FILE, dtype={"record_id": str})

methods = {
    "fixed40": "fixed40_prediction",
    "fixed50": "fixed50_prediction",
    "adaptive": "adaptive_prediction",
}

# Positive class: Unacceptable = 1
for method, col in methods.items():
    df[f"{method}_correct"] = (
        df[col] == df["true_label"]
    )

print("\nADAPTIVE ERROR ANALYSIS")
print("=" * 60)
print("Records:", len(df))
print("Unique record IDs:", df["record_id"].nunique())
print("Unique signal groups:", df["signal_hash"].nunique())

print("\nOVERALL CORRECT / INCORRECT COUNTS")
for method in methods:
    print(
        f"{method}:",
        df[f"{method}_correct"].value_counts().to_dict()
    )

# Focus on unacceptable records.
bad = df[df["true_label"] == 1].copy()

print("\nUNACCEPTABLE ECGS")
print("Total:", len(bad))

for method in methods:
    correct = int(bad[f"{method}_correct"].sum())
    print(
        f"{method}: detected {correct}/{len(bad)}; "
        f"missed {len(bad) - correct}"
    )

adaptive_rescues_40 = bad[
    (~bad["fixed40_correct"]) &
    (bad["adaptive_correct"])
]

adaptive_misses_40 = bad[
    (bad["fixed40_correct"]) &
    (~bad["adaptive_correct"])
]

print("\nUNACCEPTABLE ECGS RESCUED BY ADAPTIVE VS FIXED 40")
print("Count:", len(adaptive_rescues_40))
print(
    adaptive_rescues_40[
        [
            "record_id",
            "outer_fold",
            "mean_baseline_variation",
            "learned_threshold",
            "selected_configuration",
            "fixed40_prediction",
            "fixed50_prediction",
            "adaptive_prediction",
        ]
    ].to_string(index=False)
)

print("\nUNACCEPTABLE ECGS LOST BY ADAPTIVE VS FIXED 40")
print("Count:", len(adaptive_misses_40))
print(
    adaptive_misses_40[
        [
            "record_id",
            "outer_fold",
            "mean_baseline_variation",
            "learned_threshold",
            "selected_configuration",
            "fixed40_prediction",
            "fixed50_prediction",
            "adaptive_prediction",
        ]
    ].to_string(index=False)
)

# False alarms: acceptable ECGs predicted as unacceptable.
acceptable = df[df["true_label"] == 0].copy()

print("\nFALSE ALARMS ON ACCEPTABLE ECGS")
for method, col in methods.items():
    false_alarms = int(
        ((acceptable[col] == 1)).sum()
    )
    print(f"{method}: {false_alarms}/{len(acceptable)}")

# Save a copy with correctness indicators for later analysis.
df.to_csv(
    OUT_DIR / "adaptive_error_analysis.csv",
    index=False,
)

print("\nSaved:")
print(OUT_DIR / "adaptive_error_analysis.csv")
