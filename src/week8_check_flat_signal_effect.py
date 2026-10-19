
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

characteristics = pd.read_csv(
    ROOT / "results/week5/signal_characteristics.csv"
)

predictions = pd.read_csv(
    ROOT / "results/week8/adaptive_group_cv"
    / "out_of_fold_predictions.csv"
)

data = characteristics.merge(
    predictions[
        ["record_id", "true_label", "adaptive_prediction"]
    ],
    on="record_id",
    how="inner",
    validate="one_to_one",
)

bad = data[data["true_label"].astype(int) == 1].copy()

# Flag records whose twelve leads appear to have zero variability.
bad["flat_signal"] = bad["mean_std"].abs() < 1e-12

bad["outcome"] = bad["adaptive_prediction"].astype(int).map(
    {1: "Detected", 0: "Missed"}
)

print("\nFLAT-SIGNAL CHECK")
print("=" * 60)

print(
    bad.groupby("outcome")["flat_signal"]
    .agg(["count", "sum"])
    .to_string()
)

print("\nFLAT-SIGNAL RATE BY OUTCOME")
print("=" * 60)

print(
    (bad.groupby("outcome")["flat_signal"].mean() * 100)
    .round(2)
    .astype(str)
    .add("%")
    .to_string()
)

nonflat = bad[~bad["flat_signal"]]

features = [
    "mean_baseline_variation",
    "mean_high_frequency_ratio",
    "mean_powerline_ratio",
    "mean_rms",
    "mean_std",
]

print("\nNON-FLAT UNACCEPTABLE ECGS: DETECTED VS MISSED")
print("=" * 75)

print(
    nonflat.groupby("outcome")[features]
    .median()
    .T.round(6)
    .to_string()
)

print("\nCounts after removing flat signals:")
print(nonflat["outcome"].value_counts().to_string())

print("\nAnalysis complete. No models retrained.")
