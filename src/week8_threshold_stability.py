
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/week8/adaptive_group_cv"

folds = pd.read_csv(OUT_DIR / "fold_metrics.csv")
predictions = pd.read_csv(
    OUT_DIR / "out_of_fold_predictions.csv",
    dtype={"record_id": str},
)

thresholds = folds["learned_threshold"].to_numpy(dtype=float)

print("\nTHRESHOLD STABILITY")
print("=" * 55)
print("Mean threshold:", round(float(np.mean(thresholds)), 4))
print("Standard deviation:", round(float(np.std(thresholds, ddof=1)), 4))
print("Minimum threshold:", round(float(np.min(thresholds)), 4))
print("Maximum threshold:", round(float(np.max(thresholds)), 4))
print("Threshold range:", round(float(np.ptp(thresholds)), 4))

print("\nFREQUENCY SELECTION BY OUTER FOLD")
print(
    predictions.groupby(
        ["outer_fold", "selected_configuration"]
    ).size().unstack(fill_value=0).to_string()
)

print("\nSELECTION RATE")
print(
    predictions.groupby("outer_fold")
    ["selected_configuration"]
    .apply(lambda s: (s == "CWT_1_50_Hz").mean())
    .round(4)
    .to_string()
)

# This is descriptive only. Each fold's threshold was learned
# independently; we do not retune or evaluate a new model here.
folds[
    [
        "fold",
        "learned_threshold",
        "test_records_using_50Hz",
        "adaptive_sensitivity",
        "adaptive_specificity",
        "adaptive_f1",
    ]
].to_csv(OUT_DIR / "threshold_stability_summary.csv", index=False)

print("\nSaved threshold_stability_summary.csv")
