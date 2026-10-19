
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# -------------------- Paths --------------------

ROOT = Path(".")
RESULTS = ROOT / "results/week7/adaptive_selector"

FOLD_FILE = RESULTS / "fold_results.csv"
PREDICTION_FILE = RESULTS / "per_record_adaptive_predictions.csv"

OUTPUT = ROOT / "results/week7/selector_stability"
OUTPUT.mkdir(parents=True, exist_ok=True)


# -------------------- Metric calculation --------------------

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "records": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "sensitivity": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "specificity": tn / (tn + fp) if tn + fp else 0,
        "f1": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
    }


# -------------------- Load previous results --------------------

def main():
    if not FOLD_FILE.exists():
        raise FileNotFoundError(
            f"Missing {FOLD_FILE}. Run Experiment 7E first."
        )

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Missing {PREDICTION_FILE}. Run Experiment 7E first."
        )

    folds = pd.read_csv(FOLD_FILE)
    df = pd.read_csv(PREDICTION_FILE)

    print("\nEXPERIMENT 7F: SELECTOR STABILITY ANALYSIS")
    print("=" * 60)

    # -------------------- 1. Threshold stability --------------------

    print("\n1. LEARNED THRESHOLDS")
    print("-" * 60)

    print(
        folds[
            ["fold", "learned_threshold",
             "test_records_using_50Hz"]
        ].to_string(index=False)
    )

    thresholds = folds["learned_threshold"]

    print(f"\nMean threshold:   {thresholds.mean():.6f}")
    print(f"Median threshold: {thresholds.median():.6f}")
    print(f"Minimum threshold:{thresholds.min():.6f}")
    print(f"Maximum threshold:{thresholds.max():.6f}")
    print(f"Threshold std:    {thresholds.std(ddof=1):.6f}")

    # -------------------- 2. Configuration usage --------------------

    print("\n2. CONFIGURATION SELECTION")
    print("-" * 60)

    selection = (
        df.groupby(["outer_fold", "selected_configuration"])
        .size()
        .unstack(fill_value=0)
    )

    print(selection.to_string())

    if "CWT_1_40_Hz" not in selection.columns:
        selection["CWT_1_40_Hz"] = 0

    if "CWT_1_50_Hz" not in selection.columns:
        selection["CWT_1_50_Hz"] = 0

    selection["total"] = (
        selection["CWT_1_40_Hz"] + selection["CWT_1_50_Hz"]
    )

    selection["percent_using_50Hz"] = (
        100 * selection["CWT_1_50_Hz"] / selection["total"]
    )

    print("\nPercentage routed to 1–50 Hz:")
    print(selection["percent_using_50Hz"].round(2).to_string())

    # -------------------- 3. Direct model comparison --------------------

    print("\n3. SAME-RECORD COMPARISON")
    print("-" * 60)

    y = df["true_label"].astype(int).to_numpy()
    p40 = df["fixed40_prediction"].astype(int).to_numpy()
    p50 = df["fixed50_prediction"].astype(int).to_numpy()
    pa = df["adaptive_prediction"].astype(int).to_numpy()

    for name, pred in [
        ("Fixed-40", p40),
        ("Fixed-50", p50),
        ("Adaptive", pa),
    ]:
        scores = calculate_metrics(y, pred)
        print(
            f"{name:10s} | "
            f"Accuracy={scores['accuracy']:.4f} | "
            f"Sensitivity={scores['sensitivity']:.4f} | "
            f"Specificity={scores['specificity']:.4f} | "
            f"F1={scores['f1']:.4f}"
        )

    adaptive_differs_50 = pa != p50
    adaptive_differs_40 = pa != p40

    print(
        "\nAdaptive predictions differing from Fixed-50:",
        int(adaptive_differs_50.sum()),
    )
    print(
        "Adaptive predictions differing from Fixed-40:",
        int(adaptive_differs_40.sum()),
    )

    # Post-hoc evaluation only: test labels assess correctness here.
    # They are NOT used to select the configuration.
    correct40 = p40 == y
    correct50 = p50 == y
    correct_adaptive = pa == y

    print("\nAmong records where Adaptive differs from Fixed-50:")
    mask = adaptive_differs_50

    if mask.any():
        print("Number of differing predictions:", int(mask.sum()))
        print(
            "Adaptive correct:",
            int(correct_adaptive[mask].sum()),
        )
        print(
            "Fixed-50 correct:",
            int(correct50[mask].sum()),
        )
    else:
        print("No differing predictions.")

    # -------------------- 4. Condition-wise performance --------------------

    print("\n4. PERFORMANCE BY BASELINE-VARIATION GROUP")
    print("-" * 60)

    # Exploratory groups based on signal condition only.
    # This is a descriptive analysis, not a learned deployment rule.
    df["condition_group"] = pd.qcut(
        df["mean_baseline_variation"].rank(method="first"),
        q=3,
        labels=["Low", "Medium", "High"],
    )

    condition_rows = []

    for group in ["Low", "Medium", "High"]:
        subset = df[df["condition_group"] == group]
        y_group = subset["true_label"].astype(int).to_numpy()

        print(f"\n{group} condition: {len(subset)} ECGs")

        for name, column in [
            ("Fixed-40", "fixed40_prediction"),
            ("Fixed-50", "fixed50_prediction"),
            ("Adaptive", "adaptive_prediction"),
        ]:
            pred = subset[column].astype(int).to_numpy()
            scores = calculate_metrics(y_group, pred)

            condition_rows.append({
                "condition_group": group,
                "method": name,
                **scores,
            })

            print(
                f"{name:10s} | "
                f"Accuracy={scores['accuracy']:.4f} | "
                f"Sensitivity={scores['sensitivity']:.4f} | "
                f"F1={scores['f1']:.4f}"
            )

    # -------------------- 5. Save outputs --------------------

    selection.to_csv(OUTPUT / "configuration_usage.csv")
    pd.DataFrame(condition_rows).to_csv(
        OUTPUT / "condition_wise_results.csv", index=False
    )

    threshold_summary = pd.DataFrame([{
        "mean_threshold": thresholds.mean(),
        "median_threshold": thresholds.median(),
        "minimum_threshold": thresholds.min(),
        "maximum_threshold": thresholds.max(),
        "threshold_std": thresholds.std(ddof=1),
        "adaptive_differs_from_fixed50": int(
            adaptive_differs_50.sum()
        ),
        "adaptive_differs_from_fixed40": int(
            adaptive_differs_40.sum()
        ),
    }])

    threshold_summary.to_csv(
        OUTPUT / "threshold_stability_summary.csv", index=False
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT 7F COMPLETED")
    print("Saved results to:", OUTPUT.resolve())
    print("=" * 60)


if __name__ == "__main__":
    main()
