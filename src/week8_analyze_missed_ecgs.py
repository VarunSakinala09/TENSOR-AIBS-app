
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    ROOT
    / "results/week8/adaptive_group_cv"
    / "out_of_fold_predictions.csv"
)

CHARACTERISTICS_PATH = (
    ROOT
    / "results/week5/signal_characteristics.csv"
)

OUT_DIR = ROOT / "results/week8/missed_ecg_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def main():
    print("\nMISSED UNACCEPTABLE ECG ANALYSIS")
    print("=" * 65)

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        dtype={"record_id": str},
    )

    characteristics = pd.read_csv(
        CHARACTERISTICS_PATH,
        dtype={"record_id": str},
    )

    for df in [predictions, characteristics]:
        df["record_id"] = (
            df["record_id"].astype(str).str.strip()
        )

    if predictions["record_id"].duplicated().any():
        raise ValueError(
            "Duplicate record IDs in prediction file."
        )

    if characteristics["record_id"].duplicated().any():
        raise ValueError(
            "Duplicate record IDs in characteristics file."
        )

    required = {
        "true_label",
        "adaptive_prediction",
        "outer_fold",
    }

    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing)}"
        )

    # Validate binary labels: 0 = acceptable, 1 = unacceptable.
    predictions["true_label"] = pd.to_numeric(
        predictions["true_label"], errors="raise"
    ).astype(int)

    predictions["adaptive_prediction"] = pd.to_numeric(
        predictions["adaptive_prediction"], errors="raise"
    ).astype(int)

    if not predictions["true_label"].isin([0, 1]).all():
        raise ValueError("Unexpected true-label values.")

    if not predictions["adaptive_prediction"].isin([0, 1]).all():
        raise ValueError("Unexpected prediction values.")

    # Retain every record in the prediction dataset.
    merged = predictions.merge(
        characteristics,
        on="record_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_characteristic"),
        indicator=True,
    )

    unmatched = int((merged["_merge"] != "both").sum())

    print("Prediction records:", len(predictions))
    print("Matched characteristics:", len(merged) - unmatched)
    print("Unmatched characteristics:", unmatched)

    if unmatched:
        raise ValueError(
            "Some records lack signal characteristics. "
            "Resolve the ID mismatch before continuing."
        )

    merged = merged.drop(columns="_merge")

    # ========================================================
    # CLASSIFY EACH PREDICTION
    # ========================================================

    conditions = [
        (merged["true_label"] == 1)
        & (merged["adaptive_prediction"] == 1),

        (merged["true_label"] == 1)
        & (merged["adaptive_prediction"] == 0),

        (merged["true_label"] == 0)
        & (merged["adaptive_prediction"] == 1),

        (merged["true_label"] == 0)
        & (merged["adaptive_prediction"] == 0),
    ]

    merged["prediction_group"] = np.select(
        conditions,
        [
            "True Positive",
            "False Negative",
            "False Positive",
            "True Negative",
        ],
        default="Unknown",
    )

    merged["correct_prediction"] = (
        merged["true_label"]
        == merged["adaptive_prediction"]
    )

    # Save all records with their characteristics and errors.
    merged.to_csv(
        OUT_DIR / "all_records_with_characteristics.csv",
        index=False,
    )

    # ========================================================
    # CONFUSION-GROUP SUMMARY
    # ========================================================

    group_summary = (
        merged.groupby("prediction_group")
        .agg(
            records=("record_id", "size"),
            unique_signal_groups=("signal_hash", "nunique"),
        )
        .reset_index()
    )

    group_summary.to_csv(
        OUT_DIR / "prediction_group_counts.csv",
        index=False,
    )

    print("\nPREDICTION GROUPS")
    print("-" * 65)
    print(group_summary.to_string(index=False))

    # ========================================================
    # UNACCEPTABLE ECG ANALYSIS
    # ========================================================

    unacceptable = merged[
        merged["true_label"] == 1
    ].copy()

    detected = unacceptable[
        unacceptable["adaptive_prediction"] == 1
    ].copy()

    missed = unacceptable[
        unacceptable["adaptive_prediction"] == 0
    ].copy()

    missed.to_csv(
        OUT_DIR / "missed_unacceptable_ecgs.csv",
        index=False,
    )

    detected.to_csv(
        OUT_DIR / "detected_unacceptable_ecgs.csv",
        index=False,
    )

    print("\nUNACCEPTABLE ECGS")
    print("-" * 65)
    print("Total:", len(unacceptable))
    print("Detected:", len(detected))
    print("Missed:", len(missed))

    if len(unacceptable):
        print(
            "Sensitivity:",
            round(len(detected) / len(unacceptable), 4),
        )

    # ========================================================
    # CHARACTERISTIC COMPARISON
    # ========================================================

    characteristic_columns = [
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
        "max_powerline_ratio",
    ]

    available = [
        col for col in characteristic_columns
        if col in merged.columns
    ]

    comparison_rows = []

    groups_to_compare = {
        "Detected unacceptable": detected,
        "Missed unacceptable": missed,
        "False positive": merged[
            merged["prediction_group"] == "False Positive"
        ],
        "True negative": merged[
            merged["prediction_group"] == "True Negative"
        ],
    }

    for group_name, group_df in groups_to_compare.items():
        for column in available:
            values = pd.to_numeric(
                group_df[column], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).dropna()

            comparison_rows.append({
                "prediction_group": group_name,
                "characteristic": column,
                "records_with_value": len(values),
                "median": (
                    float(values.median())
                    if len(values) else np.nan
                ),
                "mean": (
                    float(values.mean())
                    if len(values) else np.nan
                ),
                "standard_deviation": (
                    float(values.std())
                    if len(values) > 1 else np.nan
                ),
            })

    comparison = pd.DataFrame(comparison_rows)

    comparison.to_csv(
        OUT_DIR / "characteristic_comparison.csv",
        index=False,
    )

    print("\nCHARACTERISTIC COMPARISON")
    print("-" * 65)

    print(
        comparison[
            comparison["prediction_group"].isin([
                "Detected unacceptable",
                "Missed unacceptable",
            ])
        ].round(5).to_string(index=False)
    )

    # ========================================================
    # FLAT / DEGENERATE SIGNAL CHECK
    # ========================================================

    if "mean_std" in merged.columns:
        merged["mean_std_numeric"] = pd.to_numeric(
            merged["mean_std"], errors="coerce"
        )

        flat_mask = (
            merged["mean_std_numeric"].abs() <= 1e-8
        )

        flat_summary = (
            merged.assign(
                flat_or_degenerate=flat_mask
            )
            .groupby([
                "true_label",
                "adaptive_prediction",
                "flat_or_degenerate",
            ])
            .size()
            .reset_index(name="records")
        )

        flat_summary.to_csv(
            OUT_DIR / "flat_signal_summary.csv",
            index=False,
        )

        print("\nFLAT / DEGENERATE SIGNAL CHECK")
        print("-" * 65)
        print(flat_summary.to_string(index=False))

    # ========================================================
    # FOLD-WISE MISSED RECORDS
    # ========================================================

    fold_summary = (
        unacceptable.assign(
            missed=(
                unacceptable["adaptive_prediction"] == 0
            )
        )
        .groupby("outer_fold")
        .agg(
            unacceptable_records=("record_id", "size"),
            missed_records=("missed", "sum"),
        )
        .reset_index()
    )

    fold_summary["missed_percentage"] = (
        100
        * fold_summary["missed_records"]
        / fold_summary["unacceptable_records"]
    )

    fold_summary.to_csv(
        OUT_DIR / "missed_records_by_fold.csv",
        index=False,
    )

    print("\nMISSED RECORDS BY OUTER FOLD")
    print("-" * 65)
    print(fold_summary.round(2).to_string(index=False))

    print("\nAnalysis completed.")
    print("Results saved to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
