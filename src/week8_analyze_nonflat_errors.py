
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results/week8/missed_ecg_analysis"
    / "all_records_with_characteristics.csv"
)

OUT_DIR = ROOT / "results/week8/nonflat_error_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(INPUT, dtype={"record_id": str})

    df["mean_std"] = pd.to_numeric(
        df["mean_std"], errors="coerce"
    )

    # Use the same flat-signal definition as the previous analysis.
    df["is_flat"] = df["mean_std"].abs() <= 1e-8

    # Keep non-flat unacceptable ECGs for this diagnostic only.
    nonflat_bad = df[
        (df["true_label"] == 1)
        & (~df["is_flat"])
    ].copy()

    nonflat_bad["result"] = np.where(
        nonflat_bad["adaptive_prediction"] == 1,
        "Detected",
        "Missed",
    )

    characteristics = [
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
        col for col in characteristics
        if col in nonflat_bad.columns
    ]

    print("\nNON-FLAT UNACCEPTABLE ECGS")
    print("=" * 65)
    print("Total:", len(nonflat_bad))
    print(
        nonflat_bad["result"].value_counts().to_string()
    )

    summary = (
        nonflat_bad.groupby("result")[available]
        .agg(["count", "median", "mean"])
        .round(5)
    )

    print("\nCHARACTERISTIC COMPARISON")
    print("=" * 65)
    print(summary.to_string())

    summary.to_csv(
        OUT_DIR / "nonflat_characteristic_summary.csv"
    )

    fold_summary = (
        nonflat_bad.groupby(["outer_fold", "result"])
        .size()
        .unstack(fill_value=0)
    )

    for col in ["Detected", "Missed"]:
        if col not in fold_summary.columns:
            fold_summary[col] = 0

    fold_summary["total"] = (
        fold_summary["Detected"] + fold_summary["Missed"]
    )

    fold_summary["sensitivity"] = np.where(
        fold_summary["total"] > 0,
        fold_summary["Detected"] / fold_summary["total"],
        np.nan,
    )

    fold_summary["missed_percentage"] = (
        100 * fold_summary["Missed"] / fold_summary["total"]
    )

    print("\nNON-FLAT ERRORS BY OUTER FOLD")
    print("=" * 65)
    print(fold_summary.round(4).to_string())

    fold_summary.to_csv(
        OUT_DIR / "nonflat_errors_by_fold.csv"
    )

    # Export record-level results for further inspection.
    nonflat_bad.to_csv(
        OUT_DIR / "nonflat_unacceptable_records.csv",
        index=False,
    )

    print("\nSaved results to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
