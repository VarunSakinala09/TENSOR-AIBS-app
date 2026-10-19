
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    ROOT / "results/week8/lead_quality_features"
    / "lead_quality_features.csv"
)
GROUP_PATH = (
    ROOT / "results/week4/group_aware_cv"
    / "record_signal_groups.csv"
)
OUT = ROOT / "results/week8/lead_feature_importance"
OUT.mkdir(parents=True, exist_ok=True)


def encode_labels(series):
    result = (
        series.astype(str).str.strip().str.lower()
        .map({
            "acceptable": 0,
            "accept": 0,
            "0": 0,
            "unacceptable": 1,
            "unaccept": 1,
            "1": 1,
        })
    )
    if result.isna().any():
        raise ValueError("Unknown label values")
    return result.to_numpy(dtype=int)


def main():
    df = pd.read_csv(FEATURE_PATH, dtype={"record_id": str})
    group_df = pd.read_csv(
        GROUP_PATH,
        dtype={"record_id": str, "signal_hash": str},
    )

    for frame in (df, group_df):
        frame["record_id"] = (
            frame["record_id"].astype(str).str.strip()
        )

    if df["record_id"].duplicated().any():
        raise ValueError("Duplicate feature record IDs")
    if group_df["record_id"].duplicated().any():
        raise ValueError("Duplicate group record IDs")

    df = df.set_index("record_id")
    group_df = group_df.set_index("record_id")

    ids = df.index.intersection(group_df.index).sort_values()
    df = df.loc[ids]
    group_df = group_df.loc[ids]

    y = encode_labels(df["label"])
    group_y = encode_labels(group_df["label"])

    if not np.array_equal(y, group_y):
        raise ValueError("Labels do not match")

    groups = group_df["signal_hash"].astype(str).to_numpy()

    feature_cols = [
        col for col in df.columns if col != "label"
    ]

    artifact_cols = [
        col for col in feature_cols
        if any(term in col.lower() for term in [
            "constant",
            "minus80",
            "minus_80",
            "minus-80",
        ])
    ]

    feature_cols = [
        col for col in feature_cols if col not in artifact_cols
    ]

    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    if X_df.isna().any().any():
        raise ValueError("Missing or invalid feature values")

    X = X_df.to_numpy(dtype=np.float32)

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    importance_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):
        if set(groups[train_idx]).intersection(groups[test_idx]):
            raise RuntimeError("Signal-group leakage detected")

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42 + fold,
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(X[train_idx], y[train_idx])

        result = permutation_importance(
            model,
            X[test_idx],
            y[test_idx],
            scoring="f1",
            n_repeats=10,
            random_state=100 + fold,
            n_jobs=-1,
        )

        for i, feature in enumerate(feature_cols):
            importance_rows.append({
                "fold": fold,
                "feature": feature,
                "importance_mean": result.importances_mean[i],
                "importance_std": result.importances_std[i],
            })

        print(f"Completed held-out importance for fold {fold}/5")

    importance_df = pd.DataFrame(importance_rows)
    importance_df.to_csv(
        OUT / "importance_by_fold.csv", index=False
    )

    summary = (
        importance_df.groupby("feature")
        .agg(
            mean_importance=("importance_mean", "mean"),
            fold_std=("importance_mean", "std"),
            mean_permutation_std=("importance_std", "mean"),
            positive_folds=(
                "importance_mean",
                lambda values: int((values > 0).sum()),
            ),
        )
        .sort_values("mean_importance", ascending=False)
        .reset_index()
        .round(5)
    )

    summary.to_csv(OUT / "importance_summary.csv", index=False)

    print("\nFEATURE IMPORTANCE SUMMARY")
    print("=" * 75)
    print(summary.to_string(index=False))

    print("\nSaved results to:", OUT)


if __name__ == "__main__":
    main()
