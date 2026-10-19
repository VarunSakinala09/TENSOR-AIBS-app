
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    ROOT / "results/week8/lead_quality_features"
    / "lead_quality_features.csv"
)
GROUP_PATH = (
    ROOT / "results/week4/group_aware_cv"
    / "record_signal_groups.csv"
)
OUT = ROOT / "results/week8/lead_feature_robustness"
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
        raise ValueError("Unknown label values found")
    return result.to_numpy(dtype=int)


def evaluate(y, pred):
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(
            y, pred, zero_division=0
        ),
        "sensitivity": recall_score(
            y, pred, pos_label=1, zero_division=0
        ),
        "specificity": tn / (tn + fp) if tn + fp else 0,
        "f1": f1_score(y, pred, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def make_model(seed):
    return RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )


def main():
    df = pd.read_csv(FEATURE_PATH, dtype={"record_id": str})
    groups_df = pd.read_csv(
        GROUP_PATH,
        dtype={"record_id": str, "signal_hash": str},
    )

    for frame in (df, groups_df):
        frame["record_id"] = (
            frame["record_id"].astype(str).str.strip()
        )

    if df["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs in feature file")

    if groups_df["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs in group file")

    df = df.set_index("record_id")
    groups_df = groups_df.set_index("record_id")

    common = df.index.intersection(groups_df.index).sort_values()
    df = df.loc[common]
    groups_df = groups_df.loc[common]

    y = encode_labels(df["label"])
    group_y = encode_labels(groups_df["label"])

    if not np.array_equal(y, group_y):
        raise ValueError("Feature labels and group labels differ")

    groups = groups_df["signal_hash"].astype(str).to_numpy()

    feature_cols = [
        c for c in df.columns if c != "label"
    ]
    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    if X_df.isna().any().any():
        raise ValueError("Missing or non-finite feature values")

    # Identify explicit constant-lead / -80 mV indicators.
    artifact_cols = [
        c for c in feature_cols
        if any(term in c.lower() for term in [
            "constant",
            "minus80",
            "minus_80",
            "minus-80",
        ])
    ]

    if not artifact_cols:
        raise ValueError(
            "No constant-lead or -80 mV feature columns found. "
            "Inspect the feature column names before proceeding."
        )

    reduced_cols = [
        c for c in feature_cols if c not in artifact_cols
    ]

    # Keep signal-statistics features, excluding explicit markers.
    methods = {
        "all_lead_features": feature_cols,
        "without_artifact_markers": reduced_cols,
        "statistics_only": [
            c for c in reduced_cols
            if any(term in c.lower() for term in [
                "rms", "std", "range", "amplitude",
                "variation", "slope", "power", "ratio",
                "skew", "kurtosis",
            ])
        ],
    }

    for name, cols in methods.items():
        if not cols:
            raise ValueError(f"No features available for {name}")

    print("\nLEAD-FEATURE ROBUSTNESS EXPERIMENT")
    print("=" * 60)
    print("Records:", len(y))
    print("Unique signal groups:", len(np.unique(groups)))
    print("Artifact-marker columns excluded in reduced models:")
    for col in artifact_cols:
        print(" ", col)

    for name, cols in methods.items():
        print(f"{name}: {len(cols)} features")

    X = {
        name: X_df[cols].to_numpy(dtype=np.float32)
        for name, cols in methods.items()
    }

    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=42
    )

    fold_rows = []
    oof_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X_df, y, groups), start=1
    ):
        if set(groups[train_idx]).intersection(groups[test_idx]):
            raise RuntimeError("Signal-group leakage detected")

        print(f"\nFold {fold}/5")

        for name, matrix in X.items():
            model = make_model(42 + fold)
            model.fit(matrix[train_idx], y[train_idx])
            pred = model.predict(matrix[test_idx]).astype(int)

            score = evaluate(y[test_idx], pred)
            fold_rows.append({
                "fold": fold,
                "method": name,
                **score,
            })

            for j, idx in enumerate(test_idx):
                oof_rows.append({
                    "record_id": common[idx],
                    "fold": fold,
                    "method": name,
                    "true_label": int(y[idx]),
                    "prediction": int(pred[j]),
                })

            print(
                f"  {name}: F1={score['f1']:.4f}, "
                f"Sensitivity={score['sensitivity']:.4f}, "
                f"Specificity={score['specificity']:.4f}"
            )

    folds = pd.DataFrame(fold_rows)
    oof = pd.DataFrame(oof_rows)

    folds.to_csv(OUT / "fold_metrics.csv", index=False)
    oof.to_csv(OUT / "out_of_fold_predictions.csv", index=False)

    summary = (
        folds.groupby("method")[
            ["accuracy", "precision", "sensitivity",
             "specificity", "f1"]
        ].agg(["mean", "std"]).round(4)
    )
    summary.to_csv(OUT / "fold_summary.csv")

    pooled_rows = []
    for name, part in oof.groupby("method"):
        pooled_rows.append({
            "method": name,
            **evaluate(
                part["true_label"].to_numpy(),
                part["prediction"].to_numpy(),
            ),
        })

    pooled = pd.DataFrame(pooled_rows).round(4)
    pooled.to_csv(OUT / "pooled_oof_metrics.csv", index=False)

    print("\nFOLD SUMMARY")
    print(summary.to_string())

    print("\nPOOLED OUT-OF-FOLD RESULTS")
    print(pooled.to_string(index=False))

    print("\nResults saved to:", OUT)


if __name__ == "__main__":
    main()
