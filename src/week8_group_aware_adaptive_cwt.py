
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FIXED40_PATH = ROOT / "results/week3/features_full_dataset.csv"
FIXED50_PATH = ROOT / "results/week7/cwt_d/cwt_d_features.csv"

GROUP_PATH = (
    ROOT / "results/week4/group_aware_cv"
    / "record_signal_groups.csv"
)

OUT_DIR = ROOT / "results/week8/adaptive_group_cv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "record_id"
LABEL_COL = "label"
CONDITION_COL = "mean_baseline_variation"

OUTER_FOLDS = 5
INNER_FOLDS = 3
SEED = 42
N_ESTIMATORS = 100

QUANTILES = np.linspace(0.10, 0.90, 17)


# ============================================================
# HELPERS
# ============================================================

def convert_labels(series):
    values = series.astype(str).str.strip().str.lower()

    mapping = {
        "acceptable": 0,
        "accept": 0,
        "0": 0,
        "unacceptable": 1,
        "unaccept": 1,
        "1": 1,
    }

    result = values.map(mapping)

    if result.isna().any():
        raise ValueError(
            f"Unknown labels: {values[result.isna()].unique().tolist()}"
        )

    return result.to_numpy(dtype=int)


def make_model():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced",
    )


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "sensitivity": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "specificity": (
            tn / (tn + fp) if tn + fp else 0.0
        ),
        "f1": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
    }


def load_feature_file(path, name):
    df = pd.read_csv(path, dtype={ID_COL: str})

    if ID_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"{name} is missing ID or label columns.")

    df[ID_COL] = df[ID_COL].astype(str).str.strip()

    if df[ID_COL].duplicated().any():
        raise ValueError(f"Duplicate IDs in {name}.")

    feature_cols = [
        col for col in df.columns
        if col not in [ID_COL, LABEL_COL]
    ]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        raise ValueError(
            f"{name} contains missing/non-finite feature values. "
            "Inspect the source file rather than silently filling them."
        )

    return df, feature_cols, X.to_numpy(dtype=np.float32)


# ============================================================
# GROUP-AWARE INNER PREDICTIONS
# ============================================================

def get_inner_predictions(
    X40, X50, y, groups, seed
):
    """
    Each inner validation group is excluded from its training data.
    Predictions cover every outer-training record once.
    """

    inner_cv = StratifiedGroupKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    pred40 = np.full(len(y), -1, dtype=int)
    pred50 = np.full(len(y), -1, dtype=int)

    for train_idx, valid_idx in inner_cv.split(
        X40, y, groups
    ):
        train_groups = set(groups[train_idx])
        valid_groups = set(groups[valid_idx])

        if train_groups.intersection(valid_groups):
            raise RuntimeError(
                "Signal-group leakage detected in inner CV."
            )

        model40 = make_model()
        model50 = make_model()

        model40.fit(X40[train_idx], y[train_idx])
        model50.fit(X50[train_idx], y[train_idx])

        pred40[valid_idx] = model40.predict(X40[valid_idx])
        pred50[valid_idx] = model50.predict(X50[valid_idx])

    if (pred40 < 0).any() or (pred50 < 0).any():
        raise RuntimeError(
            "Some inner-training records did not receive predictions."
        )

    return pred40, pred50


def learn_threshold(y, pred40, pred50, condition):
    """
    Choose a threshold using outer-training labels and
    group-aware inner out-of-fold predictions only.

    condition <= threshold: choose Fixed-40 prediction
    condition > threshold:  choose Fixed-50 prediction
    """

    candidates = np.unique(
        np.quantile(condition, QUANTILES)
    )

    results = []

    for threshold in candidates:
        adaptive_pred = np.where(
            condition > threshold,
            pred50,
            pred40,
        )

        results.append({
            "threshold": float(threshold),
            **calculate_metrics(y, adaptive_pred),
        })

    results_df = pd.DataFrame(results)

    best = results_df.sort_values(
        ["f1", "sensitivity", "specificity"],
        ascending=[False, False, False],
        kind="mergesort",
    ).iloc[0]

    return float(best["threshold"]), results_df


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    print("\nGROUP-AWARE NESTED ADAPTIVE CWT EXPERIMENT")
    print("=" * 60)

    df40, cols40, X40 = load_feature_file(
        FIXED40_PATH, "Fixed-40"
    )
    df50, cols50, X50 = load_feature_file(
        FIXED50_PATH, "Fixed-50"
    )

    

    groups_df = pd.read_csv(
        GROUP_PATH,
        dtype={"record_id": str, "signal_hash": str},
    )
    groups_df["record_id"] = (
        groups_df["record_id"].astype(str).str.strip()
    )

    if groups_df["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs in group mapping.")

    if groups_df["signal_hash"].isna().any():
        raise ValueError("Missing signal hashes in group mapping.")

    # Validate that each signal group has one consistent label.
    groups_df["binary_label"] = convert_labels(
        groups_df[LABEL_COL]
    )

    label_counts = groups_df.groupby("signal_hash")[
        "binary_label"
    ].nunique()

    if (label_counts > 1).any():
        raise ValueError(
            "Some identical-signal groups have conflicting labels."
        )

    # Load signal characteristics.
    char_path = ROOT / "results/week5/signal_characteristics.csv"
    chars = pd.read_csv(char_path, dtype={"record_id": str})
    chars["record_id"] = (
        chars["record_id"].astype(str).str.strip()
    )

    if chars["record_id"].duplicated().any():
        raise ValueError("Duplicate IDs in signal characteristics.")

    if CONDITION_COL not in chars.columns:
        raise ValueError(f"Missing characteristic: {CONDITION_COL}")

    # Align everything using record IDs.
    a = df40.set_index("record_id")
    b = df50.set_index("record_id")
    g = groups_df.set_index("record_id")
    c = chars.set_index("record_id")

    common_ids = (
        a.index.intersection(b.index)
        .intersection(g.index)
        .intersection(c.index)
        .sort_values()
    )

    if len(common_ids) == 0:
        raise ValueError("No common record IDs across input files.")

    a = a.loc[common_ids]
    b = b.loc[common_ids]
    g = g.loc[common_ids]
    c = c.loc[common_ids]

    y = convert_labels(a[LABEL_COL])

    if not np.array_equal(y, convert_labels(b[LABEL_COL])):
        raise ValueError("Labels differ between feature datasets.")

    if not np.array_equal(y, g["binary_label"].to_numpy()):
        raise ValueError("Labels disagree with the group mapping.")

    condition_series = pd.to_numeric(
        c[CONDITION_COL], errors="coerce"
    )

    if not np.isfinite(condition_series.to_numpy()).all():
        raise ValueError(
            "Signal condition contains missing or non-finite values."
        )

    condition = condition_series.to_numpy(dtype=float)
    groups = g["signal_hash"].to_numpy(dtype=str)

    # Reorder feature arrays to the same sorted ID order.
    X40 = a[cols40].to_numpy(dtype=np.float32)
    X50 = b[cols50].to_numpy(dtype=np.float32)

    print("Records:", len(common_ids))
    print("Unique signal groups:", len(np.unique(groups)))
    print("Acceptable:", int((y == 0).sum()))
    print("Unacceptable:", int((y == 1).sum()))
    print("Features per configuration:", X40.shape[1])

    # Same outer folds for all three methods.
    outer_cv = StratifiedGroupKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    fold_rows = []
    prediction_rows = []
    threshold_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X40, y, groups), start=1
    ):
        print(f"\n--- Outer fold {fold}/{OUTER_FOLDS} ---")

        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])

        if train_groups.intersection(test_groups):
            raise RuntimeError(
                "Signal-group leakage detected in outer CV."
            )

        X40_train, X40_test = X40[train_idx], X40[test_idx]
        X50_train, X50_test = X50[train_idx], X50[test_idx]

        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        condition_train = condition[train_idx]
        condition_test = condition[test_idx]

        # Inner OOF predictions for threshold learning.
        inner40, inner50 = get_inner_predictions(
            X40_train,
            X50_train,
            y_train,
            groups_train,
            SEED + fold,
        )

        threshold, threshold_table = learn_threshold(
            y_train,
            inner40,
            inner50,
            condition_train,
        )

        threshold_table["outer_fold"] = fold
        threshold_rows.append(threshold_table)

        # Train each model using the complete outer-training fold.
        model40 = make_model()
        model50 = make_model()

        model40.fit(X40_train, y_train)
        model50.fit(X50_train, y_train)

        pred40 = model40.predict(X40_test).astype(int)
        pred50 = model50.predict(X50_test).astype(int)

        adaptive_pred = np.where(
            condition_test > threshold,
            pred50,
            pred40,
        )

        fold_row = {
            "fold": fold,
            "train_records": len(train_idx),
            "test_records": len(test_idx),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "learned_threshold": threshold,
            "test_records_using_50Hz": int(
                (condition_test > threshold).sum()
            ),
        }

        for method, pred in [
            ("fixed40", pred40),
            ("fixed50", pred50),
            ("adaptive", adaptive_pred),
        ]:
            for metric, value in calculate_metrics(
                y_test, pred
            ).items():
                fold_row[f"{method}_{metric}"] = value

        fold_rows.append(fold_row)

        for j, idx in enumerate(test_idx):
            prediction_rows.append({
                "record_id": common_ids[idx],
                "signal_hash": groups[idx],
                "outer_fold": fold,
                "true_label": int(y_test[j]),
                "mean_baseline_variation": float(condition[idx]),
                "learned_threshold": threshold,
                "selected_configuration": (
                    "CWT_1_50_Hz"
                    if condition[idx] > threshold
                    else "CWT_1_40_Hz"
                ),
                "fixed40_prediction": int(pred40[j]),
                "fixed50_prediction": int(pred50[j]),
                "adaptive_prediction": int(adaptive_pred[j]),
            })

        print(f"Threshold: {threshold:.6f}")
        print(f"Training records: {len(train_idx)}")
        print(f"Testing records: {len(test_idx)}")
        print(
            "Adaptive sensitivity:",
            f"{fold_row['adaptive_sensitivity']:.4f}",
        )
        print(
            "Adaptive specificity:",
            f"{fold_row['adaptive_specificity']:.4f}",
        )
        print(
            "Adaptive F1:",
            f"{fold_row['adaptive_f1']:.4f}",
        )

    # --------------------------------------------------------
    # Save fold results and predictions
    # --------------------------------------------------------

    folds_df = pd.DataFrame(fold_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    thresholds_df = pd.concat(
        threshold_rows, ignore_index=True
    )

    folds_df.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    predictions_df.to_csv(
        OUT_DIR / "out_of_fold_predictions.csv", index=False
    )
    thresholds_df.to_csv(
        OUT_DIR / "threshold_search.csv", index=False
    )

    # Fold mean and standard deviation.
    summary_rows = []

    for method in ["fixed40", "fixed50", "adaptive"]:
        for metric in [
            "accuracy", "precision", "sensitivity",
            "specificity", "f1",
        ]:
            values = folds_df[f"{method}_{metric}"].to_numpy()

            summary_rows.append({
                "method": method,
                "metric": metric,
                "fold_mean": float(values.mean()),
                "fold_std": float(values.std(ddof=1)),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        OUT_DIR / "summary_results.csv", index=False
    )

    # Combined outer OOF metrics: every record is evaluated once.
    pooled_rows = []

    for method, col in [
        ("fixed40", "fixed40_prediction"),
        ("fixed50", "fixed50_prediction"),
        ("adaptive", "adaptive_prediction"),
    ]:
        scores = calculate_metrics(
            predictions_df["true_label"].to_numpy(),
            predictions_df[col].to_numpy(),
        )

        pooled_rows.append({"method": method, **scores})

    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(
        OUT_DIR / "pooled_oof_metrics.csv", index=False
    )

    print("\n" + "=" * 60)
    print("FOLD MEAN RESULTS")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    print("\nPOOLED OUT-OF-FOLD RESULTS")
    print(pooled_df.round(4).to_string(index=False))

    print("\nSaved results to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
