
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
    ROOT
    / "results/week4/group_aware_cv/record_signal_groups.csv"
)

CHARACTERISTICS_PATH = (
    ROOT / "results/week5/signal_characteristics.csv"
)

OUT_DIR = (
    ROOT
    / "results/week8/sensitivity_constrained_adaptive"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "record_id"
LABEL_COL = "label"
CONDITION_COL = "mean_baseline_variation"

OUTER_FOLDS = 5
INNER_FOLDS = 3
SEED = 42
N_ESTIMATORS = 100

QUANTILES = np.linspace(0.10, 0.90, 17)

# Specificity target: 0.95 means 95%.
MIN_SPECIFICITY = 0.95


# ============================================================
# HELPER FUNCTIONS
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
        unknown = values[result.isna()].unique().tolist()
        raise ValueError(f"Unknown labels: {unknown}")

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
        y_true,
        y_pred,
        labels=[0, 1],
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
            tn / (tn + fp) if (tn + fp) else 0.0
        ),
        "f1": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn),
    }


def load_feature_file(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} feature file not found:\n{path}"
        )

    df = pd.read_csv(path, dtype={ID_COL: str})

    if ID_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(
            f"{name} is missing ID or label columns."
        )

    df[ID_COL] = df[ID_COL].astype(str).str.strip()

    if df[ID_COL].duplicated().any():
        raise ValueError(f"Duplicate IDs found in {name}.")

    feature_cols = [
        col for col in df.columns
        if col not in [ID_COL, LABEL_COL]
    ]

    X = df[feature_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        raise ValueError(
            f"{name} contains missing or non-finite features. "
            "Inspect the input file before proceeding."
        )

    return df, feature_cols, X.to_numpy(dtype=np.float32)


# ============================================================
# GROUP-AWARE INNER OUT-OF-FOLD PREDICTIONS
# ============================================================

def get_inner_predictions(
    X40, X50, y, groups, seed
):
    """
    Generate inner out-of-fold predictions for both models.

    Each validation signal group is excluded from its
    corresponding inner-training split.
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

        pred40[valid_idx] = model40.predict(
            X40[valid_idx]
        ).astype(int)

        pred50[valid_idx] = model50.predict(
            X50[valid_idx]
        ).astype(int)

    if (pred40 < 0).any() or (pred50 < 0).any():
        raise RuntimeError(
            "Some inner-training records lack OOF predictions."
        )

    return pred40, pred50


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def evaluate_thresholds(y, pred40, pred50, condition):
    """
    Evaluate all candidate thresholds using inner OOF
    predictions and outer-training labels only.
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

        metrics = calculate_metrics(y, adaptive_pred)

        results.append({
            "threshold": float(threshold),
            **metrics,
            "meets_specificity_target": (
                metrics["specificity"] >= MIN_SPECIFICITY
            ),
        })

    return pd.DataFrame(results)


def learn_f1_threshold(y, pred40, pred50, condition):
    """
    Original adaptive selector:
    maximize F1, then sensitivity, then specificity.
    """

    results_df = evaluate_thresholds(
        y, pred40, pred50, condition
    )

    best = results_df.sort_values(
        ["f1", "sensitivity", "specificity"],
        ascending=[False, False, False],
        kind="mergesort",
    ).iloc[0]

    return float(best["threshold"]), results_df


def learn_constrained_threshold(
    y, pred40, pred50, condition
):
    """
    New selector:
    maximize sensitivity subject to inner-OOF
    specificity >= MIN_SPECIFICITY.

    Tie-breakers:
    1. Higher F1
    2. Higher specificity

    If no threshold meets the constraint, choose the
    candidate with highest specificity, then sensitivity,
    then F1. The fallback is explicitly recorded.
    """

    results_df = evaluate_thresholds(
        y, pred40, pred50, condition
    )

    eligible = results_df[
        results_df["meets_specificity_target"]
    ]

    if eligible.empty:
        best = results_df.sort_values(
            ["specificity", "sensitivity", "f1"],
            ascending=[False, False, False],
            kind="mergesort",
        ).iloc[0]

        fallback_used = True

        print(
            "WARNING: No candidate met the inner-validation "
            f"specificity target of {MIN_SPECIFICITY:.0%}. "
            "Using the highest-specificity candidate."
        )

    else:
        best = eligible.sort_values(
            ["sensitivity", "f1", "specificity"],
            ascending=[False, False, False],
            kind="mergesort",
        ).iloc[0]

        fallback_used = False

    results_df["selected_by_constrained_rule"] = (
        np.isclose(
            results_df["threshold"].to_numpy(dtype=float),
            float(best["threshold"]),
        )
    )

    results_df["fallback_used"] = fallback_used

    return float(best["threshold"]), results_df, fallback_used


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    print("\nSENSITIVITY-CONSTRAINED ADAPTIVE CWT")
    print("=" * 65)
    print(f"Minimum inner-validation specificity: "
          f"{MIN_SPECIFICITY:.0%}")

    # --------------------------------------------------------
    # Load feature datasets
    # --------------------------------------------------------

    df40, cols40, _ = load_feature_file(
        FIXED40_PATH, "Fixed-40"
    )

    df50, cols50, _ = load_feature_file(
        FIXED50_PATH, "Fixed-50"
    )

    groups_df = pd.read_csv(
        GROUP_PATH,
        dtype={"record_id": str, "signal_hash": str},
    )

    chars = pd.read_csv(
        CHARACTERISTICS_PATH,
        dtype={"record_id": str},
    )

    for frame, name in [
        (groups_df, "group mapping"),
        (chars, "signal characteristics"),
    ]:
        if ID_COL not in frame.columns:
            raise ValueError(
                f"{name} is missing {ID_COL}."
            )

        frame[ID_COL] = (
            frame[ID_COL].astype(str).str.strip()
        )

        if frame[ID_COL].duplicated().any():
            raise ValueError(
                f"Duplicate record IDs in {name}."
            )

    if "signal_hash" not in groups_df.columns:
        raise ValueError(
            "Group mapping is missing signal_hash."
        )

    if groups_df["signal_hash"].isna().any():
        raise ValueError(
            "Missing signal hashes in group mapping."
        )

    if CONDITION_COL not in chars.columns:
        raise ValueError(
            f"Missing characteristic: {CONDITION_COL}"
        )

    groups_df["binary_label"] = convert_labels(
        groups_df[LABEL_COL]
    )

    label_counts = groups_df.groupby(
        "signal_hash"
    )["binary_label"].nunique()

    if (label_counts > 1).any():
        raise ValueError(
            "Identical signal groups have conflicting labels."
        )

    # --------------------------------------------------------
    # Align records by ID
    # --------------------------------------------------------

    a = df40.set_index(ID_COL)
    b = df50.set_index(ID_COL)
    g = groups_df.set_index(ID_COL)
    c = chars.set_index(ID_COL)

    common_ids = (
        a.index.intersection(b.index)
        .intersection(g.index)
        .intersection(c.index)
        .sort_values()
    )

    if len(common_ids) == 0:
        raise ValueError(
            "No common record IDs across input files."
        )

    a = a.loc[common_ids]
    b = b.loc[common_ids]
    g = g.loc[common_ids]
    c = c.loc[common_ids]

    y = convert_labels(a[LABEL_COL])

    if not np.array_equal(
        y, convert_labels(b[LABEL_COL])
    ):
        raise ValueError(
            "Labels differ between the two feature datasets."
        )

    if not np.array_equal(
        y, g["binary_label"].to_numpy()
    ):
        raise ValueError(
            "Labels disagree with the group mapping."
        )

    condition_series = pd.to_numeric(
        c[CONDITION_COL], errors="coerce"
    )

    if not np.isfinite(
        condition_series.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Signal condition contains missing or non-finite values."
        )

    condition = condition_series.to_numpy(dtype=float)
    groups = g["signal_hash"].to_numpy(dtype=str)

    X40 = a[cols40].to_numpy(dtype=np.float32)
    X50 = b[cols50].to_numpy(dtype=np.float32)

    print("Records:", len(common_ids))
    print("Unique signal groups:", len(np.unique(groups)))
    print("Acceptable:", int((y == 0).sum()))
    print("Unacceptable:", int((y == 1).sum()))
    print("Fixed-40 features:", X40.shape[1])
    print("Fixed-50 features:", X50.shape[1])

    # --------------------------------------------------------
    # Outer group-aware cross-validation
    # --------------------------------------------------------

    outer_cv = StratifiedGroupKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    fold_rows = []
    prediction_rows = []
    threshold_rows = []
    fallback_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X40, y, groups),
        start=1,
    ):
        print(f"\n--- Outer fold {fold}/{OUTER_FOLDS} ---")

        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])

        if train_groups.intersection(test_groups):
            raise RuntimeError(
                "Signal-group leakage detected in outer CV."
            )

        X40_train = X40[train_idx]
        X40_test = X40[test_idx]

        X50_train = X50[train_idx]
        X50_test = X50[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        groups_train = groups[train_idx]
        condition_train = condition[train_idx]
        condition_test = condition[test_idx]

        # Inner OOF predictions: training data only.
        inner40, inner50 = get_inner_predictions(
            X40_train,
            X50_train,
            y_train,
            groups_train,
            SEED + fold,
        )

        # Original F1-based adaptive selector.
        f1_threshold, f1_table = learn_f1_threshold(
            y_train,
            inner40,
            inner50,
            condition_train,
        )

        # New sensitivity-constrained selector.
        constrained_threshold, constrained_table, fallback = (
            learn_constrained_threshold(
                y_train,
                inner40,
                inner50,
                condition_train,
            )
        )

        f1_table["outer_fold"] = fold
        f1_table["selector"] = "current_f1_adaptive"
        threshold_rows.append(f1_table)

        constrained_table["outer_fold"] = fold
        constrained_table["selector"] = (
            "sensitivity_constrained"
        )
        threshold_rows.append(constrained_table)

        fallback_rows.append({
            "outer_fold": fold,
            "specificity_target": MIN_SPECIFICITY,
            "constrained_threshold": constrained_threshold,
            "fallback_used": fallback,
        })

        # Train final models on the complete outer-training fold.
        model40 = make_model()
        model50 = make_model()

        model40.fit(X40_train, y_train)
        model50.fit(X50_train, y_train)

        pred40 = model40.predict(X40_test).astype(int)
        pred50 = model50.predict(X50_test).astype(int)

        # Current adaptive selector.
        current_adaptive = np.where(
            condition_test > f1_threshold,
            pred50,
            pred40,
        )

        # New constrained adaptive selector.
        constrained_adaptive = np.where(
            condition_test > constrained_threshold,
            pred50,
            pred40,
        )

        predictions = {
            "fixed40": pred40,
            "fixed50": pred50,
            "current_adaptive": current_adaptive,
            "constrained_adaptive": constrained_adaptive,
        }

        fold_row = {
            "fold": fold,
            "train_records": len(train_idx),
            "test_records": len(test_idx),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "current_f1_threshold": f1_threshold,
            "constrained_threshold": constrained_threshold,
            "constrained_fallback_used": fallback,
            "specificity_target": MIN_SPECIFICITY,
            "current_records_using_50Hz": int(
                (condition_test > f1_threshold).sum()
            ),
            "constrained_records_using_50Hz": int(
                (condition_test > constrained_threshold).sum()
            ),
        }

        for method, pred in predictions.items():
            scores = calculate_metrics(y_test, pred)

            for metric, value in scores.items():
                fold_row[f"{method}_{metric}"] = value

        fold_rows.append(fold_row)

        for j, idx in enumerate(test_idx):
            prediction_rows.append({
                "record_id": common_ids[idx],
                "signal_hash": groups[idx],
                "outer_fold": fold,
                "true_label": int(y_test[j]),
                CONDITION_COL: float(condition[idx]),
                "current_f1_threshold": f1_threshold,
                "constrained_threshold": constrained_threshold,
                "fixed40_prediction": int(pred40[j]),
                "fixed50_prediction": int(pred50[j]),
                "current_adaptive_prediction": int(
                    current_adaptive[j]
                ),
                "constrained_adaptive_prediction": int(
                    constrained_adaptive[j]
                ),
            })

        print("F1 threshold:", round(f1_threshold, 6))
        print(
            "Constrained threshold:",
            round(constrained_threshold, 6),
        )
        print("Fallback used:", fallback)

        for method in predictions:
            score = calculate_metrics(
                y_test, predictions[method]
            )
            print(
                f"{method}: "
                f"Sensitivity={score['sensitivity']:.4f}, "
                f"Specificity={score['specificity']:.4f}, "
                f"F1={score['f1']:.4f}"
            )

    # --------------------------------------------------------
    # Save fold metrics and predictions
    # --------------------------------------------------------

    folds_df = pd.DataFrame(fold_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    thresholds_df = pd.concat(
        threshold_rows, ignore_index=True
    )
    fallback_df = pd.DataFrame(fallback_rows)

    folds_df.to_csv(
        OUT_DIR / "fold_metrics.csv", index=False
    )

    predictions_df.to_csv(
        OUT_DIR / "out_of_fold_predictions.csv",
        index=False,
    )

    thresholds_df.to_csv(
        OUT_DIR / "threshold_search.csv",
        index=False,
    )

    fallback_df.to_csv(
        OUT_DIR / "fallback_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Fold means and standard deviations
    # --------------------------------------------------------

    methods = [
        "fixed40",
        "fixed50",
        "current_adaptive",
        "constrained_adaptive",
    ]

    metrics = [
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
    ]

    summary_rows = []

    for method in methods:
        for metric in metrics:
            values = folds_df[
                f"{method}_{metric}"
            ].to_numpy(dtype=float)

            summary_rows.append({
                "method": method,
                "metric": metric,
                "fold_mean": float(values.mean()),
                "fold_std": float(values.std(ddof=1)),
            })

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUT_DIR / "summary_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Pooled outer out-of-fold metrics
    # Each record is evaluated once in an outer test fold.
    # --------------------------------------------------------

    prediction_columns = {
        "fixed40": "fixed40_prediction",
        "fixed50": "fixed50_prediction",
        "current_adaptive": "current_adaptive_prediction",
        "constrained_adaptive": (
            "constrained_adaptive_prediction"
        ),
    }

    pooled_rows = []

    for method, column in prediction_columns.items():
        scores = calculate_metrics(
            predictions_df["true_label"].to_numpy(dtype=int),
            predictions_df[column].to_numpy(dtype=int),
        )

        pooled_rows.append({
            "method": method,
            **scores,
        })

    pooled_df = pd.DataFrame(pooled_rows)

    pooled_df.to_csv(
        OUT_DIR / "pooled_oof_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FOLD MEAN RESULTS")
    print("=" * 70)
    print(summary_df.round(4).to_string(index=False))

    print("\n" + "=" * 70)
    print("POOLED OUT-OF-FOLD RESULTS")
    print("=" * 70)
    print(pooled_df.round(4).to_string(index=False))

    print("\n" + "=" * 70)
    print("CONSTRAINED SELECTOR FALLBACKS")
    print("=" * 70)
    print(fallback_df.to_string(index=False))

    print("\nResults saved to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
