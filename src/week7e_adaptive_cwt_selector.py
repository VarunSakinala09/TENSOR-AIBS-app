
"""
Experiment 7E: Leakage-safe Adaptive CWT Selection

Fixed-40: CWT features generated using 1–40 Hz
Fixed-50: CWT features generated using 1–50 Hz
Adaptive: Selects between the two using baseline variation.

Run from the project root.
"""

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
from sklearn.model_selection import StratifiedKFold


# -------------------- File paths --------------------

ROOT = Path(".")
FIXED40_PATH = ROOT / "results/week3/features_full_dataset.csv"
FIXED50_PATH = ROOT / "results/week7/cwt_d/cwt_d_features.csv"
CHAR_PATH = ROOT / "results/week5/signal_characteristics.csv"

OUTPUT_DIR = ROOT / "results/week7/adaptive_selector"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------- Settings --------------------

ID_COL = "record_id"
LABEL_COL = "label"
CONDITION_COL = "mean_baseline_variation"

OUTER_FOLDS = 5
INNER_FOLDS = 3
SEED = 42
N_ESTIMATORS = 100

# Candidate thresholds are calculated from each training fold.
QUANTILES = np.linspace(0.10, 0.90, 17)


# -------------------- Helper functions --------------------

def make_model():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced",
    )


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

    labels = values.map(mapping)

    if labels.isna().any():
        unknown = values[labels.isna()].unique().tolist()
        raise ValueError(f"Unexpected labels: {unknown}")

    return labels.astype(int)


def load_features(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Run the script from the project root."
        )

    df = pd.read_csv(path)

    if ID_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(
            f"{path} must contain '{ID_COL}' and '{LABEL_COL}'."
        )

    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[LABEL_COL] = convert_labels(df[LABEL_COL])

    feature_columns = [
        col for col in df.columns
        if col not in [ID_COL, LABEL_COL]
    ]

    X = df[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    result = pd.concat(
        [
            df[[ID_COL, LABEL_COL]].reset_index(drop=True),
            X.reset_index(drop=True),
        ],
        axis=1,
    )

    print(
        f"{name}: {len(result)} records, "
        f"{len(feature_columns)} features"
    )

    return result


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else 0.0
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "sensitivity": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "specificity": specificity,
        "f1": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
    }


# -------------------- Inner cross-validation --------------------

def get_inner_predictions(X40, X50, y, seed):
    """
    Generate out-of-fold predictions within the outer training fold.
    These predictions are used to learn the adaptive threshold.
    """

    inner_cv = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    pred40 = np.zeros(len(y), dtype=int)
    pred50 = np.zeros(len(y), dtype=int)

    for train_idx, valid_idx in inner_cv.split(X40, y):
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

    return pred40, pred50


def learn_threshold(y, pred40, pred50, condition):
    """
    Learn the threshold using training-fold data only.

    Below/equal threshold -> Fixed-40 prediction
    Above threshold       -> Fixed-50 prediction

    The selection objective is F1, with sensitivity and
    specificity used to break ties.
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

        scores = calculate_metrics(y, adaptive_pred)

        results.append({
            "threshold": float(threshold),
            **scores,
        })

    results_df = pd.DataFrame(results)

    best = results_df.sort_values(
        ["f1", "sensitivity", "specificity"],
        ascending=[False, False, False],
    ).iloc[0]

    return float(best["threshold"]), results_df


# -------------------- Main experiment --------------------

def main():
    print("\nEXPERIMENT 7E: ADAPTIVE CWT SELECTION")
    print("=" * 55)

    fixed40 = load_features(FIXED40_PATH, "Fixed-40")
    fixed50 = load_features(FIXED50_PATH, "Fixed-50")

    if not CHAR_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CHAR_PATH}. Run the Week 5 "
            "signal-characteristics script first."
        )

    characteristics = pd.read_csv(CHAR_PATH)

    if ID_COL not in characteristics.columns:
        raise ValueError(
            f"'{ID_COL}' is missing from the characteristics file."
        )

    if CONDITION_COL not in characteristics.columns:
        raise ValueError(
            f"'{CONDITION_COL}' is missing from the characteristics file."
        )

    characteristics[ID_COL] = (
        characteristics[ID_COL].astype(str).str.strip()
    )

    for df, name in [
        (fixed40, "Fixed-40"),
        (fixed50, "Fixed-50"),
        (characteristics, "Characteristics"),
    ]:
        if df[ID_COL].duplicated().any():
            raise ValueError(f"Duplicate record IDs in {name}.")

    # Align records by ID, not by CSV row position.
    a = fixed40.set_index(ID_COL)
    b = fixed50.set_index(ID_COL)
    c = characteristics.set_index(ID_COL)

    common_ids = (
        a.index.intersection(b.index)
        .intersection(c.index)
        .sort_values()
    )

    a = a.loc[common_ids]
    b = b.loc[common_ids]
    c = c.loc[common_ids]

    y = a[LABEL_COL].to_numpy(dtype=int)

    if not np.array_equal(
        y, b[LABEL_COL].to_numpy(dtype=int)
    ):
        raise ValueError(
            "Labels do not match between the feature datasets."
        )

    condition = pd.to_numeric(
        c[CONDITION_COL], errors="coerce"
    ).to_numpy(dtype=float)

    if not np.isfinite(condition).all():
        median_value = np.nanmedian(condition)
        condition = np.where(
            np.isfinite(condition),
            condition,
            median_value,
        )

    X40 = a.drop(columns=[LABEL_COL]).to_numpy(
        dtype=np.float32
    )

    X50 = b.drop(columns=[LABEL_COL]).to_numpy(
        dtype=np.float32
    )

    if X40.shape[1] != X50.shape[1]:
        raise ValueError(
            f"Feature counts differ: {X40.shape[1]} vs {X50.shape[1]}"
        )

    print(f"\nAligned records: {len(common_ids)}")
    print(f"Acceptable: {(y == 0).sum()}")
    print(f"Unacceptable: {(y == 1).sum()}")
    print(f"Feature shape: {X40.shape}")

    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    fold_results = []
    prediction_results = []
    threshold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X40, y), start=1
    ):
        print(f"\n--- Outer fold {fold}/{OUTER_FOLDS} ---")

        X40_train, X40_test = X40[train_idx], X40[test_idx]
        X50_train, X50_test = X50[train_idx], X50[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        condition_train = condition[train_idx]
        condition_test = condition[test_idx]

        # Step 1: training-only inner OOF predictions.
        inner40, inner50 = get_inner_predictions(
            X40_train,
            X50_train,
            y_train,
            SEED + fold,
        )

        # Step 2: learn the threshold from the training fold.
        threshold, threshold_table = learn_threshold(
            y_train,
            inner40,
            inner50,
            condition_train,
        )

        threshold_table["outer_fold"] = fold
        threshold_results.append(threshold_table)

        # Step 3: train both models on the full outer training fold.
        model40 = make_model()
        model50 = make_model()

        model40.fit(X40_train, y_train)
        model50.fit(X50_train, y_train)

        # Step 4: predict unseen outer-test ECGs.
        pred40 = model40.predict(X40_test).astype(int)
        pred50 = model50.predict(X50_test).astype(int)

        # The learned threshold and signal condition select the model.
        adaptive_pred = np.where(
            condition_test > threshold,
            pred50,
            pred40,
        )

        fold_row = {
            "fold": fold,
            "learned_threshold": threshold,
            "test_records_using_50Hz": int(
                (condition_test > threshold).sum()
            ),
        }

        for method, predictions in [
            ("fixed40", pred40),
            ("fixed50", pred50),
            ("adaptive", adaptive_pred),
        ]:
            scores = calculate_metrics(y_test, predictions)

            for metric, value in scores.items():
                fold_row[f"{method}_{metric}"] = value

        fold_results.append(fold_row)

        print(f"Learned threshold: {threshold:.6f}")
        print(
            "Test records using 1–50 Hz: "
            f"{fold_row['test_records_using_50Hz']}/{len(test_idx)}"
        )
        print(
            f"Adaptive accuracy: {fold_row['adaptive_accuracy']:.4f}"
        )
        print(
            f"Adaptive sensitivity: {fold_row['adaptive_sensitivity']:.4f}"
        )
        print(
            f"Adaptive specificity: {fold_row['adaptive_specificity']:.4f}"
        )
        print(f"Adaptive F1: {fold_row['adaptive_f1']:.4f}")

        for j, index in enumerate(test_idx):
            prediction_results.append({
                "record_id": common_ids[index],
                "outer_fold": fold,
                "true_label": int(y_test[j]),
                "mean_baseline_variation": float(condition[index]),
                "learned_threshold": threshold,
                "selected_configuration": (
                    "CWT_1_50_Hz"
                    if condition[index] > threshold
                    else "CWT_1_40_Hz"
                ),
                "fixed40_prediction": int(pred40[j]),
                "fixed50_prediction": int(pred50[j]),
                "adaptive_prediction": int(adaptive_pred[j]),
            })

    # -------------------- Save results --------------------

    folds_df = pd.DataFrame(fold_results)

    summary_rows = []

    for method in ["fixed40", "fixed50", "adaptive"]:
        for metric in [
            "accuracy",
            "precision",
            "sensitivity",
            "specificity",
            "f1",
        ]:
            values = folds_df[f"{method}_{metric}"].to_numpy()

            summary_rows.append({
                "method": method,
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            })

    summary_df = pd.DataFrame(summary_rows)

    folds_df.to_csv(
        OUTPUT_DIR / "fold_results.csv", index=False
    )

    summary_df.to_csv(
        OUTPUT_DIR / "summary_results.csv", index=False
    )

    pd.DataFrame(prediction_results).to_csv(
        OUTPUT_DIR / "per_record_adaptive_predictions.csv",
        index=False,
    )

    pd.concat(threshold_results, ignore_index=True).to_csv(
        OUTPUT_DIR / "training_threshold_search.csv",
        index=False,
    )

    print("\n" + "=" * 55)
    print("FINAL RESULTS: MEAN ± STANDARD DEVIATION")
    print("=" * 55)

    for method in ["fixed40", "fixed50", "adaptive"]:
        print(f"\n{method.upper()}")

        for metric in [
            "accuracy",
            "precision",
            "sensitivity",
            "specificity",
            "f1",
        ]:
            row = summary_df[
                (summary_df["method"] == method)
                & (summary_df["metric"] == metric)
            ].iloc[0]

            print(
                f"{metric:12s}: "
                f"{row['mean']:.4f} ± {row['std']:.4f}"
            )

    print(f"\nResults saved in: {OUTPUT_DIR.resolve()}")
    print(
        "\nCompare ADAPTIVE against BOTH Fixed-40 and Fixed-50. "
        "If Adaptive does not beat Fixed-50, do not claim an "
        "adaptive improvement."
    )


if __name__ == "__main__":
    main()
