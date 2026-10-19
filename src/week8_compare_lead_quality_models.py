
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

LEAD_PATH = (
    ROOT / "results/week8/lead_quality_features"
    / "lead_quality_features.csv"
)

GROUP_PATH = (
    ROOT / "results/week4/group_aware_cv"
    / "record_signal_groups.csv"
)

CHAR_PATH = ROOT / "results/week5/signal_characteristics.csv"

OUT_DIR = ROOT / "results/week8/lead_quality_model_comparison"
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


def make_model(seed=SEED):
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=seed,
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
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_features(path, name):
    df = pd.read_csv(path, dtype={ID_COL: str})
    df[ID_COL] = df[ID_COL].astype(str).str.strip()

    if df[ID_COL].duplicated().any():
        raise ValueError(f"Duplicate record IDs in {name}")

    if ID_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"{name} is missing ID or label columns")

    columns = [
        c for c in df.columns
        if c not in [ID_COL, LABEL_COL]
    ]

    X = df[columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        raise ValueError(
            f"{name} has missing/non-finite feature values"
        )

    return df.set_index(ID_COL), columns, X.to_numpy(dtype=np.float32)


def get_inner_predictions(
    X40, X50, Xlead, X40_combined, X50_combined,
    y, groups, seed
):
    """
    Obtain inner out-of-fold predictions for every model.
    Each validation signal group is excluded from training.
    """

    cv = StratifiedGroupKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    predictions = {
        name: np.full(len(y), -1, dtype=int)
        for name in [
            "fixed40", "fixed50", "lead_only",
            "combined40", "combined50",
        ]
    }

    matrices = {
        "fixed40": X40,
        "fixed50": X50,
        "lead_only": Xlead,
        "combined40": X40_combined,
        "combined50": X50_combined,
    }

    for train_idx, valid_idx in cv.split(Xlead, y, groups):
        if set(groups[train_idx]).intersection(groups[valid_idx]):
            raise RuntimeError("Signal-group leakage in inner CV")

        for name, X in matrices.items():
            model = make_model(seed + len(valid_idx))
            model.fit(X[train_idx], y[train_idx])
            predictions[name][valid_idx] = model.predict(
                X[valid_idx]
            ).astype(int)

    for name, pred in predictions.items():
        if (pred < 0).any():
            raise RuntimeError(
                f"Incomplete inner predictions for {name}"
            )

    return predictions


def learn_threshold(y, pred40, pred50, condition):
    candidates = np.unique(np.quantile(condition, QUANTILES))
    rows = []

    for threshold in candidates:
        pred = np.where(condition > threshold, pred50, pred40)

        rows.append({
            "threshold": float(threshold),
            **calculate_metrics(y, pred),
        })

    table = pd.DataFrame(rows)

    best = table.sort_values(
        ["f1", "sensitivity", "specificity"],
        ascending=[False, False, False],
        kind="mergesort",
    ).iloc[0]

    return float(best["threshold"]), table


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    print("\nLEAD-QUALITY MODEL COMPARISON")
    print("=" * 65)

    df40, cols40, X40 = load_features(
        FIXED40_PATH, "Fixed-40 features"
    )
    df50, cols50, X50 = load_features(
        FIXED50_PATH, "Fixed-50 features"
    )
    dflead, lead_cols, Xlead = load_features(
        LEAD_PATH, "Lead-quality features"
    )

    groups_df = pd.read_csv(
        GROUP_PATH,
        dtype={"record_id": str, "signal_hash": str},
    )
    groups_df["record_id"] = (
        groups_df["record_id"].astype(str).str.strip()
    )

    if groups_df["record_id"].duplicated().any():
        raise ValueError("Duplicate IDs in group mapping")

    if groups_df["signal_hash"].isna().any():
        raise ValueError("Missing signal hashes")

    groups_df = groups_df.set_index("record_id")

    chars = pd.read_csv(CHAR_PATH, dtype={"record_id": str})
    chars["record_id"] = chars["record_id"].astype(str).str.strip()

    if chars["record_id"].duplicated().any():
        raise ValueError("Duplicate IDs in signal characteristics")

    chars = chars.set_index("record_id")

    if CONDITION_COL not in chars.columns:
        raise ValueError(f"Missing condition column: {CONDITION_COL}")

    common_ids = (
        df40.index.intersection(df50.index)
        .intersection(dflead.index)
        .intersection(groups_df.index)
        .intersection(chars.index)
        .sort_values()
    )

    if len(common_ids) == 0:
        raise ValueError("No common record IDs across inputs")

    df40 = df40.loc[common_ids]
    df50 = df50.loc[common_ids]
    dflead = dflead.loc[common_ids]
    groups_df = groups_df.loc[common_ids]
    chars = chars.loc[common_ids]

    y = convert_labels(df40[LABEL_COL])

    if not np.array_equal(y, convert_labels(df50[LABEL_COL])):
        raise ValueError("Labels differ between fixed CWT datasets")

    if not np.array_equal(y, convert_labels(dflead[LABEL_COL])):
        raise ValueError("Labels differ in lead-quality dataset")

    if not np.array_equal(
        y, convert_labels(groups_df[LABEL_COL])
    ):
        raise ValueError("Labels differ in group mapping")

    condition = pd.to_numeric(
        chars[CONDITION_COL], errors="coerce"
    ).to_numpy(dtype=float)

    if not np.isfinite(condition).all():
        raise ValueError("Invalid adaptive CWT condition values")

    groups = groups_df["signal_hash"].to_numpy(dtype=str)

    X40 = df40[cols40].to_numpy(dtype=np.float32)
    X50 = df50[cols50].to_numpy(dtype=np.float32)
    Xlead = dflead[lead_cols].to_numpy(dtype=np.float32)

    # The combined models use matching CWT features plus
    # lead-quality features for the same record.
    X40_combined = np.concatenate([X40, Xlead], axis=1)
    X50_combined = np.concatenate([X50, Xlead], axis=1)

    print("Records:", len(common_ids))
    print("Unique signal groups:", len(np.unique(groups)))
    print("Acceptable:", int((y == 0).sum()))
    print("Unacceptable:", int((y == 1).sum()))
    print("Fixed-40 features:", X40.shape[1])
    print("Fixed-50 features:", X50.shape[1])
    print("Lead-quality features:", Xlead.shape[1])
    print("Combined features:", X40_combined.shape[1])

    outer_cv = StratifiedGroupKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    fold_rows = []
    oof_rows = []
    threshold_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(Xlead, y, groups), start=1
    ):
        print(f"\nOuter fold {fold}/{OUTER_FOLDS}")

        if set(groups[train_idx]).intersection(groups[test_idx]):
            raise RuntimeError("Signal-group leakage in outer CV")

        y_train = y[train_idx]
        y_test = y[test_idx]

        condition_train = condition[train_idx]
        condition_test = condition[test_idx]
        groups_train = groups[train_idx]

        inner = get_inner_predictions(
            X40[train_idx],
            X50[train_idx],
            Xlead[train_idx],
            X40_combined[train_idx],
            X50_combined[train_idx],
            y_train,
            groups_train,
            SEED + fold,
        )

        # Learn thresholds only from inner OOF predictions.
        threshold_base, table_base = learn_threshold(
            y_train,
            inner["fixed40"],
            inner["fixed50"],
            condition_train,
        )

        threshold_combined, table_combined = learn_threshold(
            y_train,
            inner["combined40"],
            inner["combined50"],
            condition_train,
        )

        table_base["outer_fold"] = fold
        table_base["method"] = "baseline_adaptive"
        table_combined["outer_fold"] = fold
        table_combined["method"] = "combined_adaptive"

        threshold_rows.extend([
            table_base,
            table_combined,
        ])

        matrices = {
            "fixed40": X40,
            "fixed50": X50,
            "lead_only": Xlead,
            "combined40": X40_combined,
            "combined50": X50_combined,
        }

        test_predictions = {}

        for name, X in matrices.items():
            model = make_model(SEED + fold)
            model.fit(X[train_idx], y_train)
            test_predictions[name] = model.predict(
                X[test_idx]
            ).astype(int)

        pred_base = np.where(
            condition_test > threshold_base,
            test_predictions["fixed50"],
            test_predictions["fixed40"],
        )

        pred_combined = np.where(
            condition_test > threshold_combined,
            test_predictions["combined50"],
            test_predictions["combined40"],
        )

        methods = {
            "baseline_adaptive": pred_base,
            "lead_quality_only": test_predictions["lead_only"],
            "combined_adaptive": pred_combined,
        }

        fold_row = {
            "fold": fold,
            "train_records": len(train_idx),
            "test_records": len(test_idx),
            "train_groups": len(np.unique(groups_train)),
            "test_groups": len(np.unique(groups[test_idx])),
            "baseline_threshold": threshold_base,
            "combined_threshold": threshold_combined,
        }

        for method, pred in methods.items():
            scores = calculate_metrics(y_test, pred)

            for metric, value in scores.items():
                fold_row[f"{method}_{metric}"] = value

            for local_idx, global_idx in enumerate(test_idx):
                oof_rows.append({
                    "record_id": common_ids[global_idx],
                    "signal_hash": groups[global_idx],
                    "outer_fold": fold,
                    "true_label": int(y_test[local_idx]),
                    "method": method,
                    "prediction": int(pred[local_idx]),
                })

        fold_rows.append(fold_row)

        print(
            "Baseline adaptive F1:",
            f"{fold_row['baseline_adaptive_f1']:.4f}",
        )
        print(
            "Lead-quality-only F1:",
            f"{fold_row['lead_quality_only_f1']:.4f}",
        )
        print(
            "Combined adaptive F1:",
            f"{fold_row['combined_adaptive_f1']:.4f}",
        )

    # --------------------------------------------------------
    # Save fold metrics and outer OOF predictions
    # --------------------------------------------------------

    folds_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_rows)
    thresholds_df = pd.concat(threshold_rows, ignore_index=True)

    folds_df.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    oof_df.to_csv(OUT_DIR / "out_of_fold_predictions.csv", index=False)
    thresholds_df.to_csv(OUT_DIR / "threshold_search.csv", index=False)

    summary_rows = []
    pooled_rows = []

    methods = [
        "baseline_adaptive",
        "lead_quality_only",
        "combined_adaptive",
    ]

    metrics = [
        "accuracy", "precision", "sensitivity",
        "specificity", "f1",
    ]

    for method in methods:
        for metric in metrics:
            values = folds_df[f"{method}_{metric}"].to_numpy()

            summary_rows.append({
                "method": method,
                "metric": metric,
                "fold_mean": float(values.mean()),
                "fold_std": float(values.std(ddof=1)),
            })

        subset = oof_df[oof_df["method"] == method]

        pooled_rows.append({
            "method": method,
            **calculate_metrics(
                subset["true_label"].to_numpy(),
                subset["prediction"].to_numpy(),
            ),
        })

    summary_df = pd.DataFrame(summary_rows)
    pooled_df = pd.DataFrame(pooled_rows)

    summary_df.to_csv(OUT_DIR / "summary_results.csv", index=False)
    pooled_df.to_csv(OUT_DIR / "pooled_oof_metrics.csv", index=False)

    print("\nFOLD MEAN RESULTS")
    print("=" * 80)
    print(summary_df.round(4).to_string(index=False))

    print("\nPOOLED OUT-OF-FOLD RESULTS")
    print("=" * 80)
    print(pooled_df.round(4).to_string(index=False))

    print("\nSaved results to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
