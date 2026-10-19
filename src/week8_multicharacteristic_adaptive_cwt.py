
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
GROUP_PATH = ROOT / "results/week4/group_aware_cv/record_signal_groups.csv"
CHAR_PATH = ROOT / "results/week5/signal_characteristics.csv"

OUT_DIR = ROOT / "results/week8/multicharacteristic_adaptive"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
N_ESTIMATORS = 100

CONDITION_COL = "mean_baseline_variation"

CHAR_FEATURES = [
    "mean_baseline_variation",
    "mean_high_frequency_ratio",
    "mean_powerline_ratio",
    "mean_std",
    "mean_rms",
]


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


def make_base_model():
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
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def load_feature_file(path, name):
    df = pd.read_csv(path, dtype={"record_id": str})
    df["record_id"] = df["record_id"].str.strip()

    if df["record_id"].duplicated().any():
        raise ValueError(f"Duplicate record IDs in {name}.")

    feature_cols = [
        col for col in df.columns
        if col not in ["record_id", "label"]
    ]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        raise ValueError(f"Invalid feature values in {name}.")

    return df, feature_cols, X.to_numpy(dtype=np.float32)


def get_inner_predictions(X40, X50, y, groups, seed):
    """Generate group-aware inner out-of-fold predictions."""

    cv = StratifiedGroupKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    pred40 = np.full(len(y), -1, dtype=int)
    pred50 = np.full(len(y), -1, dtype=int)

    for train_idx, valid_idx in cv.split(X40, y, groups):
        if set(groups[train_idx]) & set(groups[valid_idx]):
            raise RuntimeError("Inner-fold group leakage detected.")

        model40 = make_base_model()
        model50 = make_base_model()

        model40.fit(X40[train_idx], y[train_idx])
        model50.fit(X50[train_idx], y[train_idx])

        pred40[valid_idx] = model40.predict(X40[valid_idx])
        pred50[valid_idx] = model50.predict(X50[valid_idx])

    if (pred40 < 0).any() or (pred50 < 0).any():
        raise RuntimeError("Some records lack inner OOF predictions.")

    return pred40, pred50


def learn_baseline_threshold(y, pred40, pred50, condition):
    """Learn the existing single-characteristic selector on training data."""

    candidates = np.unique(
        np.quantile(condition, np.linspace(0.10, 0.90, 17))
    )

    results = []

    for threshold in candidates:
        pred = np.where(condition > threshold, pred50, pred40)

        results.append({
            "threshold": float(threshold),
            **calculate_metrics(y, pred),
        })

    results.sort(
        key=lambda row: (
            row["f1"],
            row["sensitivity"],
            row["specificity"],
        ),
        reverse=True,
    )

    return results[0]["threshold"]


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    print("\nMULTICHARACTERISTIC ADAPTIVE CWT EXPERIMENT")
    print("=" * 65)

    df40, cols40, _ = load_feature_file(
        FIXED40_PATH, "Fixed 40 Hz"
    )
    df50, cols50, _ = load_feature_file(
        FIXED50_PATH, "Fixed 50 Hz"
    )

    groups_df = pd.read_csv(
        GROUP_PATH,
        dtype={"record_id": str, "signal_hash": str},
    )
    chars = pd.read_csv(CHAR_PATH, dtype={"record_id": str})

    for frame in [groups_df, chars]:
        frame["record_id"] = frame["record_id"].astype(str).str.strip()

    if groups_df["record_id"].duplicated().any():
        raise ValueError("Duplicate IDs in group mapping.")

    if chars["record_id"].duplicated().any():
        raise ValueError("Duplicate IDs in characteristics.")

    if groups_df["signal_hash"].isna().any():
        raise ValueError("Missing signal hashes.")

    missing = set(CHAR_FEATURES) - set(chars.columns)
    if missing:
        raise ValueError(f"Missing signal characteristics: {missing}")

    a = df40.set_index("record_id")
    b = df50.set_index("record_id")
    g = groups_df.set_index("record_id")
    c = chars.set_index("record_id")

    ids = (
        a.index.intersection(b.index)
        .intersection(g.index)
        .intersection(c.index)
        .sort_values()
    )

    if len(ids) == 0:
        raise ValueError("No common record IDs.")

    a, b, g, c = a.loc[ids], b.loc[ids], g.loc[ids], c.loc[ids]

    y = convert_labels(a["label"])

    if not np.array_equal(y, convert_labels(b["label"])):
        raise ValueError("Labels differ between feature files.")

    if not np.array_equal(y, convert_labels(g["label"])):
        raise ValueError("Labels disagree with group mapping.")

    if g["signal_hash"].isna().any():
        raise ValueError("Missing aligned signal hashes.")

    X40 = a[cols40].to_numpy(dtype=np.float32)
    X50 = b[cols50].to_numpy(dtype=np.float32)

    C = c[CHAR_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    if C.isna().any().any():
        raise ValueError("Invalid signal-characteristic values.")

    C = C.to_numpy(dtype=np.float32)
    condition = C[:, CHAR_FEATURES.index(CONDITION_COL)]
    groups = g["signal_hash"].to_numpy(dtype=str)

    print("Records:", len(ids))
    print("Unique signal groups:", len(np.unique(groups)))
    print("Signal characteristics:", len(CHAR_FEATURES))
    print("Acceptable:", int((y == 0).sum()))
    print("Unacceptable:", int((y == 1).sum()))

    outer_cv = StratifiedGroupKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    fold_rows = []
    prediction_rows = []
    selection_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X40, y, groups), start=1
    ):
        print(f"\n--- Outer fold {fold}/{OUTER_FOLDS} ---")

        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError("Outer-fold group leakage detected.")

        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        # Generate training-only predictions for selector learning.
        inner40, inner50 = get_inner_predictions(
            X40[train_idx],
            X50[train_idx],
            y_train,
            groups_train,
            SEED + fold,
        )

        # Existing baseline: learn its threshold from training only.
        threshold = learn_baseline_threshold(
            y_train,
            inner40,
            inner50,
            condition[train_idx],
        )

        # Learn which base model is uniquely correct on each informative
        # training example, using inner OOF predictions only.
        correct40 = inner40 == y_train
        correct50 = inner50 == y_train

        informative = correct40 != correct50

        # 0 = only 40 Hz is correct; 1 = only 50 Hz is correct.
        selector_target = correct50[informative].astype(int)

        selector_X = C[train_idx][informative]

        selector_used = False
        fallback = ""

        if (
            informative.sum() >= 10
            and len(np.unique(selector_target)) == 2
        ):
            selector = RandomForestClassifier(
                n_estimators=200,
                random_state=SEED + fold,
                n_jobs=-1,
                class_weight="balanced",
                min_samples_leaf=3,
            )

            selector.fit(selector_X, selector_target)
            selector_used = True

        else:
            # If training examples do not support learning a selector,
            # fall back to the better training-only OOF base model.
            score40 = calculate_metrics(y_train, inner40)
            score50 = calculate_metrics(y_train, inner50)

            key = ("f1", "sensitivity", "specificity")
            rank40 = tuple(score40[k] for k in key)
            rank50 = tuple(score50[k] for k in key)

            fallback = "fixed50" if rank50 > rank40 else "fixed40"

        # Train base models on all outer-training records.
        model40 = make_base_model()
        model50 = make_base_model()

        model40.fit(X40[train_idx], y_train)
        model50.fit(X50[train_idx], y_train)

        pred40 = model40.predict(X40[test_idx]).astype(int)
        pred50 = model50.predict(X50[test_idx]).astype(int)

        # Original one-characteristic adaptive selector.
        pred_baseline = np.where(
            condition[test_idx] > threshold,
            pred50,
            pred40,
        )

        # New multicharacteristic selector.
        if selector_used:
            choose50 = selector.predict(C[test_idx]).astype(bool)
            pred_multi = np.where(choose50, pred50, pred40)
        elif fallback == "fixed50":
            pred_multi = pred50.copy()
        else:
            pred_multi = pred40.copy()

        fold_row = {
            "outer_fold": fold,
            "train_records": len(train_idx),
            "test_records": len(test_idx),
            "informative_selector_training_records": int(informative.sum()),
            "selector_used": selector_used,
            "fallback": fallback,
            "baseline_threshold": threshold,
        }

        methods = [
            ("fixed40", pred40),
            ("fixed50", pred50),
            ("baseline_adaptive", pred_baseline),
            ("multicharacteristic", pred_multi),
        ]

        for name, pred in methods:
            for metric, value in calculate_metrics(y_test, pred).items():
                fold_row[f"{name}_{metric}"] = value

        fold_rows.append(fold_row)

        for j, idx in enumerate(test_idx):
            prediction_rows.append({
                "record_id": ids[idx],
                "signal_hash": groups[idx],
                "outer_fold": fold,
                "true_label": int(y_test[j]),
                "mean_baseline_variation": float(condition[idx]),
                "baseline_threshold": float(threshold),
                "fixed40_prediction": int(pred40[j]),
                "fixed50_prediction": int(pred50[j]),
                "baseline_adaptive_prediction": int(pred_baseline[j]),
                "multicharacteristic_prediction": int(pred_multi[j]),
                "selector_used": selector_used,
                "fallback": fallback,
            })

        selection_rows.append({
            "outer_fold": fold,
            "informative_training_records": int(informative.sum()),
            "choose40_training_examples": int((selector_target == 0).sum()),
            "choose50_training_examples": int((selector_target == 1).sum()),
            "selector_used": selector_used,
            "fallback": fallback,
        })

        print("Training records:", len(train_idx))
        print("Testing records:", len(test_idx))
        print("Informative selector examples:", int(informative.sum()))
        print("Selector trained:", selector_used)
        if fallback:
            print("Fallback:", fallback)
        print(
            "Multicharacteristic F1:",
            f"{fold_row['multicharacteristic_f1']:.4f}",
        )
        print(
            "Multicharacteristic sensitivity:",
            f"{fold_row['multicharacteristic_sensitivity']:.4f}",
        )

    folds_df = pd.DataFrame(fold_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    selection_df = pd.DataFrame(selection_rows)

    folds_df.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    predictions_df.to_csv(
        OUT_DIR / "out_of_fold_predictions.csv", index=False
    )
    selection_df.to_csv(
        OUT_DIR / "selector_training_summary.csv", index=False
    )

    # Report fold means and variability.
    methods = [
        "fixed40",
        "fixed50",
        "baseline_adaptive",
        "multicharacteristic",
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
            values = folds_df[f"{method}_{metric}"].to_numpy()

            summary_rows.append({
                "method": method,
                "metric": metric,
                "fold_mean": float(values.mean()),
                "fold_std": float(values.std(ddof=1)),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "summary_results.csv", index=False)

    # Pooled predictions: each record evaluated once in its outer test fold.
    pooled_rows = []

    for method, column in [
        ("fixed40", "fixed40_prediction"),
        ("fixed50", "fixed50_prediction"),
        ("baseline_adaptive", "baseline_adaptive_prediction"),
        ("multicharacteristic", "multicharacteristic_prediction"),
    ]:
        pooled_rows.append({
            "method": method,
            **calculate_metrics(
                predictions_df["true_label"].to_numpy(),
                predictions_df[column].to_numpy(),
            ),
        })

    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(OUT_DIR / "pooled_oof_metrics.csv", index=False)

    print("\n" + "=" * 65)
    print("FOLD MEAN RESULTS")
    print("=" * 65)
    print(summary_df.round(4).to_string(index=False))

    print("\nPOOLED OUT-OF-FOLD RESULTS")
    print("=" * 65)
    print(pooled_df.round(4).to_string(index=False))

    print("\nSELECTOR TRAINING SUMMARY")
    print("=" * 65)
    print(selection_df.to_string(index=False))

    print("\nSaved results to:", OUT_DIR)


if __name__ == "__main__":
    main()
