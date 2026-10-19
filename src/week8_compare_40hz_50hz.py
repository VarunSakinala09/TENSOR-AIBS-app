
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

FILES = {
    "40Hz": ROOT / "results/week3/features_full_dataset.csv",
    "50Hz": ROOT / "results/week7/cwt_d/cwt_d_features.csv",
}

GROUP_FILE = (
    ROOT / "results/week4/group_aware_cv"
    / "record_signal_groups.csv"
)

OUTPUT_DIR = ROOT / "results/week8/frequency_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

datasets = {}

for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} dataset: {path}")

    df = pd.read_csv(path, dtype={"record_id": str})

    if df["record_id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate record IDs")

    feature_cols = [
        c for c in df.columns
        if c.startswith("feature_")
    ]

    if not feature_cols:
        raise ValueError(f"No feature_* columns found in {name}")

    if not np.isfinite(df[feature_cols].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains missing or infinite features")

    datasets[name] = (df, feature_cols)

    print(f"\n{name}: {df.shape}")
    print("Feature count:", len(feature_cols))


# ------------------------------------------------------------
# 2. LOAD SIGNAL GROUPS AND ALIGN RECORDS
# ------------------------------------------------------------

groups = pd.read_csv(
    GROUP_FILE,
    dtype={"record_id": str}
)

required = {"record_id", "signal_hash"}
if not required.issubset(groups.columns):
    raise ValueError(f"Group mapping must contain {required}")

if groups["record_id"].duplicated().any():
    raise ValueError("Group mapping contains duplicate record IDs")

reference = datasets["40Hz"][0]
reference_ids = set(reference["record_id"])

if set(datasets["50Hz"][0]["record_id"]) != reference_ids:
    raise ValueError("40 Hz and 50 Hz record IDs do not match")

if set(groups["record_id"]) != reference_ids:
    raise ValueError("Feature datasets and group mapping IDs do not match")

# Use 40 Hz row order as the common record order.
reference = reference.sort_values("record_id").reset_index(drop=True)
ids = reference["record_id"]

group_lookup = groups.set_index("record_id")["signal_hash"]
signal_groups = ids.map(group_lookup)

if signal_groups.isna().any():
    raise ValueError("Some records have no signal group")

# Verify labels agree across both feature datasets.
for name, (df, _) in datasets.items():
    indexed = df.set_index("record_id").loc[ids]
    if not indexed["label"].reset_index(drop=True).equals(
        reference["label"].reset_index(drop=True)
    ):
        raise ValueError(f"Labels/order mismatch in {name}")

y = reference["label"].reset_index(drop=True)

print("\nRECORDS")
print("Total records:", len(ids))
print("Unique signal groups:", signal_groups.nunique())
print("Labels:")
print(y.value_counts().to_string())


# ------------------------------------------------------------
# 3. GENERATE COMMON DUPLICATE-AWARE FOLDS
# ------------------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

# Generate folds once and reuse the exact same folds for both models.
splits = list(cv.split(
    np.zeros(len(y)),
    y,
    groups=signal_groups
))

for fold, (train_idx, test_idx) in enumerate(splits, start=1):
    train_groups = set(signal_groups.iloc[train_idx])
    test_groups = set(signal_groups.iloc[test_idx])

    if train_groups & test_groups:
        raise RuntimeError(f"Signal-group leakage in fold {fold}")

print("\nAll folds generated; no signal groups overlap.")


# ------------------------------------------------------------
# 4. TRAIN AND EVALUATE BOTH CONFIGURATIONS
# ------------------------------------------------------------

fold_rows = []
prediction_rows = []

for name, (df, feature_cols) in datasets.items():

    # Match feature rows to the common record order.
    aligned = df.set_index("record_id").loc[ids]
    X = aligned[feature_cols].reset_index(drop=True)

    print(f"\n{'=' * 60}")
    print(f"EVALUATING {name}")
    print(f"{'=' * 60}")

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )

        print(f"\n{name} — Fold {fold}: training...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Positive class: Unacceptable
        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred,
            labels=["Acceptable", "Unacceptable"],
        ).ravel()

        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        metrics = {
            "configuration": name,
            "fold": fold,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred,
                pos_label="Unacceptable",
                zero_division=0,
            ),
            "sensitivity": recall_score(
                y_test, y_pred,
                pos_label="Unacceptable",
                zero_division=0,
            ),
            "specificity": specificity,
            "f1": f1_score(
                y_test, y_pred,
                pos_label="Unacceptable",
                zero_division=0,
            ),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "train_records": len(train_idx),
            "test_records": len(test_idx),
            "train_signal_groups": signal_groups.iloc[train_idx].nunique(),
            "test_signal_groups": signal_groups.iloc[test_idx].nunique(),
        }

        fold_rows.append(metrics)

        prediction_rows.extend({
            "configuration": name,
            "fold": fold,
            "record_id": ids.iloc[index],
            "true_label": y.iloc[index],
            "predicted_label": prediction,
        } for index, prediction in zip(test_idx, y_pred))

        print(
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"Sensitivity={metrics['sensitivity']:.4f} | "
            f"Specificity={metrics['specificity']:.4f} | "
            f"F1={metrics['f1']:.4f}"
        )


# ------------------------------------------------------------
# 5. SAVE FOLD RESULTS AND OOF PREDICTIONS
# ------------------------------------------------------------

fold_df = pd.DataFrame(fold_rows)
pred_df = pd.DataFrame(prediction_rows)

fold_df.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
pred_df.to_csv(OUTPUT_DIR / "out_of_fold_predictions.csv", index=False)


# ------------------------------------------------------------
# 6. SUMMARIZE MEAN AND STANDARD DEVIATION ACROSS FOLDS
# ------------------------------------------------------------

metric_names = [
    "accuracy", "precision", "sensitivity", "specificity", "f1"
]

summary_rows = []

for name in FILES:
    subset = fold_df[fold_df["configuration"] == name]

    row = {"configuration": name}

    for metric in metric_names:
        row[f"{metric}_mean"] = subset[metric].mean()
        row[f"{metric}_std"] = subset[metric].std(ddof=0)

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_DIR / "summary_results.csv", index=False)

print("\n" + "=" * 65)
print("FINAL DUPLICATE-AWARE FREQUENCY COMPARISON")
print("=" * 65)

print(summary.round(4).to_string(index=False))

print("\nSaved results to:", OUTPUT_DIR)
print(" - fold_metrics.csv")
print(" - out_of_fold_predictions.csv")
print(" - summary_results.csv")
