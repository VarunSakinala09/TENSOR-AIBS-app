
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
    confusion_matrix
)


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "results" / "week8" /
    "true_adaptive_rank_cv" / "tucker_core_features.npz"
)

RANK_FILE = (
    ROOT / "results" / "week2" /
    "rank_budget_all_records" / "multi_record_summary.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "week8" /
    "adaptive_rank_ablation"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5


# --------------------------------------------------
# LOAD FEATURES
# --------------------------------------------------

print("\nADAPTIVE TUCKER-RANK ABLATION")
print("=" * 65)

saved = np.load(INPUT_FILE, allow_pickle=True)

record_ids = saved["record_ids"].astype(str)
X_fixed = saved["fixed_features"].astype(np.float32)
X_adaptive = saved["adaptive_features"].astype(np.float32)
y = saved["labels"].astype(int)
groups = saved["groups"].astype(str)

rank_df = pd.read_csv(RANK_FILE)
rank_df["record_id"] = rank_df["record_id"].astype(str)

rank_df = rank_df.set_index("record_id")

missing = [
    record_id for record_id in record_ids
    if record_id not in rank_df.index
]

if missing:
    raise ValueError(
        f"{len(missing)} feature records are missing rank metadata."
    )

rank_values = np.array([
    eval_rank
    for eval_rank in rank_df.loc[
        record_ids, "selected_rank"
    ].map(
        lambda value: tuple(
            int(v.strip())
            for v in str(value).strip("()").split(",")
        )
    )
], dtype=np.float32)

if not np.isfinite(X_fixed).all():
    raise ValueError("Fixed features contain non-finite values.")

if not np.isfinite(X_adaptive).all():
    raise ValueError("Adaptive features contain non-finite values.")

if not np.isfinite(rank_values).all():
    raise ValueError("Rank metadata contains non-finite values.")

if len(np.unique(groups)) < N_SPLITS:
    raise ValueError("Not enough unique groups for cross-validation.")

print("Records:", len(record_ids))
print("Fixed features:", X_fixed.shape)
print("Adaptive features:", X_adaptive.shape)
print("Rank metadata:", rank_values.shape)
print("Acceptable:", int((y == 0).sum()))
print("Unacceptable:", int((y == 1).sum()))


# --------------------------------------------------
# FOUR COMPARISON METHODS
# --------------------------------------------------

methods = {
    "fixed_rank_core": X_fixed,
    "adaptive_rank_core": X_adaptive,
    "rank_metadata_only": rank_values,
    "adaptive_core_plus_rank_metadata": np.concatenate(
        [X_adaptive, rank_values], axis=1
    )
}

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_predictions = {
    method: np.full(len(y), -1, dtype=int)
    for method in methods
}

fold_results = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X_fixed, y, groups),
    start=1
):
    print(f"\nFold {fold}/{N_SPLITS}")

    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])

    if train_groups.intersection(test_groups):
        raise RuntimeError("Signal-group leakage detected.")

    for method_name, X in methods.items():

        model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])

        oof_predictions[method_name][test_idx] = pred

        tn, fp, fn, tp = confusion_matrix(
            y[test_idx],
            pred,
            labels=[0, 1]
        ).ravel()

        row = {
            "fold": fold,
            "method": method_name,
            "accuracy": accuracy_score(y[test_idx], pred),
            "precision": precision_score(
                y[test_idx], pred, zero_division=0
            ),
            "sensitivity": recall_score(
                y[test_idx], pred, zero_division=0
            ),
            "specificity": tn / max(tn + fp, 1),
            "f1": f1_score(
                y[test_idx], pred, zero_division=0
            ),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }

        fold_results.append(row)

        print(
            f"  {method_name}: "
            f"F1={row['f1']:.4f}, "
            f"Sensitivity={row['sensitivity']:.4f}, "
            f"Specificity={row['specificity']:.4f}"
        )


# --------------------------------------------------
# SAVE FOLD RESULTS
# --------------------------------------------------

fold_df = pd.DataFrame(fold_results)

fold_df.to_csv(
    OUTPUT_DIR / "fold_results.csv",
    index=False
)

summary = fold_df.groupby("method")[
    ["accuracy", "precision", "sensitivity", "specificity", "f1"]
].agg(["mean", "std"])

print("\nFOLD MEAN ± STANDARD DEVIATION")
print("=" * 75)
print(summary.round(4).to_string())


# --------------------------------------------------
# SAVE POOLED OUT-OF-FOLD RESULTS
# --------------------------------------------------

pooled_rows = []

for method_name, pred in oof_predictions.items():

    if np.any(pred < 0):
        raise RuntimeError(
            f"Missing predictions for {method_name}."
        )

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    pooled_rows.append({
        "method": method_name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(
            y, pred, zero_division=0
        ),
        "sensitivity": recall_score(
            y, pred, zero_division=0
        ),
        "specificity": tn / max(tn + fp, 1),
        "f1": f1_score(y, pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    })

    pd.DataFrame({
        "record_id": record_ids,
        "true_label": y,
        "predicted_label": pred
    }).to_csv(
        OUTPUT_DIR / f"{method_name}_oof.csv",
        index=False
    )

pooled_df = pd.DataFrame(pooled_rows)

pooled_df.to_csv(
    OUTPUT_DIR / "pooled_oof_results.csv",
    index=False
)

print("\nPOOLED OUT-OF-FOLD RESULTS")
print("=" * 75)
print(pooled_df.round(4).to_string(index=False))

print("\nSaved results to:")
print(OUTPUT_DIR)
