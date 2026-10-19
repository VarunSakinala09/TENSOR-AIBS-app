
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

NPZ_FILE = (
    ROOT / "results" / "week8" / "true_adaptive_rank_cv"
    / "tucker_core_features.npz"
)

GROUP_FILE = (
    ROOT / "results" / "week4" / "group_aware_cv"
    / "record_signal_groups.csv"
)

RANK_FILE = (
    ROOT / "results" / "week2" / "rank_budget_all_records"
    / "multi_record_summary.csv"
)

CURRENT_OOF = (
    ROOT / "results" / "week8" / "complete_adaptive_pipeline"
    / "fixed_core_oof_predictions.csv"
)

OUTPUT_DIR = ROOT / "results" / "week8" / "baseline_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 5
N_ESTIMATORS = 100


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
            f"Unexpected labels: {values[result.isna()].unique()}"
        )

    return result.to_numpy(dtype=int)


print("\nFIXED BASELINE AUDIT")
print("=" * 65)

saved = np.load(NPZ_FILE, allow_pickle=True)

record_ids = saved["record_ids"].astype(str)
X = saved["fixed_features"].astype(np.float32)

print("Feature matrix:", X.shape)
print("Feature values finite:", bool(np.isfinite(X).all()))

if len(record_ids) != len(X):
    raise ValueError("Record IDs and feature rows do not match.")

if len(set(record_ids)) != len(record_ids):
    raise ValueError("Duplicate record IDs in feature file.")

groups_df = pd.read_csv(GROUP_FILE)
rank_df = pd.read_csv(RANK_FILE)

for frame in [groups_df, rank_df]:
    frame["record_id"] = frame["record_id"].astype(str).str.strip()

if groups_df["record_id"].duplicated().any():
    raise ValueError("Duplicate IDs in group mapping.")

if rank_df["record_id"].duplicated().any():
    raise ValueError("Duplicate IDs in rank file.")

# Reconstruct the same record alignment used by the pipeline.
lookup = pd.DataFrame({
    "record_id": record_ids,
    "feature_index": np.arange(len(record_ids)),
})

data = (
    rank_df[["record_id", "label"]]
    .merge(
        groups_df[["record_id", "label", "signal_hash"]],
        on="record_id",
        suffixes=("_rank", "_group"),
        validate="one_to_one",
    )
    .merge(
        lookup,
        on="record_id",
        validate="one_to_one",
    )
    .sort_values("record_id")
    .reset_index(drop=True)
)

if len(data) != len(X):
    raise ValueError(
        f"Alignment count mismatch: data={len(data)}, features={len(X)}"
    )

if not (
    data["label_rank"].astype(str).str.strip().to_numpy()
    == data["label_group"].astype(str).str.strip().to_numpy()
).all():
    raise ValueError("Labels disagree between source files.")

y = convert_labels(data["label_rank"])
groups = data["signal_hash"].astype(str).to_numpy()

if pd.isna(data["signal_hash"]).any():
    raise ValueError("Missing signal group identifiers.")

X = X[data["feature_index"].to_numpy(dtype=int)]
record_ids = data["record_id"].to_numpy()

print("Aligned records:", len(data))
print("Acceptable:", int((y == 0).sum()))
print("Unacceptable:", int((y == 1).sum()))
print("Unique groups:", len(np.unique(groups)))

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

predictions = np.full(len(y), -1, dtype=int)
fold_rows = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1,
):
    if set(groups[train_idx]).intersection(groups[test_idx]):
        raise RuntimeError("Group leakage detected.")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )

    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    predictions[test_idx] = pred

    tn, fp, fn, tp = confusion_matrix(
        y[test_idx], pred, labels=[0, 1]
    ).ravel()

    row = {
        "fold": fold,
        "accuracy": accuracy_score(y[test_idx], pred),
        "precision": precision_score(
            y[test_idx], pred, zero_division=0
        ),
        "sensitivity": recall_score(
            y[test_idx], pred, zero_division=0
        ),
        "specificity": tn / max(tn + fp, 1),
        "f1": f1_score(y[test_idx], pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    fold_rows.append(row)
    print(
        f"Fold {fold}: F1={row['f1']:.4f}, "
        f"Sensitivity={row['sensitivity']:.4f}, "
        f"Specificity={row['specificity']:.4f}"
    )

if np.any(predictions < 0):
    raise RuntimeError("Some records have no out-of-fold prediction.")

tn, fp, fn, tp = confusion_matrix(
    y, predictions, labels=[0, 1]
).ravel()

summary = {
    "accuracy": accuracy_score(y, predictions),
    "precision": precision_score(
        y, predictions, zero_division=0
    ),
    "sensitivity": recall_score(
        y, predictions, zero_division=0
    ),
    "specificity": tn / max(tn + fp, 1),
    "f1": f1_score(y, predictions, zero_division=0),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
}

print("\nREPRODUCED POOLED BASELINE")
print("=" * 65)

for key, value in summary.items():
    print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(OUTPUT_DIR / "reproduced_fold_results.csv", index=False)

reproduced = pd.DataFrame({
    "record_id": record_ids,
    "true_label": y,
    "reproduced_prediction": predictions,
})

reproduced.to_csv(
    OUTPUT_DIR / "reproduced_oof_predictions.csv",
    index=False,
)

if CURRENT_OOF.exists():
    current = pd.read_csv(CURRENT_OOF)
    current["record_id"] = current["record_id"].astype(str).str.strip()

    comparison = reproduced.merge(
        current,
        on="record_id",
        suffixes=("_reproduced", "_current"),
        validate="one_to_one",
    )

    prediction_column = (
        "predicted_label"
        if "predicted_label" in comparison.columns
        else None
    )

    if prediction_column:
        matches = (
            comparison["reproduced_prediction"]
            == comparison[prediction_column]
        )

        print("\nCOMPARISON WITH COMPLETED GPU RUN")
        print("Matched record IDs:", len(comparison))
        print("Identical predictions:", int(matches.sum()))
        print("Different predictions:", int((~matches).sum()))

        comparison.to_csv(
            OUTPUT_DIR / "prediction_comparison.csv",
            index=False,
        )
else:
    print("\nCurrent GPU OOF prediction file was not found.")

print("\nAudit files saved to:", OUTPUT_DIR)
