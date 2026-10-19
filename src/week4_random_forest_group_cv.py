
from pathlib import Path
import hashlib

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

# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "raw" / "set-a"
FEATURE_FILE = ROOT / "results" / "week3" / "features_full_dataset.csv"

OUT_DIR = ROOT / "results" / "week4" / "group_aware_cv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42
N_ESTIMATORS = 100

LABELS = ["Acceptable", "Unacceptable"]
POSITIVE_LABEL = "Unacceptable"


# --------------------------------------------------
# 2. HASH RAW ECG FILES
# --------------------------------------------------

def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def find_dat_file(record_id):
    # Search both the main folder and nested dataset folders.
    matches = list(DATA_DIR.rglob(f"{record_id}.dat"))

    if not matches:
        raise FileNotFoundError(
            f"Raw signal file not found for record {record_id}"
        )

    # Prefer the file whose matching header exists.
    matches.sort(
        key=lambda p: (
            not p.with_suffix(".hea").exists(),
            len(str(p))
        )
    )

    return matches[0]


# --------------------------------------------------
# 3. METRICS
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_LABEL,
            zero_division=0
        ),
        "sensitivity": recall_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_LABEL,
            zero_division=0
        ),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_LABEL,
            zero_division=0
        ),
    }


# --------------------------------------------------
# 4. LOAD AND VALIDATE FEATURES
# --------------------------------------------------

print("=" * 65)
print("DUPLICATE-AWARE RANDOM FOREST EVALUATION")
print("=" * 65)

data = pd.read_csv(FEATURE_FILE)

data["record_id"] = data["record_id"].astype(str).str.strip()

if data["record_id"].duplicated().any():
    raise ValueError("Duplicate record IDs found in feature CSV.")

if not set(data["label"].unique()).issubset(set(LABELS)):
    raise ValueError("Unexpected label found in feature CSV.")

feature_columns = [
    column for column in data.columns
    if column.startswith("feature_")
]

if not feature_columns:
    raise ValueError("No feature columns were found.")

X = data[feature_columns].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

if X.isna().any().any():
    raise ValueError(
        "Missing or non-finite feature values found. "
        "Investigate them before training."
    )

y = data["label"].to_numpy()

print("Records:", len(data))
print("Features per record:", len(feature_columns))
print("\nLabel counts:")
print(data["label"].value_counts().to_string())


# --------------------------------------------------
# 5. GROUP IDENTICAL RAW SIGNAL FILES
# --------------------------------------------------

hashes = []

for index, record_id in enumerate(data["record_id"], start=1):
    raw_file = find_dat_file(record_id)
    hashes.append(sha256_file(raw_file))

    if index % 100 == 0:
        print(f"Hashed {index}/{len(data)} records")

data["signal_hash"] = hashes

group_sizes = data.groupby("signal_hash").size()
duplicate_groups = group_sizes[group_sizes > 1]

print("\nUnique raw signal contents:", len(group_sizes))
print("Duplicate groups:", len(duplicate_groups))

if len(duplicate_groups):
    print("\nDuplicate group sizes:")
    print(
        duplicate_groups
        .sort_values(ascending=False)
        .to_string()
    )

# A group with conflicting labels requires investigation.
label_counts_per_group = data.groupby("signal_hash")["label"].nunique()

conflicting_groups = label_counts_per_group[
    label_counts_per_group > 1
]

if len(conflicting_groups):
    conflict_rows = data[
        data["signal_hash"].isin(conflicting_groups.index)
    ][["record_id", "label", "signal_hash"]]

    conflict_rows.to_csv(
        OUT_DIR / "conflicting_duplicate_labels.csv",
        index=False
    )

    raise ValueError(
        "Identical raw signal files have conflicting labels. "
        "See conflicting_duplicate_labels.csv before continuing."
    )

groups = data["signal_hash"].to_numpy()

# Save a record-to-group audit table.
group_map = data[
    ["record_id", "label", "signal_hash"]
].copy()

group_map["group_size"] = group_map["signal_hash"].map(group_sizes)

group_map.to_csv(
    OUT_DIR / "record_signal_groups.csv",
    index=False
)


# --------------------------------------------------
# 6. STRATIFIED GROUP CROSS-VALIDATION
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_results = []
oof_predictions = np.empty(len(data), dtype=object)
prediction_counts = np.zeros(len(data), dtype=int)

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])

    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise RuntimeError(
            f"Signal leakage detected in fold {fold}."
        )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X.iloc[train_idx], y[train_idx])

    predictions = model.predict(X.iloc[test_idx])

    oof_predictions[test_idx] = predictions
    prediction_counts[test_idx] += 1

    scores = calculate_metrics(y[test_idx], predictions)

    result = {
        "fold": fold,
        "train_records": len(train_idx),
        "test_records": len(test_idx),
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        **scores
    }

    fold_results.append(result)

    print(f"\nFold {fold}")
    print("Training records:", len(train_idx))
    print("Testing records:", len(test_idx))
    print("Overlapping signal groups:", len(overlap))

    for metric, value in scores.items():
        print(f"{metric.capitalize()}: {value:.4f}")


# Every record must be tested exactly once.
if not np.all(prediction_counts == 1):
    raise RuntimeError(
        "Some records did not receive exactly one test prediction."
    )


# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

fold_df = pd.DataFrame(fold_results)

fold_df.to_csv(
    OUT_DIR / "fold_metrics.csv",
    index=False
)

pred_df = data[
    ["record_id", "label", "signal_hash"]
].copy()

pred_df["oof_prediction"] = oof_predictions

pred_df.to_csv(
    OUT_DIR / "out_of_fold_predictions.csv",
    index=False
)


# --------------------------------------------------
# 8. FINAL SUMMARY
# --------------------------------------------------

metric_columns = [
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1"
]

print("\n" + "=" * 65)
print("DUPLICATE-AWARE RESULTS")
print("=" * 65)

print("\nFold mean ± standard deviation:")

for metric in metric_columns:
    mean_value = fold_df[metric].mean()
    std_value = fold_df[metric].std(ddof=0)

    print(
        f"{metric.capitalize()}: "
        f"{mean_value:.4f} ± {std_value:.4f}"
    )

overall = calculate_metrics(y, oof_predictions)

print("\nCombined out-of-fold results:")

for metric, value in overall.items():
    print(f"{metric.capitalize()}: {value:.4f}")

print("\nFiles saved in:", OUT_DIR)
print("- fold_metrics.csv")
print("- out_of_fold_predictions.csv")
print("- record_signal_groups.csv")

print("\nEvaluation completed.")
print(
    "Note: results use group-aware folds for byte-identical raw "
    "signal files. Other forms of signal equivalence are not checked."
)
