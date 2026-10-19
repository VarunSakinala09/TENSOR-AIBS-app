
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
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "results" / "week3" / "features_full_dataset.csv"
RANK_FILE = (
    ROOT / "results" / "week2" / "rank_budget_all_records"
    / "multi_record_summary.csv"
)
GROUP_FILE = ROOT / "results" / "week4" / "group_aware_cv" / "record_signal_groups.csv"

OUT_DIR = ROOT / "results" / "week4" / "adaptive_rank_metadata_cv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42
N_ESTIMATORS = 100

print("=" * 68)
print("ADAPTIVE-RANK METADATA: GROUP-AWARE CLASSIFICATION")
print("=" * 68)

# ---------------------------------------------------------
# 1. Check input files
# ---------------------------------------------------------
for path in [FEATURE_FILE, RANK_FILE, GROUP_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

# ---------------------------------------------------------
# 2. Load fixed-rank features
# ---------------------------------------------------------
features = pd.read_csv(FEATURE_FILE)
features["record_id"] = features["record_id"].astype(str)

if features["record_id"].duplicated().any():
    raise ValueError("Duplicate record IDs in the feature CSV.")

if "label" not in features.columns:
    raise ValueError("The feature CSV has no label column.")

feature_columns = [
    c for c in features.columns
    if c not in ["record_id", "label"]
]

X_fixed = features[feature_columns].apply(
    pd.to_numeric, errors="raise"
)

if not np.isfinite(X_fixed.to_numpy(dtype=float)).all():
    raise ValueError("Fixed-rank features contain NaN or infinite values.")

y_text = features["label"].astype(str)
label_map = {"Acceptable": 0, "Unacceptable": 1}

if not set(y_text.unique()).issubset(label_map):
    raise ValueError(f"Unexpected labels: {sorted(y_text.unique())}")

y = y_text.map(label_map).to_numpy()

print(f"Fixed-rank records: {len(features)}")
print(f"Fixed-rank features: {len(feature_columns)}")

# ---------------------------------------------------------
# 3. Load adaptive-rank search results
# ---------------------------------------------------------
rank_data = pd.read_csv(RANK_FILE)
rank_data["record_id"] = rank_data["record_id"].astype(str)

if rank_data["record_id"].duplicated().any():
    raise ValueError("Duplicate record IDs in adaptive-rank results.")

required_rank_columns = [
    "record_id",
    "label",
    "selected_rank",
    "fixed_error",
    "selected_error",
    "representation_reduction_percent",
    "error_reduction_percent"
]

missing = [c for c in required_rank_columns if c not in rank_data.columns]
if missing:
    raise ValueError(f"Missing adaptive-rank columns: {missing}")

rank_data["label"] = rank_data["label"].astype(str)

# ---------------------------------------------------------
# 4. Join by record ID and validate labels
# ---------------------------------------------------------
data = features[["record_id", "label"]].merge(
    rank_data,
    on="record_id",
    how="left",
    suffixes=("_feature", "_rank"),
    validate="one_to_one"
)


missing_mask = data["selected_rank"].isna()

if missing_mask.any():
    missing_ids = data.loc[missing_mask, "record_id"].tolist()
    print(
        f"\nExcluding {len(missing_ids)} records without adaptive-rank results:"
    )
    print(missing_ids)

    data = data.loc[~missing_mask].copy().reset_index(drop=True)
    features = features[
        features["record_id"].isin(data["record_id"])
    ].copy().reset_index(drop=True)

    X_fixed = features[feature_columns].reset_index(drop=True)
    y_text = features["label"].astype(str)

    if not (data["record_id"].to_numpy() == features["record_id"].to_numpy()).all():
        raise ValueError("Record alignment failed after excluding missing records.")

    if not (data["label_feature"].to_numpy() == features["label"].to_numpy()).all():
        raise ValueError("Label alignment failed after excluding missing records.")

    y = y_text.map(label_map).to_numpy()


if not (data["label_feature"] == data["label_rank"]).all():
    raise ValueError("Label mismatch between fixed features and rank results.")

# Selected rank is stored as a string such as "(49, 17, 10)".
# Convert the three rank components into numeric features.
def parse_rank(value):
    text = str(value).strip().strip("()[]")
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid selected_rank value: {value}")
    ranks = [int(p) for p in parts]
    if any(r < 1 for r in ranks):
        raise ValueError(f"Nonpositive rank found: {value}")
    return ranks

parsed = data["selected_rank"].apply(parse_rank)

adaptive_metadata = pd.DataFrame(
    parsed.tolist(),
    columns=["adaptive_rank_time", "adaptive_rank_frequency", "adaptive_rank_leads"]
)

adaptive_metadata["adaptive_fixed_error"] = pd.to_numeric(
    data["fixed_error"], errors="raise"
)
adaptive_metadata["adaptive_selected_error"] = pd.to_numeric(
    data["selected_error"], errors="raise"
)
adaptive_metadata["adaptive_representation_reduction_pct"] = pd.to_numeric(
    data["representation_reduction_percent"], errors="raise"
)
adaptive_metadata["adaptive_error_reduction_pct"] = pd.to_numeric(
    data["error_reduction_percent"], errors="raise"
)

if not np.isfinite(adaptive_metadata.to_numpy(dtype=float)).all():
    raise ValueError("Adaptive-rank metadata contains NaN or infinite values.")

# ---------------------------------------------------------
# 5. Load duplicate-aware signal groups
# ---------------------------------------------------------
groups_data = pd.read_csv(GROUP_FILE)
groups_data["record_id"] = groups_data["record_id"].astype(str)

if groups_data["record_id"].duplicated().any():
    raise ValueError("Duplicate record IDs in signal group file.")

# Identify the signal-hash/group column robustly.
possible_group_columns = [
    c for c in groups_data.columns
    if c.lower() in ["signal_hash", "group", "group_id"]
]
if not possible_group_columns:
    raise ValueError(
        "Could not identify the signal group column in record_signal_groups.csv. "
        f"Columns found: {list(groups_data.columns)}"
    )

group_column = possible_group_columns[0]

data = data.merge(
    groups_data[["record_id", group_column]],
    on="record_id",
    how="left",
    validate="one_to_one"
)

if data[group_column].isna().any():
    missing_ids = data.loc[data[group_column].isna(), "record_id"].tolist()
    raise ValueError(
        f"{len(missing_ids)} records have no signal group. "
        f"Examples: {missing_ids[:10]}"
    )

if len(data) != len(features):
    raise ValueError("Unexpected row count after joins.")

# Align features and metadata to the merged row order.
X_fixed = X_fixed.reset_index(drop=True)
X_adaptive = pd.concat(
    [X_fixed, adaptive_metadata.reset_index(drop=True)],
    axis=1
)
y = data["label_feature"].map(label_map).to_numpy()
groups = data[group_column].astype(str).to_numpy()

print(f"Matched records: {len(data)}")
print(f"Adaptive metadata features added: {adaptive_metadata.shape[1]}")
print(f"Unique signal groups: {pd.Series(groups).nunique()}")

# Save the exact matched records and group IDs used.
data[["record_id", "label_feature", group_column]].to_csv(
    OUT_DIR / "matched_records_and_groups.csv", index=False
)

# ---------------------------------------------------------
# 6. Evaluate fixed features and fixed + adaptive metadata
#    on exactly the same group-aware folds
# ---------------------------------------------------------
cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_rows = []
oof_rows = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X_fixed, y, groups), start=1
):
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise RuntimeError(
            f"Signal group leakage detected in fold {fold}: {len(overlap)} groups"
        )

    for experiment_name, X in [
        ("fixed_rank_features", X_fixed),
        ("fixed_plus_adaptive_rank_metadata", X_adaptive)
    ]:
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(X.iloc[train_idx], y[train_idx])
        predictions = model.predict(X.iloc[test_idx])

        tn, fp, fn, tp = confusion_matrix(
            y[test_idx], predictions, labels=[0, 1]
        ).ravel()

        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        metrics = {
            "fold": fold,
            "experiment": experiment_name,
            "accuracy": accuracy_score(y[test_idx], predictions),
            "precision": precision_score(
                y[test_idx], predictions, zero_division=0
            ),
            "sensitivity": recall_score(
                y[test_idx], predictions, zero_division=0
            ),
            "specificity": specificity,
            "f1": f1_score(y[test_idx], predictions, zero_division=0),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "train_records": len(train_idx),
            "test_records": len(test_idx),
            "overlapping_signal_groups": len(overlap)
        }
        fold_rows.append(metrics)

        for idx, pred in zip(test_idx, predictions):
            oof_rows.append({
                "record_id": data.iloc[idx]["record_id"],
                "fold": fold,
                "experiment": experiment_name,
                "true_label": int(y[idx]),
                "predicted_label": int(pred)
            })

        print(
            f"Fold {fold} | {experiment_name} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"Sens={metrics['sensitivity']:.4f} | "
            f"Spec={metrics['specificity']:.4f} | "
            f"F1={metrics['f1']:.4f}"
        )

# ---------------------------------------------------------
# 7. Save fold results and combined OOF metrics
# ---------------------------------------------------------
fold_df = pd.DataFrame(fold_rows)
oof_df = pd.DataFrame(oof_rows)

fold_df.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
oof_df.to_csv(OUT_DIR / "out_of_fold_predictions.csv", index=False)

summary_rows = []

for experiment_name, part in oof_df.groupby("experiment"):
    truth = part["true_label"].to_numpy()
    pred = part["predicted_label"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()

    summary_rows.append({
        "experiment": experiment_name,
        "accuracy": accuracy_score(truth, pred),
        "precision": precision_score(truth, pred, zero_division=0),
        "sensitivity": recall_score(truth, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(truth, pred, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "combined_oof_summary.csv", index=False)

print("\n" + "=" * 68)
print("COMBINED OUT-OF-FOLD RESULTS")
print("=" * 68)
print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print(f"\nResults saved to: {OUT_DIR}")
print("Files:")
print("- fold_metrics.csv")
print("- out_of_fold_predictions.csv")
print("- combined_oof_summary.csv")
print("- matched_records_and_groups.csv")
print("\nEvaluation completed.")
