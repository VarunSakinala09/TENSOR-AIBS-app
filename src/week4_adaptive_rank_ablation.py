
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "results/week3/features_full_dataset.csv"
RANK_FILE = ROOT / "results/week2/rank_budget_all_records/multi_record_summary.csv"
GROUP_FILE = ROOT / "results/week4/group_aware_cv/record_signal_groups.csv"
OUT_DIR = ROOT / "results/week4/adaptive_rank_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
features = pd.read_csv(FEATURE_FILE)
ranks = pd.read_csv(RANK_FILE)
groups_df = pd.read_csv(GROUP_FILE)

for df in [features, ranks, groups_df]:
    df["record_id"] = df["record_id"].astype(str)

if features["record_id"].duplicated().any():
    raise ValueError("Duplicate IDs in feature file.")
if ranks["record_id"].duplicated().any():
    raise ValueError("Duplicate IDs in adaptive-rank file.")
if groups_df["record_id"].duplicated().any():
    raise ValueError("Duplicate IDs in group file.")

# Keep records with adaptive-rank results and align by ID.
data = features.merge(
    ranks, on="record_id", how="inner",
    suffixes=("_feature", "_rank"), validate="one_to_one"
)
data = data.merge(
    groups_df[["record_id", "signal_hash"]],
    on="record_id", how="inner", validate="one_to_one"
)

if len(data) != len(ranks):
    print(f"Matched records: {len(data)}; rank-search records: {len(ranks)}")

if not (data["label_feature"] == data["label_rank"]).all():
    raise ValueError("Labels disagree between feature and rank files.")

label_map = {"Acceptable": 0, "Unacceptable": 1}
if not set(data["label_feature"].unique()).issubset(label_map):
    raise ValueError("Unexpected labels.")

y = data["label_feature"].map(label_map).to_numpy()
groups = data["signal_hash"].astype(str).to_numpy()

# Recover the fixed-rank features.
feature_columns = [
    c for c in features.columns if c not in ["record_id", "label"]
]
X_fixed = data[feature_columns].apply(pd.to_numeric, errors="raise")

if not np.isfinite(X_fixed.to_numpy(dtype=float)).all():
    raise ValueError("Fixed features contain non-finite values.")

# Parse selected Tucker ranks.
def parse_rank(value):
    parts = str(value).strip().strip("()[]").split(",")
    if len(parts) != 3:
        raise ValueError(f"Invalid selected rank: {value}")
    result = [int(p.strip()) for p in parts]
    if any(v < 1 for v in result):
        raise ValueError(f"Invalid rank value: {value}")
    return result

rank_values = pd.DataFrame(
    data["selected_rank"].apply(parse_rank).tolist(),
    columns=["rank_time", "rank_frequency", "rank_leads"],
    index=data.index
)

metadata_columns = [
    "fixed_error",
    "selected_error",
    "representation_reduction_percent",
    "error_reduction_percent"
]

for col in metadata_columns:
    rank_values[col] = pd.to_numeric(data[col], errors="raise")

if not np.isfinite(rank_values.to_numpy(dtype=float)).all():
    raise ValueError("Adaptive metadata contains non-finite values.")

experiments = {
    "E0_fixed_rank": X_fixed.reset_index(drop=True),
    "E1_fixed_plus_3_ranks": pd.concat(
        [X_fixed.reset_index(drop=True),
         rank_values[["rank_time", "rank_frequency", "rank_leads"]].reset_index(drop=True)],
        axis=1
    ),
    "E2_fixed_plus_all_7_metadata": pd.concat(
        [X_fixed.reset_index(drop=True), rank_values.reset_index(drop=True)],
        axis=1
    )
}

cv = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=42
)

fold_results = []
oof_results = []

print("=" * 65)
print("ADAPTIVE-RANK ABLATION STUDY")
print("=" * 65)
print(f"Matched records: {len(data)}")
print(f"Fixed-rank features: {X_fixed.shape[1]}")
print(f"Unique signal groups: {pd.Series(groups).nunique()}")

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X_fixed, y, groups), start=1
):
    if set(groups[train_idx]) & set(groups[test_idx]):
        raise RuntimeError(f"Signal-group overlap in fold {fold}")

    for name, X in experiments.items():
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[test_idx])

        tn, fp, fn, tp = confusion_matrix(
            y[test_idx], pred, labels=[0, 1]
        ).ravel()

        row = {
            "fold": fold,
            "experiment": name,
            "accuracy": accuracy_score(y[test_idx], pred),
            "precision": precision_score(y[test_idx], pred, zero_division=0),
            "sensitivity": recall_score(y[test_idx], pred, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else 0,
            "f1": f1_score(y[test_idx], pred, zero_division=0),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp
        }
        fold_results.append(row)

        for local_i, pred_i in zip(test_idx, pred):
            oof_results.append({
                "record_id": data.iloc[local_i]["record_id"],
                "fold": fold,
                "experiment": name,
                "true_label": int(y[local_i]),
                "predicted_label": int(pred_i)
            })

        print(
            f"Fold {fold} | {name} | "
            f"Sensitivity={row['sensitivity']:.4f} | "
            f"Specificity={row['specificity']:.4f} | "
            f"F1={row['f1']:.4f}"
        )

fold_df = pd.DataFrame(fold_results)
oof_df = pd.DataFrame(oof_results)

fold_df.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
oof_df.to_csv(OUT_DIR / "out_of_fold_predictions.csv", index=False)

summary = []
for name, part in oof_df.groupby("experiment", sort=False):
    truth = part["true_label"].to_numpy()
    pred = part["predicted_label"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()

    summary.append({
        "experiment": name,
        "accuracy": accuracy_score(truth, pred),
        "precision": precision_score(truth, pred, zero_division=0),
        "sensitivity": recall_score(truth, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0,
        "f1": f1_score(truth, pred, zero_division=0),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT_DIR / "combined_oof_summary.csv", index=False)

print("\n" + "=" * 65)
print("ABLATION STUDY: COMBINED OUT-OF-FOLD RESULTS")
print("=" * 65)
print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nSaved results to: {OUT_DIR}")
