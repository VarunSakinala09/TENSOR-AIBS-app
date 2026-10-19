
from pathlib import Path
import ast

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


ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "results" / "week3" / "features_full_dataset.csv"
RANK_FILE = (
    ROOT / "results" / "week2" / "rank_budget_all_records"
    / "multi_record_summary.csv"
)
GROUP_FILE = (
    ROOT / "results" / "week4" / "group_aware_cv"
    / "record_signal_groups.csv"
)
OUT_DIR = ROOT / "results" / "week4" / "adaptive_rank_three_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 5


def parse_rank(value):
    rank = ast.literal_eval(str(value))
    rank = tuple(int(x) for x in rank)

    if len(rank) != 3 or any(x < 1 for x in rank):
        raise ValueError(f"Invalid rank: {value}")

    return rank


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
            y_true, y_pred, zero_division=0
        ),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def main():
    for path in [FEATURE_FILE, RANK_FILE, GROUP_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    base = pd.read_csv(FEATURE_FILE, dtype={"record_id": str})
    ranks = pd.read_csv(RANK_FILE, dtype={"record_id": str})
    groups = pd.read_csv(GROUP_FILE, dtype={"record_id": str})

    if base["record_id"].duplicated().any():
        raise ValueError("Duplicate IDs in baseline feature dataset.")
    if ranks["record_id"].duplicated().any():
        raise ValueError("Duplicate IDs in adaptive rank dataset.")
    if groups["record_id"].duplicated().any():
        raise ValueError("Duplicate IDs in signal-group dataset.")

    ranks = ranks[["record_id", "selected_rank"]].copy()
    parsed = ranks["selected_rank"].map(parse_rank)

    ranks[["adaptive_rank_time",
           "adaptive_rank_frequency",
           "adaptive_rank_leads"]] = pd.DataFrame(
        parsed.tolist(), index=ranks.index
    )

    groups = groups[["record_id", "signal_hash"]]

    # Use only records present in all three existing datasets.
    data = base.merge(
        ranks, on="record_id", how="inner", validate="one_to_one"
    )
    data = data.merge(
        groups, on="record_id", how="inner", validate="one_to_one"
    )

    if data.empty:
        raise ValueError("No matching records across the three datasets.")

    if data["label"].isna().any():
        raise ValueError("Missing labels found.")

    # Encode Unacceptable as the positive class.
    y = (data["label"] == "Unacceptable").astype(int).to_numpy()
    group_ids = data["signal_hash"].to_numpy()

    rank_columns = [
        "adaptive_rank_time",
        "adaptive_rank_frequency",
        "adaptive_rank_leads",
    ]
    baseline_columns = [
        c for c in base.columns
        if c.startswith("feature_")
    ]

    if len(baseline_columns) != 6000:
        raise ValueError(
            f"Expected 6000 baseline features, found {len(baseline_columns)}"
        )

    X_base = data[baseline_columns].to_numpy(dtype=np.float64)
    X_combined = np.column_stack([
        X_base,
        data[rank_columns].to_numpy(dtype=np.float64),
    ])

    if not np.isfinite(X_base).all():
        raise ValueError("Baseline features contain non-finite values.")

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    oof = {
        "E0_fixed_features": np.full(len(data), -1, dtype=int),
        "E1_fixed_plus_rank": np.full(len(data), -1, dtype=int),
    }
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X_base, y, groups=group_ids), start=1
    ):
        train_groups = set(group_ids[train_idx])
        test_groups = set(group_ids[test_idx])

        if train_groups & test_groups:
            raise RuntimeError(
                f"Signal-group leakage detected in fold {fold}"
            )

        for experiment, X in [
            ("E0_fixed_features", X_base),
            ("E1_fixed_plus_rank", X_combined),
        ]:
            model = RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            )
            model.fit(X[train_idx], y[train_idx])
            predictions = model.predict(X[test_idx])
            oof[experiment][test_idx] = predictions

            metrics = calculate_metrics(y[test_idx], predictions)
            fold_rows.append({
                "experiment": experiment,
                "fold": fold,
                "train_records": len(train_idx),
                "test_records": len(test_idx),
                **metrics,
            })

        print(f"Completed fold {fold}/{N_SPLITS}")

    if any((pred == -1).any() for pred in oof.values()):
        raise RuntimeError("Some records lack out-of-fold predictions.")

    summary_rows = []
    prediction_data = data[["record_id", "label", "signal_hash"]].copy()
    prediction_data["y_true"] = y

    for experiment, predictions in oof.items():
        metrics = calculate_metrics(y, predictions)
        summary_rows.append({"experiment": experiment, **metrics})
        prediction_data[f"{experiment}_prediction"] = predictions

    pd.DataFrame(summary_rows).to_csv(
        OUT_DIR / "overall_oof_metrics.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(
        OUT_DIR / "fold_metrics.csv", index=False
    )
    prediction_data.drop(columns=["signal_hash"]).to_csv(
        OUT_DIR / "out_of_fold_predictions.csv", index=False
    )

    print("\nOVERALL OUT-OF-FOLD RESULTS")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\nMatched records:", len(data))
    print("Unique signal groups:", data["signal_hash"].nunique())
    print("Results saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
