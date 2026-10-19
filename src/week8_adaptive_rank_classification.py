
from pathlib import Path
import re

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

FEATURE_PATH = ROOT / "results/week3/features_full_dataset.csv"
RANK_PATH = (
    ROOT / "results/week2/rank_budget_all_records"
    / "multi_record_summary.csv"
)
LEAD_PATH = (
    ROOT / "results/week8/lead_quality_features"
    / "lead_quality_features.csv"
)
GROUP_PATH = (
    ROOT / "results/week4/group_aware_cv"
    / "record_signal_groups.csv"
)

OUT = ROOT / "results/week8/adaptive_rank_classification"
OUT.mkdir(parents=True, exist_ok=True)


def normalize_ids(df):
    if "record_id" not in df.columns:
        raise ValueError(
            f"Missing record_id column. Found: {list(df.columns)}"
        )

    df["record_id"] = df["record_id"].astype(str).str.strip()

    if df["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs detected")

    return df


def encode_labels(series):
    labels = (
        series.astype(str).str.strip().str.lower()
        .map({
            "acceptable": 0,
            "accept": 0,
            "0": 0,
            "unacceptable": 1,
            "unaccept": 1,
            "1": 1,
        })
    )

    if labels.isna().any():
        raise ValueError(
            f"Unknown label values: {series[labels.isna()].unique()}"
        )

    return labels.to_numpy(dtype=int)


def find_column(df, required_words, excluded_words=()):
    """Find a column by its normalized name."""
    matches = []

    for col in df.columns:
        name = re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_")

        if all(word in name for word in required_words):
            if not any(word in name for word in excluded_words):
                matches.append(col)

    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely identify column for "
            f"{required_words}. Matches: {matches}. "
            f"Available columns: {list(df.columns)}"
        )

    return matches[0]


def parse_rank(value):
    numbers = re.findall(r"\d+", str(value))

    if len(numbers) != 3:
        raise ValueError(
            f"Could not parse a three-mode Tucker rank from: {value}"
        )

    return [int(number) for number in numbers]


def make_model(seed):
    return RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )


def evaluate(y_true, prediction):
    tn, fp, fn, tp = confusion_matrix(
        y_true, prediction, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(
            y_true, prediction, zero_division=0
        ),
        "sensitivity": recall_score(
            y_true, prediction, pos_label=1, zero_division=0
        ),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y_true, prediction, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    print("\nADAPTIVE TUCKER-RANK CLASSIFICATION")
    print("=" * 65)

    features = normalize_ids(pd.read_csv(
        FEATURE_PATH, dtype={"record_id": str}
    ))
    ranks = normalize_ids(pd.read_csv(
        RANK_PATH, dtype={"record_id": str}
    ))
    leads = normalize_ids(pd.read_csv(
        LEAD_PATH, dtype={"record_id": str}
    ))
    groups_df = normalize_ids(pd.read_csv(
        GROUP_PATH, dtype={"record_id": str}
    ))

    print("\nRank-search columns:")
    print(ranks.columns.tolist())

    # Identify the selected rank and reconstruction/size columns.
    rank_col = find_column(ranks, ["selected", "rank"])

    fixed_error_col = find_column(
        ranks, ["fixed", "error"],
        excluded_words=("size",),
    )
    selected_error_col = find_column(
        ranks, ["selected", "error"],
        excluded_words=("size",),
    )
    fixed_size_col = find_column(ranks, ["fixed", "size"])
    selected_size_col = find_column(ranks, ["selected", "size"])

    rank_features = ranks[["record_id"]].copy()

    parsed = ranks[rank_col].apply(parse_rank)
    rank_features[["selected_rank_time",
                   "selected_rank_frequency",
                   "selected_rank_leads"]] = pd.DataFrame(
        parsed.tolist(), index=ranks.index
    )

    fixed_error = pd.to_numeric(
        ranks[fixed_error_col], errors="coerce"
    )
    selected_error = pd.to_numeric(
        ranks[selected_error_col], errors="coerce"
    )
    fixed_size = pd.to_numeric(
        ranks[fixed_size_col], errors="coerce"
    )
    selected_size = pd.to_numeric(
        ranks[selected_size_col], errors="coerce"
    )

    rank_features["selected_reconstruction_error"] = selected_error
    rank_features["reconstruction_error_change"] = (
        fixed_error - selected_error
    )
    rank_features["representation_size_reduction"] = (
        fixed_size - selected_size
    )

    rank_metadata_cols = [
        c for c in rank_features.columns if c != "record_id"
    ]

    if rank_features[rank_metadata_cols].isna().any().any():
        raise ValueError(
            "Rank metadata contains missing/non-numeric values. "
            "Check the rank-search output."
        )

    # Keep all three datasets aligned by record ID.
    common = sorted(
        set(features["record_id"])
        & set(rank_features["record_id"])
        & set(leads["record_id"])
        & set(groups_df["record_id"])
    )

    if not common:
        raise ValueError("No matching record IDs across input files")

    features = features.set_index("record_id").loc[common]
    rank_features = rank_features.set_index("record_id").loc[common]
    leads = leads.set_index("record_id").loc[common]
    groups_df = groups_df.set_index("record_id").loc[common]

    y = encode_labels(features["label"])

    if not np.array_equal(y, encode_labels(leads["label"])):
        raise ValueError("Lead-feature labels do not match")

    if not np.array_equal(y, encode_labels(groups_df["label"])):
        raise ValueError("Group-mapping labels do not match")

    groups = groups_df["signal_hash"].astype(str).to_numpy()

    core_cols = [
        c for c in features.columns if c != "label"
    ]

    X_core_df = features[core_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    X_lead_df = leads.drop(columns=["label"]).apply(
        pd.to_numeric, errors="coerce"
    )

    X_core_df = X_core_df.replace([np.inf, -np.inf], np.nan)
    X_lead_df = X_lead_df.replace([np.inf, -np.inf], np.nan)

    if X_core_df.isna().any().any():
        raise ValueError("Invalid values in fixed-rank core features")

    if X_lead_df.isna().any().any():
        raise ValueError("Invalid values in lead-quality features")

    X_core = X_core_df.to_numpy(dtype=np.float32)
    X_rank = rank_features[rank_metadata_cols].to_numpy(
        dtype=np.float32
    )
    X_lead = X_lead_df.to_numpy(dtype=np.float32)

    X_combined = np.concatenate([X_core, X_rank], axis=1)

    datasets = {
        "fixed_rank_core_only": X_core,
        "fixed_core_plus_rank_metadata": X_combined,
        "lead_quality_only": X_lead,
    }

    print("\nRecords:", len(common))
    print("Unique signal groups:", len(np.unique(groups)))
    print("Fixed core features:", X_core.shape[1])
    print("Rank metadata features:", X_rank.shape[1])
    print("Lead-quality features:", X_lead.shape[1])

    rank_features.reset_index().to_csv(
        OUT / "rank_metadata_used.csv", index=False
    )

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    fold_rows = []
    prediction_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X_core, y, groups), start=1
    ):
        print(f"\nFold {fold}/5")

        if set(groups[train_idx]).intersection(groups[test_idx]):
            raise RuntimeError("Signal-group leakage detected")

        for name, matrix in datasets.items():
            model = make_model(42 + fold)
            model.fit(matrix[train_idx], y[train_idx])
            pred = model.predict(matrix[test_idx]).astype(int)

            score = evaluate(y[test_idx], pred)

            fold_rows.append({
                "fold": fold,
                "method": name,
                **score,
            })

            print(
                f"  {name}: F1={score['f1']:.4f}, "
                f"Sensitivity={score['sensitivity']:.4f}, "
                f"Specificity={score['specificity']:.4f}"
            )

            for j, global_idx in enumerate(test_idx):
                prediction_rows.append({
                    "record_id": common[global_idx],
                    "signal_hash": groups[global_idx],
                    "fold": fold,
                    "method": name,
                    "true_label": int(y[global_idx]),
                    "prediction": int(pred[j]),
                })

    folds_df = pd.DataFrame(fold_rows)
    preds_df = pd.DataFrame(prediction_rows)

    folds_df.to_csv(OUT / "fold_metrics.csv", index=False)
    preds_df.to_csv(OUT / "out_of_fold_predictions.csv", index=False)

    summary = (
        folds_df.groupby("method")[
            ["accuracy", "precision", "sensitivity",
             "specificity", "f1"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.to_csv(OUT / "fold_summary.csv")

    pooled_rows = []

    for method, part in preds_df.groupby("method"):
        pooled_rows.append({
            "method": method,
            **evaluate(
                part["true_label"].to_numpy(),
                part["prediction"].to_numpy(),
            ),
        })

    pooled = pd.DataFrame(pooled_rows).round(4)
    pooled.to_csv(OUT / "pooled_oof_metrics.csv", index=False)

    print("\nFOLD SUMMARY")
    print("=" * 75)
    print(summary.to_string())

    print("\nPOOLED OUT-OF-FOLD METRICS")
    print("=" * 75)
    print(pooled.to_string(index=False))

    print("\nSaved results to:", OUT)


if __name__ == "__main__":
    main()
