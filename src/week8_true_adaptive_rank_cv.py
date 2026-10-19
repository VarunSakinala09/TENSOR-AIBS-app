
from pathlib import Path
import ast
import time

import numpy as np
import pandas as pd
import pywt
import wfdb
from tensorly.decomposition import tucker

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "dataset" / "raw" / "set-a"

RANK_FILE = (
    ROOT / "results" / "week2" /
    "rank_budget_all_records" / "multi_record_summary.csv"
)

GROUP_FILE = (
    ROOT / "results" / "week4" /
    "group_aware_cv" / "record_signal_groups.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "week8" /
    "true_adaptive_rank_cv"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_RANK = (50, 20, 6)
MAX_RANK = (50, 36, 12)

FREQUENCIES = np.linspace(1, 40, 100)
WAVELET = "morl"

N_SPLITS = 5
RANDOM_STATE = 42


# --------------------------------------------------
# LOAD AND ALIGN RECORDS
# --------------------------------------------------

print("\nTRUE ADAPTIVE TUCKER-RANK CLASSIFICATION")
print("=" * 65)

rank_df = pd.read_csv(RANK_FILE)
group_df = pd.read_csv(GROUP_FILE)

rank_df["record_id"] = rank_df["record_id"].astype(str)
group_df["record_id"] = group_df["record_id"].astype(str)

if rank_df["record_id"].duplicated().any():
    raise ValueError("Duplicate record IDs in rank-search results.")

if group_df["record_id"].duplicated().any():
    raise ValueError("Duplicate record IDs in group mapping.")

rank_df["label"] = rank_df["label"].astype(str).str.strip()
group_df["label"] = group_df["label"].astype(str).str.strip()

data = rank_df.merge(
    group_df[["record_id", "label", "signal_hash"]],
    on="record_id",
    how="inner",
    suffixes=("_rank", "_group"),
    validate="one_to_one"
)

if not (
    data["label_rank"].to_numpy()
    == data["label_group"].to_numpy()
).all():
    raise ValueError("Label mismatch between rank results and group map.")

data["label"] = data["label_rank"]

if data["signal_hash"].isna().any():
    raise ValueError("Some records have no signal-group identifier.")

def parse_rank(value):
    rank = tuple(int(x) for x in ast.literal_eval(str(value)))

    if len(rank) != 3:
        raise ValueError(f"Invalid Tucker rank: {value}")

    if any(r < 1 for r in rank):
        raise ValueError(f"Non-positive Tucker rank: {value}")

    if any(r > m for r, m in zip(rank, MAX_RANK)):
        raise ValueError(f"Rank exceeds common feature dimensions: {value}")

    return rank


data["adaptive_rank"] = data["selected_rank"].apply(parse_rank)

data = data.sort_values("record_id").reset_index(drop=True)

y = (data["label"] == "Unacceptable").astype(int).to_numpy()
groups = data["signal_hash"].astype(str).to_numpy()

print("Aligned records:", len(data))
print("Unique signal groups:", len(np.unique(groups)))
print("Acceptable:", int((y == 0).sum()))
print("Unacceptable:", int((y == 1).sum()))
print("Fixed rank:", FIXED_RANK)
print("Adaptive core storage shape:", MAX_RANK)


# --------------------------------------------------
# BUILD CWT TENSOR
# --------------------------------------------------

def load_cwt_tensor(record_id):
    record_path = DATA_DIR / record_id

    signal, fields = wfdb.rdsamp(str(record_path))
    fs = fields["fs"]

    scales = (
        pywt.central_frequency(WAVELET)
        * fs / FREQUENCIES
    )

    scalograms = []

    for lead in range(signal.shape[1]):
        coefficients, _ = pywt.cwt(
            signal[:, lead],
            scales,
            WAVELET,
            sampling_period=1.0 / fs
        )

        scalograms.append(np.abs(coefficients).T)

    tensor = np.stack(scalograms, axis=2).astype(np.float32)

    if tensor.shape[1:] != (100, 12):
        raise ValueError(
            f"Unexpected tensor shape for {record_id}: {tensor.shape}"
        )

    return tensor


# --------------------------------------------------
# EXTRACT AND ALIGN TUCKER CORES
# --------------------------------------------------

def padded_core(tensor, rank):
    core, _ = tucker(
        tensor,
        rank=list(rank),
        init="svd",
        tol=1e-5
    )

    result = np.zeros(MAX_RANK, dtype=np.float32)

    result[
        :core.shape[0],
        :core.shape[1],
        :core.shape[2]
    ] = core.astype(np.float32)

    return result.ravel()


fixed_features = []
adaptive_features = []
feature_rows = []

start_all = time.time()

for i, row in data.iterrows():
    record_id = row["record_id"]
    adaptive_rank = row["adaptive_rank"]

    print(
        f"\nProcessing {i + 1}/{len(data)}: {record_id} "
        f"| adaptive rank={adaptive_rank}"
    )

    tensor = load_cwt_tensor(record_id)

    fixed_vector = padded_core(tensor, FIXED_RANK)

    if adaptive_rank == FIXED_RANK:
        adaptive_vector = fixed_vector.copy()
    else:
        adaptive_vector = padded_core(tensor, adaptive_rank)

    fixed_features.append(fixed_vector)
    adaptive_features.append(adaptive_vector)

    feature_rows.append({
        "record_id": record_id,
        "label": row["label"],
        "adaptive_rank": str(adaptive_rank),
        "fixed_feature_count": len(fixed_vector),
        "adaptive_feature_count": len(adaptive_vector)
    })

    if (i + 1) % 25 == 0:
        print(
            f"Progress: {i + 1}/{len(data)} records; "
            f"elapsed {(time.time() - start_all) / 60:.1f} minutes"
        )


X_fixed = np.asarray(fixed_features, dtype=np.float32)
X_adaptive = np.asarray(adaptive_features, dtype=np.float32)

np.savez_compressed(
    OUTPUT_DIR / "tucker_core_features.npz",
    record_ids=data["record_id"].to_numpy(dtype=str),
    fixed_features=X_fixed,
    adaptive_features=X_adaptive,
    labels=y,
    groups=groups
)

pd.DataFrame(feature_rows).to_csv(
    OUTPUT_DIR / "feature_extraction_summary.csv",
    index=False
)

print("\nFeature extraction completed.")
print("Fixed feature matrix:", X_fixed.shape)
print("Adaptive feature matrix:", X_adaptive.shape)


# --------------------------------------------------
# GROUP-AWARE CROSS-VALIDATION
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

methods = {
    "fixed_rank_core": X_fixed,
    "adaptive_rank_core": X_adaptive
}

oof = {
    name: np.full(len(y), -1, dtype=int)
    for name in methods
}

fold_rows = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X_fixed, y, groups),
    start=1
):
    print(f"\nFold {fold}/{N_SPLITS}")

    # Guard against identical signal groups crossing the split.
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])

    if train_groups.intersection(test_groups):
        raise RuntimeError("Signal-group leakage detected.")

    for method, X in methods.items():
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(X[train_idx], y[train_idx])
        predictions = model.predict(X[test_idx])

        oof[method][test_idx] = predictions

        tn, fp, fn, tp = confusion_matrix(
            y[test_idx],
            predictions,
            labels=[0, 1]
        ).ravel()

        metrics = {
            "fold": fold,
            "method": method,
            "accuracy": accuracy_score(y[test_idx], predictions),
            "precision": precision_score(
                y[test_idx], predictions, zero_division=0
            ),
            "sensitivity": recall_score(
                y[test_idx], predictions, zero_division=0
            ),
            "specificity": tn / max(tn + fp, 1),
            "f1": f1_score(
                y[test_idx], predictions, zero_division=0
            ),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }

        fold_rows.append(metrics)

        print(
            f"  {method}: "
            f"F1={metrics['f1']:.4f}, "
            f"Sensitivity={metrics['sensitivity']:.4f}, "
            f"Specificity={metrics['specificity']:.4f}"
        )


# --------------------------------------------------
# SAVE FOLD AND POOLED RESULTS
# --------------------------------------------------

fold_df = pd.DataFrame(fold_rows)

fold_df.to_csv(
    OUTPUT_DIR / "fold_results.csv",
    index=False
)

summary = fold_df.groupby("method")[
    ["accuracy", "precision", "sensitivity", "specificity", "f1"]
].agg(["mean", "std"])

print("\nFOLD MEAN AND STANDARD DEVIATION")
print("=" * 75)
print(summary.round(4).to_string())

pooled_rows = []

for method, predictions in oof.items():
    if np.any(predictions < 0):
        raise RuntimeError(f"Missing OOF predictions for {method}.")

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    pooled_rows.append({
        "method": method,
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
        "tp": int(tp)
    })

    pd.DataFrame({
        "record_id": data["record_id"],
        "true_label": y,
        "predicted_label": predictions
    }).to_csv(
        OUTPUT_DIR / f"{method}_oof_predictions.csv",
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

print("\nResults saved to:")
print(OUTPUT_DIR)
