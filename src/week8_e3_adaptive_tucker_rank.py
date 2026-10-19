
from pathlib import Path
import ast
import time

import numpy as np
import pandas as pd
import pywt
import wfdb
import torch
import tensorly as tl

from tensorly.decomposition import tucker
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "raw" / "set-a"

GROUP_FILE = (
    ROOT / "results" / "week4" / "group_aware_cv"
    / "record_signal_groups.csv"
)
RANK_FILE = (
    ROOT / "results" / "week2" / "rank_budget_all_records"
    / "multi_record_summary.csv"
)

OUT = ROOT / "results" / "week8" / "E3_adaptive_tucker_rank"
CACHE = OUT / "feature_cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

FS = 500
WAVELET = "morl"
FREQUENCY_COUNT = 100

FIXED_RANK = (50, 20, 6)
PAD_RANK = (50, 36, 12)

SEED = 42
N_SPLITS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tl.set_backend("pytorch")


def find_record(record_id):
    for path in [DATA_DIR / record_id, DATA_DIR / "set-a" / record_id]:
        if Path(str(path) + ".hea").exists():
            return path

    raise FileNotFoundError(f"Cannot find record: {record_id}")


def parse_rank(value):
    if isinstance(value, (tuple, list, np.ndarray)):
        rank = tuple(int(x) for x in value)
    else:
        rank = tuple(int(x) for x in ast.literal_eval(str(value)))

    if len(rank) != 3:
        raise ValueError(f"Invalid Tucker rank: {value}")

    if any(r < 1 for r in rank):
        raise ValueError(f"Rank values must be positive: {rank}")

    if any(r > cap for r, cap in zip(rank, PAD_RANK)):
        raise ValueError(f"Rank exceeds padded feature dimensions: {rank}")

    return rank


def pad_core(core):
    core = np.asarray(core, dtype=np.float32)

    if any(c > p for c, p in zip(core.shape, PAD_RANK)):
        raise ValueError(f"Core dimensions exceed padding: {core.shape}")

    padded = np.zeros(PAD_RANK, dtype=np.float32)
    padded[:core.shape[0], :core.shape[1], :core.shape[2]] = core

    return padded.ravel()


def make_features(record_id, adaptive_rank):
    cache_file = CACHE / f"{record_id}_fixed_and_adaptive.npz"

    if cache_file.exists():
        try:
            with np.load(cache_file) as saved:
                fixed = saved["fixed"].astype(np.float32)
                adaptive = saved["adaptive"].astype(np.float32)

            expected = int(np.prod(PAD_RANK))

            if fixed.size == expected and adaptive.size == expected:
                if np.isfinite(fixed).all() and np.isfinite(adaptive).all():
                    return fixed, adaptive
        except Exception:
            pass

    path = find_record(record_id)
    signal, fields = wfdb.rdsamp(str(path))

    if int(fields["fs"]) != FS or signal.shape[1] != 12:
        raise ValueError(f"Unexpected signal dimensions: {record_id}")

    # E0 and E3 use the same unfiltered ECG and fixed 1–40 Hz CWT.
    frequencies = np.linspace(1, 40, FREQUENCY_COUNT)
    scales = pywt.central_frequency(WAVELET) * FS / frequencies

    lead_scalograms = []

    for lead in range(signal.shape[1]):
        coeff, _ = pywt.cwt(
            signal[:, lead],
            scales,
            WAVELET,
            sampling_period=1.0 / FS
        )
        lead_scalograms.append(np.abs(coeff).T)

    tensor = np.stack(lead_scalograms, axis=2).astype(np.float32)
    tensor_gpu = torch.as_tensor(tensor, device=DEVICE)

    with torch.no_grad():
        fixed_core, _ = tucker(
            tensor_gpu,
            rank=list(FIXED_RANK),
            init="svd",
            tol=1e-5
        )

        adaptive_core, _ = tucker(
            tensor_gpu,
            rank=list(adaptive_rank),
            init="svd",
            tol=1e-5
        )

        fixed_core = fixed_core.detach().cpu().numpy()
        adaptive_core = adaptive_core.detach().cpu().numpy()

    fixed_features = pad_core(fixed_core)
    adaptive_features = pad_core(adaptive_core)

    if not (
        np.isfinite(fixed_features).all()
        and np.isfinite(adaptive_features).all()
    ):
        raise ValueError(f"Non-finite features for {record_id}")

    tmp = cache_file.with_suffix(".tmp.npz")

    np.savez_compressed(
        tmp,
        fixed=fixed_features,
        adaptive=adaptive_features
    )
    tmp.replace(cache_file)

    del signal, tensor, tensor_gpu, fixed_core, adaptive_core

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return fixed_features, adaptive_features


def evaluate(y_true, predictions):
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "sensitivity": recall_score(
            y_true, predictions, zero_division=0
        ),
        "specificity": tn / max(tn + fp, 1),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


print("E3: ADAPTIVE TUCKER RANK")
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

groups_df = pd.read_csv(GROUP_FILE)
rank_df = pd.read_csv(RANK_FILE)

for df in [groups_df, rank_df]:
    df["record_id"] = df["record_id"].astype(str).str.strip()

if "selected_rank" not in rank_df.columns:
    raise ValueError("Rank summary has no 'selected_rank' column.")

if rank_df["record_id"].duplicated().any():
    raise ValueError("Duplicate record IDs in rank summary.")

rank_df["adaptive_rank"] = rank_df["selected_rank"].apply(parse_rank)

# Use only records with an existing, verified rank selection.
data = (
    groups_df[["record_id", "label", "signal_hash"]]
    .merge(
        rank_df[["record_id", "adaptive_rank"]],
        on="record_id",
        how="inner",
        validate="one_to_one"
    )
    .sort_values("record_id")
    .reset_index(drop=True)
)

labels = data["label"].astype(str).str.strip().str.lower()

label_map = {
    "acceptable": 0,
    "accept": 0,
    "0": 0,
    "unacceptable": 1,
    "unaccept": 1,
    "1": 1
}

y = labels.map(label_map).to_numpy(dtype=int)

if np.isnan(y).any():
    raise ValueError("Unexpected labels.")

record_ids = data["record_id"].to_numpy()
groups = data["signal_hash"].astype(str).to_numpy()
adaptive_ranks = data["adaptive_rank"].tolist()

print("Records with rank selections:", len(data))
print("Records omitted due to missing rank selection:",
      len(groups_df) - len(data))
print("Acceptable:", int(np.sum(y == 0)))
print("Unacceptable:", int(np.sum(y == 1)))
print("Unique signal groups:", len(np.unique(groups)))
print("Fixed Tucker rank:", FIXED_RANK)
print("Adaptive rank examples:", adaptive_ranks[:5])

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)

oof_fixed = np.full(len(data), -1, dtype=int)
oof_adaptive = np.full(len(data), -1, dtype=int)

fold_rows = []
start = time.time()

for fold, (train_idx, test_idx) in enumerate(
    cv.split(np.zeros(len(y)), y, groups), start=1
):
    print(f"\nFOLD {fold}/{N_SPLITS}")

    if set(groups[train_idx]).intersection(groups[test_idx]):
        raise RuntimeError("Group leakage detected.")

    X_train_fixed, X_test_fixed = [], []
    X_train_adaptive, X_test_adaptive = [], []

    for j, idx in enumerate(train_idx, start=1):
        fixed, adaptive = make_features(
            record_ids[idx], adaptive_ranks[idx]
        )

        X_train_fixed.append(fixed)
        X_train_adaptive.append(adaptive)

        if j % 50 == 0:
            print(f"Training features: {j}/{len(train_idx)}")

    for j, idx in enumerate(test_idx, start=1):
        fixed, adaptive = make_features(
            record_ids[idx], adaptive_ranks[idx]
        )

        X_test_fixed.append(fixed)
        X_test_adaptive.append(adaptive)

        if j % 50 == 0:
            print(f"Testing features: {j}/{len(test_idx)}")

    X_train_fixed = np.asarray(X_train_fixed, dtype=np.float32)
    X_test_fixed = np.asarray(X_test_fixed, dtype=np.float32)
    X_train_adaptive = np.asarray(X_train_adaptive, dtype=np.float32)
    X_test_adaptive = np.asarray(X_test_adaptive, dtype=np.float32)

    fixed_model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    )

    adaptive_model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    )

    fixed_model.fit(X_train_fixed, y[train_idx])
    adaptive_model.fit(X_train_adaptive, y[train_idx])

    pred_fixed = fixed_model.predict(X_test_fixed)
    pred_adaptive = adaptive_model.predict(X_test_adaptive)

    oof_fixed[test_idx] = pred_fixed
    oof_adaptive[test_idx] = pred_adaptive

    fixed_result = evaluate(y[test_idx], pred_fixed)
    adaptive_result = evaluate(y[test_idx], pred_adaptive)

    fold_rows.extend([
        {
            "fold": fold,
            "method": "E0_fixed_rank",
            **fixed_result
        },
        {
            "fold": fold,
            "method": "E3_adaptive_rank",
            **adaptive_result
        }
    ])

    pd.DataFrame(fold_rows).to_csv(
        OUT / "fold_results_partial.csv", index=False
    )

    valid = oof_fixed >= 0

    pd.DataFrame({
        "record_id": record_ids[valid],
        "true_label": y[valid],
        "fixed_prediction": oof_fixed[valid],
        "adaptive_prediction": oof_adaptive[valid]
    }).to_csv(OUT / "oof_predictions_partial.csv", index=False)

    print(
        f"E0 F1={fixed_result['f1']:.4f}, "
        f"Sensitivity={fixed_result['sensitivity']:.4f}, "
        f"Specificity={fixed_result['specificity']:.4f}"
    )

    print(
        f"E3 F1={adaptive_result['f1']:.4f}, "
        f"Sensitivity={adaptive_result['sensitivity']:.4f}, "
        f"Specificity={adaptive_result['specificity']:.4f}"
    )

    print(f"Elapsed: {(time.time() - start) / 60:.1f} minutes")

if np.any(oof_fixed < 0) or np.any(oof_adaptive < 0):
    raise RuntimeError("Missing out-of-fold predictions.")

summary_rows = []

for method, predictions in [
    ("E0_fixed_rank", oof_fixed),
    ("E3_adaptive_rank", oof_adaptive)
]:
    summary_rows.append({
        "method": method,
        **evaluate(y, predictions)
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "pooled_oof_results.csv", index=False)

pd.DataFrame(fold_rows).to_csv(
    OUT / "fold_results.csv", index=False
)

pd.DataFrame({
    "record_id": record_ids,
    "true_label": y,
    "fixed_prediction": oof_fixed,
    "adaptive_prediction": oof_adaptive,
    "adaptive_rank": [str(r) for r in adaptive_ranks]
}).to_csv(OUT / "oof_predictions.csv", index=False)

pd.DataFrame({
    "record_id": record_ids,
    "adaptive_rank": [str(r) for r in adaptive_ranks]
}).to_csv(OUT / "selected_ranks.csv", index=False)

print("\nE0 vs E3 — POOLED RESULTS")
print(summary.round(4).to_string(index=False))

print("\nSaved to:", OUT)
print(f"Total time: {(time.time() - start) / 60:.1f} minutes")
