
from pathlib import Path
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
GROUP_FILE = ROOT / "results" / "week4" / "group_aware_cv" / "record_signal_groups.csv"
CHAR_FILE = ROOT / "results" / "week5" / "signal_characteristics.csv"

OUT = ROOT / "results" / "week8" / "E2_adaptive_cwt"
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


def make_features(record_id, max_frequency):
    cache_file = CACHE / f"{record_id}_cwt{max_frequency}.npz"

    if cache_file.exists():
        try:
            with np.load(cache_file) as saved:
                features = saved["features"].astype(np.float32)
            if features.size == int(np.prod(PAD_RANK)):
                return features
        except Exception:
            pass

    path = find_record(record_id)
    signal, fields = wfdb.rdsamp(str(path))

    if int(fields["fs"]) != FS or signal.shape[1] != 12:
        raise ValueError(f"Unexpected signal dimensions: {record_id}")

    # No high-pass filtering in E2.
    # Only the CWT frequency range changes.
    frequencies = np.linspace(1, max_frequency, FREQUENCY_COUNT)
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
        core, _ = tucker(
            tensor_gpu,
            rank=list(FIXED_RANK),
            init="svd",
            tol=1e-5
        )
        core = core.detach().cpu().numpy().astype(np.float32)

    padded = np.zeros(PAD_RANK, dtype=np.float32)
    padded[:core.shape[0], :core.shape[1], :core.shape[2]] = core
    features = padded.ravel()

    tmp = cache_file.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, features=features)
    tmp.replace(cache_file)

    del signal, tensor, tensor_gpu, core, padded

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return features


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


print("E2: ADAPTIVE CWT")
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

groups_df = pd.read_csv(GROUP_FILE)
chars_df = pd.read_csv(CHAR_FILE)

for df in [groups_df, chars_df]:
    df["record_id"] = df["record_id"].astype(str).str.strip()

data = (
    groups_df[["record_id", "label", "signal_hash"]]
    .merge(
        chars_df[["record_id", "mean_baseline_variation"]],
        on="record_id",
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
baseline_variation = data["mean_baseline_variation"].to_numpy(dtype=float)

if not np.isfinite(baseline_variation).all():
    raise ValueError("Invalid baseline-variation values.")

print("Records:", len(data))
print("Acceptable:", int(np.sum(y == 0)))
print("Unacceptable:", int(np.sum(y == 1)))
print("Unique signal groups:", len(np.unique(groups)))

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)

# Matched experiment:
# E0 uses fixed 40 Hz CWT.
# E2 selects 40 Hz or 50 Hz using training-fold data only.
oof_e0 = np.full(len(data), -1, dtype=int)
oof_e2 = np.full(len(data), -1, dtype=int)

fold_rows = []
start = time.time()

for fold, (train_idx, test_idx) in enumerate(
    cv.split(np.zeros(len(y)), y, groups), start=1
):
    print(f"\nFOLD {fold}/{N_SPLITS}")

    if set(groups[train_idx]).intersection(groups[test_idx]):
        raise RuntimeError("Group leakage detected.")

    # Learn the CWT-selection threshold from training records only.
    threshold = float(
        np.quantile(baseline_variation[train_idx], 0.80)
    )

    print(f"CWT selection threshold: {threshold:.6f}")

    X_train_e0, X_test_e0 = [], []
    X_train_e2, X_test_e2 = [], []

    for j, idx in enumerate(train_idx, start=1):
        rid = record_ids[idx]

        # E0: fixed 40 Hz
        X_train_e0.append(make_features(rid, 40))

        # E2: adaptively select 40 or 50 Hz
        selected_frequency = (
            50 if baseline_variation[idx] > threshold else 40
        )
        X_train_e2.append(make_features(rid, selected_frequency))

        if j % 50 == 0:
            print(f"Training features: {j}/{len(train_idx)}")

    for j, idx in enumerate(test_idx, start=1):
        rid = record_ids[idx]

        # Fixed baseline representation
        X_test_e0.append(make_features(rid, 40))

        # Apply the threshold learned from this training fold
        selected_frequency = (
            50 if baseline_variation[idx] > threshold else 40
        )
        X_test_e2.append(make_features(rid, selected_frequency))

        if j % 50 == 0:
            print(f"Testing features: {j}/{len(test_idx)}")

    X_train_e0 = np.asarray(X_train_e0, dtype=np.float32)
    X_test_e0 = np.asarray(X_test_e0, dtype=np.float32)
    X_train_e2 = np.asarray(X_train_e2, dtype=np.float32)
    X_test_e2 = np.asarray(X_test_e2, dtype=np.float32)

    model_e0 = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    )

    model_e2 = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    )

    model_e0.fit(X_train_e0, y[train_idx])
    model_e2.fit(X_train_e2, y[train_idx])

    pred_e0 = model_e0.predict(X_test_e0)
    pred_e2 = model_e2.predict(X_test_e2)

    oof_e0[test_idx] = pred_e0
    oof_e2[test_idx] = pred_e2

    result_e0 = evaluate(y[test_idx], pred_e0)
    result_e2 = evaluate(y[test_idx], pred_e2)

    fold_rows.extend([
        {
            "fold": fold,
            "method": "E0_fixed_40Hz",
            **result_e0
        },
        {
            "fold": fold,
            "method": "E2_adaptive_40_50Hz",
            **result_e2,
            "cwt_threshold": threshold
        }
    ])

    pd.DataFrame(fold_rows).to_csv(
        OUT / "fold_results_partial.csv", index=False
    )

    valid = oof_e0 >= 0
    pd.DataFrame({
        "record_id": record_ids[valid],
        "true_label": y[valid],
        "e0_prediction": oof_e0[valid],
        "e2_prediction": oof_e2[valid]
    }).to_csv(OUT / "oof_predictions_partial.csv", index=False)

    print(
        f"E0 F1={result_e0['f1']:.4f}, "
        f"Sensitivity={result_e0['sensitivity']:.4f}"
    )
    print(
        f"E2 F1={result_e2['f1']:.4f}, "
        f"Sensitivity={result_e2['sensitivity']:.4f}, "
        f"Specificity={result_e2['specificity']:.4f}"
    )
    print(f"Elapsed: {(time.time() - start) / 60:.1f} minutes")


if np.any(oof_e0 < 0) or np.any(oof_e2 < 0):
    raise RuntimeError("Missing out-of-fold predictions.")

summary_rows = []

for method, predictions in [
    ("E0_fixed_40Hz", oof_e0),
    ("E2_adaptive_40_50Hz", oof_e2)
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
    "e0_prediction": oof_e0,
    "e2_prediction": oof_e2
}).to_csv(OUT / "oof_predictions.csv", index=False)

print("\nE0 vs E2 — POOLED RESULTS")
print(summary.round(4).to_string(index=False))

print("\nSaved to:", OUT)
print(f"Total time: {(time.time() - start) / 60:.1f} minutes")
