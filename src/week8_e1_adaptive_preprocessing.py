
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pywt
import wfdb
import torch
import tensorly as tl

from scipy.signal import butter, sosfiltfilt
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
NPZ_FILE = ROOT / "results" / "week8" / "true_adaptive_rank_cv" / "tucker_core_features.npz"
OUT = ROOT / "results" / "week8" / "E1_adaptive_preprocessing"
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
    raise FileNotFoundError(f"Cannot find {record_id}")

def high_pass(signal):
    sos = butter(4, 0.5, btype="highpass", fs=FS, output="sos")
    return np.column_stack([
        sosfiltfilt(sos, signal[:, j])
        for j in range(signal.shape[1])
    ])

def make_features(record_id, use_highpass):
    cache_file = CACHE / f"{record_id}_hp{int(use_highpass)}.npz"

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

    if use_highpass:
        signal = high_pass(signal)

    frequencies = np.linspace(1, 40, FREQUENCY_COUNT)
    scales = pywt.central_frequency(WAVELET) * FS / frequencies
    lead_scalograms = []

    for lead in range(signal.shape[1]):
        coeff, _ = pywt.cwt(
            signal[:, lead], scales, WAVELET,
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

print("E1: ADAPTIVE PREPROCESSING ONLY")
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

saved = np.load(NPZ_FILE, allow_pickle=True)
fixed_ids = saved["record_ids"].astype(str)
X_fixed_raw = saved["fixed_features"].astype(np.float32)

groups_df = pd.read_csv(GROUP_FILE)
chars_df = pd.read_csv(CHAR_FILE)

for df in [groups_df, chars_df]:
    df["record_id"] = df["record_id"].astype(str).str.strip()

lookup = pd.DataFrame({
    "record_id": fixed_ids,
    "fixed_index": np.arange(len(fixed_ids))
})

data = (
    groups_df[["record_id", "label", "signal_hash"]]
    .merge(
        chars_df[["record_id", "mean_baseline_variation"]],
        on="record_id", validate="one_to_one"
    )
    .merge(lookup, on="record_id", validate="one_to_one")
    .sort_values("record_id")
    .reset_index(drop=True)
)

labels = data["label"].astype(str).str.strip().str.lower()
label_map = {
    "acceptable": 0, "accept": 0, "0": 0,
    "unacceptable": 1, "unaccept": 1, "1": 1
}
y = labels.map(label_map).to_numpy(dtype=int)

if np.isnan(y).any():
    raise ValueError("Unexpected labels.")

record_ids = data["record_id"].to_numpy()
groups = data["signal_hash"].astype(str).to_numpy()
baseline_variation = data["mean_baseline_variation"].to_numpy(dtype=float)
X_fixed = X_fixed_raw[data["fixed_index"].to_numpy(dtype=int)]

if not np.isfinite(baseline_variation).all():
    raise ValueError("Invalid baseline-variation values.")

print("Records:", len(data))
print("Feature shape:", X_fixed.shape)
print("Unique signal groups:", len(np.unique(groups)))

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS, shuffle=True, random_state=SEED
)

oof = np.full(len(data), -1, dtype=int)
fold_rows = []
start = time.time()

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X_fixed, y, groups), start=1
):
    print(f"\nFOLD {fold}/{N_SPLITS}")

    if set(groups[train_idx]).intersection(groups[test_idx]):
        raise RuntimeError("Group leakage detected.")

    # Learn the preprocessing rule using training records only.
    threshold = float(np.quantile(baseline_variation[train_idx], 0.80))
    print(f"High-pass threshold: {threshold:.6f}")

    X_train, X_test = [], []

    for j, idx in enumerate(train_idx, start=1):
        use_hp = baseline_variation[idx] > threshold
        X_train.append(make_features(record_ids[idx], use_hp))
        if j % 50 == 0:
            print(f"Training features: {j}/{len(train_idx)}")

    for j, idx in enumerate(test_idx, start=1):
        use_hp = baseline_variation[idx] > threshold
        X_test.append(make_features(record_ids[idx], use_hp))
        if j % 50 == 0:
            print(f"Testing features: {j}/{len(test_idx)}")

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(np.asarray(X_train, dtype=np.float32), y[train_idx])
    pred = model.predict(np.asarray(X_test, dtype=np.float32))
    oof[test_idx] = pred

    tn, fp, fn, tp = confusion_matrix(
        y[test_idx], pred, labels=[0, 1]
    ).ravel()

    row = {
        "fold": fold,
        "accuracy": accuracy_score(y[test_idx], pred),
        "precision": precision_score(y[test_idx], pred, zero_division=0),
        "sensitivity": recall_score(y[test_idx], pred, zero_division=0),
        "specificity": tn / max(tn + fp, 1),
        "f1": f1_score(y[test_idx], pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "highpass_threshold": threshold
    }
    fold_rows.append(row)

    pd.DataFrame(fold_rows).to_csv(
        OUT / "fold_results_partial.csv", index=False
    )
    pd.DataFrame({
        "record_id": record_ids[oof >= 0],
        "true_label": y[oof >= 0],
        "predicted_label": oof[oof >= 0]
    }).to_csv(OUT / "oof_predictions_partial.csv", index=False)

    print(
        f"E1 F1={row['f1']:.4f}, "
        f"Sensitivity={row['sensitivity']:.4f}, "
        f"Specificity={row['specificity']:.4f}"
    )
    print(f"Elapsed: {(time.time() - start) / 60:.1f} minutes")

if np.any(oof < 0):
    raise RuntimeError("Missing out-of-fold predictions.")

tn, fp, fn, tp = confusion_matrix(y, oof, labels=[0, 1]).ravel()

summary = pd.DataFrame([{
    "method": "E1_adaptive_preprocessing",
    "accuracy": accuracy_score(y, oof),
    "precision": precision_score(y, oof, zero_division=0),
    "sensitivity": recall_score(y, oof, zero_division=0),
    "specificity": tn / max(tn + fp, 1),
    "f1": f1_score(y, oof, zero_division=0),
    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
}])

summary.to_csv(OUT / "pooled_oof_results.csv", index=False)
pd.DataFrame(fold_rows).to_csv(OUT / "fold_results.csv", index=False)
pd.DataFrame({
    "record_id": record_ids,
    "true_label": y,
    "predicted_label": oof
}).to_csv(OUT / "oof_predictions.csv", index=False)

print("\nE1 POOLED RESULTS")
print(summary.round(4).to_string(index=False))
print("Saved to:", OUT)
print(f"Total time: {(time.time() - start) / 60:.1f} minutes")
