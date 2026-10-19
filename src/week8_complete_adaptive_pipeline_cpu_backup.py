
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pywt
import wfdb

from scipy.signal import butter, sosfiltfilt
from tensorly.decomposition import tucker

from sklearn.utils.extmath import randomized_svd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "dataset" / "raw" / "set-a"

GROUP_FILE = (
    ROOT / "results" / "week4" / "group_aware_cv"
    / "record_signal_groups.csv"
)

CHAR_FILE = (
    ROOT / "results" / "week5" / "signal_characteristics.csv"
)

LEAD_FILE = (
    ROOT / "results" / "week8" / "lead_quality_features"
    / "lead_quality_features.csv"
)

FIXED_CORE_FILE = (
    ROOT / "results" / "week8" / "true_adaptive_rank_cv"
    / "tucker_core_features.npz"
)

OUTPUT_DIR = ROOT / "results" / "week8" / "complete_adaptive_pipeline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 500
WAVELET = "morl"
FREQUENCY_COUNT = 100

FIXED_RANK = (50, 20, 6)
MAX_RANK = (50, 36, 12)
ENERGY_THRESHOLD = 0.95

N_SPLITS = 5
SEED = 42
N_ESTIMATORS = 100


# ============================================================
# HELPER FUNCTIONS
# ============================================================

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

    labels = values.map(mapping)

    if labels.isna().any():
        raise ValueError(
            f"Unexpected labels: {values[labels.isna()].unique()}"
        )

    return labels.astype(int)


def find_record(record_id):
    candidates = [
        DATA_DIR / record_id,
        DATA_DIR / "set-a" / record_id,
    ]

    for path in candidates:
        if Path(str(path) + ".hea").exists():
            return path

    raise FileNotFoundError(f"ECG record not found: {record_id}")


def high_pass_filter(signal):
    sos = butter(
        4,
        0.5,
        btype="highpass",
        fs=FS,
        output="sos",
    )

    return np.column_stack([
        sosfiltfilt(sos, signal[:, lead])
        for lead in range(signal.shape[1])
    ])


def create_cwt_tensor(signal, maximum_frequency):
    frequencies = np.linspace(
        1,
        maximum_frequency,
        FREQUENCY_COUNT,
    )

    scales = (
        pywt.central_frequency(WAVELET)
        * FS / frequencies
    )

    scalograms = []

    for lead in range(signal.shape[1]):
        coefficients, _ = pywt.cwt(
            signal[:, lead],
            scales,
            WAVELET,
            sampling_period=1.0 / FS,
        )

        scalograms.append(np.abs(coefficients).T)

    return np.stack(scalograms, axis=2).astype(np.float32)


def select_adaptive_rank(tensor):
    """
    Estimate a rank for each tensor mode from the leading
    singular values. Each mode has a maximum rank limit.

    The 95% target applies to the energy among the leading
    singular components examined, up to the configured cap.
    """
    ranks = []

    for mode, maximum_rank in enumerate(MAX_RANK):
        unfolded = np.moveaxis(tensor, mode, 0).reshape(
            tensor.shape[mode], -1
        )

        n_components = min(
            maximum_rank,
            min(unfolded.shape),
        )

        _, singular_values, _ = randomized_svd(
    unfolded,
    n_components=n_components,
    n_iter=3,
    random_state=SEED,
)

        energy = singular_values ** 2
        total_energy = energy.sum()

        if total_energy <= 0:
            rank = 1
        else:
            cumulative = np.cumsum(energy) / total_energy
            rank = int(
                np.searchsorted(
                    cumulative,
                    ENERGY_THRESHOLD,
                ) + 1
            )

        ranks.append(max(1, min(rank, maximum_rank)))

    return tuple(ranks)


def extract_adaptive_core(tensor, rank):
    core, _ = tucker(
        tensor,
        rank=list(rank),
        init="svd",
        tol=1e-5,
    )

    padded = np.zeros(MAX_RANK, dtype=np.float32)

    padded[
        :core.shape[0],
        :core.shape[1],
        :core.shape[2],
    ] = np.asarray(core, dtype=np.float32)

    return padded.ravel()


def make_model():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def calculate_metrics(y_true, predictions):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
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
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# LOAD AND ALIGN ALL DATA
# ============================================================

print("\nCOMPLETE ADAPTIVE ECG QUALITY PIPELINE")
print("=" * 70)

groups_df = pd.read_csv(GROUP_FILE)
chars_df = pd.read_csv(CHAR_FILE)
leads_df = pd.read_csv(LEAD_FILE)

saved = np.load(FIXED_CORE_FILE, allow_pickle=True)

fixed_ids = saved["record_ids"].astype(str)
fixed_X = saved["fixed_features"].astype(np.float32)

fixed_lookup = pd.DataFrame({
    "record_id": fixed_ids,
    "fixed_index": np.arange(len(fixed_ids)),
})

rank_df = pd.read_csv(
    ROOT / "results" / "week2" / "rank_budget_all_records"
    / "multi_record_summary.csv"
)

for frame in [groups_df, chars_df, leads_df, rank_df]:
    frame["record_id"] = frame["record_id"].astype(str).str.strip()

for frame, name in [
    (groups_df, "group mapping"),
    (chars_df, "signal characteristics"),
    (leads_df, "lead features"),
    (rank_df, "rank search"),
]:
    if frame["record_id"].duplicated().any():
        raise ValueError(f"Duplicate IDs in {name}.")

data = (
    rank_df[["record_id", "label"]]
    .merge(
        groups_df[["record_id", "label", "signal_hash"]],
        on="record_id",
        suffixes=("_rank", "_group"),
        validate="one_to_one",
    )
    .merge(
        chars_df[
            [
                "record_id",
                "mean_baseline_variation",
                "mean_high_frequency_ratio",
            ]
        ],
        on="record_id",
        validate="one_to_one",
    )
    .merge(
        leads_df.drop(columns=["label"], errors="ignore"),
        on="record_id",
        validate="one_to_one",
    )
    .merge(
        fixed_lookup,
        on="record_id",
        validate="one_to_one",
    )
)

if not (
    data["label_rank"].astype(str).str.strip().to_numpy()
    == data["label_group"].astype(str).str.strip().to_numpy()
).all():
    raise ValueError("Labels disagree between rank and group files.")

data["label"] = convert_labels(data["label_rank"]).to_numpy()
data = data.sort_values("record_id").reset_index(drop=True)

record_ids = data["record_id"].to_numpy()
y = data["label"].to_numpy(dtype=int)
group_ids = data["signal_hash"].astype(str).to_numpy()

if pd.isna(data["signal_hash"]).any():
    raise ValueError("Missing signal group identifiers.")

if not np.isfinite(
    data[
        ["mean_baseline_variation", "mean_high_frequency_ratio"]
    ].to_numpy(dtype=float)
).all():
    raise ValueError("Missing or invalid signal characteristics.")

fixed_indices = data["fixed_index"].to_numpy(dtype=int)
X_fixed = fixed_X[fixed_indices]

lead_columns = [
    column for column in leads_df.columns
    if column not in ["record_id", "label"]
]

X_lead = (
    data[lead_columns]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
)

# Imputation is fitted separately inside each outer training fold.
print("Aligned records:", len(data))
print("Fixed core shape:", X_fixed.shape)
print("Lead-quality features:", len(lead_columns))
print("Acceptable:", int((y == 0).sum()))
print("Unacceptable:", int((y == 1).sum()))
print("Unique signal groups:", len(np.unique(group_ids)))


# ============================================================
# CACHE ADAPTIVE FEATURES BY SIGNAL AND CONFIGURATION
# ============================================================

feature_cache = {}


def build_adaptive_features(
    record_id,
    use_highpass,
    maximum_frequency,
    lead_values,
):
    cache_key = (record_id, bool(use_highpass), int(maximum_frequency))

    if cache_key not in feature_cache:
        record_path = find_record(record_id)
        signal, fields = wfdb.rdsamp(str(record_path))

        if int(fields["fs"]) != FS:
            raise ValueError(
                f"Unexpected sampling frequency for {record_id}"
            )

        if signal.shape[1] != 12:
            raise ValueError(
                f"Unexpected number of leads for {record_id}"
            )

        processed = (
            high_pass_filter(signal)
            if use_highpass
            else signal
        )

        tensor = create_cwt_tensor(
            processed,
            maximum_frequency,
        )

        rank = select_adaptive_rank(tensor)
        core_features = extract_adaptive_core(tensor, rank)

        # Rank values are explicitly included in the combined features.
        rank_features = np.asarray(rank, dtype=np.float32)

        feature_cache[cache_key] = (
            core_features,
            rank_features,
            rank,
        )

    core_features, rank_features, rank = feature_cache[cache_key]

    combined = np.concatenate([
        core_features,
        rank_features,
        np.asarray(lead_values, dtype=np.float32),
    ])

    return combined, rank


# ============================================================
# GROUP-AWARE CROSS-VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

oof = {
    "fixed_core": np.full(len(data), -1, dtype=int),
    "complete_adaptive": np.full(len(data), -1, dtype=int),
}

fold_results = []
prediction_rows = []
threshold_rows = []
start_time = time.time()

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X_fixed, y, group_ids),
    start=1,
):
    print(f"\nFOLD {fold}/{N_SPLITS}")
    print("-" * 60)

    if set(group_ids[train_idx]).intersection(group_ids[test_idx]):
        raise RuntimeError("Identical signal groups cross the fold boundary.")

    # Thresholds are based on training-fold signal characteristics only.
    baseline_threshold = float(
        np.quantile(
            data.iloc[train_idx]["mean_baseline_variation"],
            0.80,
        )
    )

    frequency_threshold = float(
        np.quantile(
            data.iloc[train_idx]["mean_high_frequency_ratio"],
            0.80,
        )
    )

    print(f"High-pass threshold: {baseline_threshold:.6f}")
    print(f"50 Hz CWT threshold: {frequency_threshold:.6f}")

    # Impute lead features using training data only.
    train_leads = X_lead.iloc[train_idx]
    test_leads = X_lead.iloc[test_idx]

    medians = train_leads.median().fillna(0.0)

    train_leads = train_leads.fillna(medians).to_numpy(dtype=np.float32)
    test_leads = test_leads.fillna(medians).to_numpy(dtype=np.float32)

    adaptive_train = []
    adaptive_test = []
    selected_ranks = {}

    for local, index in enumerate(train_idx):
        row = data.iloc[index]

        use_highpass = (
            row["mean_baseline_variation"] > baseline_threshold
        )

        maximum_frequency = (
            50
            if row["mean_high_frequency_ratio"] > frequency_threshold
            else 40
        )

        vector, rank = build_adaptive_features(
            row["record_id"],
            use_highpass,
            maximum_frequency,
            train_leads[local],
        )

        adaptive_train.append(vector)
        selected_ranks[index] = rank

        if (local + 1) % 25 == 0:
            print(
                f"Training features: {local + 1}/{len(train_idx)}"
            )

    for local, index in enumerate(test_idx):
        row = data.iloc[index]

        use_highpass = (
            row["mean_baseline_variation"] > baseline_threshold
        )

        maximum_frequency = (
            50
            if row["mean_high_frequency_ratio"] > frequency_threshold
            else 40
        )

        vector, rank = build_adaptive_features(
            row["record_id"],
            use_highpass,
            maximum_frequency,
            test_leads[local],
        )

        adaptive_test.append(vector)
        selected_ranks[index] = rank

        prediction_rows.append({
            "record_id": row["record_id"],
            "fold": fold,
            "true_label": int(y[index]),
            "use_highpass": bool(use_highpass),
            "cwt_max_frequency": maximum_frequency,
            "rank_time": rank[0],
            "rank_frequency": rank[1],
            "rank_leads": rank[2],
            "baseline_threshold": baseline_threshold,
            "frequency_threshold": frequency_threshold,
        })

        if (local + 1) % 25 == 0:
            print(
                f"Testing features: {local + 1}/{len(test_idx)}"
            )

    X_adaptive_train = np.asarray(adaptive_train, dtype=np.float32)
    X_adaptive_test = np.asarray(adaptive_test, dtype=np.float32)

    # Baseline: fixed 40 Hz, fixed Tucker core features.
    fixed_model = make_model()
    fixed_model.fit(X_fixed[train_idx], y[train_idx])
    fixed_pred = fixed_model.predict(X_fixed[test_idx])

    # Complete adaptive pipeline.
    adaptive_model = make_model()
    adaptive_model.fit(X_adaptive_train, y[train_idx])
    adaptive_pred = adaptive_model.predict(X_adaptive_test)

    oof["fixed_core"][test_idx] = fixed_pred
    oof["complete_adaptive"][test_idx] = adaptive_pred

    for method, predictions in [
        ("fixed_core", fixed_pred),
        ("complete_adaptive", adaptive_pred),
    ]:
        scores = calculate_metrics(y[test_idx], predictions)

        fold_results.append({
            "fold": fold,
            "method": method,
            **scores,
        })

        print(
            f"{method}: F1={scores['f1']:.4f}, "
            f"Sensitivity={scores['sensitivity']:.4f}, "
            f"Specificity={scores['specificity']:.4f}"
        )

    threshold_rows.append({
        "fold": fold,
        "baseline_threshold": baseline_threshold,
        "frequency_threshold": frequency_threshold,
        "train_records": len(train_idx),
        "test_records": len(test_idx),
    })

    print(
        f"Elapsed time: {(time.time() - start_time) / 60:.1f} minutes"
    )


# ============================================================
# SAVE FOLD AND POOLED RESULTS
# ============================================================

fold_df = pd.DataFrame(fold_results)
fold_df.to_csv(OUTPUT_DIR / "fold_results.csv", index=False)

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(OUTPUT_DIR / "fold_thresholds.csv", index=False)

pd.DataFrame(prediction_rows).to_csv(
    OUTPUT_DIR / "adaptive_record_configurations.csv",
    index=False,
)

summary_rows = []

for method, predictions in oof.items():
    if np.any(predictions < 0):
        raise RuntimeError(f"Missing predictions for {method}")

    scores = calculate_metrics(y, predictions)
    summary_rows.append({
        "method": method,
        **scores,
    })

    pd.DataFrame({
        "record_id": record_ids,
        "true_label": y,
        "predicted_label": predictions,
    }).to_csv(
        OUTPUT_DIR / f"{method}_oof_predictions.csv",
        index=False,
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / "pooled_oof_results.csv", index=False)

fold_summary = fold_df.groupby("method")[
    ["accuracy", "precision", "sensitivity", "specificity", "f1"]
].agg(["mean", "std"])

print("\nFOLD MEAN ± STANDARD DEVIATION")
print("=" * 75)
print(fold_summary.round(4).to_string())

print("\nPOOLED OUT-OF-FOLD RESULTS")
print("=" * 75)
print(summary_df.round(4).to_string(index=False))

print("\nResults saved to:")
print(OUTPUT_DIR)
print(f"Total elapsed time: {(time.time() - start_time) / 60:.1f} minutes")
