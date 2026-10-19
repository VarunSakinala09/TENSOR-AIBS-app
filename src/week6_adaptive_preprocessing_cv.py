import os
import numpy as np
import pandas as pd
import wfdb
import pywt
import tensorly as tl

from scipy.signal import butter, sosfiltfilt

from tensorly.decomposition import tucker

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# PATHS
# =========================================================

DATA_PATH = os.path.join(
    "dataset",
    "raw",
    "set-a"
)

FEATURE_FILE = os.path.join(
    "results",
    "week3",
    "features_full_dataset.csv"
)

BASELINE_FILE = os.path.join(
    "results",
    "week6",
    "preprocessing",
    "highpass_all_records_analysis.csv"
)

OUTPUT_PATH = os.path.join(
    "results",
    "week6",
    "preprocessing"
)


# =========================================================
# PARAMETERS
# =========================================================

FS = 500

LOW_FREQUENCY = 1
HIGH_FREQUENCY = 40
NUMBER_OF_FREQUENCIES = 100

WAVELET = "morl"

HIGH_PASS_CUTOFF = 0.5

ADAPTIVE_PERCENTILE = 80


# =========================================================
# LOAD EXISTING FIXED FEATURES
# =========================================================

print("=" * 70)
print("EXPERIMENT 6: OPTIMIZED ADAPTIVE PREPROCESSING")
print("=" * 70)

print()
print("Loading existing fixed Tucker features...")

features_df = pd.read_csv(
    FEATURE_FILE
)

print(
    "Feature dataset shape:",
    features_df.shape
)


# =========================================================
# LOAD BASELINE CHARACTERISTICS
# =========================================================

print()
print("Loading baseline variation data...")

baseline_df = pd.read_csv(
    BASELINE_FILE
)

print(
    "Baseline dataset shape:",
    baseline_df.shape
)


# =========================================================
# STANDARDIZE RECORD IDs
# =========================================================

features_df["record_id"] = (
    features_df["record_id"]
    .astype(str)
)

baseline_df["record_id"] = (
    baseline_df["record_id"]
    .astype(str)
)
if features_df["record_id"].duplicated().any():
    raise ValueError("Duplicate record IDs in feature dataset.")

if baseline_df["record_id"].duplicated().any():
    raise ValueError("Duplicate record IDs in baseline dataset.")

if not set(features_df["record_id"]).issubset(
    set(baseline_df["record_id"])
):
    raise ValueError(
        "Some feature records are missing baseline characteristics."
    )

# =========================================================
# MERGE INFORMATION
# =========================================================

data_df = pd.merge(
    features_df,
    baseline_df[
        [
            "record_id",
            "raw_baseline_variation"
        ]
    ],
    on="record_id",
    how="inner"
)


print()
print(
    "Merged dataset shape:",
    data_df.shape
)
# Validate the merge before continuing.
if data_df["record_id"].duplicated().any():
    raise ValueError(
        "Duplicate record IDs found after merging datasets."
    )

if len(data_df) != len(features_df):
    raise ValueError(
        "Some feature records were lost during the merge."
    )

if data_df["raw_baseline_variation"].isna().any():
    raise ValueError(
        "Missing baseline-variation values after merging."
    )

if data_df["label"].isna().any():
    raise ValueError(
        "Missing labels found in the merged dataset."
    )

# =========================================================
# LABEL CONVERSION
# =========================================================

data_df["label"] = (
    data_df["label"]
    .astype(str)
    .str.strip()
)


label_mapping = {
    "Acceptable": 0,
    "Unacceptable": 1,
    "0": 0,
    "1": 1
}


data_df["label"] = (
    data_df["label"]
    .map(label_mapping)
)


# Remove any unsupported labels
data_df = data_df.dropna(
    subset=["label"]
)


data_df["label"] = (
    data_df["label"]
    .astype(int)
)


print()
print("Label distribution:")

print(
    data_df["label"]
    .value_counts()
    .sort_index()
)


# =========================================================
# IDENTIFY FEATURE COLUMNS
# =========================================================

feature_columns = [
    column
    for column in features_df.columns
    if column.startswith("feature_")
]

if not feature_columns:
    raise ValueError("No Tucker feature columns found.")

if not all(
    pd.api.types.is_numeric_dtype(features_df[column])
    for column in feature_columns
):
    raise ValueError("Some Tucker feature columns are not numeric.")

if features_df[feature_columns].isna().any().any():
    raise ValueError("Missing values in fixed Tucker features.")


# =========================================================
# FIXED FEATURE LOOKUP
# =========================================================

fixed_feature_lookup = (
    data_df
    .set_index("record_id")
    [feature_columns]
)


label_lookup = (
    data_df
    .set_index("record_id")
    ["label"]
)


baseline_lookup = (
    data_df
    .set_index("record_id")
    ["raw_baseline_variation"]
)


record_ids = data_df[
    "record_id"
].values

labels = data_df[
    "label"
].values


# =========================================================
# HIGH-PASS FILTER
# =========================================================

def high_pass_filter(signal):

    normalized_cutoff = (
        HIGH_PASS_CUTOFF /
        (FS / 2)
    )

    sos = butter(
        4,
        normalized_cutoff,
        btype="highpass",
        output="sos"
    )

    filtered_signal = np.zeros_like(
        signal,
        dtype=float
    )

    for lead in range(
        signal.shape[1]
    ):

        filtered_signal[:, lead] = (
            sosfiltfilt(
                sos,
                signal[:, lead]
            )
        )

    return filtered_signal


# =========================================================
# CREATE CWT TENSOR
# =========================================================

def create_cwt_tensor(signal):

    frequencies = np.linspace(
        LOW_FREQUENCY,
        HIGH_FREQUENCY,
        NUMBER_OF_FREQUENCIES
    )

    central_frequency = (
        pywt.central_frequency(
            WAVELET
        )
    )

    scales = (
        central_frequency *
        FS /
        frequencies
    )

    scalograms = []

    for lead in range(
        signal.shape[1]
    ):

        coefficients, _ = pywt.cwt(
            signal[:, lead],
            scales,
            WAVELET,
            sampling_period=1 / FS
        )

        power = np.abs(
            coefficients
        )

        scalograms.append(
            power
        )

    scalograms = np.array(
        scalograms
    )

    tensor = np.transpose(
        scalograms,
        (2, 1, 0)
    )

    return tensor


# =========================================================
# DYNAMIC TUCKER RANK
# =========================================================

def calculate_tucker_rank(
    tensor_shape
):

    time_size = tensor_shape[0]
    frequency_size = tensor_shape[1]
    lead_size = tensor_shape[2]

    time_rank = max(
        1,
        int(round(
            time_size * 0.01
        ))
    )

    frequency_rank = max(
        1,
        int(round(
            frequency_size * 0.20
        ))
    )

    lead_rank = max(
        1,
        int(round(
            lead_size * 0.50
        ))
    )

    return [
        min(time_rank, time_size),
        min(frequency_rank, frequency_size),
        min(lead_rank, lead_size)
    ]


# =========================================================
# CREATE ADAPTIVE FEATURES
# =========================================================

def create_adaptive_features(
    record_id
):

    record_path = os.path.join(
        DATA_PATH,
        record_id
    )

    record = wfdb.rdrecord(
        record_path
    )

    signal = record.p_signal

    # -----------------------------------------
    # APPLY HIGH-PASS
    # -----------------------------------------

    processed_signal = (
        high_pass_filter(
            signal
        )
    )

    # -----------------------------------------
    # CWT
    # -----------------------------------------

    tensor = create_cwt_tensor(
        processed_signal
    )

    # -----------------------------------------
    # TUCKER
    # -----------------------------------------

    rank = calculate_tucker_rank(
        tensor.shape
    )

    core, factors = tucker(
        tl.tensor(tensor),
        rank=rank,
        init="svd"
    )

    features = np.asarray(
        core
    ).flatten()

    return features


# =========================================================
# ADAPTIVE FEATURE CACHE
# =========================================================

adaptive_feature_cache = {}


def get_adaptive_features(
    record_id
):

    if record_id not in adaptive_feature_cache:

        adaptive_feature_cache[
            record_id
        ] = create_adaptive_features(
            record_id
        )

    return adaptive_feature_cache[
        record_id
    ]


# =========================================================
# CROSS VALIDATION
# =========================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


fixed_results = []
adaptive_results = []


# =========================================================
# 5-FOLD CV
# =========================================================

for fold, (
    train_index,
    test_index
) in enumerate(
    cv.split(
        record_ids,
        labels
    ),
    start=1
):

    print()
    print("=" * 70)
    print(
        f"FOLD {fold}"
    )
    print("=" * 70)


    train_ids = record_ids[
        train_index
    ]

    test_ids = record_ids[
        test_index
    ]

    y_train = labels[
        train_index
    ]

    y_test = labels[
        test_index
    ]


    # =====================================================
    # LEAKAGE-SAFE THRESHOLD
    # =====================================================

    train_baseline = np.array(
        [
            baseline_lookup[
                record_id
            ]
            for record_id in train_ids
        ]
    )


    threshold = np.percentile(
        train_baseline,
        ADAPTIVE_PERCENTILE
    )


    print()
    print(
        "Training-fold threshold:",
        threshold
    )


    # =====================================================
    # DETERMINE WHICH RECORDS NEED ADAPTIVE PROCESSING
    # =====================================================

    adaptive_train_ids = [
        record_id
        for record_id in train_ids
        if baseline_lookup[
            record_id
        ] > threshold
    ]


    adaptive_test_ids = [
        record_id
        for record_id in test_ids
        if baseline_lookup[
            record_id
        ] > threshold
    ]


    print(
        "Training ECGs requiring high-pass:",
        len(adaptive_train_ids),
        "/",
        len(train_ids)
    )

    print(
        "Testing ECGs requiring high-pass:",
        len(adaptive_test_ids),
        "/",
        len(test_ids)
    )


    # =====================================================
    # GENERATE ADAPTIVE FEATURES
    # =====================================================

    current_records = (
        adaptive_train_ids +
        adaptive_test_ids
    )

    new_records = [
        record_id
        for record_id in current_records
        if record_id not in adaptive_feature_cache
    ]


    print()
    print(
        "New adaptive ECGs to process:",
        len(new_records)
    )


    for count, record_id in enumerate(
        new_records,
        start=1
    ):

        get_adaptive_features(
            record_id
        )

        if (
            count == 1
            or count % 10 == 0
            or count == len(new_records)
        ):

            print(
                f"Adaptive features: "
                f"{count}/{len(new_records)}"
            )


    # =====================================================
    # CREATE FIXED TRAIN/TEST FEATURES
    # =====================================================

    X_fixed_train = np.array(
        [
            fixed_feature_lookup.loc[
                record_id
            ].values
            for record_id in train_ids
        ]
    )

    X_fixed_test = np.array(
        [
            fixed_feature_lookup.loc[
                record_id
            ].values
            for record_id in test_ids
        ]
    )


    # =====================================================
    # CREATE ADAPTIVE TRAIN FEATURES
    # =====================================================

    adaptive_train_features = []

    for record_id in train_ids:

        if baseline_lookup[
            record_id
        ] > threshold:

            features = (
                adaptive_feature_cache[
                    record_id
                ]
            )

        else:

            features = (
                fixed_feature_lookup.loc[
                    record_id
                ].values
            )

        adaptive_train_features.append(
            features
        )


    # =====================================================
    # CREATE ADAPTIVE TEST FEATURES
    # =====================================================

    adaptive_test_features = []

    for record_id in test_ids:

        if baseline_lookup[
            record_id
        ] > threshold:

            features = (
                adaptive_feature_cache[
                    record_id
                ]
            )

        else:

            features = (
                fixed_feature_lookup.loc[
                    record_id
                ].values
            )

        adaptive_test_features.append(
            features
        )


    X_adaptive_train = np.array(
        adaptive_train_features
    )

    X_adaptive_test = np.array(
        adaptive_test_features
    )


    print()
    print(
        "Feature shape:",
        X_fixed_train.shape
    )


    # =====================================================
    # FIXED RANDOM FOREST
    # =====================================================

    fixed_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )


    fixed_model.fit(
        X_fixed_train,
        y_train
    )


    fixed_prediction = (
        fixed_model.predict(
            X_fixed_test
        )
    )


    fixed_cm = confusion_matrix(
        y_test,
        fixed_prediction
    )


    fixed_tn, fixed_fp, fixed_fn, fixed_tp = (
        fixed_cm.ravel()
    )


    fixed_accuracy = accuracy_score(
        y_test,
        fixed_prediction
    )

    fixed_precision = precision_score(
        y_test,
        fixed_prediction,
        zero_division=0
    )

    fixed_sensitivity = recall_score(
        y_test,
        fixed_prediction,
        zero_division=0
    )

    fixed_specificity = (
        fixed_tn /
        (fixed_tn + fixed_fp)
    )

    fixed_f1 = f1_score(
        y_test,
        fixed_prediction,
        zero_division=0
    )


    fixed_results.append({

        "fold": fold,

        "accuracy": fixed_accuracy,

        "precision": fixed_precision,

        "sensitivity": fixed_sensitivity,

        "specificity": fixed_specificity,

        "f1": fixed_f1
    })


    # =====================================================
    # ADAPTIVE RANDOM FOREST
    # =====================================================

    adaptive_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )


    adaptive_model.fit(
        X_adaptive_train,
        y_train
    )


    adaptive_prediction = (
        adaptive_model.predict(
            X_adaptive_test
        )
    )


    adaptive_cm = confusion_matrix(
        y_test,
        adaptive_prediction
    )


    adaptive_tn, adaptive_fp, adaptive_fn, adaptive_tp = (
        adaptive_cm.ravel()
    )


    adaptive_accuracy = accuracy_score(
        y_test,
        adaptive_prediction
    )

    adaptive_precision = precision_score(
        y_test,
        adaptive_prediction,
        zero_division=0
    )

    adaptive_sensitivity = recall_score(
        y_test,
        adaptive_prediction,
        zero_division=0
    )

    adaptive_specificity = (
        adaptive_tn /
        (adaptive_tn + adaptive_fp)
    )

    adaptive_f1 = f1_score(
        y_test,
        adaptive_prediction,
        zero_division=0
    )


    adaptive_results.append({

        "fold": fold,

        "accuracy": adaptive_accuracy,

        "precision": adaptive_precision,

        "sensitivity": adaptive_sensitivity,

        "specificity": adaptive_specificity,

        "f1": adaptive_f1
    })


    # =====================================================
    # FOLD RESULTS
    # =====================================================

    print()
    print("FIXED BASELINE")

    print(
        f"Accuracy:    {fixed_accuracy:.4f}"
    )

    print(
        f"Precision:   {fixed_precision:.4f}"
    )

    print(
        f"Sensitivity: {fixed_sensitivity:.4f}"
    )

    print(
        f"Specificity: {fixed_specificity:.4f}"
    )

    print(
        f"F1:          {fixed_f1:.4f}"
    )


    print()
    print("ADAPTIVE PREPROCESSING")

    print(
        f"Accuracy:    {adaptive_accuracy:.4f}"
    )

    print(
        f"Precision:   {adaptive_precision:.4f}"
    )

    print(
        f"Sensitivity: {adaptive_sensitivity:.4f}"
    )

    print(
        f"Specificity: {adaptive_specificity:.4f}"
    )

    print(
        f"F1:          {adaptive_f1:.4f}"
    )


# =========================================================
# FINAL RESULTS
# =========================================================

fixed_results_df = pd.DataFrame(
    fixed_results
)

adaptive_results_df = pd.DataFrame(
    adaptive_results
)


metrics = [
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1"
]


print()
print("=" * 70)
print("FINAL 5-FOLD RESULTS")
print("=" * 70)


print()
print("FIXED BASELINE")
print("-" * 50)


for metric in metrics:

    mean = (
        fixed_results_df[
            metric
        ].mean()
    )

    std = (
        fixed_results_df[
            metric
        ].std()
    )

    print(
        f"{metric.capitalize():<15}"
        f"{mean:.4f} ± {std:.4f}"
    )


print()
print("ADAPTIVE PREPROCESSING")
print("-" * 50)


for metric in metrics:

    mean = (
        adaptive_results_df[
            metric
        ].mean()
    )

    std = (
        adaptive_results_df[
            metric
        ].std()
    )

    print(
        f"{metric.capitalize():<15}"
        f"{mean:.4f} ± {std:.4f}"
    )


# =========================================================
# DIFFERENCE
# =========================================================

print()
print("=" * 70)
print("ADAPTIVE - FIXED DIFFERENCE")
print("=" * 70)


for metric in metrics:

    fixed_mean = (
        fixed_results_df[
            metric
        ].mean()
    )

    adaptive_mean = (
        adaptive_results_df[
            metric
        ].mean()
    )

    difference = (
        adaptive_mean -
        fixed_mean
    )

    print(
        f"{metric.capitalize():<15}"
        f"{difference:+.4f}"
    )

os.makedirs(OUTPUT_PATH, exist_ok=True)
# =========================================================
# SAVE RESULTS
# =========================================================

fixed_results_df["method"] = "Fixed"

adaptive_results_df["method"] = "Adaptive"


final_results = pd.concat(
    [
        fixed_results_df,
        adaptive_results_df
    ],
    ignore_index=True
)


output_file = os.path.join(
    OUTPUT_PATH,
    "adaptive_preprocessing_cv_results.csv"
)


final_results.to_csv(
    output_file,
    index=False
)


print()
print(
    "Results saved to:"
)

print(
    output_file
)


print()
print("=" * 70)
print("EXPERIMENT 6 COMPLETED")
print("=" * 70)