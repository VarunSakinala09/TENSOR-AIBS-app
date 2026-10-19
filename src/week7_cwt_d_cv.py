import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CWT_D_FILE = os.path.join(
    BASE_DIR,
    "results",
    "week7",
    "cwt_d",
    "cwt_d_features.csv"
)

FIXED_FILE = os.path.join(
    BASE_DIR,
    "results",
    "week3",
    "features_full_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "week7",
    "cwt_d"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("EXPERIMENT 7B-2")
print("FIXED CWT vs CWT-D CLASSIFICATION")
print("=" * 70)

print("\nLoading fixed baseline features...")

fixed_df = pd.read_csv(
    FIXED_FILE
)

print(
    "Fixed dataset shape:",
    fixed_df.shape
)


print("\nLoading CWT-D features...")

cwt_d_df = pd.read_csv(
    CWT_D_FILE
)

print(
    "CWT-D dataset shape:",
    cwt_d_df.shape
)


# ============================================================
# ALIGN RECORDS
# ============================================================

fixed_df = fixed_df.sort_values(
    "record_id"
).reset_index(drop=True)

cwt_d_df = cwt_d_df.sort_values(
    "record_id"
).reset_index(drop=True)


# Make sure both datasets contain
# exactly the same recordings

if not np.array_equal(
    fixed_df["record_id"].values,
    cwt_d_df["record_id"].values
):

    raise ValueError(
        "Record IDs do not match between datasets."
    )


# Make sure labels also match

if not np.array_equal(
    fixed_df["label"].values,
    cwt_d_df["label"].values
):

    raise ValueError(
        "Labels do not match between datasets."
    )


print("\nRecord alignment check: PASSED")


# ============================================================
# PREPARE LABELS
# ============================================================

labels = fixed_df["label"]

label_mapping = {
    "Acceptable": 0,
    "Unacceptable": 1
}

y = labels.map(
    label_mapping
).values


print("\nLabel distribution:")

print(
    pd.Series(y).value_counts().sort_index()
)

print(
    "0 = Acceptable"
)

print(
    "1 = Unacceptable"
)


# ============================================================
# GET FEATURE COLUMNS
# ============================================================

fixed_feature_columns = [
    column
    for column in fixed_df.columns
    if column.startswith("feature_")
]

cwt_d_feature_columns = [
    column
    for column in cwt_d_df.columns
    if column.startswith("feature_")
]


print("\nNumber of fixed features:")
print(
    len(fixed_feature_columns)
)

print("\nNumber of CWT-D features:")
print(
    len(cwt_d_feature_columns)
)


# ============================================================
# FEATURE MATRICES
# ============================================================

X_fixed = fixed_df[
    fixed_feature_columns
].values

X_cwt_d = cwt_d_df[
    cwt_d_feature_columns
].values


print("\nFeature matrix shapes:")

print(
    "Fixed:",
    X_fixed.shape
)

print(
    "CWT-D:",
    X_cwt_d.shape
)


# ============================================================
# CROSS VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# RANDOM FOREST FUNCTION
# ============================================================

def evaluate_model(
    X,
    y,
    name
):

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    fold_results = []

    for fold, (
        train_index,
        test_index
    ) in enumerate(
        skf.split(X, y),
        start=1
    ):

        print(
            f"\nFold {fold}/5"
        )

        X_train = X[
            train_index
        ]

        X_test = X[
            test_index
        ]

        y_train = y[
            train_index
        ]

        y_test = y[
            test_index
        ]


        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )


        print(
            "Training Random Forest..."
        )

        model.fit(
            X_train,
            y_train
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        sensitivity = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )


        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        ).ravel()


        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0
        )


        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )


        fold_results.append(
            {
                "fold": fold,
                "accuracy": accuracy,
                "precision": precision,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "f1": f1
            }
        )


        print(
            f"Accuracy    : {accuracy:.4f}"
        )

        print(
            f"Precision   : {precision:.4f}"
        )

        print(
            f"Sensitivity : {sensitivity:.4f}"
        )

        print(
            f"Specificity : {specificity:.4f}"
        )

        print(
            f"F1          : {f1:.4f}"
        )


    return pd.DataFrame(
        fold_results
    )


# ============================================================
# E0 — FIXED BASELINE
# ============================================================

fixed_results = evaluate_model(
    X_fixed,
    y,
    "E0 — FIXED CWT BASELINE"
)


# ============================================================
# CWT-D
# ============================================================

cwt_d_results = evaluate_model(
    X_cwt_d,
    y,
    "CWT-D — 1–50 Hz / 100 LEVELS"
)


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def summarize(
    results
):

    metrics = [
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1"
    ]

    summary = {}

    for metric in metrics:

        summary[
            metric
        ] = results[
            metric
        ].mean()

        summary[
            metric + "_std"
        ] = results[
            metric
        ].std()

    return summary


fixed_summary = summarize(
    fixed_results
)

cwt_d_summary = summarize(
    cwt_d_results
)


# ============================================================
# FINAL COMPARISON
# ============================================================

comparison = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Sensitivity",
            "Specificity",
            "F1"
        ],

        "Fixed_Baseline": [
            fixed_summary["accuracy"],
            fixed_summary["precision"],
            fixed_summary["sensitivity"],
            fixed_summary["specificity"],
            fixed_summary["f1"]
        ],

        "CWT_D": [
            cwt_d_summary["accuracy"],
            cwt_d_summary["precision"],
            cwt_d_summary["sensitivity"],
            cwt_d_summary["specificity"],
            cwt_d_summary["f1"]
        ]
    }
)


# ============================================================
# DIFFERENCE
# ============================================================

comparison["Difference"] = (
    comparison["CWT_D"]
    -
    comparison["Fixed_Baseline"]
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

comparison_file = os.path.join(
    OUTPUT_DIR,
    "cwt_d_vs_fixed_comparison.csv"
)

fold_file = os.path.join(
    OUTPUT_DIR,
    "cwt_d_vs_fixed_fold_results.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)

fold_results = pd.concat(
    [
        fixed_results.assign(
            configuration="Fixed"
        ),

        cwt_d_results.assign(
            configuration="CWT-D"
        )
    ],
    ignore_index=True
)

fold_results.to_csv(
    fold_file,
    index=False
)


# ============================================================
# INTERPRETATION
# ============================================================

print("\n")
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

fixed_sensitivity = (
    fixed_summary["sensitivity"]
)

cwt_d_sensitivity = (
    cwt_d_summary["sensitivity"]
)


difference = (
    cwt_d_sensitivity
    -
    fixed_sensitivity
)


print(
    f"\nFixed sensitivity : "
    f"{fixed_sensitivity:.4f}"
)

print(
    f"CWT-D sensitivity : "
    f"{cwt_d_sensitivity:.4f}"
)

print(
    f"Difference        : "
    f"{difference:+.4f}"
)


if difference > 0:

    print(
        "\nCWT-D improved sensitivity."
    )

elif difference < 0:

    print(
        "\nCWT-D decreased sensitivity."
    )

else:

    print(
        "\nCWT-D produced the same sensitivity."
    )


print("\nSaved files:")

print(
    comparison_file
)

print(
    fold_file
)

print("\n")
print("=" * 70)
print("EXPERIMENT 7B-2 COMPLETE")
print("=" * 70)