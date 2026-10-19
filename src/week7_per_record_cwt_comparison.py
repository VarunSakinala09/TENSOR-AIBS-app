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

FIXED_FILE = os.path.join(
    BASE_DIR,
    "results",
    "week3",
    "features_full_dataset.csv"
)

CWT_D_FILE = os.path.join(
    BASE_DIR,
    "results",
    "week7",
    "cwt_d",
    "cwt_d_features.csv"
)

CHARACTERISTICS_FILE = os.path.join(
    BASE_DIR,
    "results",
    "week5",
    "signal_characteristics.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "week7",
    "per_record_comparison"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("EXPERIMENT 7D")
print("PER-RECORD CWT COMPARISON")
print("=" * 70)

print("\nLoading fixed CWT features...")

fixed_df = pd.read_csv(
    FIXED_FILE
)

print(
    "Fixed shape:",
    fixed_df.shape
)


print("\nLoading CWT-D features...")

cwt_d_df = pd.read_csv(
    CWT_D_FILE
)

print(
    "CWT-D shape:",
    cwt_d_df.shape
)


print("\nLoading signal characteristics...")

characteristics_df = pd.read_csv(
    CHARACTERISTICS_FILE
)

print(
    "Characteristics shape:",
    characteristics_df.shape
)


# ============================================================
# SORT AND ALIGN
# ============================================================

fixed_df = fixed_df.sort_values(
    "record_id"
).reset_index(drop=True)

cwt_d_df = cwt_d_df.sort_values(
    "record_id"
).reset_index(drop=True)

characteristics_df = characteristics_df.sort_values(
    "record_id"
).reset_index(drop=True)


# ============================================================
# CHECK RECORD IDs
# ============================================================

if not np.array_equal(
    fixed_df["record_id"].values,
    cwt_d_df["record_id"].values
):

    raise ValueError(
        "Fixed and CWT-D record IDs do not match."
    )


if not np.array_equal(
    fixed_df["record_id"].values,
    characteristics_df["record_id"].values
):

    raise ValueError(
        "Feature and characteristics record IDs do not match."
    )


# ============================================================
# CHECK LABELS
# ============================================================

if not np.array_equal(
    fixed_df["label"].values,
    cwt_d_df["label"].values
):

    raise ValueError(
        "Labels do not match."
    )


print(
    "\nRecord and label alignment: PASSED"
)


# ============================================================
# CREATE CONDITION GROUPS
# ============================================================

# We use the same condition parameter as Experiment 7C:
# mean_baseline_variation

characteristics_df["baseline_group"] = pd.qcut(
    characteristics_df["mean_baseline_variation"],
    q=3,
    labels=[
        "Low",
        "Medium",
        "High"
    ],
    duplicates="drop"
)


# ============================================================
# PREPARE FEATURES
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


X_fixed = fixed_df[
    fixed_feature_columns
].values

X_cwt_d = cwt_d_df[
    cwt_d_feature_columns
].values


# ============================================================
# LABEL ENCODING
# ============================================================

label_mapping = {
    "Acceptable": 0,
    "Unacceptable": 1
}

y = fixed_df[
    "label"
].map(
    label_mapping
).values


# ============================================================
# CROSS VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# STORAGE FOR OUT-OF-FOLD PREDICTIONS
# ============================================================

fixed_predictions = np.full(
    len(y),
    -1,
    dtype=int
)

cwt_d_predictions = np.full(
    len(y),
    -1,
    dtype=int
)

fold_numbers = np.full(
    len(y),
    -1,
    dtype=int
)


# ============================================================
# 5-FOLD TRAINING
# ============================================================

for fold, (
    train_index,
    test_index
) in enumerate(
    skf.split(X_fixed, y),
    start=1
):

    print("\n")
    print("=" * 70)
    print(f"FOLD {fold}/5")
    print("=" * 70)


    # --------------------------------------------------------
    # Store fold number
    # --------------------------------------------------------

    fold_numbers[
        test_index
    ] = fold


    # ========================================================
    # FIXED CWT MODEL
    # ========================================================

    print("\nTraining Fixed CWT model...")

    fixed_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    fixed_model.fit(
        X_fixed[train_index],
        y[train_index]
    )

    fixed_predictions[
        test_index
    ] = fixed_model.predict(
        X_fixed[test_index]
    )


    # ========================================================
    # CWT-D MODEL
    # ========================================================

    print(
        "Training CWT-D model..."
    )

    cwt_d_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    cwt_d_model.fit(
        X_cwt_d[train_index],
        y[train_index]
    )

    cwt_d_predictions[
        test_index
    ] = cwt_d_model.predict(
        X_cwt_d[test_index]
    )


    # ========================================================
    # FOLD SUMMARY
    # ========================================================

    fixed_accuracy = accuracy_score(
        y[test_index],
        fixed_predictions[test_index]
    )

    cwt_d_accuracy = accuracy_score(
        y[test_index],
        cwt_d_predictions[test_index]
    )

    print(
        f"Fixed CWT accuracy : "
        f"{fixed_accuracy:.4f}"
    )

    print(
        f"CWT-D accuracy     : "
        f"{cwt_d_accuracy:.4f}"
    )


# ============================================================
# CHECK PREDICTIONS
# ============================================================

if np.any(
    fixed_predictions == -1
):

    raise ValueError(
        "Some fixed predictions are missing."
    )

if np.any(
    cwt_d_predictions == -1
):

    raise ValueError(
        "Some CWT-D predictions are missing."
    )


print("\n")
print("=" * 70)
print("ALL OUT-OF-FOLD PREDICTIONS GENERATED")
print("=" * 70)


# ============================================================
# CREATE PER-RECORD DATASET
# ============================================================

prediction_df = pd.DataFrame(
    {
        "record_id":
            fixed_df["record_id"],

        "label":
            fixed_df["label"],

        "true_class":
            y,

        "fold":
            fold_numbers,

        "baseline_group":
            characteristics_df[
                "baseline_group"
            ],

        "mean_baseline_variation":
            characteristics_df[
                "mean_baseline_variation"
            ],

        "fixed_prediction":
            fixed_predictions,

        "cwt_d_prediction":
            cwt_d_predictions
    }
)


# ============================================================
# CORRECTNESS
# ============================================================

prediction_df[
    "fixed_correct"
] = (
    prediction_df[
        "fixed_prediction"
    ]
    ==
    prediction_df[
        "true_class"
    ]
)

prediction_df[
    "cwt_d_correct"
] = (
    prediction_df[
        "cwt_d_prediction"
    ]
    ==
    prediction_df[
        "true_class"
    ]
)


# ============================================================
# CONFIGURATION PREFERENCE
# ============================================================

def determine_preference(row):

    fixed_correct = row[
        "fixed_correct"
    ]

    cwt_d_correct = row[
        "cwt_d_correct"
    ]

    if (
        fixed_correct
        and
        not cwt_d_correct
    ):
        return "Fixed_1_40"

    elif (
        cwt_d_correct
        and
        not fixed_correct
    ):
        return "CWT_D_1_50"

    elif (
        fixed_correct
        and
        cwt_d_correct
    ):
        return "Both_Correct"

    else:
        return "Both_Wrong"


prediction_df[
    "preference"
] = prediction_df.apply(
    determine_preference,
    axis=1
)


# ============================================================
# SAVE PER-RECORD RESULTS
# ============================================================

prediction_file = os.path.join(
    OUTPUT_DIR,
    "per_record_cwt_predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# OVERALL METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1
    }


fixed_metrics = calculate_metrics(
    y,
    fixed_predictions
)

cwt_d_metrics = calculate_metrics(
    y,
    cwt_d_predictions
)


# ============================================================
# OVERALL RESULT
# ============================================================

overall_results = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Sensitivity",
            "Specificity",
            "F1"
        ],

        "Fixed_1_40": [
            fixed_metrics["accuracy"],
            fixed_metrics["precision"],
            fixed_metrics["sensitivity"],
            fixed_metrics["specificity"],
            fixed_metrics["f1"]
        ],

        "CWT_D_1_50": [
            cwt_d_metrics["accuracy"],
            cwt_d_metrics["precision"],
            cwt_d_metrics["sensitivity"],
            cwt_d_metrics["specificity"],
            cwt_d_metrics["f1"]
        ]
    }
)

overall_results[
    "Difference"
] = (
    overall_results[
        "CWT_D_1_50"
    ]
    -
    overall_results[
        "Fixed_1_40"
    ]
)


# ============================================================
# CONDITION-WISE METRICS
# ============================================================

condition_results = []

for condition in [
    "Low",
    "Medium",
    "High"
]:

    subset = prediction_df[
        prediction_df[
            "baseline_group"
        ]
        ==
        condition
    ]

    true_values = subset[
        "true_class"
    ]

    fixed_values = subset[
        "fixed_prediction"
    ]

    cwt_d_values = subset[
        "cwt_d_prediction"
    ]


    fixed_condition_metrics = calculate_metrics(
        true_values,
        fixed_values
    )

    cwt_d_condition_metrics = calculate_metrics(
        true_values,
        cwt_d_values
    )


    condition_results.append(
        {
            "condition": condition,

            "record_count": len(subset),

            "fixed_accuracy":
                fixed_condition_metrics[
                    "accuracy"
                ],

            "cwt_d_accuracy":
                cwt_d_condition_metrics[
                    "accuracy"
                ],

            "fixed_precision":
                fixed_condition_metrics[
                    "precision"
                ],

            "cwt_d_precision":
                cwt_d_condition_metrics[
                    "precision"
                ],

            "fixed_sensitivity":
                fixed_condition_metrics[
                    "sensitivity"
                ],

            "cwt_d_sensitivity":
                cwt_d_condition_metrics[
                    "sensitivity"
                ],

            "fixed_specificity":
                fixed_condition_metrics[
                    "specificity"
                ],

            "cwt_d_specificity":
                cwt_d_condition_metrics[
                    "specificity"
                ],

            "fixed_f1":
                fixed_condition_metrics[
                    "f1"
                ],

            "cwt_d_f1":
                cwt_d_condition_metrics[
                    "f1"
                ]
        }
    )


condition_results_df = pd.DataFrame(
    condition_results
)


# ============================================================
# ADD DIFFERENCES
# ============================================================

condition_results_df[
    "accuracy_difference"
] = (
    condition_results_df[
        "cwt_d_accuracy"
    ]
    -
    condition_results_df[
        "fixed_accuracy"
    ]
)

condition_results_df[
    "sensitivity_difference"
] = (
    condition_results_df[
        "cwt_d_sensitivity"
    ]
    -
    condition_results_df[
        "fixed_sensitivity"
    ]
)

condition_results_df[
    "specificity_difference"
] = (
    condition_results_df[
        "cwt_d_specificity"
    ]
    -
    condition_results_df[
        "fixed_specificity"
    ]
)

condition_results_df[
    "f1_difference"
] = (
    condition_results_df[
        "cwt_d_f1"
    ]
    -
    condition_results_df[
        "fixed_f1"
    ]
)


# ============================================================
# CONFIGURATION PREFERENCE COUNTS
# ============================================================

preference_counts = (
    prediction_df[
        [
            "baseline_group",
            "preference"
        ]
    ]
    .value_counts()
    .reset_index(
        name="count"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

overall_file = os.path.join(
    OUTPUT_DIR,
    "overall_per_record_cwt_results.csv"
)

condition_file = os.path.join(
    OUTPUT_DIR,
    "condition_wise_cwt_results.csv"
)

preference_file = os.path.join(
    OUTPUT_DIR,
    "condition_configuration_preferences.csv"
)


overall_results.to_csv(
    overall_file,
    index=False
)

condition_results_df.to_csv(
    condition_file,
    index=False
)

preference_counts.to_csv(
    preference_file,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("OVERALL OUT-OF-FOLD RESULTS")
print("=" * 70)

print(
    overall_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n")
print("=" * 70)
print("CONDITION-WISE RESULTS")
print("=" * 70)

print(
    condition_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n")
print("=" * 70)
print("CONFIGURATION PREFERENCE")
print("=" * 70)

print(
    preference_counts.to_string(
        index=False
    )
)


# ============================================================
# IMPORTANT INTERPRETATION
# ============================================================

print("\n")
print("=" * 70)
print("EXPERIMENT 7D INTERPRETATION")
print("=" * 70)

print(
    "\nEach ECG now has an out-of-fold prediction "
    "from both CWT configurations."
)

print(
    "This allows us to compare the configurations "
    "at the individual-record level."
)

print(
    "\nIMPORTANT:"
)

print(
    "A configuration being correct for one ECG "
    "does not automatically mean it should be "
    "selected adaptively using the test label."
)

print(
    "The ground-truth labels are used only to "
    "evaluate the configurations."
)

print(
    "Any final adaptive selection rule must be "
    "learned using training data only."
)

print("\nSaved files:")

print(
    prediction_file
)

print(
    overall_file
)

print(
    condition_file
)

print(
    preference_file
)

print("\n")
print("=" * 70)
print("EXPERIMENT 7D COMPLETE")
print("=" * 70)