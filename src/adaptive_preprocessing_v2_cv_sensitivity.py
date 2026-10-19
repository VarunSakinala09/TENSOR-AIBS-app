
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# 1. PATHS AND INPUT DATA
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

OLD_OUT_DIR = RESULTS / "adaptive_preprocessing_v2"

# Save new outputs separately to protect previous results.
OUT_DIR = OLD_OUT_DIR / "sensitivity_threshold_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

baseline_path = RESULTS / "week3" / "features_full_dataset.csv"
highpass_path = OLD_OUT_DIR / "highpass_features_100levels.csv"
characteristics_path = RESULTS / "week5" / "signal_characteristics.csv"

for path in (baseline_path, highpass_path, characteristics_path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

baseline = pd.read_csv(baseline_path)
highpass = pd.read_csv(highpass_path)
characteristics = pd.read_csv(characteristics_path)


# ============================================================
# 2. VALIDATE AND ALIGN RECORDS
# ============================================================

for df, name in (
    (baseline, "baseline"),
    (highpass, "high-pass"),
    (characteristics, "signal characteristics"),
):
    if "record_id" not in df.columns:
        raise ValueError(f"{name} dataset has no record_id column")

    if df["record_id"].duplicated().any():
        raise ValueError(f"Duplicate record IDs in {name} dataset")

if set(baseline["record_id"]) != set(highpass["record_id"]):
    raise ValueError("Baseline and high-pass records do not match")

baseline = baseline.sort_values("record_id").reset_index(drop=True)

highpass = (
    highpass.set_index("record_id")
    .reindex(baseline["record_id"])
    .reset_index()
)

characteristics = (
    characteristics.set_index("record_id")
    .reindex(baseline["record_id"])
    .reset_index()
)

if highpass["label"].isna().any():
    raise ValueError("Missing labels after aligning records")

if not np.array_equal(
    baseline["label"].to_numpy(),
    highpass["label"].to_numpy(),
):
    raise ValueError("Labels differ between baseline and high-pass datasets")

if characteristics["mean_baseline_variation"].isna().any():
    raise ValueError("Missing baseline-variation values")

base_cols = [
    c for c in baseline.columns if c.startswith("feature_")
]

high_cols = [
    c for c in highpass.columns if c.startswith("feature_")
]

if not base_cols or base_cols != high_cols:
    raise ValueError("Baseline and high-pass feature columns do not match")

X_base = baseline[base_cols].to_numpy(dtype=float)
X_high = highpass[high_cols].to_numpy(dtype=float)

y = baseline["label"].astype(str).to_numpy()

variation = characteristics[
    "mean_baseline_variation"
].to_numpy(dtype=float)

if set(np.unique(y)) != {"Acceptable", "Unacceptable"}:
    raise ValueError(f"Unexpected labels: {np.unique(y)}")

if not (
    np.isfinite(X_base).all()
    and np.isfinite(X_high).all()
    and np.isfinite(variation).all()
):
    raise ValueError("Non-finite values found in input data")

if set(baseline["record_id"]) != set(characteristics["record_id"]):
    raise ValueError(
        "Baseline and signal-characteristics record IDs do not match"
    )


# ============================================================
# 3. METRICS AND MODEL FUNCTIONS
# ============================================================

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=["Acceptable", "Unacceptable"],
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            pos_label="Unacceptable",
            zero_division=0,
        ),
        "sensitivity": recall_score(
            y_true,
            y_pred,
            pos_label="Unacceptable",
            zero_division=0,
        ),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(
            y_true,
            y_pred,
            pos_label="Unacceptable",
            zero_division=0,
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_model():
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )


def get_unacceptable_probability(model, X):
    classes = list(model.classes_)

    if "Unacceptable" not in classes:
        raise ValueError("Unacceptable class missing from model")

    index = classes.index("Unacceptable")

    return model.predict_proba(X)[:, index]


def probability_to_labels(probabilities, threshold):
    return np.where(
        probabilities >= threshold,
        "Unacceptable",
        "Acceptable",
    )


# ============================================================
# 4. CROSS-VALIDATION SETUP
# ============================================================

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# ============================================================
# 5. EXPERIMENT A:
#    BASELINE VS FIXED HIGH-PASS VS EXISTING ADAPTIVE METHOD
# ============================================================

print("\n" + "=" * 65)
print("EXPERIMENT A: PREPROCESSING COMPARISON")
print("=" * 65)

old_predictions = {
    name: np.empty(len(y), dtype=object)
    for name in ("Baseline", "Fixed_HighPass", "Adaptive_80th")
}

old_fold_rows = []

for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X_base, y), start=1
):
    cutoff = np.percentile(variation[train_idx], 80)

    train_use_high = variation[train_idx] > cutoff
    test_use_high = variation[test_idx] > cutoff

    X_adapt_train = X_base[train_idx].copy()
    X_adapt_test = X_base[test_idx].copy()

    X_adapt_train[train_use_high] = X_high[
        train_idx[train_use_high]
    ]

    X_adapt_test[test_use_high] = X_high[
        test_idx[test_use_high]
    ]

    configurations = {
        "Baseline": (
            X_base[train_idx],
            X_base[test_idx],
        ),
        "Fixed_HighPass": (
            X_high[train_idx],
            X_high[test_idx],
        ),
        "Adaptive_80th": (
            X_adapt_train,
            X_adapt_test,
        ),
    }

    for name, (X_train, X_test) in configurations.items():
        model = make_model()
        model.fit(X_train, y[train_idx])

        predictions = model.predict(X_test)
        old_predictions[name][test_idx] = predictions

        old_fold_rows.append({
            "fold": fold,
            "configuration": name,
            "preprocessing_threshold": cutoff,
            "test_records_filtered": (
                int(test_use_high.sum())
                if name == "Adaptive_80th"
                else 0
            ),
            **calculate_metrics(y[test_idx], predictions),
        })

    print(f"Fold {fold}/5 complete")


old_fold_df = pd.DataFrame(old_fold_rows)

old_fold_df.to_csv(
    OUT_DIR / "preprocessing_comparison_fold_metrics.csv",
    index=False,
)

old_oof = pd.DataFrame({
    "record_id": baseline["record_id"],
    "label": y,
    "mean_baseline_variation": variation,
})

for name, predictions in old_predictions.items():
    old_oof[f"{name}_prediction"] = predictions

old_oof.to_csv(
    OUT_DIR / "preprocessing_comparison_oof_predictions.csv",
    index=False,
)

old_summary = pd.DataFrame([
    {
        "configuration": name,
        **calculate_metrics(y, predictions),
    }
    for name, predictions in old_predictions.items()
])

old_summary.to_csv(
    OUT_DIR / "preprocessing_comparison_pooled_metrics.csv",
    index=False,
)

print("\nPREPROCESSING COMPARISON — POOLED OOF")
print(old_summary.round(4).to_string(index=False))


# ============================================================
# 6. EXPERIMENT B:
#    PROBABILITY THRESHOLD SELECTION
#
#    Goal: maximize sensitivity while targeting specificity >= 95%.
#    Threshold selection uses inner OOF predictions only.
# ============================================================

print("\n" + "=" * 65)
print("EXPERIMENT B: SENSITIVITY-PRIORITIZED PROBABILITY THRESHOLD")
print("=" * 65)

TARGET_SPECIFICITY = 0.95
INNER_SPLITS = 3

PROBABILITY_THRESHOLDS = np.round(
    np.arange(0.05, 0.951, 0.025),
    3,
)

probability_oof_predictions = np.empty(len(y), dtype=object)
probability_oof_scores = np.full(len(y), np.nan)

threshold_rows = []
probability_fold_rows = []

for outer_fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X_base, y), start=1
):
    X_train = X_base[train_idx]
    y_train = y[train_idx]

    # --------------------------------------------------------
    # Inner cross-validation:
    # Obtain training-only out-of-fold probabilities.
    # --------------------------------------------------------

    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=100 + outer_fold,
    )

    inner_probabilities = np.full(len(train_idx), np.nan)

    for inner_train_idx, inner_valid_idx in inner_cv.split(
        X_train, y_train
    ):
        inner_model = make_model()

        inner_model.fit(
            X_train[inner_train_idx],
            y_train[inner_train_idx],
        )

        inner_probabilities[inner_valid_idx] = (
            get_unacceptable_probability(
                inner_model,
                X_train[inner_valid_idx],
            )
        )

    if not np.isfinite(inner_probabilities).all():
        raise RuntimeError(
            "Some inner out-of-fold probabilities were not assigned"
        )

    # --------------------------------------------------------
    # Choose threshold using inner OOF predictions.
    # Do not use outer test labels to choose it.
    # --------------------------------------------------------

    candidate_rows = []

    for probability_cutoff in PROBABILITY_THRESHOLDS:
        inner_predictions = probability_to_labels(
            inner_probabilities,
            probability_cutoff,
        )

        candidate_rows.append({
            "probability_threshold": float(probability_cutoff),
            **calculate_metrics(y_train, inner_predictions),
        })

    candidate_df = pd.DataFrame(candidate_rows)

    eligible = candidate_df[
        candidate_df["specificity"] >= TARGET_SPECIFICITY
    ]

    if not eligible.empty:
        # Primary goal: sensitivity.
        # Tie-break: F1, precision, then specificity.
        chosen = eligible.sort_values(
            ["sensitivity", "f1", "precision", "specificity"],
            ascending=[False, False, False, False],
        ).iloc[0]

        target_met = True

    else:
        # No candidate met the target.
        # Report the best specificity-first fallback.
        chosen = candidate_df.sort_values(
            ["specificity", "sensitivity", "f1", "precision"],
            ascending=[False, False, False, False],
        ).iloc[0]

        target_met = False

    probability_cutoff = float(
        chosen["probability_threshold"]
    )

    # --------------------------------------------------------
    # Evaluate on the untouched outer test fold.
    # --------------------------------------------------------

    outer_model = make_model()

    outer_model.fit(
        X_base[train_idx],
        y[train_idx],
    )

    test_probabilities = get_unacceptable_probability(
        outer_model,
        X_base[test_idx],
    )

    test_predictions = probability_to_labels(
        test_probabilities,
        probability_cutoff,
    )

    probability_oof_predictions[test_idx] = test_predictions
    probability_oof_scores[test_idx] = test_probabilities

    test_metrics = calculate_metrics(
        y[test_idx],
        test_predictions,
    )

    threshold_rows.append({
        "outer_fold": outer_fold,
        "selected_probability_threshold": probability_cutoff,
        "inner_specificity_target_met": target_met,
        "inner_sensitivity": float(chosen["sensitivity"]),
        "inner_specificity": float(chosen["specificity"]),
        "inner_precision": float(chosen["precision"]),
        "inner_f1": float(chosen["f1"]),
    })

    probability_fold_rows.append({
        "fold": outer_fold,
        "probability_threshold": probability_cutoff,
        "inner_specificity_target_met": target_met,
        **test_metrics,
    })

    print(
        f"Fold {outer_fold}/5 | "
        f"probability cutoff={probability_cutoff:.3f} | "
        f"inner target met={target_met} | "
        f"test sensitivity={test_metrics['sensitivity']:.4f} | "
        f"test specificity={test_metrics['specificity']:.4f}"
    )


# ============================================================
# 7. SAVE RESULTS WITHOUT OVERWRITING OLD EXPERIMENTS
# ============================================================

threshold_df = pd.DataFrame(threshold_rows)
fold_metrics_df = pd.DataFrame(probability_fold_rows)

threshold_df.to_csv(
    OUT_DIR / "probability_thresholds_by_fold.csv",
    index=False,
)

fold_metrics_df.to_csv(
    OUT_DIR / "probability_threshold_fold_metrics.csv",
    index=False,
)

probability_oof_df = pd.DataFrame({
    "record_id": baseline["record_id"],
    "label": y,
    "predicted_probability_unacceptable": probability_oof_scores,
    "selected_prediction": probability_oof_predictions,
})

probability_oof_df.to_csv(
    OUT_DIR / "probability_threshold_oof_predictions.csv",
    index=False,
)

pooled_metrics = calculate_metrics(
    y,
    probability_oof_predictions,
)

pooled_summary_df = pd.DataFrame([{
    "configuration": "Probability_Threshold_Selector",
    **pooled_metrics,
}])

pooled_summary_df.to_csv(
    OUT_DIR / "probability_threshold_pooled_metrics.csv",
    index=False,
)


# ============================================================
# 8. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("PROBABILITY-THRESHOLD SELECTOR — POOLED OOF")
print("=" * 65)

print(pooled_summary_df.round(4).to_string(index=False))

print("\nFOLD-MEAN METRICS")

print(
    fold_metrics_df[
        [
            "accuracy",
            "precision",
            "sensitivity",
            "specificity",
            "f1",
        ]
    ].mean().round(4)
)

print("\nSELECTED THRESHOLDS")

print(
    threshold_df[
        [
            "outer_fold",
            "selected_probability_threshold",
            "inner_specificity_target_met",
            "inner_sensitivity",
            "inner_specificity",
        ]
    ].to_string(index=False)
)

print("\nConfusion matrix counts for pooled predictions:")
print(
    "TP:", pooled_metrics["tp"],
    "| FN:", pooled_metrics["fn"],
    "| TN:", pooled_metrics["tn"],
    "| FP:", pooled_metrics["fp"],
)

print(f"\nNew experiment results saved to:\n{OUT_DIR}")
print("Previous experiment result files were not overwritten.")


# ============================================================
# EXPERIMENT C: NEARBY PROBABILITY CUTOFF COMPARISON
# Exploratory analysis using existing outer-fold OOF scores.
# Does not overwrite previous experiment outputs.
# ============================================================

print("\n" + "=" * 65)
print("EXPERIMENT C: PROBABILITY CUTOFF COMPARISON")
print("=" * 65)

cutoffs_to_compare = [0.250, 0.275, 0.300, 0.325, 0.350]

cutoff_results = []

for cutoff in cutoffs_to_compare:
    predictions = probability_to_labels(
        probability_oof_scores,
        cutoff
    )

    metrics = calculate_metrics(y, predictions)

    cutoff_results.append({
        "probability_cutoff": cutoff,
        **metrics
    })

cutoff_df = pd.DataFrame(cutoff_results)

cutoff_output = OUT_DIR / "nearby_probability_cutoff_comparison.csv"
cutoff_df.to_csv(cutoff_output, index=False)

print("\nPOOLED OUT-OF-FOLD CUTOFF COMPARISON")
print(
    cutoff_df[
        [
            "probability_cutoff",
            "accuracy",
            "precision",
            "sensitivity",
            "specificity",
            "f1",
            "tp",
            "fn",
            "tn",
            "fp",
        ]
    ].round(4).to_string(index=False)
)

print("\nSaved to:", cutoff_output)
print(
    "\nNOTE: This table is exploratory. "
    "Do not select a final cutoff from this table and report "
    "its same-data performance as an unbiased estimate."
)


# ============================================================
# EXPERIMENT D: HIGHER INNER SPECIFICITY TARGET
# Reuses the existing baseline features and model functions.
# Does not overwrite previous experiment results.
# ============================================================

print("\n" + "=" * 65)
print("EXPERIMENT D: SENSITIVITY-FIRST WITH 97% INNER SPECIFICITY")
print("=" * 65)

TARGET_SPECIFICITY_D = 0.97

OUT_D = OUT_DIR / "specificity_buffer_experiment"
OUT_D.mkdir(parents=True, exist_ok=True)

cutoffs_D = np.round(np.arange(0.05, 0.951, 0.025), 3)

predictions_D = np.empty(len(y), dtype=object)
probabilities_D = np.full(len(y), np.nan)

fold_rows_D = []
threshold_rows_D = []

for outer_fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X_base, y), start=1
):
    X_train = X_base[train_idx]
    y_train = y[train_idx]

    inner_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=200 + outer_fold,
    )

    inner_prob = np.full(len(train_idx), np.nan)

    # Inner out-of-fold probabilities: training data only.
    for inner_train, inner_valid in inner_cv.split(X_train, y_train):
        model = make_model()

        model.fit(
            X_train[inner_train],
            y_train[inner_train],
        )

        inner_prob[inner_valid] = get_unacceptable_probability(
            model,
            X_train[inner_valid],
        )

    if not np.isfinite(inner_prob).all():
        raise RuntimeError("Missing inner out-of-fold probabilities")

    candidate_rows = []

    for cutoff in cutoffs_D:
        inner_pred = probability_to_labels(inner_prob, cutoff)

        candidate_rows.append({
            "cutoff": float(cutoff),
            **calculate_metrics(y_train, inner_pred),
        })

    candidate_df = pd.DataFrame(candidate_rows)

    eligible = candidate_df[
        candidate_df["specificity"] >= TARGET_SPECIFICITY_D
    ]

    if not eligible.empty:
        # Maximize sensitivity while meeting the inner target.
        chosen = eligible.sort_values(
            ["sensitivity", "f1", "precision", "specificity"],
            ascending=[False, False, False, False],
        ).iloc[0]

        target_met = True
    else:
        # Explicit fallback if no candidate meets 97%.
        chosen = candidate_df.sort_values(
            ["specificity", "sensitivity", "f1", "precision"],
            ascending=[False, False, False, False],
        ).iloc[0]

        target_met = False

    cutoff = float(chosen["cutoff"])

    # Fit only on the outer training fold.
    outer_model = make_model()
    outer_model.fit(X_base[train_idx], y[train_idx])

    test_prob = get_unacceptable_probability(
        outer_model,
        X_base[test_idx],
    )

    test_pred = probability_to_labels(test_prob, cutoff)

    predictions_D[test_idx] = test_pred
    probabilities_D[test_idx] = test_prob

    metrics = calculate_metrics(y[test_idx], test_pred)

    fold_rows_D.append({
        "fold": outer_fold,
        "cutoff": cutoff,
        "inner_target_met": target_met,
        **metrics,
    })

    threshold_rows_D.append({
        "fold": outer_fold,
        "cutoff": cutoff,
        "inner_specificity": float(chosen["specificity"]),
        "inner_sensitivity": float(chosen["sensitivity"]),
        "inner_target_met": target_met,
        "test_specificity": metrics["specificity"],
        "test_sensitivity": metrics["sensitivity"],
    })

    print(
        f"Fold {outer_fold}/5 | cutoff={cutoff:.3f} | "
        f"inner target met={target_met} | "
        f"test sensitivity={metrics['sensitivity']:.4f} | "
        f"test specificity={metrics['specificity']:.4f}"
    )

# Save this experiment separately.
fold_df_D = pd.DataFrame(fold_rows_D)
threshold_df_D = pd.DataFrame(threshold_rows_D)

fold_df_D.to_csv(
    OUT_D / "fold_metrics.csv",
    index=False,
)

threshold_df_D.to_csv(
    OUT_D / "selected_thresholds.csv",
    index=False,
)

oof_df_D = pd.DataFrame({
    "record_id": baseline["record_id"],
    "label": y,
    "predicted_probability_unacceptable": probabilities_D,
    "prediction": predictions_D,
})

oof_df_D.to_csv(
    OUT_D / "oof_predictions.csv",
    index=False,
)

pooled_D = calculate_metrics(y, predictions_D)

pd.DataFrame([pooled_D]).to_csv(
    OUT_D / "pooled_metrics.csv",
    index=False,
)

print("\nEXPERIMENT D — POOLED OOF RESULTS")
print(pd.DataFrame([pooled_D]).round(4).to_string(index=False))

print("\nFOLD-MEAN RESULTS")
print(
    fold_df_D[
        ["accuracy", "precision", "sensitivity", "specificity", "f1"]
    ].mean().round(4)
)

print("\nSELECTED THRESHOLDS")
print(threshold_df_D.to_string(index=False))

print("\nSaved new results to:", OUT_D)
