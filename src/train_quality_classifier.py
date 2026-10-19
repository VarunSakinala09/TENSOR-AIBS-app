
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT / "results" / "week3"
    / "features_full_dataset.csv"
)

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "ecg_quality_fixed_tucker.joblib"
INFO_FILE = MODEL_DIR / "ecg_quality_fixed_tucker_info.csv"

SEED = 42

def main():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}"
        )

    df = pd.read_csv(FEATURE_FILE)

    if "label" not in df.columns:
        raise ValueError(
            "The feature CSV has no 'label' column."
        )

    labels_text = df["label"].astype(str).str.strip().str.lower()

    label_map = {
        "acceptable": 0,
        "accept": 0,
        "0": 0,
        "unacceptable": 1,
        "unaccept": 1,
        "1": 1,
    }

    y = labels_text.map(label_map)

    if y.isna().any():
        raise ValueError(
            "Unknown label values: "
            f"{labels_text[y.isna()].unique().tolist()}"
        )

    y = y.to_numpy(dtype=np.int32)

    feature_columns = [
        column for column in df.columns
        if column.startswith("feature_")
    ]

    if not feature_columns:
        raise ValueError(
            "No feature_ columns were found. "
            "Inspect the CSV columns before training."
        )

    X_df = df[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    if X_df.isna().any().any():
        raise ValueError(
            "Feature columns contain missing or nonnumeric values."
        )

    X = X_df.to_numpy(dtype=np.float32)

    if not np.isfinite(X).all():
        raise ValueError(
            "Feature columns contain infinite or invalid values."
        )

    if X.shape[1] != 6000:
        raise ValueError(
            f"Expected 6000 fixed Tucker features, got {X.shape[1]}. "
            "Do not train until the feature pipeline is verified."
        )

    print("Dataset:", FEATURE_FILE)
    print("Feature matrix:", X.shape)
    print("Acceptable:", int(np.sum(y == 0)))
    print("Unacceptable:", int(np.sum(y == 1)))

    # Hold out a stratified test split for a basic diagnostic.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nHold-out diagnostic (not the final research CV result)")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Acceptable", "Unacceptable"],
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    # Fit the deployable demonstration model on all labelled records.
    # This final fitted model is for the app, not an unbiased test score.
    model.fit(X, y)

    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "n_features": int(X.shape[1]),
        "feature_method": "fixed CWT 1-40 Hz, Tucker rank (50, 20, 6)",
        "class_mapping": {
            0: "Acceptable",
            1: "Unacceptable",
        },
        "training_records": int(len(y)),
    }

    joblib.dump(artifact, MODEL_FILE)

    pd.DataFrame([{
        "model_file": str(MODEL_FILE),
        "feature_count": X.shape[1],
        "training_records": len(y),
        "acceptable_records": int(np.sum(y == 0)),
        "unacceptable_records": int(np.sum(y == 1)),
        "feature_method": artifact["feature_method"],
    }]).to_csv(INFO_FILE, index=False)

    print("\nSaved model:", MODEL_FILE)
    print("Saved model information:", INFO_FILE)
    print("Classifier is ready for fixed-rank feature inference.")


if __name__ == "__main__":
    main()
