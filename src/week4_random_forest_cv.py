import pandas as pd
import numpy as np

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
# 1. LOAD DATASET
# ============================================================

data_path = "results/week3/features_full_dataset.csv"

data = pd.read_csv(data_path)

print("\nDATASET")
print("-------")
print("Dataset shape:", data.shape)


# ============================================================
# 2. SELECT FEATURES DYNAMICALLY
# ============================================================

feature_columns = [
    column for column in data.columns
    if column.startswith("feature_")
]

X = data[feature_columns]
y = data["label"]

print("\nFEATURES")
print("--------")
print("Number of features:", len(feature_columns))
print("Feature matrix shape:", X.shape)


# ============================================================
# 3. LABEL DISTRIBUTION
# ============================================================

print("\nLABELS")
print("------")
print("Number of labels:", len(y))

print("\nClass distribution:")
print(y.value_counts())


# ============================================================
# 4. STRATIFIED 5-FOLD CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Store results from every fold
accuracy_scores = []
precision_scores = []
recall_scores = []
specificity_scores = []
f1_scores = []


# ============================================================
# 5. TRAIN AND TEST EACH FOLD
# ============================================================

for fold, (train_index, test_index) in enumerate(cv.split(X, y), start=1):

    print("\n" + "=" * 55)
    print("FOLD", fold)
    print("=" * 55)

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )


    print("\nTraining Random Forest...")

    model.fit(X_train, y_train)

    print("Training completed.")


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred = model.predict(X_test)


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="Unacceptable",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="Unacceptable",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="Unacceptable",
        zero_division=0
    )


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["Acceptable", "Unacceptable"]
    )

    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]


    # Specificity = TN / (TN + FP)

    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 0


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    specificity_scores.append(specificity)
    f1_scores.append(f1)


    # --------------------------------------------------------
    # Print fold results
    # --------------------------------------------------------

    print("\nFold Results")
    print("------------")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"Specificity:   {specificity:.4f}")
    print(f"F1-score:      {f1:.4f}")


    print("\nConfusion Matrix")
    print("----------------")
    print("              Acceptable  Unacceptable")
    print(
        f"Acceptable    {tn:10d}  {fp:12d}"
    )
    print(
        f"Unacceptable  {fn:10d}  {tp:12d}"
    )


# ============================================================
# 6. FINAL CROSS-VALIDATION RESULTS
# ============================================================

print("\n")
print("=" * 55)
print("FINAL 5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 55)


print("\nAccuracy")
print("--------")
print(
    f"Mean: {np.mean(accuracy_scores):.4f}"
)
print(
    f"Std:  {np.std(accuracy_scores):.4f}"
)


print("\nPrecision - Unacceptable")
print("------------------------")
print(
    f"Mean: {np.mean(precision_scores):.4f}"
)
print(
    f"Std:  {np.std(precision_scores):.4f}"
)


print("\nRecall / Sensitivity - Unacceptable")
print("------------------------------------")
print(
    f"Mean: {np.mean(recall_scores):.4f}"
)
print(
    f"Std:  {np.std(recall_scores):.4f}"
)


print("\nSpecificity")
print("-----------")
print(
    f"Mean: {np.mean(specificity_scores):.4f}"
)
print(
    f"Std:  {np.std(specificity_scores):.4f}"
)


print("\nF1-score - Unacceptable")
print("-----------------------")
print(
    f"Mean: {np.mean(f1_scores):.4f}"
)
print(
    f"Std:  {np.std(f1_scores):.4f}"
)


# ============================================================
# 7. SUMMARY TABLE
# ============================================================

results = pd.DataFrame({
    "Fold": range(1, 6),
    "Accuracy": accuracy_scores,
    "Precision": precision_scores,
    "Sensitivity": recall_scores,
    "Specificity": specificity_scores,
    "F1": f1_scores
})

print("\n")
print("=" * 55)
print("FOLD-WISE SUMMARY")
print("=" * 55)

print(results.to_string(index=False))


print("\n")
print("=" * 55)
print("BASELINE COMPLETE")
print("=" * 55)