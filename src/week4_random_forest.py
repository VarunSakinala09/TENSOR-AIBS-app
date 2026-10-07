import os

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==================================================
# PROJECT PATHS
# ==================================================

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

feature_file = os.path.join(
    project_root,
    "results",
    "week3",
    "features_full_dataset.csv"
)


# ==================================================
# LOAD FEATURE DATASET
# ==================================================

data = pd.read_csv(feature_file)

print("\nDATASET")
print("-------")
print("Dataset shape:", data.shape)


# ==================================================
# SEPARATE FEATURES AND LABEL
# ==================================================

feature_columns = [
    column
    for column in data.columns
    if column.startswith("feature_")
]

X = data[feature_columns]

y = data["label"]


print("\nFEATURES")
print("--------")
print("Feature matrix shape:", X.shape)

print("\nLABELS")
print("------")
print("Number of labels:", y.shape[0])
print("\nClass distribution:")
print(y.value_counts())


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTRAIN / TEST SPLIT")
print("------------------")

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==================================================
# RANDOM FOREST CLASSIFIER
# ==================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


print("\nTRAINING RANDOM FOREST...")
print("-------------------------")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ==================================================
# PREDICTION
# ==================================================

y_pred = model.predict(X_test)


# ==================================================
# EVALUATION
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="Unacceptable"
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="Unacceptable"
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="Unacceptable"
)


# ==================================================
# CONFUSION MATRIX
# ==================================================

labels = ["Acceptable", "Unacceptable"]

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

true_acceptable = matrix[0, 0]
false_unacceptable = matrix[0, 1]

false_acceptable = matrix[1, 0]
true_unacceptable = matrix[1, 1]


specificity = (
    true_acceptable
    / (true_acceptable + false_unacceptable)
)


# ==================================================
# RESULTS
# ==================================================

print("\n" + "=" * 50)
print("RANDOM FOREST RESULTS")
print("=" * 50)

print(
    f"Accuracy:    {accuracy:.4f}"
)

print(
    f"Precision:   {precision:.4f}"
)

print(
    f"Recall:      {recall:.4f}"
)

print(
    f"Specificity: {specificity:.4f}"
)

print(
    f"F1-score:    {f1:.4f}"
)


# ==================================================
# CONFUSION MATRIX
# ==================================================

print("\nCONFUSION MATRIX")
print("----------------")

print("Rows = Actual")
print("Columns = Predicted")

print(
    pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )
)


# ==================================================
# CLASSIFICATION REPORT
# ==================================================

print("\nCLASSIFICATION REPORT")
print("---------------------")

print(
    classification_report(
        y_test,
        y_pred
    )
)