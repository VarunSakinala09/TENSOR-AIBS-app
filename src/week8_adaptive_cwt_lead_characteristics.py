
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]

PATH40 = ROOT / "results/week3/features_full_dataset.csv"
PATH50 = ROOT / "results/week7/cwt_d/cwt_d_features.csv"
PATH_LEAD = (
    ROOT / "results/week8/lead_quality_features"
    / "lead_quality_features.csv"
)
PATH_GROUPS = (
    ROOT / "results/week4/group_aware_cv"
    / "record_signal_groups.csv"
)

OUT = ROOT / "results/week8/adaptive_cwt_lead_characteristics"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
N_TREES = 100


def encode_labels(series):
    labels = (
        series.astype(str).str.strip().str.lower()
        .map({
            "acceptable": 0,
            "accept": 0,
            "0": 0,
            "unacceptable": 1,
            "unaccept": 1,
            "1": 1,
        })
    )

    if labels.isna().any():
        raise ValueError(
            f"Unknown labels found: "
            f"{series[labels.isna()].unique()}"
        )

    return labels.to_numpy(dtype=int)


def load_features(path):
    df = pd.read_csv(path, dtype={"record_id": str})
    df["record_id"] = df["record_id"].astype(str).str.strip()

    if df["record_id"].duplicated().any():
        raise ValueError(f"Duplicate record IDs in {path}")

    y = encode_labels(df["label"])
    cols = [c for c in df.columns if c not in ["record_id", "label"]]

    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        raise ValueError(f"Missing or invalid values in {path}")

    return df["record_id"].to_numpy(), X.to_numpy(dtype=np.float32), y


def make_model(seed):
    return RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )


def evaluate(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(
            y_true, pred, zero_division=0
        ),
        "sensitivity": recall_score(
            y_true, pred, pos_label=1, zero_division=0
        ),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y_true, pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_rule(y, pred40, pred50, values):
    """
    Learn whether to use 50 Hz or 40 Hz from training-only
    inner out-of-fold predictions.

    Both directions are tested:
      1. Use 50 Hz when characteristic <= threshold.
      2. Use 50 Hz when characteristic >= threshold.

    Selection priority: F1, sensitivity, specificity.
    """
    thresholds = np.unique(
        np.quantile(values, np.linspace(0.05, 0.95, 19))
    )

    best = None

    for threshold in thresholds:
        for direction in ["le", "ge"]:
            if direction == "le":
                choose50 = values <= threshold
            else:
                choose50 = values >= threshold

            pred = np.where(choose50, pred50, pred40)
            score = evaluate(y, pred)

            candidate = (
                score["f1"],
                score["sensitivity"],
                score["specificity"],
            )

            if best is None or candidate > best["score"]:
                best = {
                    "threshold": float(threshold),
                    "direction": direction,
                    "score": candidate,
                    "inner_metrics": score,
                }

    return best


def apply_rule(pred40, pred50, values, rule):
    if rule["direction"] == "le":
        choose50 = values <= rule["threshold"]
    else:
        choose50 = values >= rule["threshold"]

    return np.where(choose50, pred50, pred40).astype(int)


def main():
    print("\nADAPTIVE CWT USING LEAD CHARACTERISTICS")
    print("=" * 65)

    ids40, X40, y40 = load_features(PATH40)
    ids50, X50, y50 = load_features(PATH50)
    idslead, Xlead, ylead = load_features(PATH_LEAD)

    def index_by_id(ids):
        return {record_id: i for i, record_id in enumerate(ids)}

    map40 = index_by_id(ids40)
    map50 = index_by_id(ids50)
    maplead = index_by_id(idslead)

    groups_df = pd.read_csv(
        PATH_GROUPS,
        dtype={"record_id": str, "signal_hash": str},
    )
    groups_df["record_id"] = (
        groups_df["record_id"].astype(str).str.strip()
    )

    if groups_df["record_id"].duplicated().any():
        raise ValueError("Duplicate IDs in group mapping")

    groups_df = groups_df.set_index("record_id")

    common = sorted(
        set(map40)
        .intersection(map50)
        .intersection(maplead)
        .intersection(groups_df.index)
    )

    idx40 = np.array([map40[r] for r in common])
    idx50 = np.array([map50[r] for r in common])
    idxlead = np.array([maplead[r] for r in common])

    X40 = X40[idx40]
    X50 = X50[idx50]
    Xlead = Xlead[idxlead]

    y = y40[idx40]

    if not np.array_equal(y, y50[idx50]):
        raise ValueError("40 Hz and 50 Hz labels differ")

    if not np.array_equal(y, ylead[idxlead]):
        raise ValueError("Lead-quality labels differ")

    group_labels = encode_labels(groups_df.loc[common, "label"])
    if not np.array_equal(y, group_labels):
        raise ValueError("Group mapping labels differ")

    groups = (
        groups_df.loc[common, "signal_hash"]
        .astype(str).to_numpy()
    )

    # These characteristics are used only to select CWT settings.
    lead_df = pd.read_csv(
        PATH_LEAD, dtype={"record_id": str}
    )
    lead_df["record_id"] = (
        lead_df["record_id"].astype(str).str.strip()
    )
    lead_df = lead_df.set_index("record_id").loc[common]

    characteristics = {
        "mean_lead_std": pd.to_numeric(
            lead_df["mean_lead_std"], errors="coerce"
        ).to_numpy(dtype=float),
        "mean_lead_range": pd.to_numeric(
            lead_df["mean_lead_range"], errors="coerce"
        ).to_numpy(dtype=float),
    }

    for name, values in characteristics.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Invalid values in {name}")

    print("Records:", len(y))
    print("Unique signal groups:", len(np.unique(groups)))
    print("40 Hz feature count:", X40.shape[1])
    print("50 Hz feature count:", X50.shape[1])
    print("Lead-quality feature count:", Xlead.shape[1])

    outer_cv = StratifiedGroupKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    fold_rows = []
    prediction_rows = []
    rule_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X40, y, groups), start=1
    ):
        print(f"\nOuter fold {fold}/{OUTER_FOLDS}")

        if set(groups[train_idx]).intersection(groups[test_idx]):
            raise RuntimeError("Signal-group leakage detected")

        y_train = y[train_idx]
        y_test = y[test_idx]

        # Generate inner out-of-fold predictions using only
        # the outer training records.
        inner40 = np.full(len(train_idx), -1, dtype=int)
        inner50 = np.full(len(train_idx), -1, dtype=int)

        inner_cv = StratifiedGroupKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=SEED + fold,
        )

        for inner_train, inner_valid in inner_cv.split(
            X40[train_idx],
            y_train,
            groups[train_idx],
        ):
            global_train = train_idx[inner_train]
            global_valid = train_idx[inner_valid]

            model40 = make_model(SEED + fold)
            model40.fit(X40[global_train], y[global_train])
            inner40[inner_valid] = model40.predict(
                X40[global_valid]
            )

            model50 = make_model(SEED + fold)
            model50.fit(X50[global_train], y[global_train])
            inner50[inner_valid] = model50.predict(
                X50[global_valid]
            )

        if np.any(inner40 < 0) or np.any(inner50 < 0):
            raise RuntimeError("Incomplete inner OOF predictions")

        # Learn the selection rule using outer-training data only.
        learned_rules = {}

        for characteristic, values in characteristics.items():
            rule = select_rule(
                y_train,
                inner40,
                inner50,
                values[train_idx],
            )
            learned_rules[characteristic] = rule

            rule_rows.append({
                "fold": fold,
                "characteristic": characteristic,
                "threshold": rule["threshold"],
                "direction": rule["direction"],
                "inner_f1": rule["inner_metrics"]["f1"],
                "inner_sensitivity": (
                    rule["inner_metrics"]["sensitivity"]
                ),
                "inner_specificity": (
                    rule["inner_metrics"]["specificity"]
                ),
            })

            print(
                f"  {characteristic}: use 50 Hz when "
                f"value {rule['direction']} "
                f"{rule['threshold']:.6f}"
            )

        # Train final fold models on the complete outer-training set.
        final40 = make_model(SEED + fold)
        final40.fit(X40[train_idx], y_train)
        pred40 = final40.predict(X40[test_idx]).astype(int)

        final50 = make_model(SEED + fold)
        final50.fit(X50[train_idx], y_train)
        pred50 = final50.predict(X50[test_idx]).astype(int)

        final_lead = make_model(SEED + fold)
        final_lead.fit(Xlead[train_idx], y_train)
        pred_lead = final_lead.predict(Xlead[test_idx]).astype(int)

        predictions = {
            "fixed40": pred40,
            "fixed50": pred50,
            "lead_quality_only": pred_lead,
        }

        for characteristic, rule in learned_rules.items():
            predictions[f"adaptive_{characteristic}"] = apply_rule(
                pred40,
                pred50,
                characteristics[characteristic][test_idx],
                rule,
            )

        for method, pred in predictions.items():
            score = evaluate(y_test, pred)

            fold_rows.append({
                "fold": fold,
                "method": method,
                **score,
            })

            for j, global_idx in enumerate(test_idx):
                prediction_rows.append({
                    "record_id": common[global_idx],
                    "signal_hash": groups[global_idx],
                    "fold": fold,
                    "method": method,
                    "true_label": int(y[global_idx]),
                    "prediction": int(pred[j]),
                })

            print(
                f"  {method}: F1={score['f1']:.4f}, "
                f"Sensitivity={score['sensitivity']:.4f}, "
                f"Specificity={score['specificity']:.4f}"
            )

    folds_df = pd.DataFrame(fold_rows)
    preds_df = pd.DataFrame(prediction_rows)
    rules_df = pd.DataFrame(rule_rows)

    folds_df.to_csv(OUT / "fold_metrics.csv", index=False)
    preds_df.to_csv(OUT / "out_of_fold_predictions.csv", index=False)
    rules_df.to_csv(OUT / "learned_rules.csv", index=False)

    summary = (
        folds_df.groupby("method")[
            ["accuracy", "precision", "sensitivity",
             "specificity", "f1"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.to_csv(OUT / "fold_summary.csv")

    pooled_rows = []
    for method, part in preds_df.groupby("method"):
        pooled_rows.append({
            "method": method,
            **evaluate(
                part["true_label"].to_numpy(),
                part["prediction"].to_numpy(),
            ),
        })

    pooled = pd.DataFrame(pooled_rows).round(4)
    pooled.to_csv(OUT / "pooled_oof_metrics.csv", index=False)

    print("\nFOLD SUMMARY")
    print("=" * 75)
    print(summary.to_string())

    print("\nPOOLED OUT-OF-FOLD METRICS")
    print("=" * 75)
    print(pooled.to_string(index=False))

    print("\nLEARNED RULES")
    print(rules_df.to_string(index=False))

    print("\nSaved results to:", OUT)


if __name__ == "__main__":
    main()
