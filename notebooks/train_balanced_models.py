#!/usr/bin/env python3
"""
Train imbalance-aware models for one feature set.

Models:
- weighted logistic regression
- RF baseline
- RF undersample
- RF SMOTE
- XGB baseline
- XGB undersample
- XGB SMOTE

Metrics:
- macro precision
- macro recall
- macro F1
- balanced accuracy

Outputs:
  results/balanced_models/<feature_set>/
    metrics_summary.csv
    confusion_<model_variant>.csv
    confusion_<model_variant>.png
    feature_importance_<model_variant>.csv
    feature_importance_<model_variant>.png
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


MODEL_READY = Path("results/model_ready")
OUT_ROOT = Path("results/balanced_models")


def load_X(feature_set: str) -> pd.DataFrame:
    if feature_set == "theory":
        return pd.read_parquet(MODEL_READY / "X_theory.parquet")
    elif feature_set == "all":
        return pd.read_parquet(MODEL_READY / "X_all.parquet")
    elif feature_set == "select":
        return pd.read_parquet(MODEL_READY / "X_select.parquet")
    else:
        raise ValueError("feature_set must be theory, all, or select")


def save_confusion_matrix(cm, labels, out_csv, out_png, title):
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(out_csv)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def save_feature_importance(model, feature_names, out_csv, out_png, title, kind="rf"):
    if kind == "rf":
        imp = model.feature_importances_
    elif kind == "xgb":
        booster = model.get_booster()
        gain = booster.get_score(importance_type="gain")
        imp = np.array([gain.get(f, 0.0) for f in feature_names])
    else:
        return

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": imp
    }).sort_values("importance", ascending=False)

    df.to_csv(out_csv, index=False)

    top = df.head(15).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def run_cv_model(X, y, groups, model_variant, out_dir):
    gkf = GroupKFold(n_splits=5)

    labels_orig = sorted(pd.unique(y))
    all_true = []
    all_pred = []

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    first_fold_done = False

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)

        if model_variant == "logreg_weighted":
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42
                ))
            ])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        elif model_variant in {"rf_original", "rf_undersample", "rf_smote"}:
            X_fit = X_train.copy()
            y_fit = y_train.copy()

            if model_variant == "rf_undersample":
                rus = RandomUnderSampler(random_state=42)
                X_fit, y_fit = rus.fit_resample(X_fit, y_fit)

            elif model_variant == "rf_smote":
                sm = SMOTE(random_state=42)
                X_fit, y_fit = sm.fit_resample(X_fit, y_fit)

            clf = RandomForestClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=8,
                max_features="sqrt",
                class_weight="balanced" if model_variant == "rf_original" else None,
                random_state=42,
                n_jobs=-1
            )
            clf.fit(X_fit, y_fit)
            y_pred = clf.predict(X_test)

            if not first_fold_done:
                save_feature_importance(
                    clf,
                    list(X.columns),
                    out_dir / f"feature_importance_{model_variant}.csv",
                    out_dir / f"feature_importance_{model_variant}.png",
                    f"{model_variant} feature importance",
                    kind="rf"
                )

        elif model_variant in {"xgb_original", "xgb_undersample", "xgb_smote"}:
            X_fit = X_train.copy()
            y_fit_enc = y_train_enc.copy()

            if model_variant == "xgb_undersample":
                rus = RandomUnderSampler(random_state=42)
                X_fit, y_fit_enc = rus.fit_resample(X_fit, y_fit_enc)

            elif model_variant == "xgb_smote":
                sm = SMOTE(random_state=42)
                X_fit, y_fit_enc = sm.fit_resample(X_fit, y_fit_enc)

            clf = XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=42,
                subsample=0.6,
                reg_lambda=2.0,
                min_child_weight=5,
                max_depth=3,
                learning_rate=0.01,
                gamma=0.2,
                colsample_bytree=0.7
            )

            if model_variant == "xgb_original":
                sw = compute_sample_weight(class_weight="balanced", y=y_train_enc)
                clf.fit(X_fit, y_fit_enc, sample_weight=sw)
            else:
                clf.fit(X_fit, y_fit_enc)

            y_pred_enc = clf.predict(X_test)
            y_pred = le.inverse_transform(y_pred_enc)

            if not first_fold_done:
                save_feature_importance(
                    clf,
                    list(X.columns),
                    out_dir / f"feature_importance_{model_variant}.csv",
                    out_dir / f"feature_importance_{model_variant}.png",
                    f"{model_variant} feature importance",
                    kind="xgb"
                )

        else:
            raise ValueError(f"Unknown model variant: {model_variant}")

        first_fold_done = True
        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

    metrics = {
        "Model Variant": model_variant,
        "Macro Precision": precision_score(all_true, all_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(all_true, all_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(all_true, all_pred, average="macro", zero_division=0),
        "Balanced Acc": balanced_accuracy_score(all_true, all_pred),
    }

    cm = confusion_matrix(all_true, all_pred, labels=labels_orig)
    save_confusion_matrix(
        cm,
        labels_orig,
        out_dir / f"confusion_{model_variant}.csv",
        out_dir / f"confusion_{model_variant}.png",
        f"Confusion Matrix - {model_variant}"
    )

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=["theory", "all", "select"], required=True)
    args = parser.parse_args()

    out_dir = OUT_ROOT / args.feature_set
    out_dir.mkdir(parents=True, exist_ok=True)

    X = load_X(args.feature_set)
    y = pd.read_parquet(MODEL_READY / "y.parquet")["EventLabel"]
    groups = pd.read_parquet(MODEL_READY / "groups.parquet")["SubjectID"]

    print("[LOAD]", args.feature_set, X.shape, y.shape, groups.shape)
    print(y.value_counts().sort_index())

    model_variants = [
        "logreg_weighted",
        "rf_original",
        "rf_undersample",
        "rf_smote",
        "xgb_original",
        "xgb_undersample",
        "xgb_smote",
    ]

    rows = []
    for mv in model_variants:
        print(f"\n[RUN] {mv}")
        metrics = run_cv_model(X, y, groups, mv, out_dir)
        rows.append(metrics)
        print(metrics)

    summary = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)

    print("\nSaved:", out_dir / "metrics_summary.csv")


if __name__ == "__main__":
    main()