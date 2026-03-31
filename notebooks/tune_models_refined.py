#!/usr/bin/env python3
"""
Refined tuning using Optuna for:
- weighted Logistic Regression
- Random Forest
- Random Forest + SMOTE
- Random Forest + undersampling
- XGBoost

Uses:
- GroupKFold (5 folds)
- Primary optimization target: macro recall
- Also records macro precision, macro F1, balanced accuracy

Outputs:
results/tuning_refined/<feature_set>/
    leaderboard.csv
    best_models.json
    study_<model>.csv
    label_mapping.json
"""

from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    balanced_accuracy_score,
    make_scorer,
)
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBClassifier


MODEL_READY = Path("results/model_ready")
OUT_ROOT = Path("results/tuning_refined")

warnings.filterwarnings("ignore")

macro_f1 = make_scorer(f1_score, average="macro")
macro_recall = make_scorer(recall_score, average="macro", zero_division=0)
macro_precision = make_scorer(precision_score, average="macro", zero_division=0)
bal_acc = make_scorer(balanced_accuracy_score)

SCORERS = {
    "macro_f1": macro_f1,
    "macro_recall": macro_recall,
    "macro_precision": macro_precision,
    "bal_acc": bal_acc,
}


def load_X(feature_set: str):
    if feature_set == "theory":
        x_path = MODEL_READY / "X_theory.parquet"
    elif feature_set == "all":
        x_path = MODEL_READY / "X_all.parquet"
    elif feature_set == "select":
        x_path = MODEL_READY / "X_select.parquet"
    else:
        raise ValueError("feature_set must be theory, all, or select")

    y_path = MODEL_READY / "y.parquet"
    g_path = MODEL_READY / "groups.parquet"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing: {y_path}")
    if not g_path.exists():
        raise FileNotFoundError(f"Missing: {g_path}")

    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)["EventLabel"].to_numpy()
    groups = pd.read_parquet(g_path)["SubjectID"].to_numpy()
    return X, y, groups


def encode_labels(y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    orig_to_enc = {int(orig): int(enc) for enc, orig in enumerate(le.classes_)}
    enc_to_orig = {int(enc): int(orig) for enc, orig in enumerate(le.classes_)}
    return y_enc, orig_to_enc, enc_to_orig


def evaluate_estimator(estimator, X, y, groups, fit_params=None):
    cv = GroupKFold(n_splits=5)
    fit_params = fit_params or {}

    scores = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        groups=groups,
        cv=cv,
        scoring=SCORERS,
        n_jobs=-1,
        params=fit_params,
        return_train_score=False,
        error_score=np.nan,
    )

    out = {
        "macro_precision": float(np.nanmean(scores["test_macro_precision"])),
        "macro_recall": float(np.nanmean(scores["test_macro_recall"])),
        "macro_f1": float(np.nanmean(scores["test_macro_f1"])),
        "bal_acc": float(np.nanmean(scores["test_bal_acc"])),
    }

    if any(np.isnan(v) for v in out.values()):
        raise RuntimeError("Evaluation returned NaN metrics.")

    return out


def optimize_logreg(X, y, groups, n_trials=30, seed=42):
    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 1e2, log=True)
        cw_option = trial.suggest_categorical("class_weight_option", ["balanced", "w3", "w4", "w5"])

        if cw_option == "balanced":
            class_weight = "balanced"
        elif cw_option == "w3":
            class_weight = {0: 1.0, 1: 3.0, 2: 1.0, 5: 3.0}
        elif cw_option == "w4":
            class_weight = {0: 1.0, 1: 4.0, 2: 1.0, 5: 4.0}
        else:
            class_weight = {0: 1.0, 1: 5.0, 2: 1.0, 5: 5.0}

        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=6000,
                solver="lbfgs",
                random_state=seed,
                C=C,
                class_weight=class_weight,
            ))
        ])

        scores = evaluate_estimator(estimator, X, y, groups)
        trial.set_user_attr("macro_f1", scores["macro_f1"])
        trial.set_user_attr("macro_precision", scores["macro_precision"])
        trial.set_user_attr("bal_acc", scores["bal_acc"])
        return scores["macro_recall"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best = study.best_trial
    attrs = best.user_attrs

    return study, {
        "model": "logreg_optuna",
        "best_macro_precision": attrs["macro_precision"],
        "best_macro_recall": best.value,
        "best_macro_f1": attrs["macro_f1"],
        "best_bal_acc": attrs["bal_acc"],
        "best_params": best.params,
    }


def optimize_rf(X, y, groups, n_trials=40, seed=42):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 300, 1200, step=100)
        max_depth = trial.suggest_categorical("max_depth", [10, 14, 18, 24, None])
        min_samples_split = trial.suggest_categorical("min_samples_split", [2, 5, 10])
        min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [1, 2, 4, 8])
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5])
        cw_option = trial.suggest_categorical(
            "class_weight_option",
            ["balanced", "balanced_subsample", "w3", "w4"]
        )

        if cw_option == "balanced":
            class_weight = "balanced"
        elif cw_option == "balanced_subsample":
            class_weight = "balanced_subsample"
        elif cw_option == "w3":
            class_weight = {0: 1.0, 1: 3.0, 2: 1.0, 5: 3.0}
        else:
            class_weight = {0: 1.0, 1: 4.0, 2: 1.0, 5: 4.0}

        estimator = Pipeline([
            ("clf", RandomForestClassifier(
                random_state=seed,
                n_jobs=-1,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight=class_weight,
            ))
        ])

        scores = evaluate_estimator(estimator, X, y, groups)
        trial.set_user_attr("macro_f1", scores["macro_f1"])
        trial.set_user_attr("macro_precision", scores["macro_precision"])
        trial.set_user_attr("bal_acc", scores["bal_acc"])
        return scores["macro_recall"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best = study.best_trial
    attrs = best.user_attrs

    return study, {
        "model": "rf_optuna",
        "best_macro_precision": attrs["macro_precision"],
        "best_macro_recall": best.value,
        "best_macro_f1": attrs["macro_f1"],
        "best_bal_acc": attrs["bal_acc"],
        "best_params": best.params,
    }


def optimize_rf_smote(X, y, groups, n_trials=30, seed=42):
    def objective(trial):
        sampling_strategy = trial.suggest_categorical(
            "sampling_strategy",
            ["auto", "not majority"]
        )
        n_estimators = trial.suggest_int("n_estimators", 300, 900, step=100)
        max_depth = trial.suggest_categorical("max_depth", [10, 14, 18, 24])
        min_samples_split = trial.suggest_categorical("min_samples_split", [2, 5, 10])
        min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [1, 2, 4, 8])
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5])

        estimator = ImbPipeline([
            ("smote", SMOTE(random_state=seed, sampling_strategy=sampling_strategy)),
            ("clf", RandomForestClassifier(
                random_state=seed,
                n_jobs=-1,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
            ))
        ])

        scores = evaluate_estimator(estimator, X, y, groups)
        trial.set_user_attr("macro_f1", scores["macro_f1"])
        trial.set_user_attr("macro_precision", scores["macro_precision"])
        trial.set_user_attr("bal_acc", scores["bal_acc"])
        return scores["macro_recall"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best = study.best_trial
    attrs = best.user_attrs

    return study, {
        "model": "rf_smote_optuna",
        "best_macro_precision": attrs["macro_precision"],
        "best_macro_recall": best.value,
        "best_macro_f1": attrs["macro_f1"],
        "best_bal_acc": attrs["bal_acc"],
        "best_params": best.params,
    }


def optimize_rf_under(X, y, groups, n_trials=25, seed=42):
    def objective(trial):
        sampling_strategy = trial.suggest_categorical(
            "sampling_strategy",
            ["auto", "not minority", "majority"]
        )
        n_estimators = trial.suggest_int("n_estimators", 300, 900, step=100)
        max_depth = trial.suggest_categorical("max_depth", [10, 14, 18, 24])
        min_samples_split = trial.suggest_categorical("min_samples_split", [2, 5, 10])
        min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [1, 2, 4, 8])
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5])

        estimator = ImbPipeline([
            ("under", RandomUnderSampler(random_state=seed, sampling_strategy=sampling_strategy)),
            ("clf", RandomForestClassifier(
                random_state=seed,
                n_jobs=-1,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
            ))
        ])

        scores = evaluate_estimator(estimator, X, y, groups)
        trial.set_user_attr("macro_f1", scores["macro_f1"])
        trial.set_user_attr("macro_precision", scores["macro_precision"])
        trial.set_user_attr("bal_acc", scores["bal_acc"])
        return scores["macro_recall"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best = study.best_trial
    attrs = best.user_attrs

    return study, {
        "model": "rf_undersample_optuna",
        "best_macro_precision": attrs["macro_precision"],
        "best_macro_recall": best.value,
        "best_macro_f1": attrs["macro_f1"],
        "best_bal_acc": attrs["bal_acc"],
        "best_params": best.params,
    }


def optimize_xgb(X, y_orig, groups, n_trials=40, seed=42):
    y_enc, _, _ = encode_labels(y_orig)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_enc)

    def objective(trial):
        estimator = Pipeline([
            ("clf", XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=seed,
                n_jobs=-1,
                n_estimators=trial.suggest_int("n_estimators", 300, 1300, step=100),
                learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                max_depth=trial.suggest_int("max_depth", 2, 6),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                min_child_weight=trial.suggest_categorical("min_child_weight", [1, 2, 5, 10]),
                gamma=trial.suggest_float("gamma", 0.0, 0.2),
                reg_lambda=trial.suggest_categorical("reg_lambda", [0.5, 1.0, 2.0, 5.0]),
            ))
        ])

        scores = evaluate_estimator(
            estimator,
            X,
            y_enc,
            groups,
            fit_params={"clf__sample_weight": sample_weights},
        )
        trial.set_user_attr("macro_f1", scores["macro_f1"])
        trial.set_user_attr("macro_precision", scores["macro_precision"])
        trial.set_user_attr("bal_acc", scores["bal_acc"])
        return scores["macro_recall"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best = study.best_trial
    attrs = best.user_attrs

    return study, {
        "model": "xgb_optuna",
        "best_macro_precision": attrs["macro_precision"],
        "best_macro_recall": best.value,
        "best_macro_f1": attrs["macro_f1"],
        "best_bal_acc": attrs["bal_acc"],
        "best_params": best.params,
    }


def save_study(study, out_path):
    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    df.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=["theory", "all", "select"], required=True)
    args = parser.parse_args()

    out_dir = OUT_ROOT / args.feature_set
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups = load_X(args.feature_set)
    _, orig_to_enc, enc_to_orig = encode_labels(y)

    print(f"[LOAD] Feature set={args.feature_set} X={X.shape} y={y.shape} groups={groups.shape}")
    print("[LOAD] Class counts:", pd.Series(y).value_counts().sort_index().to_dict())

    studies_and_results = []

    tasks = [
        ("logreg_optuna", optimize_logreg, "study_logreg_optuna.csv"),
        ("rf_optuna", optimize_rf, "study_rf_optuna.csv"),
        ("rf_smote_optuna", optimize_rf_smote, "study_rf_smote_optuna.csv"),
        ("rf_undersample_optuna", optimize_rf_under, "study_rf_undersample_optuna.csv"),
        ("xgb_optuna", optimize_xgb, "study_xgb_optuna.csv"),
    ]

    for name, fn, csv_name in tasks:
        try:
            print(f"\n[OPTIMIZE] {name}")
            study, result = fn(X, y, groups)
            save_study(study, out_dir / csv_name)
            studies_and_results.append(result)
            print(
                f"[BEST] {result['model']}: "
                f"precision={result['best_macro_precision']:.4f}, "
                f"recall={result['best_macro_recall']:.4f}, "
                f"macro_f1={result['best_macro_f1']:.4f}, "
                f"bal_acc={result['best_bal_acc']:.4f}"
            )
            print(result["best_params"])
        except Exception as e:
            print(f"[WARN] {name} failed: {e}")

    if not studies_and_results:
        raise SystemExit("All optimization tasks failed.")

    summary = pd.DataFrame(studies_and_results).sort_values(
        ["best_macro_recall", "best_macro_f1", "best_bal_acc"],
        ascending=False
    )
    summary.to_csv(out_dir / "leaderboard.csv", index=False)

    best_models = {
        row["model"]: {
            "best_macro_precision": row["best_macro_precision"],
            "best_macro_recall": row["best_macro_recall"],
            "best_macro_f1": row["best_macro_f1"],
            "best_bal_acc": row["best_bal_acc"],
            "best_params": row["best_params"],
        }
        for row in studies_and_results
    }
    (out_dir / "best_models.json").write_text(
        json.dumps(best_models, indent=2, default=str),
        encoding="utf-8"
    )

    label_mapping = {
        "orig_to_enc": orig_to_enc,
        "enc_to_orig": enc_to_orig,
    }
    (out_dir / "label_mapping.json").write_text(
        json.dumps(label_mapping, indent=2),
        encoding="utf-8"
    )

    print("\nSaved Optuna tuning results to:", out_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()