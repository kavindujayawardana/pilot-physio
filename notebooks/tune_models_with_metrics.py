#!/usr/bin/env python3
"""
Original-style tuning script, but also saves:
- Macro Precision
- Macro Recall
- Macro F1
- Balanced Accuracy

Outputs:
results/tuning_with_metrics/<feature_set>/
    leaderboard.csv
    best_models.json
    label_mapping.json
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    make_scorer,
    f1_score,
    recall_score,
    precision_score,
    balanced_accuracy_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


MODEL_READY = Path("results/model_ready")
OUT_ROOT = Path("results/tuning_with_metrics")


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


def load_feature_set(feature_set: str):
    if feature_set == "theory":
        x_path = MODEL_READY / "X_theory.parquet"
    elif feature_set == "all":
        x_path = MODEL_READY / "X_all.parquet"
    elif feature_set == "select":
        x_path = MODEL_READY / "X_select.parquet"
    else:
        raise ValueError("feature_set must be one of: theory, all, select")

    y_path = MODEL_READY / "y.parquet"
    g_path = MODEL_READY / "groups.parquet"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing file: {y_path}")
    if not g_path.exists():
        raise FileNotFoundError(f"Missing file: {g_path}")

    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)["EventLabel"].to_numpy()
    groups = pd.read_parquet(g_path)["SubjectID"].to_numpy()

    if not (len(X) == len(y) == len(groups)):
        raise ValueError(
            f"Length mismatch: X={len(X)}, y={len(y)}, groups={len(groups)}"
        )

    return X, y, groups


def encode_labels(y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    orig_to_enc = {int(orig): int(enc) for enc, orig in enumerate(le.classes_)}
    enc_to_orig = {int(enc): int(orig) for enc, orig in enumerate(le.classes_)}
    return y_enc, orig_to_enc, enc_to_orig


def run_search(
    name,
    estimator,
    param_distributions,
    X,
    y,
    groups,
    out_dir,
    n_iter=25,
    seed=42,
    fit_params=None,
):
    cv = GroupKFold(n_splits=5)
    fit_params = fit_params or {}

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=SCORERS,
        refit="macro_f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=seed,
        return_train_score=False,
        error_score=np.nan,
    )

    search.fit(X, y, groups=groups, **fit_params)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(out_dir / f"cv_results_{name}.csv", index=False)

    valid = cv_results.dropna(
        subset=[
            "mean_test_macro_precision",
            "mean_test_macro_recall",
            "mean_test_macro_f1",
            "mean_test_bal_acc",
        ]
    ).copy()

    if valid.empty:
        raise RuntimeError(f"All parameter combinations failed for {name}")

    valid = valid.sort_values(
        ["mean_test_macro_f1", "mean_test_macro_recall", "mean_test_bal_acc"],
        ascending=False,
    )
    best_row = valid.iloc[0]

    result = {
        "model": name,
        "best_macro_precision": float(best_row["mean_test_macro_precision"]),
        "best_macro_recall": float(best_row["mean_test_macro_recall"]),
        "best_macro_f1": float(best_row["mean_test_macro_f1"]),
        "best_bal_acc": float(best_row["mean_test_bal_acc"]),
        "best_params": search.best_params_,
    }

    print(
        f"[BEST] {name}: "
        f"precision={result['best_macro_precision']:.4f}, "
        f"recall={result['best_macro_recall']:.4f}, "
        f"macro_f1={result['best_macro_f1']:.4f}, "
        f"bal_acc={result['best_bal_acc']:.4f}"
    )
    print(result["best_params"])

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-set",
        choices=["theory", "all", "select"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", type=int, default=5)  # kept for compatibility
    args = parser.parse_args()

    out_dir = OUT_ROOT / args.feature_set
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y_orig, groups = load_feature_set(args.feature_set)
    y_enc, orig_to_enc, enc_to_orig = encode_labels(y_orig)

    print(
        f"[LOAD] Feature set={args.feature_set}  "
        f"X={X.shape}  y_orig={y_orig.shape}  groups={groups.shape}"
    )
    print("[LOAD] Label counts (orig):", pd.Series(y_orig).value_counts().sort_index().to_dict())
    print("[LOAD] Label counts (enc): ", pd.Series(y_enc).value_counts().sort_index().to_dict())
    print("[MAP]  orig_to_enc:", orig_to_enc)

    results = []

    # --------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=args.seed,
        ))
    ])

    logreg_params = {
        "clf__C": np.logspace(-3, 2, 20),
    }

    results.append(
        run_search(
            name="logreg",
            estimator=logreg,
            param_distributions=logreg_params,
            X=X,
            y=y_orig,
            groups=groups,
            out_dir=out_dir,
            n_iter=20,
            seed=args.seed,
        )
    )

    # --------------------------------------------------
    # Random Forest
    # --------------------------------------------------
    rf = Pipeline([
        ("clf", RandomForestClassifier(
            random_state=args.seed,
            n_jobs=-1,
        ))
    ])

    rf_params = {
        "clf__n_estimators": [100, 200, 300, 400, 500],
        "clf__max_depth": [5, 10, 14, 18, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__max_features": ["sqrt", "log2", 0.3, 0.5],
        "clf__class_weight": ["balanced", "balanced_subsample", None],
    }

    results.append(
        run_search(
            name="rf",
            estimator=rf,
            param_distributions=rf_params,
            X=X,
            y=y_orig,
            groups=groups,
            out_dir=out_dir,
            n_iter=25,
            seed=args.seed,
        )
    )

    # --------------------------------------------------
    # XGBoost
    # --------------------------------------------------
    xgb = Pipeline([
        ("clf", XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=args.seed,
            n_jobs=-1,
        ))
    ])

    xgb_params = {
        "clf__n_estimators": [100, 200, 300, 500, 700],
        "clf__max_depth": [2, 3, 4, 5, 6],
        "clf__learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1],
        "clf__subsample": [0.6, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.5, 0.7, 0.8, 1.0],
        "clf__min_child_weight": [1, 2, 5, 10],
        "clf__gamma": [0.0, 0.05, 0.1, 0.2],
        "clf__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    xgb_sample_weights = compute_sample_weight(class_weight="balanced", y=y_enc)

    results.append(
        run_search(
            name="xgb",
            estimator=xgb,
            param_distributions=xgb_params,
            X=X,
            y=y_enc,
            groups=groups,
            out_dir=out_dir,
            n_iter=25,
            seed=args.seed,
            fit_params={"clf__sample_weight": xgb_sample_weights},
        )
    )

    summary = pd.DataFrame(results).sort_values(
        ["best_macro_recall", "best_macro_f1", "best_bal_acc"],
        ascending=False,
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
        for row in results
    }

    (out_dir / "best_models.json").write_text(
        json.dumps(best_models, indent=2, default=str),
        encoding="utf-8",
    )

    label_mapping = {
        "orig_to_enc": orig_to_enc,
        "enc_to_orig": enc_to_orig,
    }
    (out_dir / "label_mapping.json").write_text(
        json.dumps(label_mapping, indent=2),
        encoding="utf-8",
    )

    print("\n[OK] Saved to:", out_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()