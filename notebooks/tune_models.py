#!/usr/bin/env python3
"""
tune_models.py

Hyperparameter tuning (GroupKFold) for multiple models over a chosen feature set.

Feature sets supported:
  - theory  -> results/model_ready/X_theory.parquet
  - all     -> results/model_ready/X_all.parquet
  - select  -> results/model_ready/X_select.parquet

Targets/groups expected in:
  - results/model_ready/y.parquet        (column: EventLabel)
  - results/model_ready/groups.parquet   (column: SubjectID)

Outputs (per feature set) written to:
  results/tuning/<feature_set>/
    - leaderboard.csv
    - best_models.json
    - label_mapping.json

Notes:
- Handles class imbalance via class weights:
    * LogisticRegression: class_weight='balanced'
    * RandomForest: class_weight='balanced'
    * XGBoost: sample_weight using sklearn's compute_sample_weight
    * CatBoost (optional): class_weights list (aligned to encoded classes)
- XGBoost expects classes 0..K-1, so we label-encode y for XGB (and CatBoost for consistency).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import make_scorer, f1_score, balanced_accuracy_score

# Optional models
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False


# -----------------------------
# Scoring
# -----------------------------
macro_f1_scorer = make_scorer(f1_score, average="macro")
bal_acc_scorer = make_scorer(balanced_accuracy_score)

SCORERS = {
    "macro_f1": macro_f1_scorer,
    "bal_acc": bal_acc_scorer,
}


# -----------------------------
# Helpers
# -----------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_model_ready(model_ready: Path, feature_set: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load X, y, groups for a given feature_set."""
    if feature_set == "theory":
        x_path = model_ready / "X_theory.parquet"
    elif feature_set == "all":
        x_path = model_ready / "X_all.parquet"
    elif feature_set == "select":
        x_path = model_ready / "X_select.parquet"
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    y_path = model_ready / "y.parquet"
    g_path = model_ready / "groups.parquet"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing X file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing y file: {y_path}")
    if not g_path.exists():
        raise FileNotFoundError(f"Missing groups file: {g_path}")

    X = pd.read_parquet(x_path)
    y_df = pd.read_parquet(y_path)
    g_df = pd.read_parquet(g_path)

    if "EventLabel" not in y_df.columns:
        raise ValueError("y.parquet must contain column 'EventLabel'")
    if "SubjectID" not in g_df.columns:
        raise ValueError("groups.parquet must contain column 'SubjectID'")

    y = y_df["EventLabel"].to_numpy()
    groups = g_df["SubjectID"].to_numpy()

    if len(X) != len(y) or len(y) != len(groups):
        raise ValueError(
            f"Row mismatch: X={len(X)}, y={len(y)}, groups={len(groups)}"
        )

    return X, y, groups


def summarize_counts(y: np.ndarray) -> Dict[Any, int]:
    vals, cnts = np.unique(y, return_counts=True)
    return {int(v) if np.issubdtype(type(v), np.integer) or str(v).isdigit() else v: int(c) for v, c in zip(vals, cnts)}


def build_label_mapping(y_orig: np.ndarray) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
    """
    Encode labels to 0..K-1 and return:
      y_enc,
      orig_to_enc,
      enc_to_orig
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y_orig)

    # LabelEncoder stores classes_ sorted; map originals -> encoded indices
    classes = le.classes_
    orig_to_enc = {int(orig): int(enc) for enc, orig in enumerate(classes)}
    enc_to_orig = {int(enc): int(orig) for enc, orig in enumerate(classes)}
    return y_enc, orig_to_enc, enc_to_orig


@dataclass
class ModelSpec:
    name: str
    estimator: Any
    param_distributions: Dict[str, Any]
    needs_scaling: bool
    use_encoded_y: bool  # True for models that require 0..K-1


def build_models_and_spaces(random_state: int = 42) -> List[ModelSpec]:
    """
    Define models + random search spaces.
    Keep spaces reasonably small for MSc timeline.
    """
    specs: List[ModelSpec] = []

    # Logistic Regression (multinomial by default if solver supports it)
    # class_weight='balanced' for imbalance
    logreg = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=None,  # lbfgs ignores n_jobs; keep None for compatibility
        random_state=random_state,
    )
    # NOTE: sklearn 1.8+ deprecates setting 'penalty' explicitly for non-elasticnet;
    # we tune only C and solver here to avoid warnings.
    specs.append(
        ModelSpec(
            name="logreg",
            estimator=logreg,
            param_distributions={
                "clf__C": np.logspace(-4, 2, 20),
                # if you want to try saga for potential l1/elasticnet later:
                # "clf__solver": ["lbfgs", "saga"]
                "clf__solver": ["lbfgs"],
            },
            needs_scaling=True,
            use_encoded_y=False,
        )
    )

    # Random Forest
    rf = RandomForestClassifier(
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    specs.append(
        ModelSpec(
            name="rf",
            estimator=rf,
            param_distributions={
                "clf__n_estimators": [200, 400, 800],
                "clf__max_depth": [None, 6, 10, 14, 18],
                "clf__max_features": [0.2, 0.3, 0.5, "sqrt"],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4, 8],
            },
            needs_scaling=False,
            use_encoded_y=False,
        )
    )

    # XGBoost (optional)
    if _HAS_XGB:
        xgb = XGBClassifier(
            objective="multi:softprob",
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
        specs.append(
            ModelSpec(
                name="xgb",
                estimator=xgb,
                param_distributions={
                    "clf__max_depth": [2, 3, 4, 5],
                    "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
                    "clf__subsample": [0.6, 0.8, 0.9, 1.0],
                    "clf__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
                    "clf__min_child_weight": [1, 2, 5, 10],
                    "clf__gamma": [0.0, 0.05, 0.1, 0.2],
                    "clf__reg_lambda": [0.5, 1.0, 2.0, 5.0],
                    # n_estimators intentionally omitted (use low lr + defaults)
                },
                needs_scaling=False,
                use_encoded_y=True,  # must be 0..K-1
            )
        )
    else:
        print("[WARN] xgboost not installed; skipping XGBClassifier.")

    # CatBoost (optional)
    if _HAS_CATBOOST:
        # We'll pass class_weights at fit-time (depends on label encoding).
        cb = CatBoostClassifier(
            loss_function="MultiClass",
            random_seed=random_state,
            verbose=False,
        )
        specs.append(
            ModelSpec(
                name="catboost",
                estimator=cb,
                param_distributions={
                    "clf__depth": [3, 4, 5, 6, 7],
                    "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
                    "clf__l2_leaf_reg": [1, 3, 5, 7, 10],
                    "clf__iterations": [300, 600, 1000],
                },
                needs_scaling=False,
                use_encoded_y=True,  # for consistent class_weights handling
            )
        )
    else:
        print("[WARN] catboost not installed; skipping CatBoostClassifier.")

    return specs


def make_pipeline(spec: ModelSpec) -> Pipeline:
    steps = []
    if spec.needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", spec.estimator))
    return Pipeline(steps)


def fit_params_for_model(
    spec: ModelSpec,
    y_used: np.ndarray,
    y_enc: np.ndarray,
    orig_to_enc: Dict[int, int],
) -> Dict[str, Any]:
    """
    Return fit_params for RandomizedSearchCV.fit() depending on model.
    We handle class imbalance primarily through:
      - class_weight='balanced' inside logreg/rf
      - sample_weight for xgb
      - class_weights for catboost
    """
    fit_params: Dict[str, Any] = {}

    if spec.name == "xgb":
        # XGB expects encoded labels; supply per-sample weights for imbalance
        w = compute_sample_weight(class_weight="balanced", y=y_enc)
        fit_params["clf__sample_weight"] = w

    if spec.name == "catboost":
        # CatBoost supports class_weights as a list aligned to class indices 0..K-1
        # Compute balanced weights from encoded y
        classes, counts = np.unique(y_enc, return_counts=True)
        n_classes = len(classes)
        total = counts.sum()
        # balanced weight: total/(n_classes*count_c)
        cw = [float(total / (n_classes * c)) for c in counts]
        # Ensure list length = n_classes and ordered by class index
        # (LabelEncoder classes are 0..K-1 in order)
        fit_params["clf__class_weights"] = cw

    return fit_params


def run_tuning_one(
    X: pd.DataFrame,
    y_orig: np.ndarray,
    y_enc: np.ndarray,
    groups: np.ndarray,
    spec: ModelSpec,
    n_iter: int,
    splits: int,
    seed: int,
    n_jobs: int,
) -> Dict[str, Any]:
    """
    Tune a single model spec using GroupKFold and macro_f1 as refit metric.
    Returns dict of best results for leaderboard + best params.
    """
    cv = GroupKFold(n_splits=splits)
    pipe = make_pipeline(spec)

    # Choose which y to use
    if spec.use_encoded_y:
        y_used = y_enc
        y_for_fit_label = "enc"
    else:
        y_used = y_orig
        y_for_fit_label = "orig"

    # Fit params (weights)
    _, orig_to_enc, _ = build_label_mapping(y_orig)  # for signature; mapping stable
    fit_params = fit_params_for_model(spec, y_used=y_used, y_enc=y_enc, orig_to_enc=orig_to_enc)

    print(f"\n[TUNE] {spec.name}  y_for_fit={y_for_fit_label}  (n_iter={n_iter}, splits={splits})")

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=spec.param_distributions,
        n_iter=n_iter,
        scoring=SCORERS,
        refit="macro_f1",
        cv=cv,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )

    search.fit(X, y_used, groups=groups, **fit_params)

    best_macro_f1 = float(search.best_score_)
    # Also report best balanced accuracy from the same best params
    # Recompute via cv_results_ entry matching best_index_
    best_idx = int(search.best_index_)
    mean_bal_acc = float(search.cv_results_["mean_test_bal_acc"][best_idx])

    best_params = search.best_params_

    print(f"[BEST] {spec.name}: macro_f1={best_macro_f1:.4f}  bal_acc={mean_bal_acc:.4f}")
    print(f"[BEST] params: {best_params}")

    return {
        "Model": spec.name,
        "CV Macro F1": best_macro_f1,
        "Balanced Acc": mean_bal_acc,
        "Best Params": best_params,
    }


def save_outputs(out_dir: Path, leaderboard: pd.DataFrame, best_models: Dict[str, Any], label_mapping: Dict[str, Any]) -> None:
    _ensure_dir(out_dir)
    leaderboard_path = out_dir / "leaderboard.csv"
    best_models_path = out_dir / "best_models.json"
    mapping_path = out_dir / "label_mapping.json"

    leaderboard.to_csv(leaderboard_path, index=False)
    best_models_path.write_text(json.dumps(best_models, indent=2, default=str))
    mapping_path.write_text(json.dumps(label_mapping, indent=2))

    print(f"\n[OK] Saved: {leaderboard_path}")
    print(f"[OK] Saved: {best_models_path}")
    print(f"[OK] Saved: {mapping_path}")


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-set",
        choices=["theory", "all", "select"],
        required=True,
        help="Which feature set to tune on",
    )
    parser.add_argument(
        "--model-ready",
        type=str,
        default="results/model_ready",
        help="Path to model-ready parquet folder",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/tuning",
        help="Output folder for tuning results",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    model_ready = Path(args.model_ready)
    out_root = Path(args.out)
    out_dir = out_root / args.feature_set

    X, y_orig, groups = load_model_ready(model_ready=model_ready, feature_set=args.feature_set)
    y_enc, orig_to_enc, enc_to_orig = build_label_mapping(y_orig)

    print(f"[LOAD] Feature set={args.feature_set}  X={X.shape}  y_orig={y_orig.shape}  groups={groups.shape}")
    print(f"[LOAD] Label counts (orig): {summarize_counts(y_orig)}")
    print(f"[LOAD] Label counts (enc):  {summarize_counts(y_enc)}")
    print(f"[MAP]  orig_to_enc: {orig_to_enc}")

    specs = build_models_and_spaces(random_state=args.seed)

    rows: List[Dict[str, Any]] = []
    best_models: Dict[str, Any] = {}

    for spec in specs:
        res = run_tuning_one(
            X=X,
            y_orig=y_orig,
            y_enc=y_enc,
            groups=groups,
            spec=spec,
            n_iter=args.n_iter,
            splits=args.splits,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        rows.append({
            "Feature Set": args.feature_set,
            "Model": res["Model"],
            "CV Macro F1": res["CV Macro F1"],
            "Balanced Acc": res["Balanced Acc"],
        })
        best_models[res["Model"]] = {
            "cv_macro_f1": res["CV Macro F1"],
            "balanced_acc": res["Balanced Acc"],
            "best_params": res["Best Params"],
            # record if model used encoded labels
            "y_used": "enc" if next(s for s in specs if s.name == res["Model"]).use_encoded_y else "orig",
        }

    leaderboard = pd.DataFrame(rows).sort_values(["CV Macro F1", "Balanced Acc"], ascending=False)

    label_mapping = {
        "orig_to_enc": orig_to_enc,
        "enc_to_orig": enc_to_orig,
        "note": "XGBoost (and CatBoost if enabled) were tuned using encoded labels 0..K-1. Others used original labels.",
    }

    save_outputs(out_dir=out_dir, leaderboard=leaderboard, best_models=best_models, label_mapping=label_mapping)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())