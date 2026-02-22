#!/usr/bin/env python3
"""
Hyperparameter tuning (per model) with imbalance-aware GroupKFold CV.

Fixes:
- XGBoost requires class labels to be contiguous 0..K-1.
  We encode y -> y_enc and save mapping to decode later.

Usage:
  python notebooks/tune_models.py --feature-set theory
  python notebooks/tune_models.py --feature-set all

Inputs:
  results/model_ready/X_theory.parquet
  results/model_ready/X_all.parquet
  results/model_ready/y.parquet               (must contain column EventLabel)
  results/model_ready/groups.parquet          (must contain column SubjectID)

Outputs:
  results/tuning/<feature_set>/
    - label_mapping.json
    - best_models.json
    - leaderboard.csv
    - cv_results_<model>.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

_HAS_XGB = True
try:
    from xgboost import XGBClassifier
except Exception:
    _HAS_XGB = False

_HAS_CAT = True
try:
    from catboost import CatBoostClassifier
except Exception:
    _HAS_CAT = False


# -----------------------------
# Utilities
# -----------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_parquet_series(path: Path, col: str) -> pd.Series:
    df = pd.read_parquet(path)
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' in {path}, got {list(df.columns)}")
    return df[col]


def load_model_ready(model_ready_dir: Path, feature_set: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    feature_set = feature_set.lower().strip()
    if feature_set not in {"theory", "all"}:
        raise ValueError("--feature-set must be one of: theory, all")

    x_path = model_ready_dir / ("X_theory.parquet" if feature_set == "theory" else "X_all.parquet")
    y_path = model_ready_dir / "y.parquet"
    g_path = model_ready_dir / "groups.parquet"

    X = pd.read_parquet(x_path)
    y = read_parquet_series(y_path, "EventLabel")
    groups = read_parquet_series(g_path, "SubjectID")

    if not (len(X) == len(y) == len(groups)):
        raise ValueError(f"Shape mismatch: X={X.shape}, y={y.shape}, groups={groups.shape}")

    return X, y, groups


def compute_class_weights(y: pd.Series) -> dict[int, float]:
    classes = np.array(sorted(pd.unique(y)))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y.to_numpy())
    return {int(c): float(w) for c, w in zip(classes, weights)}


def make_sample_weights(y: pd.Series, cw: dict[int, float]) -> np.ndarray:
    return np.asarray([cw[int(v)] for v in y.to_numpy()], dtype=float)


def macro_f1():
    return make_scorer(f1_score, average="macro")


def bal_acc():
    return make_scorer(balanced_accuracy_score)


def encode_labels(y: pd.Series) -> tuple[pd.Series, dict[int, int], dict[int, int]]:
    """
    Map original labels (e.g., 0,1,2,5) -> contiguous (0..K-1).
    Returns (y_enc, orig_to_enc, enc_to_orig)
    """
    orig = sorted(pd.unique(y))
    orig_to_enc = {int(o): int(i) for i, o in enumerate(orig)}
    enc_to_orig = {int(i): int(o) for i, o in enumerate(orig)}
    y_enc = y.map(orig_to_enc).astype(int)
    return y_enc, orig_to_enc, enc_to_orig


# -----------------------------
# Model specs
# -----------------------------
def build_model_specs(
    y_orig: pd.Series,
    y_enc: pd.Series,
    seed: int,
) -> list[dict]:
    """
    Each spec:
      {
        "name": str,
        "estimator": estimator,
        "param_distributions": dict,
        "y_for_fit": "orig"|"enc",
        "fit_params_fn": callable|None
      }
    """
    cw_orig = compute_class_weights(y_orig)
    cw_enc = compute_class_weights(y_enc)

    specs: list[dict] = []

    # Logistic Regression (use original y; sklearn handles non-contiguous classes)
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            class_weight=cw_orig,
            random_state=seed,
        ))
    ])
    lr_space = {
        "clf__C": np.logspace(-3, 3, 20),
        "clf__solver": ["lbfgs"],
        # remove penalty from search to avoid sklearn warning spam
    }
    specs.append({
        "name": "logreg",
        "estimator": lr,
        "param_distributions": lr_space,
        "y_for_fit": "orig",
        "fit_params_fn": None,
    })

    # Random Forest (use original y)
    rf = RandomForestClassifier(
        random_state=seed,
        n_jobs=-1,
        class_weight=cw_orig,
    )
    rf_space = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [None, 6, 10, 14, 18, 24],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
    }
    specs.append({
        "name": "rf",
        "estimator": rf,
        "param_distributions": rf_space,
        "y_for_fit": "orig",
        "fit_params_fn": None,
    })

    # XGBoost (must use encoded y)
    if _HAS_XGB:
        xgb = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_estimators=500,
            learning_rate=0.05,
        )
        xgb_space = {
            "max_depth": [3, 4, 5, 6, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 2, 5, 10],
            "gamma": [0, 0.05, 0.1, 0.2],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        }

        def xgb_fit_params() -> dict:
            sw = make_sample_weights(y_enc, cw_enc)
            return {"sample_weight": sw}

        specs.append({
            "name": "xgb",
            "estimator": xgb,
            "param_distributions": xgb_space,
            "y_for_fit": "enc",
            "fit_params_fn": xgb_fit_params,
        })
    else:
        print("[WARN] xgboost not installed; skipping XGBClassifier.")

    # CatBoost (optional)
    if _HAS_CAT:
        cat = CatBoostClassifier(
            loss_function="MultiClass",
            random_seed=seed,
            verbose=False,
        )
        cat_space = {
            "depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5, 7, 9],
            "iterations": [300, 600, 900, 1200],
        }

        def cat_fit_params() -> dict:
            # CatBoost expects class_weights ordered by class indices 0..K-1
            classes = sorted(pd.unique(y_enc))
            cw_list = [float(cw_enc[int(c)]) for c in classes]
            return {"class_weights": cw_list}

        specs.append({
            "name": "catboost",
            "estimator": cat,
            "param_distributions": cat_space,
            "y_for_fit": "enc",
            "fit_params_fn": cat_fit_params,
        })
    else:
        print("[WARN] catboost not installed; skipping CatBoostClassifier.")

    return specs


# -----------------------------
# Tuning runner
# -----------------------------
def run_tuning(
    X: pd.DataFrame,
    y_orig: pd.Series,
    y_enc: pd.Series,
    groups: pd.Series,
    out_dir: Path,
    seed: int,
    n_iter: int,
    n_splits: int,
    n_jobs: int,
    label_maps: dict,
) -> None:
    ensure_dir(out_dir)

    # Save mapping so later evaluation can decode XGB/CatBoost predictions back to original labels
    (out_dir / "label_mapping.json").write_text(json.dumps(label_maps, indent=2), encoding="utf-8")

    cv = GroupKFold(n_splits=n_splits)

    scoring = {
        "macro_f1": macro_f1(),
        "bal_acc": bal_acc(),
    }

    specs = build_model_specs(y_orig=y_orig, y_enc=y_enc, seed=seed)

    leaderboard_rows = []
    best_models = {
        "created_utc": utc_now(),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "label_counts_orig": {int(k): int(v) for k, v in y_orig.value_counts().to_dict().items()},
        "label_counts_enc": {int(k): int(v) for k, v in y_enc.value_counts().to_dict().items()},
        "cv": "GroupKFold",
        "n_splits": int(n_splits),
        "refit_metric": "macro_f1",
        "models": {},
    }

    for spec in specs:
        name = spec["name"]
        est = spec["estimator"]
        space = spec["param_distributions"]

        y_fit = y_orig if spec["y_for_fit"] == "orig" else y_enc

        fit_params = {}
        if spec["fit_params_fn"] is not None:
            fit_params = spec["fit_params_fn"]()

        print(f"\n[TUNE] {name}  y_for_fit={spec['y_for_fit']}  (n_iter={n_iter}, splits={n_splits})")

        search = RandomizedSearchCV(
            estimator=est,
            param_distributions=space,
            n_iter=min(n_iter, len(list(space.values())[0]) if name == "logreg" else n_iter),
            scoring=scoring,
            refit="macro_f1",
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
            random_state=seed,
            return_train_score=False,
            error_score="raise",  # fail fast so you can debug
        )

        search.fit(X, y_fit, groups=groups, **fit_params)

        cv_res = pd.DataFrame(search.cv_results_)
        cv_path = out_dir / f"cv_results_{name}.csv"
        cv_res.to_csv(cv_path, index=False)

        best_idx = int(search.best_index_)
        best_macro = float(cv_res.loc[best_idx, "mean_test_macro_f1"])
        best_bal = float(cv_res.loc[best_idx, "mean_test_bal_acc"])
        best_params = search.best_params_

        print(f"[BEST] {name}: macro_f1={best_macro:.4f}  bal_acc={best_bal:.4f}")
        print(f"[BEST] params: {best_params}")

        leaderboard_rows.append({
            "model": name,
            "y_for_fit": spec["y_for_fit"],
            "best_macro_f1": best_macro,
            "best_bal_acc": best_bal,
            "best_params": json.dumps(best_params, default=str),
        })

        best_models["models"][name] = {
            "y_for_fit": spec["y_for_fit"],
            "best_macro_f1": best_macro,
            "best_bal_acc": best_bal,
            "best_params": best_params,
            "cv_results_csv": str(cv_path),
        }

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("best_macro_f1", ascending=False)
    leaderboard_path = out_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    best_json_path = out_dir / "best_models.json"
    best_json_path.write_text(json.dumps(best_models, indent=2, default=str), encoding="utf-8")

    print(f"\n[OK] Saved: {leaderboard_path}")
    print(f"[OK] Saved: {best_json_path}")
    print(f"[OK] Saved: {out_dir / 'label_mapping.json'}")


# -----------------------------
# CLI
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", required=True, choices=["theory", "all"])
    ap.add_argument("--model-ready", default="results/model_ready")
    ap.add_argument("--out", default="results/tuning")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-iter", type=int, default=25)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    model_ready_dir = Path(args.model_ready).resolve()
    out_dir = Path(args.out).resolve() / args.feature_set

    X, y_orig, groups = load_model_ready(model_ready_dir, args.feature_set)
    y_enc, orig_to_enc, enc_to_orig = encode_labels(y_orig)

    label_maps = {
        "orig_to_enc": orig_to_enc,
        "enc_to_orig": enc_to_orig,
        "orig_labels_sorted": sorted(orig_to_enc.keys()),
    }

    print(f"[LOAD] Feature set={args.feature_set}  X={X.shape}  y_orig={y_orig.shape}  groups={groups.shape}")
    print(f"[LOAD] Label counts (orig): {y_orig.value_counts().to_dict()}")
    print(f"[LOAD] Label counts (enc):  {y_enc.value_counts().to_dict()}")
    print(f"[MAP]  orig_to_enc: {orig_to_enc}")

    run_tuning(
        X=X,
        y_orig=y_orig,
        y_enc=y_enc,
        groups=groups,
        out_dir=out_dir,
        seed=args.seed,
        n_iter=args.n_iter,
        n_splits=args.splits,
        n_jobs=args.n_jobs,
        label_maps=label_maps,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())