#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier


BASE = Path("results")
MODEL_READY = BASE / "model_ready"
TUNING_ALL = BASE / "tuning" / "all"
OUT_DIR = BASE / "select_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_shap_mean_abs_by_feature(explainer, X):
    sv = explainer.shap_values(X)

    if isinstance(sv, list):
        return np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)

    sv = np.asarray(sv)

    if sv.ndim == 2:
        return np.abs(sv).mean(axis=0)

    if sv.ndim == 3:
        if sv.shape[1] == X.shape[1]:
            return np.abs(sv).mean(axis=(0, 2))
        else:
            return np.abs(sv).mean(axis=(0, 1))

    raise ValueError(f"Unexpected SHAP shape: {sv.shape}")


def main():
    print("[LOAD] X_all and y")
    X = pd.read_parquet(MODEL_READY / "X_all.parquet")
    y = pd.read_parquet(MODEL_READY / "y.parquet")["EventLabel"]

    print("X shape:", X.shape)

    label_map = json.loads((TUNING_ALL / "label_mapping.json").read_text())
    orig_to_enc = {int(k): int(v) for k, v in label_map["orig_to_enc"].items()}
    y_enc = y.map(orig_to_enc)

    best_models = json.loads((TUNING_ALL / "best_models.json").read_text())
    xgb_params = best_models["models"]["xgb"]["best_params"]

    print("[LOAD] Best XGB params:", xgb_params)

    print("[TRAIN] Final XGB")
    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        **xgb_params
    )
    model.fit(X, y_enc)

    print("[GAIN] Extracting importance")
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")

    gain_df = pd.DataFrame({
        "feature": list(gain.keys()),
        "gain": list(gain.values())
    })
    gain_df.to_csv(OUT_DIR / "xgb_gain_importance.csv", index=False)

    print("[SHAP] Computing SHAP")
    explainer = shap.TreeExplainer(model)
    shap_mean = compute_shap_mean_abs_by_feature(explainer, X)

    shap_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": shap_mean
    }).sort_values("mean_abs_shap", ascending=False)

    shap_df.to_csv(OUT_DIR / "shap_importance.csv", index=False)

    TOP_N = 15
    select_features = shap_df.head(TOP_N)["feature"].tolist()

    print("[SELECT] Top features:")
    for f in select_features:
        print(" -", f)

    (OUT_DIR / "select_features.json").write_text(
        json.dumps(select_features, indent=2)
    )

    X_select = X[select_features]
    X_select.to_parquet(MODEL_READY / "X_select.parquet")

    print("\n✅ SELECT features created successfully")


if __name__ == "__main__":
    main()
