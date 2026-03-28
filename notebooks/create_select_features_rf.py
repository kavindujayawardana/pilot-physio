#!/usr/bin/env python3

"""
Create SELECT feature set using Random Forest importance on ALL features.

Outputs:
  results/select_features_rf/
      rf_feature_importance.csv
      select_features.json
  results/model_ready/X_select.parquet
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE = Path("results")
MODEL_READY = BASE / "model_ready"
OUT_DIR = BASE / "select_features_rf"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("[LOAD] X_all and y")
    X = pd.read_parquet(MODEL_READY / "X_all.parquet")
    y = pd.read_parquet(MODEL_READY / "y.parquet")["EventLabel"]

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("[TRAIN] Random Forest on ALL features")
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)

    print("[IMPORTANCE] Extracting RF feature importance")
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(OUT_DIR / "rf_feature_importance.csv", index=False)

    TOP_N = 15
    print(f"[SELECT] Taking top {TOP_N} features")
    select_features = importance_df.head(TOP_N)["feature"].tolist()

    for f in select_features:
        print(" -", f)

    (OUT_DIR / "select_features.json").write_text(
        json.dumps(select_features, indent=2)
    )

    print("[SAVE] Creating X_select.parquet")
    X_select = X[select_features].copy()
    X_select.to_parquet(MODEL_READY / "X_select.parquet")

    print("\n✅ RF-based SELECT features created successfully")
    print("Saved importance CSV to:", OUT_DIR / "rf_feature_importance.csv")
    print("Saved select list to:", OUT_DIR / "select_features.json")
    print("Saved X_select to:", MODEL_READY / "X_select.parquet")


if __name__ == "__main__":
    main()