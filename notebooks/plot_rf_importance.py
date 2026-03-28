#!/usr/bin/env python3

"""
Plot ALL Random Forest feature importances
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Paths
IN_PATH = Path("results/select_features_rf/rf_feature_importance.csv")
OUT_PATH = Path("results/select_features_rf/rf_feature_importance_all.png")

print("[LOAD] Feature importance CSV")
df = pd.read_csv(IN_PATH)

# Sort (should already be sorted, but just in case)
df = df.sort_values("importance", ascending=False)

# Reverse for horizontal plotting
df_plot = df.iloc[::-1]

print("[PLOT] Creating FULL feature importance plot")

# Make figure tall enough for all features
plt.figure(figsize=(10, len(df) * 0.25))

plt.barh(df_plot["feature"], df_plot["importance"])

plt.xlabel("Importance")
plt.title("Random Forest Feature Importance (All Features)")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200)

print("✅ Saved plot to:", OUT_PATH)