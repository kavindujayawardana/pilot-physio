#!/usr/bin/env python3

"""
Select Top 20 features from RF importance
"""

import json
from pathlib import Path
import pandas as pd

# Paths
IN_PATH = Path("results/select_features_rf/rf_feature_importance.csv")
OUT_PATH = Path("results/select_features_rf/select_features.json")

print("[LOAD] RF feature importance")
df = pd.read_csv(IN_PATH)

# Sort by importance
df = df.sort_values("importance", ascending=False)

# Select top 20
top_features = df["feature"].head(20).tolist()

print("\nTop 20 features:")
for f in top_features:
    print(" -", f)

# Save JSON
OUT_PATH.write_text(json.dumps(top_features, indent=2))

print("\n✅ Saved:", OUT_PATH)
print("Total features:", len(top_features))