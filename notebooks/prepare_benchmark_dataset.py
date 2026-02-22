#!/usr/bin/env python3
"""
STEP 11 — Prepare Final Benchmark Dataset

This script:
1. Loads all subjects (features_with_indices preferred)
2. Keeps only CA / DA / SS
3. Removes:
   - Transition windows
   - Bad windows
   - Rows with missing values
4. Separates:
   - X (features)
   - y (EventLabel)
   - groups (SubjectID)
5. Saves clean dataset ready for modelling
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------
FEATURE_ROOT = Path("features")
CONFIG_PATH = Path("config/feature_sets.json")
OUTPUT_ROOT = Path("results/model_ready")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

BENCHMARK_TYPES = {"CA", "DA", "SS"}

# ----------------------------
# Load feature set config
# ----------------------------
cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
theory_features = cfg["dataset_A_theory"]
metadata_exclude = set(cfg["dataset_B_all_policy"]["exclude_columns_exact"])

# ----------------------------
# Load all subjects
# ----------------------------
dfs = []

for subj_dir in FEATURE_ROOT.iterdir():
    if subj_dir.is_dir():
        p1 = subj_dir / "features_with_indices.parquet"
        p2 = subj_dir / "features.parquet"

        if p1.exists():
            df = pd.read_parquet(p1)
        elif p2.exists():
            df = pd.read_parquet(p2)
        else:
            continue

        dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

print("Initial rows:", len(data))

# ----------------------------
# 11.1 FILTERING
# ----------------------------

# Keep only benchmark sessions
data = data[data["SessionType"].isin(BENCHMARK_TYPES)]
print("After benchmark filter:", len(data))

# Remove transition windows
if "IsTransition" in data.columns:
    data = data[data["IsTransition"] == False]
print("After removing transitions:", len(data))

# Remove bad windows
if "IsBadWindow" in data.columns:
    data = data[data["IsBadWindow"] == False]
print("After removing bad windows:", len(data))

# Drop rows with missing values
data = data.dropna()
print("After dropping missing rows:", len(data))

# ----------------------------
# 11.2 Separate target and groups
# ----------------------------

y = data["EventLabel"].copy()
groups = data["SubjectID"].copy()

# ----------------------------
# Dataset A — Theory
# ----------------------------
X_theory = data[theory_features].copy()

# ----------------------------
# Dataset B — All
# ----------------------------
all_features = [
    c for c in data.columns
    if c not in metadata_exclude and pd.api.types.is_numeric_dtype(data[c])
]

X_all = data[all_features].copy()

print("\nFinal shapes:")
print("Theory:", X_theory.shape)
print("All:", X_all.shape)
print("Target:", y.shape)

# ----------------------------
# Save outputs
# ----------------------------

X_theory.to_parquet(OUTPUT_ROOT / "X_theory.parquet", index=False)
X_all.to_parquet(OUTPUT_ROOT / "X_all.parquet", index=False)
y.to_frame("EventLabel").to_parquet(OUTPUT_ROOT / "y.parquet", index=False)
groups.to_frame("SubjectID").to_parquet(OUTPUT_ROOT / "groups.parquet", index=False)

print("\nSaved model-ready datasets to:", OUTPUT_ROOT)