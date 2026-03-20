#!/usr/bin/env python3
"""
Prepare final benchmark dataset

Manager-updated version:
- KEEP transition windows
- Remove IsBadWindow == True
- Drop rows with missing values
- Keep only benchmark sessions (CA, DA, SS)

Outputs:
  results/model_ready/
    X_theory.parquet
    X_all.parquet
    y.parquet
    groups.parquet
"""

from pathlib import Path
import json
import pandas as pd

FEATURE_ROOT = Path("features")
CONFIG_PATH = Path("config/feature_sets.json")
OUTPUT_ROOT = Path("results/model_ready")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

BENCHMARK_TYPES = {"CA", "DA", "SS"}


def main():
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    theory_features = cfg["dataset_A_theory"]
    metadata_exclude = set(cfg["dataset_B_all_policy"]["exclude_columns_exact"])

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

    if not dfs:
        raise SystemExit("No feature parquet files found.")

    data = pd.concat(dfs, ignore_index=True)

    print("Initial rows:", len(data))

    # Keep only benchmark sessions
    data = data[data["SessionType"].isin(BENCHMARK_TYPES)].copy()
    print("After benchmark filter:", len(data))

    # KEEP transition windows
    print("Keeping transition windows:", len(data))

    # Remove bad windows
    if "IsBadWindow" in data.columns:
        before = len(data)
        data = data[data["IsBadWindow"] == False].copy()
        print("After removing bad windows:", len(data), f"(removed {before - len(data)})")
    else:
        print("IsBadWindow column not found; skipping bad-window filtering.")

    # Drop rows with missing values
    before = len(data)
    data = data.dropna().copy()
    print("After dropping missing rows:", len(data), f"(removed {before - len(data)})")

    # Target and groups
    y = data["EventLabel"].copy()
    groups = data["SubjectID"].copy()

    # Theory feature set
    missing_theory = [f for f in theory_features if f not in data.columns]
    if missing_theory:
        raise SystemExit(f"Missing theory features: {missing_theory}")

    X_theory = data[theory_features].copy()

    # All feature set
    all_features = [
        c for c in data.columns
        if c not in metadata_exclude and pd.api.types.is_numeric_dtype(data[c])
    ]
    X_all = data[all_features].copy()

    print("\nFinal shapes:")
    print("Theory:", X_theory.shape)
    print("All:", X_all.shape)
    print("Target:", y.shape)
    print("Groups:", groups.shape)

    print("\nClass counts:")
    print(y.value_counts().sort_index())

    # Save outputs
    X_theory.to_parquet(OUTPUT_ROOT / "X_theory.parquet", index=False)
    X_all.to_parquet(OUTPUT_ROOT / "X_all.parquet", index=False)
    y.to_frame("EventLabel").to_parquet(OUTPUT_ROOT / "y.parquet", index=False)
    groups.to_frame("SubjectID").to_parquet(OUTPUT_ROOT / "groups.parquet", index=False)

    print("\nSaved model-ready datasets to:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()