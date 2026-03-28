#!/usr/bin/env python3

"""
Compare tuned models across feature sets
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path("results/tuning")

rows = []

print("\n=== Tuned Model Comparison ===\n")

for fs_dir in BASE_DIR.iterdir():
    if not fs_dir.is_dir():
        continue

    leaderboard_path = fs_dir / "leaderboard.csv"

    if not leaderboard_path.exists():
        continue

    df = pd.read_csv(leaderboard_path)

    # -----------------------------
    # 🔧 FIX COLUMN NAMES
    # -----------------------------
    df = df.rename(columns={
        "model": "Model",
        "best_macro_f1": "CV Macro F1",
        "best_bal_acc": "Balanced Acc"
    })

    # If Model column still missing, try fallback
    if "Model" not in df.columns:
        print(f"[WARN] No 'Model' column in {leaderboard_path}")
        print("Columns found:", df.columns.tolist())
        continue

    df["Feature Set"] = fs_dir.name

    df = df[["Model", "CV Macro F1", "Balanced Acc", "Feature Set"]]

    rows.append(df)

# Combine all
final_df = pd.concat(rows, ignore_index=True)

# Sort by best model
final_df = final_df.sort_values("CV Macro F1", ascending=False)

print(final_df.to_string(index=False))

# Save
out_path = BASE_DIR / "model_comparison.csv"
final_df.to_csv(out_path, index=False)

print("\nSaved to:", out_path)