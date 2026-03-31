#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

sources = [
    ("original", Path("results/tuning")),
    ("refined", Path("results/tuning_refined")),
]

rows = []

def normalize_columns(df):
    rename_map = {}

    for col in df.columns:
        c = col.lower()

        if c in ["model"]:
            rename_map[col] = "Model"

        elif "precision" in c:
            rename_map[col] = "Macro Precision"

        elif "recall" in c:
            rename_map[col] = "Macro Recall"

        elif "f1" in c:
            rename_map[col] = "Macro F1"

        elif "bal" in c:
            rename_map[col] = "Balanced Acc"

    df = df.rename(columns=rename_map)

    # Ensure all columns exist
    for col in ["Model", "Macro Precision", "Macro Recall", "Macro F1", "Balanced Acc"]:
        if col not in df.columns:
            df[col] = pd.NA

    return df


for source_name, base in sources:
    if not base.exists():
        continue

    for fs_dir in base.iterdir():
        if not fs_dir.is_dir():
            continue

        leaderboard = fs_dir / "leaderboard.csv"
        if not leaderboard.exists():
            continue

        df = pd.read_csv(leaderboard)
        df = normalize_columns(df)

        df["Feature Set"] = fs_dir.name
        df["Source"] = source_name

        df = df[[
            "Model",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Balanced Acc",
            "Feature Set",
            "Source"
        ]]

        rows.append(df)

if not rows:
    raise SystemExit("No tuning results found.")

final_df = pd.concat(rows, ignore_index=True)

# Sort by importance (your case = recall first)
final_df = final_df.sort_values(
    ["Macro Recall", "Macro F1", "Balanced Acc"],
    ascending=False,
    na_position="last"
)

print("\n=== FINAL MODEL COMPARISON (Original + Refined) ===\n")
print(final_df.to_string(index=False))

out_path = Path("results/final_model_comparison.csv")
final_df.to_csv(out_path, index=False)

print("\nSaved to:", out_path)