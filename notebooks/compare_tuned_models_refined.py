#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

BASE = Path("results/tuning_refined")

rows = []

for fs_dir in BASE.iterdir():
    if not fs_dir.is_dir():
        continue

    leaderboard = fs_dir / "leaderboard.csv"
    if not leaderboard.exists():
        continue

    df = pd.read_csv(leaderboard)

    df = df.rename(columns={
        "model": "Model",
        "best_macro_precision": "Macro Precision",
        "best_macro_recall": "Macro Recall",
        "best_macro_f1": "Macro F1",
        "best_bal_acc": "Balanced Acc",
    })

    df["Feature Set"] = fs_dir.name
    df = df[["Model", "Macro Precision", "Macro Recall", "Macro F1", "Balanced Acc", "Feature Set"]]
    rows.append(df)

if not rows:
    raise SystemExit("No refined tuning results found.")

final_df = pd.concat(rows, ignore_index=True)
final_df = final_df.sort_values(
    ["Macro Recall", "Macro F1", "Balanced Acc"],
    ascending=False
)

print("\n=== Refined Tuned Model Comparison ===\n")
print(final_df.to_string(index=False))

out_path = BASE / "model_comparison.csv"
final_df.to_csv(out_path, index=False)

print("\nSaved to:", out_path)