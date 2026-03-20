#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

BASE = Path("results/balanced_models")

rows = []

for fs_dir in BASE.iterdir():
    if fs_dir.is_dir():
        summary = fs_dir / "metrics_summary.csv"
        if summary.exists():
            df = pd.read_csv(summary)
            df["Feature Set"] = fs_dir.name
            rows.append(df)

if not rows:
    raise SystemExit("No balanced-model results found.")

final_df = pd.concat(rows, ignore_index=True)
final_df = final_df.sort_values(["Macro F1", "Balanced Acc"], ascending=False)

print("\n=== Balanced Model Comparison ===\n")
print(final_df.to_string(index=False))

final_df.to_csv(BASE / "model_comparison.csv", index=False)
print("\nSaved to:", BASE / "model_comparison.csv")