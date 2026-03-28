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

# Keep only columns needed for model comparison
wanted = [
    "Feature Set",
    "Model Variant",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Balanced Acc",
]
final_df = final_df[wanted]

# Sort primarily by recall if that is your priority,
# then by Macro F1, then Balanced Acc
final_df = final_df.sort_values(
    ["Macro Recall", "Macro F1", "Balanced Acc"],
    ascending=False
)

print("\n=== Balanced Model Comparison (Recall-Focused) ===\n")
print(final_df.to_string(index=False))

out_path = BASE / "model_comparison_with_recall.csv"
final_df.to_csv(out_path, index=False)

print("\nSaved to:", out_path)