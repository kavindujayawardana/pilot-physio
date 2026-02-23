#!/usr/bin/env python3
"""
Step 14 — Compare tuned models across feature sets.

Creates:
  results/tuning/model_comparison.csv
"""

from pathlib import Path
import pandas as pd


def load_leaderboard(path: Path, feature_set: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["model", "best_macro_f1", "best_bal_acc"]].copy()
    df["feature_set"] = feature_set
    return df


def main():
    base = Path("results/tuning")

    theory_path = base / "theory" / "leaderboard.csv"
    all_path = base / "all" / "leaderboard.csv"

    df_theory = load_leaderboard(theory_path, "theory")
    df_all = load_leaderboard(all_path, "all")

    df = pd.concat([df_theory, df_all], ignore_index=True)

    df = df.rename(columns={
        "model": "Model",
        "feature_set": "Feature Set",
        "best_macro_f1": "CV Macro F1",
        "best_bal_acc": "Balanced Acc"
    })

    df = df.sort_values("CV Macro F1", ascending=False)

    out_path = base / "model_comparison.csv"
    df.to_csv(out_path, index=False)

    print("\n=== Tuned Model Comparison ===\n")
    print(df.to_string(index=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()