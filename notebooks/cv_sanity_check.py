#!/usr/bin/env python3
"""
CV Sanity Check — Step 12

Purpose
-------
Verify your CV split logic is correct (no SubjectID leakage) BEFORE training.

What it does
------------
1) Loads model-ready y + groups from results/model_ready/
2) Prints:
   - number of rows
   - number of unique subjects
   - overall class distribution
3) Runs Leave-One-Group-Out (LOSO) outer CV:
   - prints first 2 outer folds (train/test subject IDs + class balance)
4) For the FIRST outer fold only:
   - runs inner GroupKFold(5) on the training set
   - prints fold sizes + subject overlap checks

How to run
----------
python notebooks/cv_sanity_check.py

Notes
-----
- This script does NOT train any model.
- It only checks splits and distributions.
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut, GroupKFold


MODEL_READY_DIR = Path("results/model_ready")


def load_series_parquet(path: Path, expected_name: str) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_parquet(path)
    if expected_name not in df.columns:
        raise ValueError(f"{path} does not contain column '{expected_name}'. Found: {df.columns.tolist()}")

    return df[expected_name]


def pct(s: pd.Series) -> pd.Series:
    return (s.value_counts(normalize=True).sort_index() * 100).round(2)


def main() -> int:
    y_path = MODEL_READY_DIR / "y.parquet"
    g_path = MODEL_READY_DIR / "groups.parquet"

    y = load_series_parquet(y_path, "EventLabel").astype(int)
    groups = load_series_parquet(g_path, "SubjectID").astype(str)

    if len(y) != len(groups):
        raise ValueError(f"Length mismatch: y={len(y)} vs groups={len(groups)}")

    print("\n=== DATA SUMMARY ===")
    print(f"Rows: {len(y):,}")
    print(f"Unique subjects: {groups.nunique()}")
    print(f"Subjects: {sorted(groups.unique().tolist())}")

    print("\nOverall class counts:")
    print(y.value_counts().sort_index())

    print("\nOverall class %:")
    print(pct(y))

    # -----------------------------
    # OUTER CV: Leave-One-Subject-Out
    # -----------------------------
    logo = LeaveOneGroupOut()

    print("\n=== OUTER CV (LeaveOneGroupOut / LOSO) — First 2 folds ===")
    outer_folds = list(logo.split(np.zeros(len(y)), y, groups))
    if not outer_folds:
        raise RuntimeError("No outer folds produced. Check groups content.")

    for fold_i, (train_idx, test_idx) in enumerate(outer_folds[:2], start=1):
        train_subj = sorted(groups.iloc[train_idx].unique().tolist())
        test_subj = sorted(groups.iloc[test_idx].unique().tolist())

        print(f"\n[Outer fold {fold_i}]")
        print(f"  Train subjects ({len(train_subj)}): {train_subj}")
        print(f"  Test subject ({len(test_subj)}): {test_subj}")
        assert len(test_subj) == 1, "LOSO should hold out exactly 1 subject per fold."

        # No leakage check
        overlap = set(train_subj).intersection(set(test_subj))
        print(f"  Subject overlap (should be empty): {overlap}")

        # Class balance check
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        print("  Train class %:")
        print("  ", pct(y_train).to_dict())
        print("  Test class %:")
        print("  ", pct(y_test).to_dict())

    # -----------------------------
    # INNER CV: GroupKFold on training portion of first outer fold
    # -----------------------------
    train_idx, test_idx = outer_folds[0]
    y_train = y.iloc[train_idx]
    g_train = groups.iloc[train_idx]

    print("\n=== INNER CV (GroupKFold) on FIRST outer-fold TRAIN set ===")
    gkf = GroupKFold(n_splits=5)

    inner_splits = list(gkf.split(np.zeros(len(y_train)), y_train, g_train))
    print(f"Inner folds produced: {len(inner_splits)}")

    for j, (tr_i, va_i) in enumerate(inner_splits, start=1):
        tr_subj = set(g_train.iloc[tr_i].unique().tolist())
        va_subj = set(g_train.iloc[va_i].unique().tolist())
        overlap = tr_subj.intersection(va_subj)

        print(f"\n[Inner fold {j}]")
        print(f"  Train rows: {len(tr_i):,} | Val rows: {len(va_i):,}")
        print(f"  Train subjects: {len(tr_subj)} | Val subjects: {len(va_subj)}")
        print(f"  Subject overlap (should be empty): {overlap}")

        # Optional: class balance summary
        print("  Train class %:", pct(y_train.iloc[tr_i]).to_dict())
        print("  Val class %:", pct(y_train.iloc[va_i]).to_dict())

    # -----------------------------
    # Save a tiny CV plan artifact (optional, local only)
    # -----------------------------
    out_dir = Path("results/cv_plan")
    out_dir.mkdir(parents=True, exist_ok=True)

    outer_plan = [
        {
            "fold": k + 1,
            "test_subject": groups.iloc[test_idx].unique().tolist()[0],
        }
        for k, (_, test_idx) in enumerate(outer_folds)
    ]
    (out_dir / "outer_folds.json").write_text(json.dumps(outer_plan, indent=2), encoding="utf-8")

    print("\n[OK] CV sanity check completed.")
    print(f"[OK] Saved outer fold plan to: {out_dir / 'outer_folds.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())