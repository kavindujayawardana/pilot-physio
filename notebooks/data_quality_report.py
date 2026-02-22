#!/usr/bin/env python3
"""
Data Quality Report (Pre-filtering)
Counts:
- total windows (optionally benchmark-only)
- transition windows
- bad windows
- missing values (per feature + summary)
- how many rows would be dropped by each rule

Usage examples:
  python notebooks/data_quality_report.py --features ./features --benchmark-only
  python notebooks/data_quality_report.py --features ./features --out ./results/feature_diagnostics --benchmark-only --purity 0.8
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


BENCHMARK_TYPES = {"CA", "DA", "SS"}


def load_all_features(features_root: Path) -> pd.DataFrame:
    dfs = []
    for subj_dir in sorted([p for p in features_root.iterdir() if p.is_dir()]):
        f = subj_dir / "features.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No features.parquet found under: {features_root}")

    return pd.concat(dfs, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-filter data quality report for windowed features.")
    parser.add_argument("--features", required=True, help="Path to features/ directory")
    parser.add_argument("--out", default="./results/feature_diagnostics", help="Output folder for CSV summary")
    parser.add_argument("--benchmark-only", action="store_true", help="Keep only CA/DA/SS sessions")
    parser.add_argument("--purity", type=float, default=0.80, help="Label purity threshold (used if IsTransition missing)")
    args = parser.parse_args()

    features_root = Path(args.features).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading features from: {features_root}")
    data = load_all_features(features_root)
    print(f"[INFO] Loaded rows: {len(data):,} | cols: {len(data.columns):,}")

    # Benchmark filter
    if args.benchmark_only:
        if "SessionType" not in data.columns:
            raise ValueError("SessionType column not found. Cannot apply --benchmark-only.")
        before = len(data)
        data = data[data["SessionType"].isin(BENCHMARK_TYPES)].copy()
        print(f"[INFO] Benchmark-only filter: {before:,} -> {len(data):,}")

    # Identify transition windows
    transition_count = None
    if "IsTransition" in data.columns:
        transition_count = int(data["IsTransition"].fillna(False).astype(bool).sum())
        transition_rule_used = "IsTransition column"
    elif "LabelFrac" in data.columns:
        transition_count = int((pd.to_numeric(data["LabelFrac"], errors="coerce") < args.purity).sum())
        transition_rule_used = f"LabelFrac < {args.purity}"
    else:
        transition_count = 0
        transition_rule_used = "No IsTransition/LabelFrac found (treated as 0 transitions)"

    # Bad windows
    bad_count = 0
    if "IsBadWindow" in data.columns:
        bad_count = int(data["IsBadWindow"].fillna(False).astype(bool).sum())

    # Choose feature columns (exclude metadata-like columns)
    META_LIKE = {
        "WindowID", "SubjectID", "CrewID", "SessionName", "SessionType", "BenchmarkTask",
        "StartIdx", "EndIdx", "StartTime", "EndTime",
        "EventLabel", "LabelFrac",
        "BadSampleFrac", "IsBadWindow", "IsTransition",
        "raw_data_pointer", "channels",
        "window_len_s", "step_s", "transition_keep_frac", "created_utc",
        "_source_file"
    }
    feature_cols = [c for c in data.columns if c not in META_LIKE]

    # Missingness summary (per feature)
    missing_per_feature = data[feature_cols].isna().sum().sort_values(ascending=False)
    missing_per_feature = missing_per_feature[missing_per_feature > 0]

    # Rows with any missing feature
    rows_any_missing = int(data[feature_cols].isna().any(axis=1).sum())

    # Simulate how many rows remain after each filter
    total_rows = len(data)

    # Apply transition removal
    if "IsTransition" in data.columns:
        mask_no_transition = data["IsTransition"].fillna(False).astype(bool) == False
    elif "LabelFrac" in data.columns:
        mask_no_transition = pd.to_numeric(data["LabelFrac"], errors="coerce") >= args.purity
    else:
        mask_no_transition = pd.Series([True] * len(data), index=data.index)

    after_transition = int(mask_no_transition.sum())

    # Apply bad removal
    if "IsBadWindow" in data.columns:
        mask_no_bad = data["IsBadWindow"].fillna(False).astype(bool) == False
    else:
        mask_no_bad = pd.Series([True] * len(data), index=data.index)

    after_transition_bad = int((mask_no_transition & mask_no_bad).sum())

    # Apply missing removal
    mask_no_missing = data[feature_cols].isna().any(axis=1) == False
    after_all = int((mask_no_transition & mask_no_bad & mask_no_missing).sum())

    # Print a clean console report
    print("\n================ DATA QUALITY REPORT ================")
    print(f"Rows in scope: {total_rows:,}")
    print(f"Transition rule used: {transition_rule_used}")
    print(f"Transition windows: {transition_count:,} ({transition_count/total_rows*100:.2f}%)")
    print(f"Bad windows: {bad_count:,} ({bad_count/total_rows*100:.2f}%)")
    print(f"Rows with ANY missing feature: {rows_any_missing:,} ({rows_any_missing/total_rows*100:.2f}%)")
    print("-----------------------------------------------------")
    print(f"Rows after removing transitions only: {after_transition:,}")
    print(f"Rows after removing transitions + bad: {after_transition_bad:,}")
    print(f"Rows after removing transitions + bad + missing: {after_all:,}")
    print("=====================================================\n")

    # Save outputs
    summary_df = pd.DataFrame([{
        "rows_total": total_rows,
        "transition_rule_used": transition_rule_used,
        "transition_count": transition_count,
        "transition_pct": (transition_count / total_rows * 100) if total_rows else 0,
        "bad_count": bad_count,
        "bad_pct": (bad_count / total_rows * 100) if total_rows else 0,
        "rows_any_missing": rows_any_missing,
        "missing_pct": (rows_any_missing / total_rows * 100) if total_rows else 0,
        "rows_after_transition_only": after_transition,
        "rows_after_transition_and_bad": after_transition_bad,
        "rows_after_all_filters": after_all,
        "purity_threshold_used": args.purity,
        "benchmark_only": bool(args.benchmark_only),
    }])

    summary_path = out_root / "data_quality_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    missing_path = out_root / "missing_counts_per_feature.csv"
    missing_per_feature.to_csv(missing_path, header=["missing_count"])

    print(f"[OK] Saved summary: {summary_path}")
    print(f"[OK] Saved missing-per-feature: {missing_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())