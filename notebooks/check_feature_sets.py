#!/usr/bin/env python3
"""
Check Feature Sets (Step 10 — Freeze Feature Sets)

What this script does:
1) Loads ONE subject’s feature table to inspect column availability
   - Prefers: features/<subject>/features_with_indices.parquet
   - Fallback: features/<subject>/features.parquet
2) Checks whether your THEORY feature list is present
3) Builds the ALL-features list by excluding metadata columns
4) Prints a clean summary (present/missing, counts, previews)

Usage:
  python notebooks/check_feature_sets.py
  python notebooks/check_feature_sets.py --subject 22
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


# ----------------------------
# Configuration
# ----------------------------
FEATURE_ROOT = Path("features")

PREFERRED = "features_with_indices.parquet"
FALLBACK = "features.parquet"

# Update this list as your "Dataset A — Theory"
# (This matches the ~12–15 theory features we discussed, plus SAI/CAI indices.)
THEORY_FEATURES: List[str] = [
    # EEG aggregated
    "frontal_mean_alpha",
    "frontal_asymmetry_alpha",
    "theta_beta_ratio",
    # Cardiovascular
    "HR_mean",
    "HR_slope",
    # HRV (30s)
    "HRV_RMSSD_30s",
    "HRV_SDNN_30s",
    # GSR
    "GSR_mean",
    "GSR_std",
    "GSR_peak_count",
    # Respiration
    "breathing_rate_10s_bpm",
    # Anxiety indices (final names in features_with_indices.parquet)
    "SAI",
    "CAI",
]

# Columns to exclude when building "ALL" (Dataset B)
# (These are not modelling features; they’re IDs/labels/quality flags/pointers/parameters.)
METADATA_COLS = {
    "WindowID",
    "SubjectID",
    "CrewID",
    "SessionName",
    "SessionType",
    "BenchmarkTask",
    "StartIdx",
    "EndIdx",
    "StartTime",
    "EndTime",
    "EventLabel",
    "LabelFrac",
    "BadSampleFrac",
    "IsBadWindow",
    "IsTransition",
    "raw_data_pointer",
    "channels",
    "window_len_s",
    "step_s",
    "transition_keep_frac",
    "created_utc",
    # Just in case older versions had these:
    "_source_file",
}


def pick_subject_dir(subject: str | None) -> Path:
    """Pick a subject directory. If subject is None, pick the first numeric-ish folder found."""
    if subject is not None:
        d = FEATURE_ROOT / str(subject)
        if not d.exists():
            raise SystemExit(f"[ERROR] Subject folder not found: {d}")
        return d

    # Auto-pick the first directory under features/
    dirs = [p for p in FEATURE_ROOT.iterdir() if p.is_dir()]
    if not dirs:
        raise SystemExit(f"[ERROR] No subject folders found under: {FEATURE_ROOT.resolve()}")

    # Prefer folders with numeric names if possible
    dirs_sorted = sorted(dirs, key=lambda p: (not p.name.isdigit(), p.name))
    return dirs_sorted[0]


def load_features_table(subject_dir: Path) -> pd.DataFrame:
    preferred = subject_dir / PREFERRED
    fallback = subject_dir / FALLBACK

    if preferred.exists():
        path = preferred
    elif fallback.exists():
        path = fallback
    else:
        raise SystemExit(
            f"[ERROR] No feature parquet found in {subject_dir}.\n"
            f"Expected one of:\n  - {preferred}\n  - {fallback}"
        )

    df = pd.read_parquet(path)
    print(f"\nLoaded: {path}  (rows={len(df):,}, cols={df.shape[1]:,})")
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default=None, help="SubjectID folder under features/ (e.g., 22). If omitted, auto-picks.")
    args = parser.parse_args()

    if not FEATURE_ROOT.exists():
        raise SystemExit(f"[ERROR] features/ directory not found at: {FEATURE_ROOT.resolve()}")

    subject_dir = pick_subject_dir(args.subject)
    df = load_features_table(subject_dir)

    cols = list(df.columns)

    # ----------------------------
    # THEORY feature check
    # ----------------------------
    present = [c for c in THEORY_FEATURES if c in cols]
    missing = [c for c in THEORY_FEATURES if c not in cols]

    print("\n=== THEORY FEATURE CHECK ===")
    print(f"Present: {len(present)}")
    print(f"Missing: {len(missing)}")
    if missing:
        print("Missing list:", missing)

    # ----------------------------
    # ALL features (exclude metadata)
    # ----------------------------
    all_features = [
        c for c in cols
        if (c not in METADATA_COLS) and pd.api.types.is_numeric_dtype(df[c])
    ]

    print("\n=== ALL FEATURES (EXCL METADATA) ===")
    print(f"Total columns in file: {len(cols)}")
    print(f"All-features count: {len(all_features)}")
    print("First 30 all-features:", all_features[:30])

    # Quick sanity: are theory features included in ALL?
    # (They should be, unless some are non-numeric or missing.)
    theory_missing_in_all = [c for c in present if c not in all_features]
    if theory_missing_in_all:
        print("\n[WARN] These THEORY features are present in the table but not in ALL-features (likely non-numeric dtype):")
        print(theory_missing_in_all)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())