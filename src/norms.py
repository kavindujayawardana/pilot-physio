#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


BENCHMARK_SUFFIXES = ("_CA_preproc.parquet", "_DA_preproc.parquet", "_SS_preproc.parquet")

EXCLUDE_COLS = {
    "TimeSecs", "Event", "IsBadWindow",
    "IsBadSample", "IsFlatline", "IsSaturated"
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_mean_std(x: np.ndarray) -> tuple[float, float]:
    x = x.astype(np.float64)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if not np.isfinite(sd) or sd == 0.0:
        sd = 1.0
    return mu, sd


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute per-subject normalization stats from benchmark sessions only.")
    parser.add_argument("--preprocessed", required=True, help="Path to preprocessed/ directory")
    parser.add_argument("--out", required=True, help="Output root (e.g., processed/)")
    args = parser.parse_args()

    preproc_root = Path(args.preprocessed).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted([p for p in preproc_root.iterdir() if p.is_dir()])
    if not subject_dirs:
        print(f"[ERROR] No subject directories found in {preproc_root}")
        return 1

    for subj_dir in subject_dirs:
        subject_id = subj_dir.name

        # 1) Collect benchmark sessions only
        bench_files = []
        for f in sorted(subj_dir.glob("*_preproc.parquet")):
            if any(f.name.endswith(suf) for suf in BENCHMARK_SUFFIXES):
                bench_files.append(f)

        if not bench_files:
            print(f"[WARN] No benchmark sessions for subject {subject_id}. Skipping.")
            continue

        print(f"[NORM] Subject {subject_id}: {len(bench_files)} benchmark sessions")

        # 2) Load and concatenate
        dfs = []
        for f in bench_files:
            df = pd.read_parquet(f)
            dfs.append(df)

        all_df = pd.concat(dfs, axis=0, ignore_index=True)

        # 3) Select numeric columns only (exclude TimeSecs/Event/flags)
        numeric_cols = [
            c for c in all_df.columns
            if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(all_df[c])
        ]

        if not numeric_cols:
            print(f"[WARN] No numeric signal columns for subject {subject_id}. Skipping.")
            continue

        norms = {
            "subject": int(subject_id) if subject_id.isdigit() else subject_id,
            "computed_from_sessions": [f.name for f in bench_files],
            "timestamp_utc": utc_now(),
            "method": "zscore_mean_std",
            "target": "apply to features/model inputs; computed only from CA/DA/SS",
            "channels": {}
        }

        # 4) Compute mean/std per channel
        for col in numeric_cols:
            mu, sd = safe_mean_std(all_df[col].to_numpy())
            norms["channels"][col] = {"mean": mu, "std": sd}

        # 5) Save norms.json to processed/<subject>/norms.json
        out_subj_dir = out_root / subject_id
        out_subj_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_subj_dir / "norms.json"
        out_path.write_text(json.dumps(norms, indent=2), encoding="utf-8")

        print(f"[OK] Saved {out_path}")

    print("\nDone computing per-subject norms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
