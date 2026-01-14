#!/usr/bin/env python3
"""
src/indices.py — Step 8: Anxiety indices (SAI/CAI)

Reads:
  features/<subject>/features.parquet

Writes:
  features/<subject>/features_with_indices.parquet
  features/indices_spec.json  (if missing; created once)

Key rules:
- Normalization is PER SUBJECT.
- Normalization stats are computed ONLY from benchmark sessions (CA, DA, SS).
- LOFT is excluded from stats to avoid leakage.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
BENCHMARK_TYPES = {"CA", "DA", "SS"}  # exclude LOFT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def zscore_with_stats(x: pd.Series, mu: float, sd: float) -> pd.Series:
    return (x - mu) / (sd + EPS)


def mean_std_from_benchmark(df: pd.DataFrame, col: str) -> Tuple[float, float]:
    """
    Compute mean and std using benchmark rows only (CA/DA/SS).
    Returns (mu, sd). If sd=0 or not enough data, sd becomes 1 to avoid blowups.
    """
    bench = df[df["SessionType"].isin(BENCHMARK_TYPES)]
    s = pd.to_numeric(bench[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 5:
        # Not enough values -> fallback (avoid crashing)
        return float(s.mean()) if len(s) else 0.0, 1.0
    mu = float(s.mean())
    sd = float(s.std(ddof=1))
    if not np.isfinite(sd) or sd < EPS:
        sd = 1.0
    return mu, sd


def ensure_indices_spec(out_root: Path) -> None:
    spec_path = out_root / "indices_spec.json"
    if spec_path.exists():
        return

    spec = {
        "version": "1.0",
        "created_utc": utc_now(),
        "normalization": {
            "scope": "per_subject",
            "stats_source": "benchmark_only",
            "benchmark_session_types": sorted(list(BENCHMARK_TYPES)),
            "exclude_session_types": ["LOFT"],
            "zscore": "z = (x - mu_subject) / (sd_subject + 1e-12)",
            "index_standardization": "Final indices are z-scored again using benchmark windows only"
        },
        "indices": {
            "SAI": {
                "name": "Somatic Anxiety Index",
                "definition": "SAI_raw = z(GSR_phasic_rate) + z(HR_mean) + z(resp_irregularity); SAI = z(SAI_raw)",
                "components": [
                    {"feature": "GSR_phasic_rate", "source": "SCR_count_10s / 10.0", "weight": 1.0},
                    {"feature": "HR_mean", "source": "HR_mean", "weight": 1.0},
                    {"feature": "respiration_irregularity", "source": "R_std (proxy)", "weight": 1.0}
                ],
                "notes": [
                    "GSR phasic rate computed from SCR_count_10s converted to peaks per second.",
                    "Respiration irregularity approximated using R_std due to absence of breath-interval variability features."
                ]
            },
            "CAI": {
                "name": "Cognitive Anxiety Index",
                "definition": "CAI_raw = z(frontal_theta) - z(frontal_alpha); CAI = z(CAI_raw)",
                "components": [
                    {"feature": "frontal_theta", "source": "mean of FP1/FP2/F3/F4/Fz theta band power", "weight": 1.0, "sign": "+"},
                    {"feature": "frontal_alpha", "source": "frontal_mean_alpha", "weight": 1.0, "sign": "-"}
                ]
            }
        }
    }

    spec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)


def compute_indices_for_subject(features_path: Path, out_subject_dir: Path) -> None:
    df = pd.read_parquet(features_path)

    # Basic safety checks
    required = {"SessionType", "HR_mean", "SCR_count_10s", "R_std", "frontal_mean_alpha"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {features_path}: {missing}")

    # --- Build component features ---
    # GSR phasic rate: SCR peaks per second over 10s
    df["GSR_phasic_rate"] = pd.to_numeric(df["SCR_count_10s"], errors="coerce") / 10.0

    # Resp irregularity proxy
    df["respiration_irregularity"] = pd.to_numeric(df["R_std"], errors="coerce")

    # Frontal theta (compute from available EEG theta columns)
    frontal_theta_cols = ["EEG_FP1_theta", "EEG_FP2_theta", "EEG_F3_theta", "EEG_F4_theta", "EEG_Fz_theta"]
    for c in frontal_theta_cols:
        if c not in df.columns:
            raise ValueError(f"Missing frontal theta column: {c} in {features_path}")

    df["frontal_theta"] = df[frontal_theta_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # Frontal alpha is already computed
    df["frontal_alpha"] = pd.to_numeric(df["frontal_mean_alpha"], errors="coerce")

    # --- Compute per-subject benchmark stats for each component ---
    stats: Dict[str, Dict[str, float]] = {}
    for col in ["GSR_phasic_rate", "HR_mean", "respiration_irregularity", "frontal_theta", "frontal_alpha"]:
        mu, sd = mean_std_from_benchmark(df, col)
        stats[col] = {"mean": mu, "std": sd}

    # --- z-score components using benchmark μ/σ ---
    df["z_GSR_phasic_rate"] = zscore_with_stats(df["GSR_phasic_rate"], stats["GSR_phasic_rate"]["mean"], stats["GSR_phasic_rate"]["std"])
    df["z_HR_mean"] = zscore_with_stats(df["HR_mean"], stats["HR_mean"]["mean"], stats["HR_mean"]["std"])
    df["z_resp_irregularity"] = zscore_with_stats(df["respiration_irregularity"], stats["respiration_irregularity"]["mean"], stats["respiration_irregularity"]["std"])

    df["z_frontal_theta"] = zscore_with_stats(df["frontal_theta"], stats["frontal_theta"]["mean"], stats["frontal_theta"]["std"])
    df["z_frontal_alpha"] = zscore_with_stats(df["frontal_alpha"], stats["frontal_alpha"]["mean"], stats["frontal_alpha"]["std"])

    # --- Build raw indices ---
    df["SAI_raw"] = df["z_GSR_phasic_rate"] + df["z_HR_mean"] + df["z_resp_irregularity"]
    df["CAI_raw"] = df["z_frontal_theta"] - df["z_frontal_alpha"]

    # --- Standardize indices again using benchmark windows only ---
    sai_mu, sai_sd = mean_std_from_benchmark(df.assign(SAI_raw=df["SAI_raw"]), "SAI_raw")
    cai_mu, cai_sd = mean_std_from_benchmark(df.assign(CAI_raw=df["CAI_raw"]), "CAI_raw")

    df["SAI"] = zscore_with_stats(df["SAI_raw"], sai_mu, sai_sd)
    df["CAI"] = zscore_with_stats(df["CAI_raw"], cai_mu, cai_sd)

    # Validity flags (useful later)
    df["SAI_valid"] = df[["GSR_phasic_rate", "HR_mean", "respiration_irregularity"]].notna().all(axis=1)
    df["CAI_valid"] = df[["frontal_theta", "frontal_alpha"]].notna().all(axis=1)

    # Save per-subject stats for transparency (optional but very useful)
    stats_payload = {
        "subject": str(df["SubjectID"].iloc[0]) if "SubjectID" in df.columns and len(df) else out_subject_dir.name,
        "created_utc": utc_now(),
        "benchmark_session_types": sorted(list(BENCHMARK_TYPES)),
        "component_stats": stats,
        "index_stats": {
            "SAI_raw": {"mean": sai_mu, "std": sai_sd},
            "CAI_raw": {"mean": cai_mu, "std": cai_sd}
        }
    }
    out_subject_dir.mkdir(parents=True, exist_ok=True)
    with open(out_subject_dir / "indices_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2)

    out_path = out_subject_dir / "features_with_indices.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[OK] Saved {out_path} ({len(df)} rows, {df.shape[1]} cols)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Path to features/ root (contains <subject>/features.parquet)")
    ap.add_argument("--out", required=True, help="Output root (usually ./features)")
    args = ap.parse_args()

    features_root = Path(args.features).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    ensure_indices_spec(out_root)

    # Find all per-subject features.parquet
    files = sorted(features_root.rglob("features.parquet"))
    if not files:
        print(f"[ERROR] No features.parquet found under {features_root}")
        return 1

    for fp in files:
        # subject directory is parent name
        subject_dir = fp.parent.name
        out_subject_dir = out_root / subject_dir
        try:
            compute_indices_for_subject(fp, out_subject_dir)
        except Exception as e:
            print(f"[ERROR] Failed {fp}: {type(e).__name__}: {e}")

    print("\nDone indices.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
