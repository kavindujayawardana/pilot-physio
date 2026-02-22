#!/usr/bin/env python3
"""
Feature Diagnostics Script
- Missing values summary
- Low-variance / constant feature check
- Correlation matrix (heatmap + CSV)
- VIF (multicollinearity) table

Usage:
  python notebooks/feature_diagnostics.py \
    --features ./features \
    --out ./results/feature_diagnostics \
    --benchmark-only \
    --vif-mode base   # base | indices | both
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor


BENCHMARK_TYPES = {"CA", "DA", "SS"}
PREFERRED = "features_with_indices.parquet"
FALLBACK = "features.parquet"

# ---- Your "theory" features (no metadata) ----
THEORY_BASE = [
    # EEG aggregated
    "frontal_mean_alpha",
    "frontal_asymmetry_alpha",
    "theta_beta_ratio",
    # HR / HRV
    "HR_mean",
    "HR_slope",
    "HRV_RMSSD_30s",
    "HRV_SDNN_30s",
    "HRV_pNN50_30s",
    # GSR
    "GSR_mean",
    "GSR_std",
    "GSR_slope",
    "GSR_peak_count",
    # Respiration
    "breathing_rate_10s_bpm",
]

# Indices (your indices.py may name these SAI/CAI or SAI_z/CAI_z)
INDEX_CANDIDATES = ["SAI", "CAI", "SAI_z", "CAI_z"]

# Optional roll/lag features to include if you want (kept off by default)
OPTIONAL_LAGS = [
    "HR_mean_roll5s_mean",
    "HR_mean_roll10s_mean",
    "HR_mean_lag1s",
    "HR_mean_lag5s",
    "HR_mean_lag10s",
    "GSR_mean_roll5s_mean",
    "GSR_mean_roll10s_mean",
    "GSR_mean_lag1s",
    "GSR_mean_lag5s",
    "GSR_mean_lag10s",
]


def load_all_features(features_root: Path) -> pd.DataFrame:
    dfs = []
    for subj_dir in sorted([p for p in features_root.iterdir() if p.is_dir()]):
        preferred = subj_dir / PREFERRED
        fallback = subj_dir / FALLBACK
        if preferred.exists():
            df = pd.read_parquet(preferred)
            df["_source_file"] = PREFERRED
            dfs.append(df)
        elif fallback.exists():
            df = pd.read_parquet(fallback)
            df["_source_file"] = FALLBACK
            dfs.append(df)
    if not dfs:
        raise SystemExit(f"[ERROR] No feature files found under {features_root}")
    return pd.concat(dfs, ignore_index=True)


def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype == "object":
            continue
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def save_missingness(df: pd.DataFrame, out_dir: Path) -> None:
    miss = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": df.isna().mean() * 100.0
    }).sort_values("missing_pct", ascending=False)
    miss.to_csv(out_dir / "missingness_all_columns.csv")
    # also only numeric
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    miss_num = miss.loc[num_cols].sort_values("missing_pct", ascending=False)
    miss_num.to_csv(out_dir / "missingness_numeric.csv")


def save_variance_checks(df: pd.DataFrame, out_dir: Path) -> None:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    d = df[num_cols].copy()
    # replace inf
    d = d.replace([np.inf, -np.inf], np.nan)

    stats = pd.DataFrame({
        "std": d.std(skipna=True),
        "var": d.var(skipna=True),
        "unique_values": d.nunique(dropna=True),
        "zero_pct": (d == 0).mean(skipna=True) * 100.0,
    }).sort_values("std", ascending=True)

    stats.to_csv(out_dir / "numeric_variance_summary.csv")

    low_var = stats[(stats["unique_values"] <= 1) | (stats["std"].fillna(0) < 1e-12)]
    low_var.to_csv(out_dir / "constant_or_low_variance_features.csv")


def corr_and_heatmap(df: pd.DataFrame, cols: list[str], out_dir: Path, tag: str) -> None:
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if d.shape[0] < 50:
        print(f"[WARN] Too few complete rows for correlation ({tag}). Rows={d.shape[0]}")
        return

    corr = d.corr(numeric_only=True)
    corr.to_csv(out_dir / f"correlation_{tag}.csv")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title(f"Correlation heatmap ({tag})")
    plt.tight_layout()
    plt.savefig(out_dir / f"correlation_{tag}.png")
    plt.close()


def compute_vif(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    VIF requires a complete numeric matrix with no NaNs.
    We:
      - keep cols
      - drop rows with NaN
      - compute VIF per feature
    """
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if d.shape[0] < 50:
        return pd.DataFrame({"feature": cols, "VIF": np.nan, "rows_used": d.shape[0]})

    X = d.to_numpy(dtype=float, copy=False)
    vifs = []
    for i, col in enumerate(cols):
        try:
            v = variance_inflation_factor(X, i)
        except Exception:
            v = np.nan
        vifs.append(v)

    out = pd.DataFrame({"feature": cols, "VIF": vifs})
    out["rows_used"] = d.shape[0]
    out = out.sort_values("VIF", ascending=False)
    return out


def pick_feature_set(df: pd.DataFrame, mode: str, include_lags: bool) -> tuple[list[str], str]:
    base = [c for c in THEORY_BASE if c in df.columns]

    indices_present = [c for c in INDEX_CANDIDATES if c in df.columns]
    # choose ONE naming convention if both exist
    # (prefer z versions if present)
    if "SAI_z" in indices_present or "CAI_z" in indices_present:
        indices = [c for c in ["SAI_z", "CAI_z"] if c in df.columns]
    else:
        indices = [c for c in ["SAI", "CAI"] if c in df.columns]

    lags = [c for c in OPTIONAL_LAGS if c in df.columns] if include_lags else []

    if mode == "base":
        return base + lags, "base"
    if mode == "indices":
        # indices only + EEG/HRV etc? Usually you keep indices AND non-overlapping predictors.
        # Here we do: indices + base features EXCLUDING the exact base components used to build indices
        # to avoid perfect collinearity.
        # SAI uses (GSR_peak_count, HR_mean, breathing_rate_10s_bpm)
        exclude_for_sai = {"GSR_peak_count", "HR_mean", "breathing_rate_10s_bpm"}
        # CAI uses (theta_beta_ratio, frontal_mean_alpha)
        exclude_for_cai = {"theta_beta_ratio", "frontal_mean_alpha"}

        keep_base = [c for c in base if c not in (exclude_for_sai | exclude_for_cai)]
        return indices + keep_base + lags, "indices"
    if mode == "both":
        # WARNING: likely high VIF, but allowed for diagnostics
        return indices + base + lags, "both"

    raise ValueError("--vif-mode must be one of: base | indices | both")


def main() -> int:
    parser = argparse.ArgumentParser(description="Feature diagnostics: missingness, variance, correlations, VIF.")
    parser.add_argument("--features", required=True, help="Path to features/ directory")
    parser.add_argument("--out", default="./results/feature_diagnostics", help="Output directory")
    parser.add_argument("--benchmark-only", action="store_true", help="Use benchmark sessions only (CA/DA/SS)")
    parser.add_argument("--vif-mode", default="base", choices=["base", "indices", "both"],
                        help="Which feature set to use for correlation/VIF")
    parser.add_argument("--include-lags", action="store_true", help="Include lag/roll features in VIF/corr")
    args = parser.parse_args()

    features_root = Path(args.features).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_features(features_root)
    data = safe_numeric(data)

    if args.benchmark_only:
        if "SessionType" not in data.columns:
            raise SystemExit("[ERROR] SessionType not found; cannot filter benchmark-only.")
        data = data[data["SessionType"].isin(BENCHMARK_TYPES)].copy()

    # ---- 1) Missingness on everything (light summary EDA for all features) ----
    save_missingness(data, out_dir)

    # ---- 2) Variance / constants on everything (light summary EDA for all features) ----
    save_variance_checks(data, out_dir)

    # ---- 3) Full EDA diagnostics on modelling feature set (theory features / indices) ----
    feature_set, tag = pick_feature_set(data, args.vif_mode, args.include_lags)
    if len(feature_set) < 3:
        raise SystemExit(f"[ERROR] Too few features available for {tag}: {feature_set}")

    # Correlation + heatmap
    corr_and_heatmap(data, feature_set, out_dir, tag=tag)

    # VIF
    vif_df = compute_vif(data, feature_set)
    vif_df.to_csv(out_dir / f"vif_{tag}.csv", index=False)

    # Quick text summary
    top_vif = vif_df.head(20)
    top_vif.to_csv(out_dir / f"vif_{tag}_top20.csv", index=False)

    print(f"[OK] Saved diagnostics to: {out_dir}")
    print(f"[OK] Feature set ({tag}) size: {len(feature_set)}")
    print(f"[OK] VIF rows used: {int(vif_df['rows_used'].iloc[0]) if len(vif_df) else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())