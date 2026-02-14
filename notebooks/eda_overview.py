#!/usr/bin/env python3
"""
EDA Overview Script (Step 9 — Exploratory Data Analysis)

What this script does
---------------------
A) Full EDA on THEORY features (≈12–20) including SAI/CAI:
   - Histograms by EventLabel
   - Boxplots by EventLabel
   - Correlation heatmap (theory features)
   - ACF up to 60s (pooled and per subject where feasible)
   - Per-subject mean/std + outlier z-score summaries
   - Time series plots (per subject): SAI/CAI + HR/GSR + state trace
   - Time-in-each-state summaries (durations)
   - State transition summaries (counts, typical durations)

B) Light "global diagnostics" EDA on ALL numeric features:
   - Missingness (%)
   - Variance (near-zero variance detection)
   - Extreme outlier rates (|z| > 3)
   - Duplicate row check
   - Class balance (counts + duration)

Inputs
------
Reads per subject:
  - features/<subject>/features_with_indices.parquet  (preferred)
  - else features/<subject>/features.parquet

Outputs
-------
Writes to:
  results/eda/
    theory/...
    global/...

Notes
-----
- By default, most plots use benchmark sessions only (CA, DA, SS).
- Robust: skips missing columns (e.g., if SAI/CAI not present yet).
- If your windowing kept transitions, and you have an "IsTransition" column,
  this script will summarize them; otherwise it proceeds without it.

Run
---
python notebooks/eda_overview.py --features ./features --out ./results/eda
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf


# ----------------------------
# Defaults (edit if needed)
# ----------------------------
BENCHMARK_TYPES = {"CA", "DA", "SS"}

# Your theory features (use what exists in your current features.parquet)
THEORY_FEATURES = [
    # EEG (aggregated)
    "frontal_mean_alpha",
    "frontal_asymmetry_alpha",
    "theta_beta_ratio",

    # Cardiovascular
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
    "R_mean",
    "R_std",
    "R_slope",
    "R_peak_count",

    # Anxiety indices (depending on your indices.py output)
    "SAI",
    "CAI",
    "SAI_z",
    "CAI_z",
]

# ACF settings: if step=1s, then 60 lags ≈ 60 seconds
ACF_LAGS = 60

# Outlier threshold for global diagnostics
Z_OUTLIER = 3.0

# How many subjects to render full time-series panels for (keeps runtime manageable)
MAX_SUBJECT_PLOTS = 6


# ----------------------------
# Helpers
# ----------------------------
def mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pick_feature_file(subject_dir: Path) -> Path | None:
    preferred = subject_dir / "features_with_indices.parquet"
    fallback = subject_dir / "features.parquet"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    return None


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def rle_durations(labels: np.ndarray, step_seconds: float = 1.0) -> pd.DataFrame:
    """
    Run-length encoding: durations of consecutive identical labels.
    Returns a dataframe with columns: label, run_len, duration_s
    """
    if labels.size == 0:
        return pd.DataFrame(columns=["label", "run_len", "duration_s"])

    changes = np.where(labels[1:] != labels[:-1])[0] + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, labels.size]
    run_len = (ends - starts).astype(int)
    run_labels = labels[starts]

    return pd.DataFrame({
        "label": run_labels,
        "run_len": run_len,
        "duration_s": run_len * step_seconds
    })


def save_simple_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, out_png: Path) -> None:
    plt.figure(figsize=(7, 4))
    series.sort_index().plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="./features", help="Path to features/")
    parser.add_argument("--out", default="./results/eda", help="Output folder")
    parser.add_argument("--include-loft", action="store_true",
                        help="If set, EDA includes LOFT sessions too (default: benchmark only for most plots).")
    parser.add_argument("--max-subject-plots", type=int, default=MAX_SUBJECT_PLOTS)
    args = parser.parse_args()

    features_root = Path(args.features).resolve()
    out_root = Path(args.out).resolve()

    # Folder layout
    theory_root = out_root / "theory"
    global_root = out_root / "global"

    folders = {
        # Theory EDA
        "t01_hist": theory_root / "01_histograms",
        "t02_box": theory_root / "02_boxplots",
        "t03_corr": theory_root / "03_correlations",
        "t04_class": theory_root / "04_class_balance",
        "t05_acf": theory_root / "05_acf",
        "t06_subject": theory_root / "06_subject_summaries",
        "t07_timeseries": theory_root / "07_timeseries",
        "t08_transitions": theory_root / "08_state_transitions",

        # Global diagnostics
        "g01_missing": global_root / "01_missingness",
        "g02_variance": global_root / "02_variance",
        "g03_outliers": global_root / "03_outliers",
        "g04_duplicates": global_root / "04_duplicates",
        "g05_allcorr": global_root / "05_all_feature_summaries",
    }
    for p in folders.values():
        mkdir(p)

    # ----------------------------
    # Load all subject features
    # ----------------------------
    dfs = []
    for subject_dir in sorted([p for p in features_root.iterdir() if p.is_dir()]):
        fpath = pick_feature_file(subject_dir)
        if fpath is None:
            continue
        df = pd.read_parquet(fpath)
        df["_source_file"] = fpath.name
        dfs.append(df)

    if not dfs:
        raise SystemExit(f"[ERROR] No feature files found under {features_root}")

    data = pd.concat(dfs, ignore_index=True)

    # Basic required columns checks
    required_cols = ["SubjectID", "SessionType", "SessionName", "EventLabel"]
    for c in required_cols:
        if c not in data.columns:
            print(f"[WARN] Missing column '{c}' in combined data. Some plots/summaries may be skipped.")

    # Decide benchmark vs include-loft
    if args.include_loft:
        analysis_df = data.copy()
        analysis_tag = "all_sessions"
    else:
        if "SessionType" in data.columns:
            analysis_df = data[data["SessionType"].isin(BENCHMARK_TYPES)].copy()
        else:
            analysis_df = data.copy()
        analysis_tag = "benchmark_only"

    # If your pipeline adds IsTransition (recommended), use it for summaries.
    has_transition = "IsTransition" in analysis_df.columns
    if has_transition:
        transition_counts = analysis_df["IsTransition"].value_counts(dropna=False)
        transition_counts.to_csv(folders["t04_class"] / f"transition_counts_{analysis_tag}.csv")
    else:
        transition_counts = None

    # ----------------------------
    # Identify numeric columns
    # ----------------------------
    numeric_cols = [c for c in analysis_df.columns if pd.api.types.is_numeric_dtype(analysis_df[c])]
    # Remove obvious metadata-like numeric columns (keep StartTime etc. if you want)
    # We'll still include them in global diagnostics unless you want to exclude.

    # Theory feature intersection
    theory_features = [f for f in THEORY_FEATURES if f in analysis_df.columns]
    if not theory_features:
        print("[WARN] None of the THEORY_FEATURES were found in the dataset.")
        # Continue anyway: global diagnostics still useful.

    # Save which theory features were used
    pd.Series(theory_features).to_csv(folders["t06_subject"] / "theory_features_used.csv", index=False)

    # Ensure numeric coercion for theory features
    analysis_df = safe_numeric(analysis_df, theory_features)

    # ----------------------------
    # THEORY EDA (Full)
    # ----------------------------

    # (1) Class balance (counts + approximate duration)
    if "EventLabel" in analysis_df.columns:
        counts = analysis_df["EventLabel"].value_counts()
        counts.to_csv(folders["t04_class"] / f"event_counts_{analysis_tag}.csv")
        save_simple_bar(
            counts,
            f"Event counts ({analysis_tag})",
            "EventLabel",
            "Windows",
            folders["t04_class"] / f"event_counts_{analysis_tag}.png",
        )

        # Duration approximation: windows updated every step_s.
        # If you have step_s column, use it; else assume 1s update (your current setup).
        if "step_s" in analysis_df.columns:
            step_s = float(pd.to_numeric(analysis_df["step_s"], errors="coerce").dropna().mode().iloc[0])
        else:
            step_s = 1.0

        duration_s = counts * step_s
        duration_s.to_csv(folders["t04_class"] / f"event_duration_seconds_{analysis_tag}.csv")

        save_simple_bar(
            duration_s,
            f"Approx duration by state (seconds) ({analysis_tag})",
            "EventLabel",
            "Seconds",
            folders["t04_class"] / f"event_duration_seconds_{analysis_tag}.png",
        )

    # (2) Histograms by EventLabel (theory features)
    if theory_features and "EventLabel" in analysis_df.columns:
        for feat in theory_features:
            # Skip extremely sparse columns
            s = pd.to_numeric(analysis_df[feat], errors="coerce")
            if s.notna().sum() < 200:
                continue

            plt.figure(figsize=(7, 4))
            sns.histplot(
                data=analysis_df,
                x=feat,
                hue="EventLabel",
                bins=40,
                kde=True,
                stat="density",
                common_norm=False,
            )
            plt.title(f"Histogram: {feat} by EventLabel ({analysis_tag})")
            plt.tight_layout()
            plt.savefig(folders["t01_hist"] / f"hist_{feat}_{analysis_tag}.png")
            plt.close()

    # (3) Boxplots by EventLabel (theory features)
    if theory_features and "EventLabel" in analysis_df.columns:
        for feat in theory_features:
            s = pd.to_numeric(analysis_df[feat], errors="coerce")
            if s.notna().sum() < 200:
                continue

            plt.figure(figsize=(7, 4))
            sns.boxplot(data=analysis_df, x="EventLabel", y=feat)
            plt.title(f"Boxplot: {feat} vs EventLabel ({analysis_tag})")
            plt.tight_layout()
            plt.savefig(folders["t02_box"] / f"box_{feat}_{analysis_tag}.png")
            plt.close()

    # (4) Correlation heatmap (theory features only)
    if len(theory_features) >= 2:
        corr = analysis_df[theory_features].corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
        plt.title(f"Theory feature correlation ({analysis_tag})")
        plt.tight_layout()
        plt.savefig(folders["t03_corr"] / f"theory_corr_{analysis_tag}.png")
        plt.close()

    # (5) ACF pooled for key features (up to 60s)
    acf_features = [f for f in ["HR_mean", "GSR_mean", "breathing_rate_10s_bpm", "SAI", "CAI", "SAI_z", "CAI_z"] if f in analysis_df.columns]
    for feat in acf_features:
        s = pd.to_numeric(analysis_df[feat], errors="coerce").dropna()
        if len(s) < 500:
            continue
        plt.figure(figsize=(8, 4))
        plot_acf(s, lags=ACF_LAGS)
        plt.title(f"ACF (pooled): {feat} ({analysis_tag})")
        plt.tight_layout()
        plt.savefig(folders["t05_acf"] / f"acf_pooled_{feat}_{analysis_tag}.png")
        plt.close()

    # (6) Per-subject mean/std and outlier summaries (theory features)
    if theory_features and "SubjectID" in analysis_df.columns:
        subj_stats = analysis_df.groupby("SubjectID")[theory_features].agg(["mean", "std"])
        subj_stats.to_csv(folders["t06_subject"] / f"subject_mean_std_{analysis_tag}.csv")

        # Outlier summary: z-score per subject across benchmark pooled distribution
        # (use global mean/std of benchmark/all_sessions to compute z; count extreme points per subject)
        mu = analysis_df[theory_features].mean(numeric_only=True)
        sd = analysis_df[theory_features].std(numeric_only=True).replace(0, np.nan)

        z = (analysis_df[theory_features] - mu) / sd
        extreme = (np.abs(z) > Z_OUTLIER)

        extreme_rate = extreme.groupby(analysis_df["SubjectID"]).mean(numeric_only=True) * 100.0
        extreme_rate.to_csv(folders["t06_subject"] / f"subject_extreme_zpct_{analysis_tag}.csv")

    # (7) Time in each state + transition summaries (per subject/session)
    # This needs EventLabel and SessionName ideally.
    if "EventLabel" in analysis_df.columns and "SubjectID" in analysis_df.columns and "SessionName" in analysis_df.columns:
        # Determine step seconds
        if "step_s" in analysis_df.columns:
            step_s = float(pd.to_numeric(analysis_df["step_s"], errors="coerce").dropna().mode().iloc[0])
        else:
            step_s = 1.0

        durations_rows = []
        transitions_rows = []

        # Sort for stable run-length encoding:
        sort_cols = [c for c in ["SubjectID", "SessionName", "StartTime", "StartIdx"] if c in analysis_df.columns]
        df_sorted = analysis_df.sort_values(sort_cols) if sort_cols else analysis_df

        for (sid, sess), g in df_sorted.groupby(["SubjectID", "SessionName"]):
            labels = pd.to_numeric(g["EventLabel"], errors="coerce").fillna(0).to_numpy(dtype=int)

            # Time in each state (windows * step_s)
            vc = pd.Series(labels).value_counts()
            for label, cnt in vc.items():
                durations_rows.append({
                    "SubjectID": sid,
                    "SessionName": sess,
                    "EventLabel": int(label),
                    "windows": int(cnt),
                    "duration_s": float(cnt * step_s),
                })

            # Transition/run-length
            runs = rle_durations(labels, step_seconds=step_s)
            if not runs.empty:
                transitions_rows.append({
                    "SubjectID": sid,
                    "SessionName": sess,
                    "num_transitions": int((runs.shape[0] - 1) if runs.shape[0] > 0 else 0),
                    "mean_run_duration_s": float(runs["duration_s"].mean()),
                    "median_run_duration_s": float(runs["duration_s"].median()),
                })

        durations_df = pd.DataFrame(durations_rows)
        transitions_df = pd.DataFrame(transitions_rows)

        durations_df.to_csv(folders["t08_transitions"] / f"time_in_state_by_session_{analysis_tag}.csv", index=False)
        transitions_df.to_csv(folders["t08_transitions"] / f"transition_summary_by_session_{analysis_tag}.csv", index=False)

        # Aggregate time in each state across all subjects/sessions
        agg_dur = durations_df.groupby("EventLabel")["duration_s"].sum().sort_index()
        agg_dur.to_csv(folders["t08_transitions"] / f"time_in_state_total_{analysis_tag}.csv")
        save_simple_bar(
            agg_dur,
            f"Total time in each state (seconds) ({analysis_tag})",
            "EventLabel",
            "Seconds",
            folders["t08_transitions"] / f"time_in_state_total_{analysis_tag}.png",
        )

    # (8) Time series panels: SAI/CAI (or z versions) + HR/GSR + state trace
    if "SubjectID" in analysis_df.columns and ("StartTime" in analysis_df.columns or "StartIdx" in analysis_df.columns):
        time_col = "StartTime" if "StartTime" in analysis_df.columns else "StartIdx"

        plot_subjects = list(analysis_df["SubjectID"].dropna().unique())[: args.max_subject_plots]

        # Choose which index columns exist
        idx_cols = [c for c in ["SAI_z", "CAI_z", "SAI", "CAI"] if c in analysis_df.columns]
        phys_cols = [c for c in ["HR_mean", "GSR_mean", "breathing_rate_10s_bpm"] if c in analysis_df.columns]

        for sid in plot_subjects:
            g = analysis_df[analysis_df["SubjectID"] == sid].sort_values([c for c in [time_col] if c in analysis_df.columns])

            # Plot: state over time
            if "EventLabel" in g.columns:
                plt.figure(figsize=(12, 2.8))
                plt.plot(g[time_col], pd.to_numeric(g["EventLabel"], errors="coerce"), linewidth=1)
                plt.title(f"State over time — Subject {sid} ({analysis_tag})")
                plt.xlabel(time_col)
                plt.ylabel("EventLabel")
                plt.tight_layout()
                plt.savefig(folders["t07_timeseries"] / f"state_subject_{sid}_{analysis_tag}.png")
                plt.close()

            # Plot each index / physiological feature separately (keeps simple)
            for feat in idx_cols + phys_cols:
                s = pd.to_numeric(g[feat], errors="coerce")
                if s.notna().sum() < 200:
                    continue
                plt.figure(figsize=(12, 3.2))
                plt.plot(g[time_col], s, linewidth=1)
                plt.title(f"{feat} time series — Subject {sid} ({analysis_tag})")
                plt.xlabel(time_col)
                plt.ylabel(feat)
                plt.tight_layout()
                plt.savefig(folders["t07_timeseries"] / f"{feat}_subject_{sid}_{analysis_tag}.png")
                plt.close()

    # ----------------------------
    # GLOBAL DIAGNOSTICS (Light EDA on ALL features)
    # ----------------------------
    # Use the same analysis_df (benchmark only unless include-loft set).
    # We’ll focus on numeric columns excluding obvious IDs where helpful.
    numeric_df = analysis_df.select_dtypes(include=[np.number]).copy()

    # (G1) Missingness
    missing_pct = (numeric_df.isna().mean() * 100.0).sort_values(ascending=False)
    missing_pct.to_csv(folders["g01_missing"] / f"missingness_pct_{analysis_tag}.csv")

    # (G2) Variance (near-zero variance)
    var = numeric_df.var(numeric_only=True).sort_values(ascending=True)
    var.to_csv(folders["g02_variance"] / f"variance_{analysis_tag}.csv")

    near_zero = var[var.fillna(0) < 1e-8]
    near_zero.to_csv(folders["g02_variance"] / f"near_zero_variance_{analysis_tag}.csv")

    # (G3) Extreme outlier rates: |z| > 3
    # Avoid division by 0
    g_mu = numeric_df.mean(numeric_only=True)
    g_sd = numeric_df.std(numeric_only=True).replace(0, np.nan)
    z = (numeric_df - g_mu) / g_sd
    extreme_rate = (np.abs(z) > Z_OUTLIER).mean(numeric_only=True) * 100.0
    extreme_rate.sort_values(ascending=False).to_csv(folders["g03_outliers"] / f"extreme_z_rate_pct_{analysis_tag}.csv")

    # (G4) Duplicate row check (entire row duplicates)
    dup_count = int(analysis_df.duplicated().sum())
    (folders["g04_duplicates"] / f"duplicates_{analysis_tag}.txt").write_text(
        f"Duplicate full rows: {dup_count}\n",
        encoding="utf-8"
    )

    # (G5) Save a concise “global feature summary” table
    summary = pd.DataFrame({
        "missing_pct": missing_pct,
        "variance": var.reindex(missing_pct.index),
        "extreme_z_pct": extreme_rate.reindex(missing_pct.index),
    })
    summary.to_csv(folders["g05_allcorr"] / f"global_feature_summary_{analysis_tag}.csv")

    # ----------------------------
    # Done
    # ----------------------------
    print(f"\n[EDA] Completed ({analysis_tag}). Outputs saved to: {out_root}")
    if transition_counts is not None:
        print("[EDA] Transition summary found (IsTransition). Saved counts.")
    return 0


if __name__ == "__main__":
    # Silence a few noisy warnings during plotting
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    raise SystemExit(main())