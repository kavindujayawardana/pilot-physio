#!/usr/bin/env python3
"""
EDA Overview Script
Step 9 — Exploratory Data Analysis

Reads (per subject):
  - features/<subject>/features_with_indices.parquet  (preferred, if exists)
  - else features/<subject>/features.parquet

Writes:
  results/eda/<step_name>/*

Notes:
- Benchmark sessions only (CA, DA, SS)
- Script is robust: skips features not present (e.g., SAI/CAI if indices not computed yet)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

# ----------------------------
# Configuration
# ----------------------------
FEATURE_ROOT = Path("features")
EDA_ROOT = Path("results/eda")
BENCHMARK_TYPES = {"CA", "DA", "SS"}

# Prefer indices if present
PREFERRED_FEATURE_FILENAME = "features_with_indices.parquet"
FALLBACK_FEATURE_FILENAME = "features.parquet"

# Your intended "key" features (script will filter to only those that exist)
KEY_FEATURES = [
    "HR_mean",
    "GSR_mean",
    "frontal_mean_alpha",
    "theta_beta_ratio",
    "SAI",
    "CAI",
]

# ----------------------------
# Create EDA subfolders
# ----------------------------
folders = {
    "01_histograms": EDA_ROOT / "01_histograms",
    "02_boxplots": EDA_ROOT / "02_boxplots",
    "03_correlations": EDA_ROOT / "03_correlations",
    "04_event_balance": EDA_ROOT / "04_event_balance",
    "05_acf": EDA_ROOT / "05_acf",
    "06_subject_variance": EDA_ROOT / "06_subject_variance",
    "07_event_locked": EDA_ROOT / "07_event_locked",
    "08_event_study": EDA_ROOT / "08_event_study",
    "09_timeseries": EDA_ROOT / "09_timeseries",
}
for f in folders.values():
    f.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load and combine features (robust)
# ----------------------------
dfs = []

# For each subject folder under features/
for subject_dir in sorted([p for p in FEATURE_ROOT.iterdir() if p.is_dir()]):
    preferred = subject_dir / PREFERRED_FEATURE_FILENAME
    fallback = subject_dir / FALLBACK_FEATURE_FILENAME

    if preferred.exists():
        df = pd.read_parquet(preferred)
        df["_source_file"] = PREFERRED_FEATURE_FILENAME
        dfs.append(df)
    elif fallback.exists():
        df = pd.read_parquet(fallback)
        df["_source_file"] = FALLBACK_FEATURE_FILENAME
        dfs.append(df)
    else:
        # no usable file for this subject
        continue

if not dfs:
    raise SystemExit(f"[ERROR] No feature files found under {FEATURE_ROOT}")

data = pd.concat(dfs, ignore_index=True)

# Benchmark only
benchmark = data[data["SessionType"].isin(BENCHMARK_TYPES)].copy()

# ----------------------------
# Select only features that exist in the loaded dataset
# ----------------------------
available_features = [f for f in KEY_FEATURES if f in benchmark.columns]
missing_features = [f for f in KEY_FEATURES if f not in benchmark.columns]

# Save a small log so you can show progress
pd.Series(available_features).to_csv(folders["04_event_balance"] / "available_features_used.csv", index=False)
pd.Series(missing_features).to_csv(folders["04_event_balance"] / "missing_features_skipped.csv", index=False)

if not available_features:
    raise SystemExit("[ERROR] None of KEY_FEATURES exist in the loaded data. Check your feature files.")

print("[EDA] Using features:", available_features)
if missing_features:
    print("[EDA] Skipping missing features:", missing_features)

# ----------------------------
# 1. Histograms per feature
# ----------------------------
for feat in available_features:
    if "EventLabel" not in benchmark.columns:
        continue

    plt.figure(figsize=(6, 4))
    sns.histplot(
        data=benchmark,
        x=feat,
        hue="EventLabel",
        bins=40,
        kde=True,
        stat="density",
        common_norm=False,
    )
    plt.title(f"Histogram: {feat} by EventLabel")
    plt.tight_layout()
    plt.savefig(folders["01_histograms"] / f"hist_{feat}.png")
    plt.close()

# ----------------------------
# 2. Boxplots
# ----------------------------
for feat in available_features:
    if "EventLabel" not in benchmark.columns:
        continue

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=benchmark, x="EventLabel", y=feat)
    plt.title(f"Boxplot: {feat} vs EventLabel")
    plt.tight_layout()
    plt.savefig(folders["02_boxplots"] / f"box_{feat}.png")
    plt.close()

# ----------------------------
# 3. Correlation heatmap
# ----------------------------
corr_df = benchmark[available_features].corr(numeric_only=True)
plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, cmap="coolwarm", center=0, annot=False)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(folders["03_correlations"] / "feature_correlation_heatmap.png")
plt.close()

# ----------------------------
# 4. Event balance & duration
# ----------------------------
if "EventLabel" in benchmark.columns:
    event_counts = benchmark["EventLabel"].value_counts()
    event_counts.to_csv(folders["04_event_balance"] / "event_counts.csv")

    plt.figure(figsize=(6, 4))
    event_counts.sort_index().plot(kind="bar")
    plt.title("Event Window Counts (Benchmark)")
    plt.xlabel("EventLabel")
    plt.ylabel("Number of Windows")
    plt.tight_layout()
    plt.savefig(folders["04_event_balance"] / "event_counts.png")
    plt.close()

# ----------------------------
# 5. ACF plots (temporal dependence)
# ----------------------------
for feat in [f for f in ["HR_mean", "SAI", "CAI"] if f in benchmark.columns]:
    series = pd.to_numeric(benchmark[feat], errors="coerce").dropna()
    if len(series) < 200:
        continue
    plot_acf(series, lags=60)
    plt.title(f"ACF: {feat}")
    plt.tight_layout()
    plt.savefig(folders["05_acf"] / f"acf_{feat}.png")
    plt.close()

# ----------------------------
# 6. Per-subject variance summaries
# ----------------------------
if "SubjectID" in benchmark.columns:
    subject_stats = benchmark.groupby("SubjectID")[available_features].agg(["mean", "std"])
    subject_stats.to_csv(folders["06_subject_variance"] / "subject_feature_stats.csv")

# ----------------------------
# 7. Visual traces around events (window-based approximation)
# ----------------------------
# Mean trace by label over time
if "StartTime" in benchmark.columns and "EventLabel" in benchmark.columns:
    for feat in [f for f in ["SAI", "CAI"] if f in benchmark.columns]:
        plt.figure(figsize=(8, 4))
        sns.lineplot(
            data=benchmark,
            x="StartTime",
            y=feat,
            hue="EventLabel",
            estimator="mean",
            errorbar=None,
        )
        plt.title(f"Event-aligned Mean Trace: {feat}")
        plt.tight_layout()
        plt.savefig(folders["07_event_locked"] / f"event_locked_{feat}.png")
        plt.close()

# ----------------------------
# 8. Preliminary event-study plots (same as above but kept separate folder)
# ----------------------------
if "StartTime" in benchmark.columns and "EventLabel" in benchmark.columns:
    for feat in [f for f in ["SAI", "CAI"] if f in benchmark.columns]:
        plt.figure(figsize=(8, 4))
        sns.lineplot(
            data=benchmark,
            x="StartTime",
            y=feat,
            hue="EventLabel",
            estimator="mean",
            errorbar=None,
        )
        plt.title(f"Event Study Plot: {feat}")
        plt.tight_layout()
        plt.savefig(folders["08_event_study"] / f"event_study_{feat}.png")
        plt.close()

# ----------------------------
# 9. Time series per subject
# ----------------------------
if "SubjectID" in benchmark.columns and "StartTime" in benchmark.columns:
    for sid, df_s in benchmark.groupby("SubjectID"):
        for feat in [f for f in ["SAI", "CAI"] if f in benchmark.columns]:
            plt.figure(figsize=(10, 3))
            plt.plot(df_s["StartTime"], df_s[feat], linewidth=1)
            plt.title(f"{feat} Time Series — Subject {sid}")
            plt.xlabel("Time (s)")
            plt.ylabel(feat)
            plt.tight_layout()
            plt.savefig(folders["09_timeseries"] / f"{feat}_subject_{sid}.png")
            plt.close()

print("\n[EDA] Completed. Outputs saved to results/eda/")
