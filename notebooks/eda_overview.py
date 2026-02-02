#!/usr/bin/env python3
"""
EDA Overview Script (Extended)
Step 9 — Exploratory Data Analysis (EDA)

Reads (per subject):
  - features/<subject>/features_with_indices.parquet  (preferred, if exists)
  - else features/<subject>/features.parquet

Writes:
  results/eda/<step_name>/*

Includes:
  01) Histograms per feature (per label)
  02) Boxplots: feature vs EventLabel
  03) Correlation heatmap (features)
  04) Event counts + approximate cumulative duration per label
  05) ACF up to 60s for multiple variables (pooled + per-subject optional)
  06) Per-subject variance/outlier summaries
  07) Visual traces around event onsets (simple window-based view)
  08) Preliminary event-study plot of mean SAI & CAI around events
  09) Time series plots (features over time per subject/session)
  10) Time series plot of EventLabel (state timeline) + state transition counts
  11) Data integrity: missing values, duplicates, infs, basic outlier counts

Notes:
- Benchmark sessions only by default (CA, DA, SS). LOFT excluded unless you change it.
- Robust: skips missing features (e.g., SAI/CAI if indices not computed).
- Assumes windowed features already exist and include: SubjectID, SessionName, SessionType,
  StartTime, EventLabel, etc. If some columns are missing, those outputs are skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# seaborn is fine for EDA here (your environment already has it)
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf


# ----------------------------
# Configuration
# ----------------------------
FEATURE_ROOT = Path("features")
EDA_ROOT = Path("results/eda")
BENCHMARK_TYPES = {"CA", "DA", "SS"}  # benchmark only

PREFERRED_FEATURE_FILENAME = "features_with_indices.parquet"
FALLBACK_FEATURE_FILENAME = "features.parquet"

# Your intended "key" features (script will filter to only those that exist)
KEY_FEATURES = [
    "HR_mean",
    "HR_slope",
    "GSR_mean",
    "GSR_peak_count",
    "frontal_mean_alpha",
    "frontal_asymmetry_alpha",
    "theta_beta_ratio",
    "breathing_rate_10s_bpm",
    "SCL_mean_10s",
    "SCR_count_10s",
    "SAI",
    "CAI",
]

# ACF settings
ACF_MAX_SECONDS = 60          # plot up to 60 seconds lag
DEFAULT_STEP_SECONDS = 1.0    # if step_s exists, we use it; else assume 1s
MIN_SERIES_LEN_FOR_ACF = 300  # avoid noisy tiny series

# Outlier rule (simple, transparent): |z| > 3 (pooled), per feature
OUTLIER_Z = 3.0


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_step_seconds(df: pd.DataFrame) -> float:
    if "step_s" in df.columns:
        v = pd.to_numeric(df["step_s"], errors="coerce")
        if v.notna().any():
            return float(v.dropna().iloc[0])
    return DEFAULT_STEP_SECONDS


def robust_read_features(subject_dir: Path) -> Tuple[pd.DataFrame | None, str | None]:
    preferred = subject_dir / PREFERRED_FEATURE_FILENAME
    fallback = subject_dir / FALLBACK_FEATURE_FILENAME

    if preferred.exists():
        df = pd.read_parquet(preferred)
        return df, PREFERRED_FEATURE_FILENAME
    if fallback.exists():
        df = pd.read_parquet(fallback)
        return df, FALLBACK_FEATURE_FILENAME
    return None, None


def compute_state_durations(df: pd.DataFrame, time_col: str = "StartTime", label_col: str = "EventLabel") -> pd.DataFrame:
    """
    Compute "time in each state" and number of episodes using the window-level timeline.

    Assumes rows are window-starts; durations approximated by step size.
    """
    if time_col not in df.columns or label_col not in df.columns:
        return pd.DataFrame()

    df = df.sort_values([time_col]).copy()
    step_s = get_step_seconds(df)
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")

    # total time = number of windows * step_s
    counts = df[label_col].value_counts(dropna=False).sort_index()
    total_seconds = counts * step_s

    # episodes: count transitions into label
    labels = df[label_col].to_numpy()
    episodes = {}
    if len(labels) > 0:
        prev = labels[0]
        episodes[int(prev) if pd.notna(prev) else -1] = 1
        for cur in labels[1:]:
            if cur != prev:
                k = int(cur) if pd.notna(cur) else -1
                episodes[k] = episodes.get(k, 0) + 1
                prev = cur

    out = pd.DataFrame({
        "EventLabel": counts.index.astype(str),
        "window_count": counts.values,
        "approx_duration_seconds": total_seconds.values,
        "approx_duration_minutes": total_seconds.values / 60.0,
        "episodes": [episodes.get(int(x) if x != "nan" else -1, 0) for x in counts.index.astype(str)],
    })
    return out


def plot_state_timeline(df: pd.DataFrame, out_path: Path, title: str) -> None:
    """
    Plot EventLabel vs StartTime as a step plot (state transitions over time).
    """
    if "StartTime" not in df.columns or "EventLabel" not in df.columns:
        return

    d = df.sort_values("StartTime").copy()
    t = pd.to_numeric(d["StartTime"], errors="coerce")
    y = pd.to_numeric(d["EventLabel"], errors="coerce")

    mask = t.notna() & y.notna()
    t = t[mask]
    y = y[mask]

    if len(t) < 5:
        return

    plt.figure(figsize=(12, 3))
    plt.step(t, y, where="post", linewidth=1)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("EventLabel (state)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_timeseries(df: pd.DataFrame, feat: str, out_path: Path, title: str) -> None:
    if "StartTime" not in df.columns or feat not in df.columns:
        return
    d = df.sort_values("StartTime").copy()
    t = pd.to_numeric(d["StartTime"], errors="coerce")
    x = pd.to_numeric(d[feat], errors="coerce")
    mask = t.notna() & x.notna()
    t = t[mask]
    x = x[mask]
    if len(t) < 5:
        return

    plt.figure(figsize=(12, 3))
    plt.plot(t, x, linewidth=1)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(feat)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> int:
    # ----------------------------
    # Create EDA subfolders
    # ----------------------------
    folders = {
        "00_data_integrity": EDA_ROOT / "00_data_integrity",
        "01_histograms": EDA_ROOT / "01_histograms",
        "02_boxplots": EDA_ROOT / "02_boxplots",
        "03_correlations": EDA_ROOT / "03_correlations",
        "04_event_balance": EDA_ROOT / "04_event_balance",
        "05_acf": EDA_ROOT / "05_acf",
        "06_subject_variance": EDA_ROOT / "06_subject_variance",
        "07_event_locked": EDA_ROOT / "07_event_locked",
        "08_event_study": EDA_ROOT / "08_event_study",
        "09_timeseries": EDA_ROOT / "09_timeseries",
        "10_state_transitions": EDA_ROOT / "10_state_transitions",
    }
    for f in folders.values():
        safe_mkdir(f)

    # ----------------------------
    # Load and combine features (robust)
    # ----------------------------
    dfs = []
    sources = []
    subject_dirs = [p for p in FEATURE_ROOT.iterdir() if p.is_dir()]

    for subject_dir in sorted(subject_dirs):
        df, src = robust_read_features(subject_dir)
        if df is None:
            continue
        df["_source_file"] = src
        dfs.append(df)
        sources.append((subject_dir.name, src))

    if not dfs:
        raise SystemExit(f"[ERROR] No feature files found under {FEATURE_ROOT}")

    data = pd.concat(dfs, ignore_index=True)

    # Benchmark only
    if "SessionType" in data.columns:
        benchmark = data[data["SessionType"].isin(BENCHMARK_TYPES)].copy()
    else:
        benchmark = data.copy()

    # ----------------------------
    # Data integrity checks (missing, dupes, infs)
    # ----------------------------
    integrity_rows = []

    # duplicates
    if "WindowID" in benchmark.columns:
        dup_count = int(benchmark["WindowID"].duplicated().sum())
    else:
        dup_count = -1

    # missing / inf summary over key columns (if present)
    cols_to_check = ["WindowID", "SubjectID", "SessionName", "SessionType", "StartTime", "EventLabel"] + KEY_FEATURES
    cols_to_check = [c for c in cols_to_check if c in benchmark.columns]

    missing_counts = benchmark[cols_to_check].isna().sum().sort_values(ascending=False)
    missing_counts.to_csv(folders["00_data_integrity"] / "missing_counts.csv")

    # inf counts
    inf_counts = {}
    for c in cols_to_check:
        if pd.api.types.is_numeric_dtype(benchmark[c]):
            arr = pd.to_numeric(benchmark[c], errors="coerce").to_numpy()
            inf_counts[c] = int(np.isinf(arr).sum())
    pd.Series(inf_counts).sort_values(ascending=False).to_csv(folders["00_data_integrity"] / "inf_counts.csv")

    integrity_rows.append({"metric": "duplicate_WindowID_count", "value": dup_count})
    integrity = pd.DataFrame(integrity_rows)
    integrity.to_csv(folders["00_data_integrity"] / "integrity_summary.csv", index=False)

    # ----------------------------
    # Select only features that exist
    # ----------------------------
    available_features = [f for f in KEY_FEATURES if f in benchmark.columns]
    missing_features = [f for f in KEY_FEATURES if f not in benchmark.columns]

    pd.Series(available_features).to_csv(folders["04_event_balance"] / "available_features_used.csv", index=False)
    pd.Series(missing_features).to_csv(folders["04_event_balance"] / "missing_features_skipped.csv", index=False)

    if not available_features:
        raise SystemExit("[ERROR] None of KEY_FEATURES exist in the loaded data. Check your feature files.")

    print("[EDA] Using features:", available_features)
    if missing_features:
        print("[EDA] Skipping missing features:", missing_features)

    # ----------------------------
    # 1) Histograms per feature (per label)
    # ----------------------------
    if "EventLabel" in benchmark.columns:
        for feat in available_features:
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
    # 2) Boxplots: feature vs EventLabel
    # ----------------------------
    if "EventLabel" in benchmark.columns:
        for feat in available_features:
            plt.figure(figsize=(6, 4))
            sns.boxplot(data=benchmark, x="EventLabel", y=feat)
            plt.title(f"Boxplot: {feat} vs EventLabel")
            plt.tight_layout()
            plt.savefig(folders["02_boxplots"] / f"box_{feat}.png")
            plt.close()

    # ----------------------------
    # 3) Correlation heatmap (features)
    # ----------------------------
    corr_df = benchmark[available_features].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation Heatmap (Selected Features)")
    plt.tight_layout()
    plt.savefig(folders["03_correlations"] / "feature_correlation_heatmap.png")
    plt.close()
    corr_df.to_csv(folders["03_correlations"] / "feature_correlation_matrix.csv")

    # ----------------------------
    # 4) Event count & cumulative duration per label (imbalance)
    # ----------------------------
    if "EventLabel" in benchmark.columns:
        event_counts = benchmark["EventLabel"].value_counts().sort_index()
        event_counts.to_csv(folders["04_event_balance"] / "event_counts.csv")

        plt.figure(figsize=(6, 4))
        event_counts.plot(kind="bar")
        plt.title("Event Window Counts (Benchmark)")
        plt.xlabel("EventLabel")
        plt.ylabel("Number of Windows")
        plt.tight_layout()
        plt.savefig(folders["04_event_balance"] / "event_counts.png")
        plt.close()

        # approximate duration using step_s
        step_s = get_step_seconds(benchmark)
        event_duration_s = (event_counts * step_s).rename("approx_duration_seconds")
        event_duration_s.to_csv(folders["04_event_balance"] / "event_duration_seconds.csv")

        plt.figure(figsize=(6, 4))
        (event_duration_s / 60.0).plot(kind="bar")
        plt.title("Approx. Cumulative Duration per Label (minutes)")
        plt.xlabel("EventLabel")
        plt.ylabel("Minutes (approx.)")
        plt.tight_layout()
        plt.savefig(folders["04_event_balance"] / "event_duration_minutes.png")
        plt.close()

    # ----------------------------
    # 5) ACF plots up to 60s (multiple variables)
    # ----------------------------
    # Convert lag seconds -> lag steps using step_s
    step_s = get_step_seconds(benchmark)
    lags_steps = max(1, int(round(ACF_MAX_SECONDS / step_s)))

    acf_features = [f for f in available_features if f in {"HR_mean", "GSR_mean", "theta_beta_ratio", "frontal_mean_alpha", "SAI", "CAI"}]
    for feat in acf_features:
        series = pd.to_numeric(benchmark[feat], errors="coerce").dropna()
        if len(series) < MIN_SERIES_LEN_FOR_ACF:
            continue
        plot_acf(series, lags=lags_steps)
        plt.title(f"ACF (pooled benchmark): {feat}  (up to ~{ACF_MAX_SECONDS}s)")
        plt.tight_layout()
        plt.savefig(folders["05_acf"] / f"acf_pooled_{feat}.png")
        plt.close()

    # ----------------------------
    # 6) Per-subject variance & outlier summaries
    # ----------------------------
    if "SubjectID" in benchmark.columns:
        # mean/std per subject
        subject_stats = benchmark.groupby("SubjectID")[available_features].agg(["mean", "std"])
        subject_stats.to_csv(folders["06_subject_variance"] / "subject_feature_stats_mean_std.csv")

        # outlier counts per subject per feature using pooled z-score
        outlier_counts = []
        for feat in available_features:
            x = pd.to_numeric(benchmark[feat], errors="coerce")
            mu = float(x.mean(skipna=True))
            sd = float(x.std(skipna=True))
            if not np.isfinite(sd) or sd == 0:
                continue
            z = (x - mu) / sd
            tmp = benchmark[["SubjectID"]].copy()
            tmp["is_outlier"] = (np.abs(z) > OUTLIER_Z) & z.notna()
            c = tmp.groupby("SubjectID")["is_outlier"].sum().rename(feat)
            outlier_counts.append(c)

        if outlier_counts:
            out_df = pd.concat(outlier_counts, axis=1).fillna(0).astype(int)
            out_df.to_csv(folders["06_subject_variance"] / f"subject_outlier_counts_zgt{OUTLIER_Z}.csv")

    # ----------------------------
    # 7) Visual sample traces around event onsets (window-based approximation)
    # ----------------------------
    # Here we do a simple global lineplot of feature vs StartTime grouped by EventLabel,
    # which gives intuition of temporal patterns by state.
    if "StartTime" in benchmark.columns and "EventLabel" in benchmark.columns:
        for feat in [f for f in ["SAI", "CAI"] if f in benchmark.columns]:
            plt.figure(figsize=(10, 4))
            sns.lineplot(
                data=benchmark,
                x="StartTime",
                y=feat,
                hue="EventLabel",
                estimator="mean",
                errorbar=None,
            )
            plt.title(f"Mean {feat} over time by EventLabel (window timeline)")
            plt.tight_layout()
            plt.savefig(folders["07_event_locked"] / f"mean_trace_{feat}_by_label.png")
            plt.close()

    # ----------------------------
    # 8) Preliminary event-study plot (mean SAI/CAI around events)
    # ----------------------------
    # A simple and robust proxy: plot mean by EventLabel (already in boxplots), plus
    # a per-session average time series for each label is harder without onset markers.
    # We'll keep a label-wise mean barplot for SAI/CAI for clarity.
    if "EventLabel" in benchmark.columns:
        for feat in [f for f in ["SAI", "CAI"] if f in benchmark.columns]:
            means = benchmark.groupby("EventLabel")[feat].mean(numeric_only=True).sort_index()
            plt.figure(figsize=(6, 4))
            means.plot(kind="bar")
            plt.title(f"Mean {feat} by EventLabel (benchmark)")
            plt.xlabel("EventLabel")
            plt.ylabel(f"Mean {feat}")
            plt.tight_layout()
            plt.savefig(folders["08_event_study"] / f"mean_{feat}_by_label.png")
            plt.close()
            means.to_csv(folders["08_event_study"] / f"mean_{feat}_by_label.csv")

    # ----------------------------
    # 9) Time series plots per subject
    # ----------------------------
    if {"SubjectID", "StartTime"}.issubset(benchmark.columns):
        for sid, df_s in benchmark.groupby("SubjectID"):
            # plot state timeline
            if "EventLabel" in df_s.columns:
                plot_state_timeline(
                    df_s,
                    folders["10_state_transitions"] / f"state_timeline_subject_{sid}.png",
                    title=f"State timeline (EventLabel) — Subject {sid} (benchmark)",
                )

            # plot a few key features as time series
            for feat in [f for f in ["HR_mean", "GSR_mean", "SAI", "CAI"] if f in df_s.columns]:
                plot_feature_timeseries(
                    df_s,
                    feat,
                    folders["09_timeseries"] / f"{feat}_subject_{sid}.png",
                    title=f"{feat} time series — Subject {sid} (benchmark)",
                )

            # compute per-subject time in state (pooled across sessions)
            dur = compute_state_durations(df_s)
            if not dur.empty:
                dur.to_csv(folders["10_state_transitions"] / f"time_in_state_subject_{sid}.csv", index=False)

    # ----------------------------
    # 10) Time series plots per session + transition summaries
    # ----------------------------
    if {"SubjectID", "SessionName", "StartTime"}.issubset(benchmark.columns):
        transition_rows = []

        for (sid, sess), df_ss in benchmark.groupby(["SubjectID", "SessionName"]):
            # state timeline per session
            if "EventLabel" in df_ss.columns:
                plot_state_timeline(
                    df_ss,
                    folders["10_state_transitions"] / f"state_timeline_{sid}_{sess}.png",
                    title=f"State timeline — Subject {sid}, Session {sess}",
                )

                # transition counts
                d = df_ss.sort_values("StartTime").copy()
                y = pd.to_numeric(d["EventLabel"], errors="coerce").dropna().to_numpy()
                n_trans = int(np.sum(y[1:] != y[:-1])) if len(y) > 1 else 0
                transition_rows.append({
                    "SubjectID": sid,
                    "SessionName": sess,
                    "num_transitions": n_trans,
                    "num_windows": len(d),
                })

        if transition_rows:
            trans_df = pd.DataFrame(transition_rows)
            trans_df.to_csv(folders["10_state_transitions"] / "transition_counts_per_session.csv", index=False)

    print("\n[EDA] Completed. Outputs saved to results/eda/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())