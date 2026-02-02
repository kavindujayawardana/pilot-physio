#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks
import neurokit2 as nk


# -----------------------------
# Helpers
# -----------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_type_from_name(session_name: str) -> str:
    parts = session_name.split("_")
    return parts[-1].upper() if len(parts) >= 2 else "UNKNOWN"


def is_benchmark(session_type: str) -> bool:
    return session_type in {"CA", "DA", "SS"}


def safe_float(x) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float("nan")


def linear_slope(y: np.ndarray) -> float:
    """Simple least-squares slope over equally spaced samples."""
    if y is None or len(y) < 2:
        return float("nan")
    y = np.asarray(y, dtype=np.float64)
    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=np.nanmedian(y))
    x = np.arange(len(y), dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = (x * x).sum()
    if denom <= 0:
        return float("nan")
    return float((x * y).sum() / denom)


def log10p(power: float, eps: float = 1e-12) -> float:
    return float(np.log10(power + eps))


def bandpower_log(
    x: np.ndarray,
    fs: float,
    fmin: float,
    fmax: float,
    nperseg: Optional[int] = None,
) -> float:
    """
    Welch PSD band power (integrated) then log10(power + eps).
    Uses np.trapezoid for integration (avoids deprecated np.trapz).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8:
        return float("nan")
    freqs, psd = welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg if nperseg is not None else min(256, x.size),
        noverlap=None,
        detrend="constant",
        scaling="density",
    )
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return float("nan")
    p = float(np.trapezoid(psd[mask], freqs[mask]))
    return log10p(p)


def parse_channels(ch_str: str) -> List[str]:
    return [c.strip() for c in str(ch_str).split(",") if c.strip()]


def get_idx(cols: List[str], name: str) -> Optional[int]:
    try:
        return cols.index(name)
    except ValueError:
        return None


def compute_rr_from_rpeaks(rpeaks_s: np.ndarray) -> np.ndarray:
    """RR intervals in seconds from R-peak times in seconds."""
    rpeaks_s = np.asarray(rpeaks_s, dtype=np.float64)
    if rpeaks_s.size < 2:
        return np.array([], dtype=np.float64)
    rr = np.diff(rpeaks_s)
    # remove non-physiological / invalid RR intervals
    rr = rr[np.isfinite(rr)]
    rr = rr[(rr > 0.25) & (rr < 2.5)]  # 24–240 bpm approx
    return rr


def hrv_metrics_from_rr(rr: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute RMSSD, SDNN, pNN50 from RR (seconds).
    - RMSSD: sqrt(mean(diff(RR)^2))
    - SDNN: std(RR)
    - pNN50: %(|diff(RR)| > 0.05 s)
    """
    rr = np.asarray(rr, dtype=np.float64)
    if rr.size < 3:
        return (float("nan"), float("nan"), float("nan"))

    sdnn = float(np.std(rr, ddof=1)) if rr.size >= 2 else float("nan")

    drr = np.diff(rr)
    if drr.size < 2:
        rmssd = float("nan")
        pnn50 = float("nan")
    else:
        rmssd = float(np.sqrt(np.mean(drr * drr)))
        pnn50 = float(100.0 * np.mean(np.abs(drr) > 0.05))

    return (rmssd, sdnn, pnn50)


# -----------------------------
# R-peaks loading (best practice)
# -----------------------------
def load_rpeaks_json(preprocessed_root: Path, subject_id: str, session_name: str) -> Optional[np.ndarray]:
    """
    Expected file:
      preprocessed/<subject>/<session>_rpeaks.json
    Format (recommended):
      {"rpeaks_times_s": [..]}   OR directly a list.
    """
    p = preprocessed_root / subject_id / f"{session_name}_rpeaks.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            # allow several keys
            for k in ("rpeaks_times_s", "rpeaks_times", "rpeaks", "ECG_R_Peaks_s"):
                if k in obj:
                    arr = np.asarray(obj[k], dtype=np.float64)
                    return arr[np.isfinite(arr)]
            return None
        if isinstance(obj, list):
            arr = np.asarray(obj, dtype=np.float64)
            return arr[np.isfinite(arr)]
    except Exception:
        return None
    return None


def compute_rpeaks_from_preprocessed_session(df_pre: pd.DataFrame, fs: float) -> Optional[np.ndarray]:
    """
    Fallback if rpeaks json is missing:
    detect R-peaks from preprocessed ECG (whole session once).
    """
    if "ECG" not in df_pre.columns:
        return None
    ecg = pd.to_numeric(df_pre["ECG"], errors="coerce").to_numpy(dtype=np.float64)
    ecg = np.nan_to_num(ecg, nan=np.nanmedian(ecg))
    try:
        _, info = nk.ecg_peaks(ecg, sampling_rate=fs)
        r = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int64)
        if r.size == 0:
            return None
        return (r.astype(np.float64) / fs)
    except Exception:
        return None


# -----------------------------
# Feature computation per window
# -----------------------------
EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

FRONTAL_CHANNELS = ["EEG_FP1", "EEG_FP2", "EEG_F3", "EEG_F4", "EEG_FZ"]


def compute_window_features(
    x: np.ndarray,            # shape (C, S)
    cols: List[str],          # length C
    fs: float,
) -> Dict[str, float]:
    """
    Compute features from the *window signal only* (2s/5s/whatever is present).
    """
    feats: Dict[str, float] = {}

    # ----- EEG bandpowers per EEG channel -----
    eeg_idxs = [(i, c) for i, c in enumerate(cols) if c.startswith("EEG_")]
    for i, ch in eeg_idxs:
        sig = x[i, :]
        for band, (f1, f2) in EEG_BANDS.items():
            feats[f"{ch}_{band}"] = bandpower_log(sig, fs, f1, f2)

    # ----- Aggregated EEG features -----
    # frontal_mean_alpha
    alpha_vals = []
    for name in FRONTAL_CHANNELS:
        # allow Fz to be either EEG_Fz or EEG_FZ
        name2 = name if name in cols else name.replace("FZ", "Fz")
        idx = get_idx(cols, name2)
        if idx is None:
            continue
        alpha_vals.append(feats.get(f"{name2}_alpha", float("nan")))
    if len(alpha_vals) > 0:
        alpha_vals = [v for v in alpha_vals if np.isfinite(v)]
        feats["frontal_mean_alpha"] = float(np.mean(alpha_vals)) if alpha_vals else float("nan")
    else:
        feats["frontal_mean_alpha"] = float("nan")

    # frontal_asymmetry_alpha = log(alpha_F3) - log(alpha_F4)
    f3 = "EEG_F3" if "EEG_F3" in cols else "EEG_F3"
    f4 = "EEG_F4" if "EEG_F4" in cols else "EEG_F4"
    a_f3 = feats.get(f"{f3}_alpha", float("nan"))
    a_f4 = feats.get(f"{f4}_alpha", float("nan"))
    feats["frontal_asymmetry_alpha"] = float(a_f3 - a_f4) if (np.isfinite(a_f3) and np.isfinite(a_f4)) else float("nan")

    # theta_beta_ratio = mean(theta)/mean(beta) (on frontal channels)
    theta_vals = []
    beta_vals = []
    for name in FRONTAL_CHANNELS:
        name2 = name if name in cols else name.replace("FZ", "Fz")
        t = feats.get(f"{name2}_theta", float("nan"))
        b = feats.get(f"{name2}_beta", float("nan"))
        if np.isfinite(t):
            theta_vals.append(t)
        if np.isfinite(b):
            beta_vals.append(b)
    if theta_vals and beta_vals:
        # because these are log-powers, a strict ratio in linear domain would be 10**t / 10**b
        # but for a stable scalar we use difference in log domain:
        feats["theta_beta_ratio"] = float(np.mean(theta_vals) - np.mean(beta_vals))
    else:
        feats["theta_beta_ratio"] = float("nan")

    # ----- Simple GSR / Resp within-window stats (if present) -----
    gsr_idx = get_idx(cols, "GSR")
    if gsr_idx is not None:
        g = x[gsr_idx, :].astype(np.float64)
        feats["GSR_mean"] = float(np.mean(g))
        feats["GSR_std"] = float(np.std(g))
        feats["GSR_slope"] = linear_slope(g)
        # Peaks as a rough phasic proxy (window-only)
        peaks, _ = find_peaks(g, distance=max(1, int(0.25 * fs)))
        feats["GSR_peak_count"] = float(len(peaks))
    else:
        feats["GSR_mean"] = feats["GSR_std"] = feats["GSR_slope"] = feats["GSR_peak_count"] = float("nan")

    r_idx = get_idx(cols, "R")
    if r_idx is not None:
        r = x[r_idx, :].astype(np.float64)
        feats["R_mean"] = float(np.mean(r))
        feats["R_std"] = float(np.std(r))
        feats["R_slope"] = linear_slope(r)
        peaks, _ = find_peaks(r, distance=max(1, int(0.75 * fs)))  # ~0.75s min distance
        feats["R_peak_count"] = float(len(peaks))
    else:
        feats["R_mean"] = feats["R_std"] = feats["R_slope"] = feats["R_peak_count"] = float("nan")

    return feats


def compute_10s_resp_and_gsr_features(
    df_pre: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    fs: float,
) -> Dict[str, float]:
    """
    Compute trailing 10s metrics from the *preprocessed continuous session*.
    Uses the 10 seconds ending at window end_idx.
    """
    out: Dict[str, float] = {"breathing_rate_10s_bpm": float("nan"),
                             "SCL_mean_10s": float("nan"),
                             "SCR_count_10s": float("nan")}

    n10 = int(10.0 * fs)
    s0 = max(0, end_idx - n10)
    s1 = end_idx

    # Resp
    if "R" in df_pre.columns:
        r = pd.to_numeric(df_pre["R"].iloc[s0:s1], errors="coerce").to_numpy(dtype=np.float64)
        r = np.nan_to_num(r, nan=np.nanmedian(r))
        peaks, _ = find_peaks(r, distance=max(1, int(0.75 * fs)))
        dur_s = (s1 - s0) / fs if (s1 > s0) else 0.0
        if dur_s > 0 and len(peaks) >= 1:
            out["breathing_rate_10s_bpm"] = float((len(peaks) / dur_s) * 60.0)

    # GSR: treat SCL as mean GSR (tonic proxy) in last 10s; SCR count from peaks
    if "GSR" in df_pre.columns:
        g = pd.to_numeric(df_pre["GSR"].iloc[s0:s1], errors="coerce").to_numpy(dtype=np.float64)
        g = np.nan_to_num(g, nan=np.nanmedian(g))
        out["SCL_mean_10s"] = float(np.mean(g))
        peaks, _ = find_peaks(g, distance=max(1, int(0.25 * fs)))
        out["SCR_count_10s"] = float(len(peaks))

    return out


def compute_hr_hrv_features_trailing(
    rpeaks_s: Optional[np.ndarray],
    t_end: float,
    t_start: float,
    hrv_window_s: float = 30.0,
) -> Dict[str, float]:
    """
    Near-real-time friendly:
    - HR_mean & HR_slope from beats whose RR / HR samples lie inside the current window [t_start, t_end]
    - HRV metrics computed over trailing 30s interval [t_end - 30s, t_end]
    """
    out = {
        "HR_mean": float("nan"),
        "HR_slope": float("nan"),
        "RR_count_window": float("nan"),
        "HRV_RMSSD_30s": float("nan"),
        "HRV_SDNN_30s": float("nan"),
        "HRV_pNN50_30s": float("nan"),
    }

    if rpeaks_s is None or len(rpeaks_s) < 2 or not np.isfinite(t_end):
        return out

    rp = np.asarray(rpeaks_s, dtype=np.float64)
    rp = rp[np.isfinite(rp)]
    if rp.size < 2:
        return out

    # HR samples: assign instantaneous HR at each RR interval midpoint
    rr = np.diff(rp)
    mids = (rp[:-1] + rp[1:]) / 2.0
    valid = (rr > 0.25) & (rr < 2.5) & np.isfinite(rr) & np.isfinite(mids)
    rr = rr[valid]
    mids = mids[valid]
    if rr.size == 0:
        return out

    hr_inst = 60.0 / rr

    # Window HR (current window)
    in_win = (mids >= t_start) & (mids <= t_end)
    if np.any(in_win):
        hr_win = hr_inst[in_win]
        out["RR_count_window"] = float(hr_win.size)
        out["HR_mean"] = float(np.mean(hr_win))
        out["HR_slope"] = linear_slope(hr_win)
    else:
        out["RR_count_window"] = 0.0

    # HRV trailing 30s
    t0 = t_end - float(hrv_window_s)
    # select r-peaks within [t0, t_end]
    mask = (rp >= t0) & (rp <= t_end)
    rp30 = rp[mask]
    rr30 = compute_rr_from_rpeaks(rp30)
    rmssd, sdnn, pnn50 = hrv_metrics_from_rr(rr30)

    out["HRV_RMSSD_30s"] = rmssd
    out["HRV_SDNN_30s"] = sdnn
    out["HRV_pNN50_30s"] = pnn50

    return out


def add_rolls_and_lags(
    df: pd.DataFrame,
    feature_cols: List[str],
    roll_windows_s: List[int] = [5, 10],
    lag_seconds: List[int] = [1, 5, 10],
    step_s: float = 1.0,
) -> pd.DataFrame:
    """
    Adds rolling means and lags within each session (time-ordered).
    Assumes windows are updated every `step_s` seconds (default 1s).
    Rolling 5s mean = mean of last 5 rows if step=1.
    """
    df = df.sort_values(["SessionName", "StartTime"]).reset_index(drop=True)

    def per_session(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()

        # rolling means
        for w in roll_windows_s:
            k = max(1, int(round(w / step_s)))
            for c in feature_cols:
                g[f"{c}_roll{w}s_mean"] = g[c].rolling(window=k, min_periods=1).mean()

        # lags (shift by N rows)
        for lag in lag_seconds:
            k = max(1, int(round(lag / step_s)))
            for c in feature_cols:
                g[f"{c}_lag{lag}s"] = g[c].shift(k)

        return g

    return df.groupby("SessionName", group_keys=False).apply(per_session)


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="Extract features from windowed physiological signals.")
    p.add_argument("--windows", required=True, help="Path to windows/ (contains <subject>/windows.parquet)")
    p.add_argument("--preprocessed", required=True, help="Path to preprocessed/ (for continuous context + rpeaks)")
    p.add_argument("--out", default="./features", help="Output features directory")
    p.add_argument("--fs", type=float, default=256.0, help="Sampling rate (Hz)")
    p.add_argument("--hrv-window", type=float, default=30.0, help="HRV trailing window length in seconds")
    args = p.parse_args()

    fs = float(args.fs)
    windows_root = Path(args.windows).resolve()
    preproc_root = Path(args.preprocessed).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    subj_tables = sorted(windows_root.glob("*/windows.parquet"))
    if not subj_tables:
        print(f"[ERROR] No windows.parquet found under {windows_root}")
        return 1

    for wtab in subj_tables:
        subject_id = wtab.parent.name
        dfw = pd.read_parquet(wtab)
        if dfw.empty:
            print(f"[WARN] {subject_id}: windows table empty -> skipping")
            continue

        # Ensure these exist (your windowing now keeps transitions but marks them)
        if "IsTransition" not in dfw.columns:
            # Backward compatible: if not present, infer from LabelFrac and transition_keep_frac if available
            if "LabelFrac" in dfw.columns and "transition_keep_frac" in dfw.columns:
                dfw["IsTransition"] = dfw["LabelFrac"] < dfw["transition_keep_frac"]
            else:
                dfw["IsTransition"] = False

        # Cache: preprocessed session dfs + rpeaks per session
        preproc_cache: Dict[str, pd.DataFrame] = {}
        rpeaks_cache: Dict[str, Optional[np.ndarray]] = {}

        rows_out: List[Dict] = []

        # Iterate windows
        for _, row in dfw.iterrows():
            session_name = str(row["SessionName"])
            s_type = session_type_from_name(session_name)

            # Load window array
            pointer = Path(str(row["raw_data_pointer"]))
            x = np.load(pointer)  # shape (C, S)
            cols = parse_channels(row["channels"])

            # Basic per-window features
            feats = compute_window_features(x, cols, fs)

            # Load continuous preprocessed session (for 10s features + HRV from r-peaks)
            if session_name not in preproc_cache:
                pq = preproc_root / subject_id / f"{session_name}_preproc.parquet"
                if pq.exists():
                    preproc_cache[session_name] = pd.read_parquet(pq)
                else:
                    preproc_cache[session_name] = pd.DataFrame()

            df_pre = preproc_cache[session_name]

            # R-peaks: best practice (load saved); fallback (detect once from session)
            if session_name not in rpeaks_cache:
                rp = load_rpeaks_json(preproc_root, subject_id, session_name)
                if rp is None and not df_pre.empty:
                    rp = compute_rpeaks_from_preprocessed_session(df_pre, fs)
                rpeaks_cache[session_name] = rp

            # HR/HRV (near-real-time: trailing)
            t_start = safe_float(row["StartTime"])
            t_end = safe_float(row["EndTime"])
            hrv = compute_hr_hrv_features_trailing(
                rpeaks_cache[session_name],
                t_end=t_end,
                t_start=t_start,
                hrv_window_s=float(args.hrv_window),
            )
            feats.update(hrv)

            # 10s respiration / GSR context
            if not df_pre.empty:
                start_idx = int(row["StartIdx"])
                end_idx = int(row["EndIdx"])
                feats.update(compute_10s_resp_and_gsr_features(df_pre, start_idx, end_idx, fs))
            else:
                feats.setdefault("breathing_rate_10s_bpm", float("nan"))
                feats.setdefault("SCL_mean_10s", float("nan"))
                feats.setdefault("SCR_count_10s", float("nan"))

            # Output row: keep window metadata (for dependency control) + features
            out_row = {
                "WindowID": row["WindowID"],
                "SubjectID": row["SubjectID"],
                "CrewID": row.get("CrewID", None),
                "SessionName": session_name,
                "SessionType": s_type,
                "BenchmarkTask": is_benchmark(s_type),
                "StartIdx": int(row["StartIdx"]),
                "EndIdx": int(row["EndIdx"]),
                "StartTime": safe_float(row["StartTime"]),
                "EndTime": safe_float(row["EndTime"]),
                "EventLabel": int(row["EventLabel"]),
                "LabelFrac": safe_float(row.get("LabelFrac", np.nan)),
                "IsTransition": bool(row.get("IsTransition", False)),
                "BadSampleFrac": safe_float(row.get("BadSampleFrac", np.nan)),
                "IsBadWindow": bool(row.get("IsBadWindow", False)),
            }
            out_row.update(feats)
            rows_out.append(out_row)

        dff = pd.DataFrame(rows_out)

        # Add rolling means + lags (for 1s update cadence)
        base_for_rolls = [
            "HR_mean", "HR_slope",
            "GSR_mean", "GSR_std", "GSR_peak_count",
            "R_mean", "R_std", "R_peak_count",
            "HRV_RMSSD_30s", "HRV_SDNN_30s", "HRV_pNN50_30s",
            "frontal_mean_alpha", "frontal_asymmetry_alpha", "theta_beta_ratio",
        ]
        base_for_rolls = [c for c in base_for_rolls if c in dff.columns]
        if base_for_rolls:
            dff = add_rolls_and_lags(
                dff,
                feature_cols=base_for_rolls,
                roll_windows_s=[5, 10],
                lag_seconds=[1, 5, 10],
                step_s=1.0,   # you are updating every 1 second
            )

        # Save
        out_dir = out_root / subject_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "features.parquet"
        dff.to_parquet(out_file, index=False)

        print(f"[OK] Saved {out_file} ({len(dff)} windows, {len(dff.columns)} cols)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())