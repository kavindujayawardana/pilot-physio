#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks

EPS = 1e-12


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_channels(ch_str: str) -> list[str]:
    if not isinstance(ch_str, str) or not ch_str.strip():
        return []
    return [c.strip() for c in ch_str.split(",") if c.strip()]


def is_eeg_channel(name: str) -> bool:
    return name.upper().startswith("EEG_")


def bandpower_log10(sig: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    if sig is None:
        return np.nan
    sig = np.asarray(sig, dtype=np.float64)
    if sig.size < 8 or not np.isfinite(sig).all():
        return np.nan

    nperseg = min(256, sig.size)
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan

    p = np.trapz(psd[mask], freqs[mask])
    return float(np.log10(p + EPS))


def eeg_features(signal_map: dict[str, np.ndarray], fs: float) -> dict[str, float]:
    feats: dict[str, float] = {}

    eeg_chs = [k for k in signal_map.keys() if is_eeg_channel(k)]
    if not eeg_chs:
        feats["frontal_mean_alpha"] = np.nan
        feats["frontal_asymmetry_alpha"] = np.nan
        feats["theta_beta_ratio"] = np.nan
        return feats

    bands = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
    }

    for ch in eeg_chs:
        sig = signal_map.get(ch)
        for band_name, (fmin, fmax) in bands.items():
            feats[f"{ch}_{band_name}"] = bandpower_log10(sig, fs, fmin, fmax)

    frontal_candidates = ["EEG_FP1", "EEG_FP2", "EEG_F3", "EEG_F4", "EEG_FZ"]
    frontal_alpha_vals = [feats.get(f"{c}_alpha", np.nan) for c in frontal_candidates]
    frontal_alpha_vals = [v for v in frontal_alpha_vals if np.isfinite(v)]
    feats["frontal_mean_alpha"] = float(np.mean(frontal_alpha_vals)) if frontal_alpha_vals else np.nan

    a_f3 = feats.get("EEG_F3_alpha", np.nan)
    a_f4 = feats.get("EEG_F4_alpha", np.nan)
    feats["frontal_asymmetry_alpha"] = float(a_f3 - a_f4) if np.isfinite(a_f3) and np.isfinite(a_f4) else np.nan

    theta_vals = [feats.get(f"{c}_theta", np.nan) for c in eeg_chs]
    beta_vals = [feats.get(f"{c}_beta", np.nan) for c in eeg_chs]
    theta_vals = [v for v in theta_vals if np.isfinite(v)]
    beta_vals = [v for v in beta_vals if np.isfinite(v)]
    feats["theta_beta_ratio"] = float(np.mean(theta_vals) - np.mean(beta_vals)) if theta_vals and beta_vals else np.nan

    return feats


def gsr_basic_features(gsr: np.ndarray, fs: float) -> dict[str, float]:
    feats = {"GSR_mean": np.nan, "GSR_std": np.nan, "GSR_slope": np.nan, "GSR_peak_count": np.nan}
    if gsr is None:
        return feats
    gsr = np.asarray(gsr, dtype=np.float64)
    if gsr.size < 8 or not np.isfinite(gsr).all():
        return feats

    feats["GSR_mean"] = float(np.mean(gsr))
    feats["GSR_std"] = float(np.std(gsr))

    try:
        t = np.arange(gsr.size) / fs
        a, _b = np.polyfit(t, gsr, deg=1)
        feats["GSR_slope"] = float(a)
    except Exception:
        feats["GSR_slope"] = np.nan

    x = gsr - np.mean(gsr)
    mad = np.median(np.abs(x - np.median(x))) + EPS
    peaks, _ = find_peaks(x, height=2.5 * mad, distance=int(0.2 * fs))
    feats["GSR_peak_count"] = float(peaks.size)
    return feats


def resp_basic_features(resp: np.ndarray, fs: float) -> dict[str, float]:
    feats = {"R_mean": np.nan, "R_std": np.nan, "R_slope": np.nan, "R_peak_count": np.nan}
    if resp is None:
        return feats
    resp = np.asarray(resp, dtype=np.float64)
    if resp.size < 8 or not np.isfinite(resp).all():
        return feats

    feats["R_mean"] = float(np.mean(resp))
    feats["R_std"] = float(np.std(resp))

    try:
        t = np.arange(resp.size) / fs
        a, _b = np.polyfit(t, resp, deg=1)
        feats["R_slope"] = float(a)
    except Exception:
        feats["R_slope"] = np.nan

    x = resp - np.mean(resp)
    mad = np.median(np.abs(x - np.median(x))) + EPS
    peaks, _ = find_peaks(x, height=2.0 * mad, distance=int(0.4 * fs))
    feats["R_peak_count"] = float(peaks.size)
    return feats


# ---------------------------
# ECG / HRV (BEST PRACTICE)
# ---------------------------

def load_rpeaks_samples(preprocessed_root: Path, subject_id: str, session_name: str) -> np.ndarray:
    """
    Loads R-peaks (sample indices) saved during preprocessing.
    Expected file:
      preprocessed/<subject>/<session>_rpeaks.json
    Returns empty array if missing.
    """
    fp = preprocessed_root / subject_id / f"{session_name}_rpeaks.json"
    if not fp.exists():
        return np.array([], dtype=int)

    try:
        obj = json.loads(fp.read_text(encoding="utf-8"))
        arr = obj.get("rpeaks_samples", [])
        return np.asarray(arr, dtype=int)
    except Exception:
        return np.array([], dtype=int)


def rr_intervals_from_rpeaks(rpeaks_samples: np.ndarray, fs: float) -> np.ndarray:
    """RR intervals in seconds from consecutive R-peak sample indices."""
    r = np.asarray(rpeaks_samples, dtype=np.int64)
    if r.size < 2:
        return np.array([], dtype=np.float64)
    rr = np.diff(r) / fs
    rr = rr[np.isfinite(rr) & (rr > 0)]
    return rr.astype(np.float64)


def hrv_metrics_from_rr(rr: np.ndarray) -> dict[str, float]:
    """
    True HRV metrics from RR (seconds):
    - RMSSD = sqrt(mean(diff(RR)^2))
    - SDNN = std(RR)
    - pNN50 = fraction(|diff(RR)| > 0.05) * 100
    Uses NaN if insufficient data.
    """
    out = {"HRV_RMSSD_30s": np.nan, "HRV_SDNN_30s": np.nan, "HRV_pNN50_30s": np.nan}
    rr = np.asarray(rr, dtype=np.float64)

    if rr.size >= 2:
        out["HRV_SDNN_30s"] = float(np.std(rr, ddof=1)) if rr.size >= 3 else float(np.std(rr))

    if rr.size >= 3:
        drr = np.diff(rr)
        out["HRV_RMSSD_30s"] = float(np.sqrt(np.mean(drr ** 2)))
        out["HRV_pNN50_30s"] = float((np.mean(np.abs(drr) > 0.05) * 100.0))

    return out


def ecg_features_from_rpeaks(rpeaks_samples: np.ndarray, fs: float, start_idx: int, end_idx: int) -> dict[str, float]:
    """
    Window-level HR metrics using true R-peaks:
    - HR_mean (within window) = 60 / median(RR_window)
    - HR_slope within window: slope of instantaneous HR across beats
    - HRV metrics computed over rolling 30s ending at window end.
    """
    feats = {
        "HR_mean": np.nan,
        "HR_slope": np.nan,
        "RR_count_window": np.nan,
        "HRV_RMSSD_30s": np.nan,
        "HRV_SDNN_30s": np.nan,
        "HRV_pNN50_30s": np.nan,
    }

    r = np.asarray(rpeaks_samples, dtype=np.int64)
    if r.size < 2:
        feats["RR_count_window"] = 0.0
        return feats

    # R-peaks inside window
    in_win = r[(r >= start_idx) & (r < end_idx)]
    rr_win = rr_intervals_from_rpeaks(in_win, fs)
    feats["RR_count_window"] = float(rr_win.size)

    if rr_win.size >= 1:
        feats["HR_mean"] = float(60.0 / (np.median(rr_win) + EPS))

    # HR slope: need at least 3 RR intervals to fit reliably
    if rr_win.size >= 3:
        hr_inst = 60.0 / (rr_win + EPS)
        # time for each interval: use times of later peak
        t = (in_win[1:] / fs)[: hr_inst.size]
        try:
            a, _b = np.polyfit(t, hr_inst, deg=1)
            feats["HR_slope"] = float(a)
        except Exception:
            feats["HR_slope"] = np.nan

    # Rolling 30s HRV ending at window end
    lookback = int(30 * fs)
    lb_start = max(0, end_idx - lookback)
    in_30s = r[(r >= lb_start) & (r < end_idx)]
    rr_30s = rr_intervals_from_rpeaks(in_30s, fs)
    feats.update(hrv_metrics_from_rr(rr_30s))

    return feats


def add_rolling_and_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["SessionName", "StartIdx"]).reset_index(drop=True)

    base_cols = [
        "HR_mean", "HR_slope",
        "GSR_mean", "GSR_std", "GSR_peak_count",
        "R_mean", "R_std", "R_peak_count",
        # HRV already computed as 30s “true” metrics, but you can still smooth if you want:
        "HRV_RMSSD_30s", "HRV_SDNN_30s", "HRV_pNN50_30s",
    ]

    roll_windows = {5: "5s", 10: "10s"}

    for col in base_cols:
        if col not in df.columns:
            continue
        for w, tag in roll_windows.items():
            df[f"{col}_roll{tag}_mean"] = (
                df.groupby("SessionName")[col]
                .transform(lambda s: s.rolling(w, min_periods=max(1, w // 3)).mean())
            )

    # Breathing rate proxy over 10s
    if "R_peak_count" in df.columns:
        peaks10 = df.groupby("SessionName")["R_peak_count"].transform(lambda s: s.rolling(10, min_periods=3).sum())
        df["breathing_rate_10s_bpm"] = (peaks10 / 10.0) * 60.0
    else:
        df["breathing_rate_10s_bpm"] = np.nan

    # Tonic-like proxies
    if "GSR_mean" in df.columns:
        df["SCL_mean_10s"] = df.groupby("SessionName")["GSR_mean"].transform(lambda s: s.rolling(10, min_periods=3).mean())
    else:
        df["SCL_mean_10s"] = np.nan

    if "GSR_peak_count" in df.columns:
        df["SCR_count_10s"] = df.groupby("SessionName")["GSR_peak_count"].transform(lambda s: s.rolling(10, min_periods=3).sum())
    else:
        df["SCR_count_10s"] = np.nan

    # Lags
    lag_steps = [1, 5, 10]
    lag_cols = [
        "HR_mean",
        "HRV_RMSSD_30s",
        "HRV_SDNN_30s",
        "HRV_pNN50_30s",
        "GSR_mean",
        "R_mean",
        "frontal_mean_alpha",
        "frontal_asymmetry_alpha",
        "theta_beta_ratio",
    ]
    for col in lag_cols:
        if col not in df.columns:
            continue
        for k in lag_steps:
            df[f"{col}_lag{k}s"] = df.groupby("SessionName")[col].shift(k)

    return df


def extract_features_for_subject(subject_dir: Path, out_subject_dir: Path, preprocessed_root: Path, fs: float) -> None:
    windows_file = subject_dir / "windows.parquet"
    if not windows_file.exists():
        return

    dfw = pd.read_parquet(windows_file)
    out_subject_dir.mkdir(parents=True, exist_ok=True)

    if dfw.empty:
        print(f"[WARN] {subject_dir.name}: empty windows.parquet")
        return

    rows_out: list[dict] = []

    # We load rpeaks once per session
    rpeaks_cache: dict[str, np.ndarray] = {}

    for i in range(len(dfw)):
        row = dfw.iloc[i]
        session_name = str(row.get("SessionName"))

        # load rpeaks for this session
        if session_name not in rpeaks_cache:
            rpeaks_cache[session_name] = load_rpeaks_samples(preprocessed_root, subject_dir.name, session_name)

        rpeaks_samples = rpeaks_cache[session_name]

        ch_list = parse_channels(row.get("channels", ""))
        pointer = row.get("raw_data_pointer", None)

        meta = {
            "WindowID": row.get("WindowID"),
            "SubjectID": row.get("SubjectID"),
            "CrewID": row.get("CrewID"),
            "SessionName": session_name,
            "SessionType": row.get("SessionType"),
            "BenchmarkTask": row.get("BenchmarkTask"),
            "StartIdx": int(row.get("StartIdx")),
            "EndIdx": int(row.get("EndIdx")),
            "StartTime": row.get("StartTime"),
            "EndTime": row.get("EndTime"),
            "EventLabel": row.get("EventLabel"),
            "LabelFrac": row.get("LabelFrac"),
            "BadSampleFrac": row.get("BadSampleFrac"),
            "IsBadWindow": row.get("IsBadWindow"),
        }

        # Load window array
        x = None
        if isinstance(pointer, str) and pointer:
            try:
                x = np.load(pointer)
            except Exception:
                x = None

        signal_map: dict[str, np.ndarray] = {}
        if x is not None and x.ndim == 2 and len(ch_list) == x.shape[0]:
            signal_map = {name: x[idx, :] for idx, name in enumerate(ch_list)}

        feats: dict[str, float] = {}
        feats.update(eeg_features(signal_map, fs))

        # ECG/HRV: fully correct from saved R-peaks
        feats.update(ecg_features_from_rpeaks(rpeaks_samples, fs, meta["StartIdx"], meta["EndIdx"]))

        # GSR + Resp basics
        feats.update(gsr_basic_features(signal_map.get("GSR", None), fs))
        feats.update(resp_basic_features(signal_map.get("R", None), fs))

        rows_out.append({**meta, **feats})

    dff = pd.DataFrame(rows_out)
    dff = add_rolling_and_lags(dff)

    out_path = out_subject_dir / "features.parquet"
    dff.to_parquet(out_path, index=False)
    print(f"[OK] Saved {out_path} ({dff.shape[0]} windows, {dff.shape[1]} cols)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True, help="Path to windows/ directory")
    ap.add_argument("--preprocessed", required=True, help="Path to preprocessed/ directory (for *_rpeaks.json)")
    ap.add_argument("--out", required=True, help="Path to features/ output directory")
    ap.add_argument("--fs", type=float, default=256.0, help="Sampling rate (Hz), default 256")
    ap.add_argument("--subjects", default="", help="Comma-separated subject IDs to process (optional)")
    args = ap.parse_args()

    windows_root = Path(args.windows).resolve()
    preprocessed_root = Path(args.preprocessed).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    subjects_filter = set()
    if args.subjects.strip():
        subjects_filter = {s.strip() for s in args.subjects.split(",") if s.strip()}

    subject_dirs = sorted([p for p in windows_root.iterdir() if p.is_dir()])
    if subjects_filter:
        subject_dirs = [p for p in subject_dirs if p.name in subjects_filter]

    if not subject_dirs:
        print(f"[ERROR] No subject folders found under {windows_root}")
        return 1

    for subj_dir in subject_dirs:
        out_subj_dir = out_root / subj_dir.name
        extract_features_for_subject(subj_dir, out_subj_dir, preprocessed_root, fs=float(args.fs))

    manifest = {
        "created_utc": utc_now(),
        "windows_root": str(windows_root),
        "preprocessed_root": str(preprocessed_root),
        "out_root": str(out_root),
        "fs": float(args.fs),
        "subjects": [p.name for p in subject_dirs],
    }
    (out_root / "features_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
