#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import mne
import neurokit2 as nk
from scipy.signal import butter, filtfilt, find_peaks


DEFAULT_FS = 256.0


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def estimate_fs(timesecs: pd.Series) -> float:
    dt = timesecs.diff().dropna()
    if len(dt) == 0:
        return np.nan
    return float(1.0 / dt.median())


def butter_lowpass(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, x)


def butter_bandpass(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    return filtfilt(b, a, x)


def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd == 0 or np.isnan(sd):
        return np.zeros_like(x)
    return (x - mu) / sd


def mark_bad_windows(
    data_mat: np.ndarray,  # shape: (channels, samples)
    fs: float,
    win_s: float = 2.0,
    step_s: float = 1.0,
    z_thresh: float = 6.0,
) -> np.ndarray:
    """
    Returns a boolean mask over samples: True where sample belongs to a bad window.
    """
    n_ch, n = data_mat.shape
    win = int(win_s * fs)
    step = int(step_s * fs)
    if win <= 1 or step <= 0 or n < win:
        return np.zeros(n, dtype=bool)

    # zscore each channel globally (session-level)
    zmat = np.vstack([zscore(data_mat[c, :]) for c in range(n_ch)])

    bad_sample = np.zeros(n, dtype=bool)
    for start in range(0, n - win + 1, step):
        end = start + win
        mx = np.nanmax(np.abs(zmat[:, start:end]))
        if mx > z_thresh:
            bad_sample[start:end] = True
    return bad_sample


def save_subject_metadata(subject_dir: Path, payload: dict) -> None:
    subject_dir.mkdir(parents=True, exist_ok=True)
    meta_path = subject_dir / "metadata.json"
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess physiological signals per session.")
    parser.add_argument("--processed", required=True, help="Input processed dir (e.g., ./processed)")
    parser.add_argument("--out", default="./preprocessed", help="Output preprocessed dir")
    parser.add_argument("--fs", type=float, default=DEFAULT_FS, help="Target sampling rate (default 256)")
    parser.add_argument("--do-ica", action="store_true", help="Optional: run ICA on EEG (advanced)")
    args = parser.parse_args()

    processed_dir = Path(args.processed).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(processed_dir.glob("*/*_clean.parquet"))
    if not parquet_files:
        print(f"[ERROR] No _clean.parquet files found under {processed_dir}")
        return 2

    for pq in parquet_files:
        subject_id = pq.parent.name
        session_name = pq.stem.replace("_clean", "")
        print(f"[PREP] {subject_id} / {session_name}")

        df = pd.read_parquet(pq)

        # Ensure required columns exist
        if "TimeSecs" not in df.columns:
            print(f"[WARN] Missing TimeSecs in {pq}, skipping.")
            continue
        if "Event" not in df.columns:
            df["Event"] = 0

        fs_est = estimate_fs(df["TimeSecs"])
        fs_target = float(args.fs)

        # Identify channels
        # EEG channels: everything except TimeSecs/Event and known non-EEG signals
        non_eeg = {"TimeSecs", "Event", "ECG", "GSR", "R"}
        eeg_cols = [c for c in df.columns if c not in non_eeg]
        ecg_col = "ECG" if "ECG" in df.columns else None
        gsr_col = "GSR" if "GSR" in df.columns else None
        r_col = "R" if "R" in df.columns else None

        # Build continuous matrix for resampling/filtering (only numeric channels)
        signal_cols = eeg_cols + [c for c in [ecg_col, gsr_col, r_col] if c is not None]
        sig = df[signal_cols].apply(pd.to_numeric, errors="coerce").to_numpy().T  # (ch, samples)

        # -------- Step 1: Resample if needed --------
        resampled = False
        if np.isfinite(fs_est) and abs(fs_est - fs_target) > 0.5:
            # If difference is large, it might be metadata issue; still resample carefully
            pass

        if np.isfinite(fs_est) and abs(fs_est - fs_target) > 0.01:
            # Small variation -> resample to exact 256 Hz
            sig = mne.filter.resample(sig, down=fs_est, up=fs_target, npad="auto")
            resampled = True

            # Recreate TimeSecs to match new sample count
            n_new = sig.shape[1]
            df = df.iloc[:n_new].copy() if len(df) >= n_new else df.reindex(range(n_new)).copy()
            df["TimeSecs"] = np.arange(n_new) / fs_target

        # -------- Step 2: EEG filtering --------
        # Bandpass 0.5–45, FIR zero-phase
        if len(eeg_cols) > 0:
            eeg_idx = [signal_cols.index(c) for c in eeg_cols]
            eeg_data = sig[eeg_idx, :]
            eeg_data = mne.filter.filter_data(
                eeg_data, sfreq=fs_target, l_freq=0.5, h_freq=45.0,
                method="fir", phase="zero"
            )
            # Notch 60 Hz (and optionally 120)
            eeg_data = mne.filter.notch_filter(eeg_data, Fs=fs_target, freqs=[60.0], method="fir", phase="zero")
            sig[eeg_idx, :] = eeg_data

            # Optional common average reference
            # If you want: subtract mean across EEG channels at each time
            # eeg_data = eeg_data - eeg_data.mean(axis=0, keepdims=True)
            # sig[eeg_idx, :] = eeg_data

            # Optional ICA (advanced)
            removed_components = []
            if args.do_ica:
                info = mne.create_info(ch_names=eeg_cols, sfreq=fs_target, ch_types=["eeg"] * len(eeg_cols))
                raw = mne.io.RawArray(eeg_data, info, verbose=False)
                ica = mne.preprocessing.ICA(n_components=min(20, len(eeg_cols)), random_state=42)
                try:
                    ica.fit(raw)
                    # Without EOG, we can't auto-detect EOG comps reliably
                    # So we do not remove automatically here.
                    removed_components = []
                except Exception:
                    removed_components = []
                # Put back EEG (unchanged if no removal)
                sig[eeg_idx, :] = raw.get_data()

        else:
            removed_components = []

        # -------- Step 3: ECG bandpass + R-peaks --------
        rpeaks_times = []
        if ecg_col is not None:
            ecg_i = signal_cols.index(ecg_col)
            ecg_f = butter_bandpass(sig[ecg_i, :], fs_target, 0.5, 40.0, order=4)
            sig[ecg_i, :] = ecg_f

            try:
                signals, info = nk.ecg_peaks(ecg_f, sampling_rate=fs_target)
                rpeaks = info.get("ECG_R_Peaks", [])
                rpeaks_times = (np.array(rpeaks) / fs_target).tolist()
            except Exception:
                rpeaks_times = []

        # -------- Step 4: GSR detrend + tonic/phasic --------
        scr_peaks_times = []
        gsr_tonic = None
        gsr_phasic = None

        if gsr_col is not None:
            gsr_i = signal_cols.index(gsr_col)
            gsr_raw = sig[gsr_i, :]

            # Detrend: high-pass via simple approach (subtract low-pass tonic baseline)
            tonic = butter_lowpass(gsr_raw, fs_target, cutoff_hz=0.05, order=2)
            gsr_detrended = gsr_raw - tonic
            # NeuroKit decomposition
            try:
                phasic_df = nk.eda_phasic(gsr_raw, sampling_rate=fs_target)
                # common keys: "EDA_Tonic", "EDA_Phasic"
                if "EDA_Tonic" in phasic_df.columns:
                    gsr_tonic = phasic_df["EDA_Tonic"].to_numpy()
                if "EDA_Phasic" in phasic_df.columns:
                    gsr_phasic = phasic_df["EDA_Phasic"].to_numpy()

                # SCR peaks (from processed signal)
                eda_signals, eda_info = nk.eda_peaks(gsr_raw, sampling_rate=fs_target)
                peaks = eda_info.get("SCR_Peaks", [])
                scr_peaks_times = (np.array(peaks) / fs_target).tolist()
            except Exception:
                gsr_tonic = tonic
                gsr_phasic = gsr_detrended
                scr_peaks_times = []

            sig[gsr_i, :] = gsr_raw  # keep filtered raw in main channel; tonic/phasic saved separately

        # -------- Step 5: Respiration low-pass + breath peaks --------
        breath_peaks_times = []
        breath_rate_bpm = None

        if r_col is not None:
            r_i = signal_cols.index(r_col)
            r_raw = sig[r_i, :]
            r_f = butter_lowpass(r_raw, fs_target, cutoff_hz=5.0, order=4)
            sig[r_i, :] = r_f

            # Breath peaks: enforce a minimum distance (e.g., 0.8s)
            min_dist = int(0.8 * fs_target)
            peaks, _ = find_peaks(r_f, distance=min_dist)
            breath_peaks_times = (peaks / fs_target).tolist()

            # Breath rate estimate (bpm) using average peak-to-peak
            if len(peaks) >= 2:
                intervals = np.diff(peaks) / fs_target  # seconds per breath
                mean_interval = float(np.mean(intervals))
                breath_rate_bpm = 60.0 / mean_interval if mean_interval > 0 else None

        # -------- Step 6: Artifact detection (z-score windows) --------
        bad_mask = mark_bad_windows(sig, fs_target, win_s=2.0, step_s=1.0, z_thresh=6.0)

        # -------- Step 7: Save outputs --------
        out_subject_dir = out_dir / subject_id
        out_subject_dir.mkdir(parents=True, exist_ok=True)

        # Rebuild dataframe columns from sig matrix
        out_df = pd.DataFrame({col: sig[i, :] for i, col in enumerate(signal_cols)})
        out_df["TimeSecs"] = df["TimeSecs"].values[: len(out_df)]
        out_df["Event"] = df["Event"].values[: len(out_df)]
        out_df["IsBadWindow"] = bad_mask[: len(out_df)]

        # Add tonic/phasic if available
        if gsr_tonic is not None:
            out_df["GSR_Tonic"] = gsr_tonic[: len(out_df)]
        if gsr_phasic is not None:
            out_df["GSR_Phasic"] = gsr_phasic[: len(out_df)]

        out_file = out_subject_dir / f"{session_name}_preproc.parquet"
        out_df.to_parquet(out_file, index=False)

        # Save/update metadata.json for this subject (simple approach: overwrite with latest run summary)
        meta = {
            "subjectID": subject_id,
            "sampling_rate_target": fs_target,
            "ingestion_source": str(pq),
            "preprocessing_timestamp": utc_now(),
            "resampled": resampled,
            "fs_estimated": fs_est,
            "filters": {
                "EEG_bandpass_hz": [0.5, 45.0],
                "EEG_notch_hz": [60.0],
                "ECG_bandpass_hz": [0.5, 40.0],
                "RESP_lowpass_hz": 5.0,
                "GSR_detrend_lowpass_hz": 0.05,
            },
            "ica": {
                "performed": bool(args.do_ica),
                "n_components": 20,
                "random_state": 42,
                "removed_components": removed_components,
            },
            "detectors": {
                "bad_window_rule": {"win_s": 2.0, "step_s": 1.0, "z_thresh": 6.0},
                "rpeak_count": len(rpeaks_times),
                "scr_peak_count": len(scr_peaks_times),
                "breath_peak_count": len(breath_peaks_times),
                "breath_rate_bpm_estimate": breath_rate_bpm,
            },
            "events_present": sorted(out_df["Event"].dropna().astype(int).unique().tolist()),
        }
        save_subject_metadata(out_subject_dir, meta)

        print(f"[OK] Saved {out_file}")

    print("\nDone preprocessing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
