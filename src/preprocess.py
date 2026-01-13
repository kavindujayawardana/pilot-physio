#!/usr/bin/env python3
"""
src/preprocess.py — Step 4: Preprocessing (signal cleaning)

Input:
  processed/<subject>/<session>_clean.parquet

Output:
  preprocessed/<subject>/<session>_preproc.parquet
  preprocessed/<subject>/<session>_rpeaks.json
  preprocessed/<subject>/<session>_metadata.json

Key guarantees:
- Skips macOS junk files (._*, __MACOSX, hidden).
- Coerces signal columns to float to avoid MNE "object dtype" crashes.
- Saves R-peaks from continuous ECG (best practice for correct HRV later).
- Does NOT perform per-subject normalization here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# Optional dependencies
try:
    import mne
except Exception:
    mne = None

try:
    import neurokit2 as nk
except Exception:
    nk = None


EPS = 1e-12


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_hidden_or_junk(part) -> bool:
    """
    Works for both Path objects and strings (Path.parts yields strings).
    """
    name = part.name if hasattr(part, "name") else str(part)

    if name == "__MACOSX":
        return True
    if name.startswith("._"):
        return True
    if name.startswith("."):
        return True
    return False


def butter_bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    nyq = 0.5 * fs
    lowc = max(low / nyq, 1e-6)
    highc = min(high / nyq, 0.999999)
    b, a = butter(order, [lowc, highc], btype="band")
    return filtfilt(b, a, x)


def butter_lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 2) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    nyq = 0.5 * fs
    c = min(cutoff / nyq, 0.999999)
    b, a = butter(order, c, btype="low")
    return filtfilt(b, a, x)


def butter_highpass(x: np.ndarray, fs: float, cutoff: float, order: int = 2) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    nyq = 0.5 * fs
    c = max(cutoff / nyq, 1e-6)
    b, a = butter(order, c, btype="high")
    return filtfilt(b, a, x)


def session_type_from_name(session_name: str) -> str:
    parts = session_name.split("_")
    if len(parts) >= 2:
        return parts[-1].upper()
    return "UNKNOWN"


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Convert selected columns to float (errors -> NaN), then replace inf with NaN.
    """
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)
    return out


@dataclass
class SessionMeta:
    subject_id: str
    session_name: str
    session_type: str
    fs: float
    n_samples: int
    duration_s: float
    channels: list[str]
    created_utc: str
    notes: list[str]
    eeg_filter: dict[str, Any]
    ecg_filter: dict[str, Any]
    gsr_filter: dict[str, Any]
    resp_filter: dict[str, Any]
    ica: dict[str, Any]
    rpeaks_count: int


def preprocess_one_session(
    in_path: Path,
    processed_root: Path,
    out_root: Path,
    fs_target: float,
    do_ica: bool,
    notch_120: bool,
) -> None:
    subject_id = in_path.parent.name
    session_name = in_path.stem.replace("_clean", "")
    session_type = session_type_from_name(session_name)

    print(f"[PREP] {subject_id} / {session_name}")

    # ✅ FIX: define output directory EARLY and ALWAYS
    out_subject_dir = out_root / str(subject_id)
    out_subject_dir.mkdir(parents=True, exist_ok=True)

    notes: list[str] = []

    df = pd.read_parquet(in_path)

    # Ensure TimeSecs exists (should exist from ingestion)
    if "TimeSecs" not in df.columns:
        notes.append("TimeSecs missing; inferred using fs_target and row index.")
        df["TimeSecs"] = np.arange(len(df), dtype=np.float64) / float(fs_target)

    # Ensure Event exists
    if "Event" not in df.columns:
        notes.append("Event missing; set to baseline 0.")
        df["Event"] = 0

    # Identify signal columns
    meta_cols = {"TimeSecs", "Event"}
    if "IsBadSample" in df.columns:
        meta_cols.add("IsBadSample")

    # Candidate signals = everything except meta
    signal_cols = [c for c in df.columns if c not in meta_cols]

    # Coerce signals to numeric floats (prevents MNE object dtype crash)
    df = coerce_numeric(df, signal_cols)
    if df[signal_cols].isna().any().any():
        notes.append("NaNs in signals filled with 0 prior to filtering.")
        df[signal_cols] = df[signal_cols].fillna(0.0)

    # Build matrix (C x N)
    sig = df[signal_cols].to_numpy(dtype=np.float64).T
    N = sig.shape[1]
    fs = float(fs_target)

    # Identify modality columns
    eeg_cols = [c for c in signal_cols if c.upper().startswith("EEG_")]
    ecg_col = "ECG" if "ECG" in signal_cols else None
    gsr_col = "GSR" if "GSR" in signal_cols else None
    resp_col = "R" if "R" in signal_cols else None

    # -------------------------
    # EEG filtering (MNE FIR)
    # -------------------------
    eeg_filter_params = {
        "bandpass": [0.5, 45.0],
        "notch": [60.0] + ([120.0] if notch_120 else []),
        "method": "mne.filter.filter_data (FIR, zero-phase)",
        "applied": False,
    }

    if eeg_cols and mne is not None:
        eeg_idx = [signal_cols.index(c) for c in eeg_cols]
        eeg_data = sig[eeg_idx, :].astype(np.float64, copy=False)

        eeg_data = mne.filter.filter_data(
            eeg_data,
            sfreq=fs,
            l_freq=0.5,
            h_freq=45.0,
            method="fir",
            phase="zero",
            verbose=True,
        )

        eeg_data = mne.filter.notch_filter(
            eeg_data,
            Fs=fs,
            freqs=eeg_filter_params["notch"],
            method="fir",
            phase="zero",
            verbose=True,
        )

        sig[eeg_idx, :] = eeg_data
        eeg_filter_params["applied"] = True
    elif eeg_cols and mne is None:
        notes.append("MNE not installed/available; EEG filtering skipped.")

    # -------------------------
    # ECG bandpass + R-peaks
    # -------------------------
    ecg_filter_params = {"bandpass": [0.5, 40.0], "method": "butter+filtfilt", "applied": False}
    rpeaks_samples: list[int] = []
    rpeaks_times_s: list[float] = []

    if ecg_col is not None:
        ecg_i = signal_cols.index(ecg_col)
        ecg_f = butter_bandpass(sig[ecg_i, :], fs, 0.5, 40.0, order=4)
        sig[ecg_i, :] = ecg_f
        ecg_filter_params["applied"] = True

        if nk is None:
            notes.append("NeuroKit2 not installed/available; R-peak detection skipped.")
        else:
            try:
                _signals, info = nk.ecg_peaks(ecg_f, sampling_rate=fs)
                rpeaks_samples = [int(x) for x in info.get("ECG_R_Peaks", [])]
                rpeaks_times_s = (np.array(rpeaks_samples, dtype=float) / fs).tolist()
            except Exception as e:
                notes.append(f"R-peak detection failed: {type(e).__name__}: {e}")
                rpeaks_samples = []
                rpeaks_times_s = []

    # ✅ Save R-peaks JSON artifact (best practice)
    rpeaks_payload = {
        "subject": str(subject_id),
        "session": str(session_name),
        "fs": float(fs),
        "rpeaks_samples": rpeaks_samples,
        "rpeaks_times_s": rpeaks_times_s,
        "created_utc": utc_now(),
        "method": "neurokit2.ecg_peaks" if nk is not None else "none",
    }
    rpeaks_path = out_subject_dir / f"{session_name}_rpeaks.json"
    with open(rpeaks_path, "w", encoding="utf-8") as f:
        json.dump(rpeaks_payload, f, indent=2)

    # -------------------------
    # GSR processing (gentle HP)
    # -------------------------
    gsr_filter_params = {"highpass": 0.05, "method": "butter+filtfilt", "applied": False}
    if gsr_col is not None:
        gsr_i = signal_cols.index(gsr_col)
        sig[gsr_i, :] = butter_highpass(sig[gsr_i, :], fs, cutoff=0.05, order=2)
        gsr_filter_params["applied"] = True

    # -------------------------
    # Resp processing (LP ~5Hz)
    # -------------------------
    resp_filter_params = {"lowpass": 5.0, "method": "butter+filtfilt", "applied": False}
    if resp_col is not None:
        r_i = signal_cols.index(resp_col)
        sig[r_i, :] = butter_lowpass(sig[r_i, :], fs, cutoff=5.0, order=2)
        resp_filter_params["applied"] = True

    # -------------------------
    # Optional ICA (not auto-removing)
    # -------------------------
    ica_info: dict[str, Any] = {"ran": False}
    if do_ica and eeg_cols and mne is not None:
        try:
            eeg_idx = [signal_cols.index(c) for c in eeg_cols]
            if len(eeg_idx) >= 5:
                info = mne.create_info(
                    ch_names=[signal_cols[i] for i in eeg_idx],
                    sfreq=fs,
                    ch_types=["eeg"] * len(eeg_idx),
                )
                raw = mne.io.RawArray(sig[eeg_idx, :], info, verbose=False)

                ica = mne.preprocessing.ICA(n_components=20, random_state=42, max_iter="auto")
                ica.fit(raw, verbose=False)

                # No EOG channels -> no auto exclusion
                raw_clean = ica.apply(raw.copy(), verbose=False)
                sig[eeg_idx, :] = raw_clean.get_data()

                ica_info = {"ran": True, "n_components": 20, "excluded": [], "note": "ICA applied, no auto exclusions."}
            else:
                ica_info = {"ran": False, "note": "ICA skipped (insufficient EEG channels)."}
        except Exception as e:
            ica_info = {"ran": False, "note": f"ICA failed: {type(e).__name__}: {e}"}
            notes.append(ica_info["note"])

    # -------------------------
    # Build output dataframe
    # -------------------------
    df_out = pd.DataFrame(sig.T, columns=signal_cols)

    # Preserve TimeSecs and Event
    df_out["TimeSecs"] = pd.to_numeric(df["TimeSecs"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if df_out["TimeSecs"].isna().any():
        notes.append("TimeSecs had NaNs/inf; filled using index/fs.")
        df_out["TimeSecs"] = np.arange(N, dtype=np.float64) / fs

    # Event: treat missing/NaN as baseline 0
    df_out["Event"] = pd.to_numeric(df["Event"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)

    # Column order
    ordered = ["TimeSecs", "Event"] + [c for c in signal_cols if c not in {"TimeSecs", "Event"}]
    df_out = df_out[ordered]

    duration_s = float(df_out["TimeSecs"].iloc[-1] - df_out["TimeSecs"].iloc[0]) if N > 1 else 0.0

    out_parq = out_subject_dir / f"{session_name}_preproc.parquet"
    df_out.to_parquet(out_parq, index=False)

    meta = SessionMeta(
        subject_id=str(subject_id),
        session_name=str(session_name),
        session_type=str(session_type),
        fs=float(fs),
        n_samples=int(N),
        duration_s=float(duration_s),
        channels=list(df_out.columns),
        created_utc=utc_now(),
        notes=notes,
        eeg_filter=eeg_filter_params,
        ecg_filter=ecg_filter_params,
        gsr_filter=gsr_filter_params,
        resp_filter=resp_filter_params,
        ica=ica_info,
        rpeaks_count=int(len(rpeaks_samples)),
    )

    meta_path = out_subject_dir / f"{session_name}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    print(f"[OK] Saved {out_parq}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True, help="Path to processed/ directory")
    ap.add_argument("--out", required=True, help="Path to preprocessed/ output directory")
    ap.add_argument("--fs", type=float, default=256.0, help="Target sampling rate (Hz), default 256")
    ap.add_argument("--do-ica", action="store_true", help="Run ICA on EEG (optional)")
    ap.add_argument("--notch-120", action="store_true", help="Also notch 120 Hz (optional)")
    ap.add_argument("--subjects", default="", help="Comma-separated subject IDs to process (optional)")
    args = ap.parse_args()

    processed_root = Path(args.processed).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not processed_root.exists():
        print(f"[ERROR] processed directory not found: {processed_root}")
        return 1

    subjects_filter = set()
    if args.subjects.strip():
        subjects_filter = {s.strip() for s in args.subjects.split(",") if s.strip()}

    all_files = sorted(processed_root.rglob("*_clean.parquet"))
    # ✅ FIX: p.parts yields strings, is_hidden_or_junk now supports strings
    clean_files = [p for p in all_files if not any(is_hidden_or_junk(x) for x in p.parts)]

    if subjects_filter:
        clean_files = [p for p in clean_files if p.parent.name in subjects_filter]

    if not clean_files:
        print(f"[ERROR] No *_clean.parquet files found under {processed_root}")
        return 1

    for p in clean_files:
        try:
            preprocess_one_session(
                in_path=p,
                processed_root=processed_root,
                out_root=out_root,
                fs_target=float(args.fs),
                do_ica=bool(args.do_ica),
                notch_120=bool(args.notch_120),
            )
        except Exception as e:
            print(f"[ERROR] Failed {p}: {type(e).__name__}: {e}")

    print("\nDone preprocessing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
