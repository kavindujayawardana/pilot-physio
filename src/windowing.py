#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


FS = 256.0  # nominal sampling rate (Hz)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_type_from_name(session_name: str) -> str:
    # expected patterns: "22_CA", "22_LOFT", "5_SS"
    parts = session_name.split("_")
    return parts[-1].upper() if len(parts) >= 2 else "UNKNOWN"


def is_benchmark(session_type: str) -> bool:
    return session_type in {"CA", "DA", "SS"}


def load_subject_crew_id(processed_root: Path, subject_id: str) -> str | None:
    meta_path = processed_root / subject_id / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return meta.get("crewID")
        except Exception:
            return None
    return None


def clean_event_column(df: pd.DataFrame, subject_id: str, session_name: str) -> pd.DataFrame:
    """
    Make Event safe and deterministic:
    - if missing -> create all zeros
    - coerce to numeric
    - replace +/-inf with NaN
    - fill NaN with 0 (baseline)
    - convert to int
    """
    if "Event" not in df.columns:
        print(f"[WARN] {subject_id}/{session_name}: missing Event column -> creating all zeros")
        df["Event"] = 0
        return df

    raw = pd.to_numeric(df["Event"], errors="coerce")
    arr = raw.to_numpy(dtype=float, copy=False)
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())

    if (n_nan + n_inf) > 0:
        print(f"[WARN] {subject_id}/{session_name}: Event had {n_nan} NaN and {n_inf} inf values -> filled with 0")

    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)
    df["Event"] = raw
    return df


def ensure_timesecs(df: pd.DataFrame, subject_id: str, session_name: str) -> pd.DataFrame:
    if "TimeSecs" not in df.columns:
        raise ValueError(f"{subject_id}/{session_name}: missing TimeSecs column")

    df["TimeSecs"] = pd.to_numeric(df["TimeSecs"], errors="coerce")
    if df["TimeSecs"].isna().any():
        print(f"[WARN] {subject_id}/{session_name}: TimeSecs contains NaNs -> forward-fill/backfill applied")
        df["TimeSecs"] = df["TimeSecs"].ffill().bfill()

    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Window and label preprocessed sessions (supports keeping transition windows)."
    )
    parser.add_argument("--preprocessed", required=True, help="Path to preprocessed/")
    parser.add_argument("--processed", required=True, help="Path to processed/ (for metadata like crewID)")
    parser.add_argument("--out", default="./windows", help="Output windows directory")

    # ✅ Changed defaults for near real-time:
    parser.add_argument("--window-len", type=float, default=5.0, help="Window length in seconds (default: 5.0)")
    parser.add_argument("--step", type=float, default=1.0, help="Step size in seconds (default: 1.0)")

    parser.add_argument(
        "--transition-threshold",
        type=float,
        default=0.20,
        help="Transition overlap threshold (0.20 means purity must be >= 0.80).",
    )
    parser.add_argument(
        "--bad-frac-threshold",
        type=float,
        default=0.20,
        help="Mark window bad if fraction of bad samples exceeds this.",
    )
    args = parser.parse_args()

    preproc_root = Path(args.preprocessed).resolve()
    processed_root = Path(args.processed).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    win_len_samples = int(round(args.window_len * FS))          # 5.0s -> 1280
    step_samples = int(round(args.step * FS))                   # 1.0s -> 256
    keep_frac = 1.0 - float(args.transition_threshold)          # 0.20 -> 0.80

    # Find all preprocessed session files
    session_files = sorted(preproc_root.glob("*/*_preproc.parquet"))
    if not session_files:
        print(f"[ERROR] No _preproc.parquet files found under {preproc_root}")
        return 1

    subject_rows: dict[str, list[dict]] = {}

    for pq in session_files:
        subject_id = pq.parent.name
        session_name = pq.stem.replace("_preproc", "")
        s_type = session_type_from_name(session_name)
        benchmark = is_benchmark(s_type)

        # Safety: skip macOS junk sessions if they ever appear
        if session_name.startswith("._") or pq.name.startswith("._"):
            print(f"[SKIP] macOS metadata file: {subject_id} / {session_name}")
            continue

        print(f"[WIN] {subject_id} / {session_name} (type={s_type})")

        df = pd.read_parquet(pq)

        # Clean required columns
        df = ensure_timesecs(df, subject_id, session_name)
        df = clean_event_column(df, subject_id, session_name)

        # Ensure bad flag exists (treated as per-sample mask: True = bad sample)
        if "IsBadWindow" not in df.columns:
            df["IsBadWindow"] = False

        # Identify numeric signal columns excluding metadata
        exclude = {"TimeSecs", "Event", "IsBadWindow"}
        signal_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not signal_cols:
            print(f"[WARN] {subject_id}/{session_name}: No numeric signal columns found -> skipping session")
            continue

        # Prepare output folder for arrays
        session_out_dir = out_root / subject_id / session_name
        session_out_dir.mkdir(parents=True, exist_ok=True)

        crew_id = load_subject_crew_id(processed_root, subject_id)

        N = len(df)
        if N < win_len_samples:
            print(f"[WARN] {subject_id}/{session_name}: too short for one window (N={N}) -> skipping")
            continue

        # Pre-grab arrays for speed
        event_arr = df["Event"].to_numpy(dtype=np.int32, copy=False)            # safe ints, no NaNs
        bad_arr = df["IsBadWindow"].astype(float).to_numpy(copy=False)          # 0/1 mask
        time_arr = df["TimeSecs"].to_numpy(dtype=np.float64, copy=False)
        sig_arr = df[signal_cols].to_numpy(dtype=np.float32, copy=False)        # shape: (N, C)

        for start in range(0, N - win_len_samples + 1, step_samples):
            end = start + win_len_samples

            labels = event_arr[start:end]

            # Label distribution within window
            uniq, cnts = np.unique(labels, return_counts=True)
            if len(uniq) == 0:
                continue

            top_i = int(np.argmax(cnts))
            top_label = int(uniq[top_i])
            frac = float(cnts[top_i] / win_len_samples)

            # ✅ KEEP transition windows, but flag them
            is_transition = frac < keep_frac

            # Bad window fraction
            bad_frac = float(bad_arr[start:end].mean())
            is_bad_window = bad_frac > float(args.bad_frac_threshold)

            start_time = float(time_arr[start])
            end_time = float(time_arr[end - 1])

            window_id = f"{subject_id}_{session_name}_{start}"

            # Save window data: (channels x samples)
            window_mat = sig_arr[start:end].T  # (C, win_len_samples)
            win_path = session_out_dir / f"win_{start:06d}.npy"
            np.save(win_path, window_mat)

            row = {
                "WindowID": window_id,
                "SubjectID": subject_id,
                "CrewID": crew_id,
                "SessionName": session_name,
                "SessionType": s_type,
                "BenchmarkTask": benchmark,
                "StartIdx": start,
                "EndIdx": end,
                "StartTime": start_time,
                "EndTime": end_time,

                # Label info
                "EventLabel": top_label,
                "LabelFrac": frac,
                "IsTransition": bool(is_transition),
                "purity_threshold": float(keep_frac),

                # Quality info
                "BadSampleFrac": bad_frac,
                "IsBadWindow": bool(is_bad_window),

                # Storage pointers
                "raw_data_pointer": str(win_path),
                "channels": ",".join(signal_cols),

                # Window config
                "window_len_s": float(args.window_len),
                "step_s": float(args.step),
                "transition_keep_frac": float(keep_frac),

                "created_utc": utc_now(),
            }

            subject_rows.setdefault(subject_id, []).append(row)

    # Write per-subject windows.parquet
    for subject_id, rows in subject_rows.items():
        out_subj_dir = out_root / subject_id
        out_subj_dir.mkdir(parents=True, exist_ok=True)

        out_table = pd.DataFrame(rows)
        out_file = out_subj_dir / "windows.parquet"
        out_table.to_parquet(out_file, index=False)

        # Helpful counts
        n_total = len(out_table)
        n_trans = int(out_table["IsTransition"].sum()) if "IsTransition" in out_table.columns else 0
        print(f"[OK] Saved {out_file} ({n_total} windows; transitions={n_trans})")

    print("\nDone windowing.")
    print("Reminder for modeling: filter to IsTransition==False (and usually IsBadWindow==False).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())