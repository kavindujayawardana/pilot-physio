#!/usr/bin/env python3
"""
Ingestion script:
- Reads NASA subject ZIP files containing CSVs
- Standardizes columns
- Fixes Event labels (3/4 -> 0)
- Converts ADC counts to physical units (multipliers provided)
- Writes cleaned parquet per session
- Appends ingest_log.csv
- Writes metadata.json per subject directory
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------- Configuration ----------
DEFAULT_SAMPLING_RATE = 256  # Hz (used only if TimeSecs must be inferred)

ADC_CONVERSIONS = {
    "ECG": 0.012215,
    "R": 0.2384186,
    "GSR": 0.2384186,
}

EVENT_INVALID_TO_ZERO = {3, 4}

# ----------------------------------


@dataclass
class SessionLogRow:
    ingestion_timestamp: str
    zip_file: str
    csv_file: str
    subject_id: str
    crew_id: Optional[str]
    session_name: str
    sampling_rate_hz: Optional[float]
    time_inferred: bool
    duration_seconds: Optional[float]
    num_samples: int
    columns: str
    event_counts: str
    notes: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def standardize_colname(name: str) -> str:
    """
    Standardize column names:
    - strip whitespace
    - replace '-' with '_'
    - collapse multiple spaces
    - consistent casing (keep original letters but normalize common issues)
    """
    name = name.strip()
    name = name.replace("-", "_")
    name = re.sub(r"\s+", "_", name)
    # Many NASA datasets have mixed casing; pick a consistent one:
    # We'll keep original but normalize to exact matching later by case-insensitive mapping.
    return name


def drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def robust_read_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Try reading CSV using common separators.
    We avoid asking the user to know the delimiter.
    """
    # Try utf-8 then latin-1, common in datasets
    encodings = ["utf-8", "latin-1"]
    seps = [",", ";", "\t", "|"]

    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                return pd.read_csv(BytesIO(file_bytes), sep=sep, engine="python", encoding=enc)
            except Exception as e:
                last_err = e
                continue

    raise RuntimeError(f"Failed to read CSV with fallback separators/encodings. Last error: {last_err}")


def ensure_timesecs(df: pd.DataFrame, sampling_rate: float) -> Tuple[pd.DataFrame, bool, Optional[float]]:
    """
    Ensure TimeSecs exists.
    If missing, infer using index / sampling_rate.
    Returns (df, time_inferred, duration_seconds)
    """
    cols_lower = {c.lower(): c for c in df.columns}
    if "timesecs" in cols_lower:
        # Ensure correct column name is exactly TimeSecs
        original = cols_lower["timesecs"]
        if original != "TimeSecs":
            df = df.rename(columns={original: "TimeSecs"})
        # duration (if numeric)
        duration = None
        try:
            duration = float(df["TimeSecs"].iloc[-1] - df["TimeSecs"].iloc[0])
            if duration < 0:
                duration = None
        except Exception:
            duration = None
        return df, False, duration

    # Infer
    n = len(df)
    df = df.copy()
    df["TimeSecs"] = [i / sampling_rate for i in range(n)]
    duration = (n - 1) / sampling_rate if n > 0 else 0.0
    return df, True, duration


def coerce_event(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, int], str]:
    """
    Convert Event column to int and map 3/4 -> 0.
    Returns (df, event_counts_dict, notes)
    """
    notes = ""
    cols_lower = {c.lower(): c for c in df.columns}
    if "event" not in cols_lower:
        # If there is no Event column, create baseline 0 and note it.
        df = df.copy()
        df["Event"] = 0
        notes = "Event column missing; created Event=0."
        return df, {0: int(len(df))}, notes

    original = cols_lower["event"]
    if original != "Event":
        df = df.rename(columns={original: "Event"})

    df = df.copy()

    # Coerce to numeric then fill invalid with 0
    df["Event"] = pd.to_numeric(df["Event"], errors="coerce").fillna(0).astype(int)

    # Map invalid labels
    mask_invalid = df["Event"].isin(list(EVENT_INVALID_TO_ZERO))
    if mask_invalid.any():
        df.loc[mask_invalid, "Event"] = 0
        notes = "Mapped Event labels 3/4 -> 0."

    counts = df["Event"].value_counts(dropna=False).to_dict()
    # ensure python int keys
    counts = {int(k): int(v) for k, v in counts.items()}
    return df, counts, notes


def apply_adc_conversions(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply given multipliers if columns exist (case-insensitive match).
    Returns (df, applied_columns)
    """
    applied = []
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    for col, factor in ADC_CONVERSIONS.items():
        key = col.lower()
        if key in cols_lower:
            actual = cols_lower[key]
            # Rename to canonical spelling if needed (ECG/R/GSR)
            if actual != col:
                df = df.rename(columns={actual: col})
            df[col] = pd.to_numeric(df[col], errors="coerce") * factor
            applied.append(col)

    return df, applied


def extract_subject_and_session(zip_path: Path, csv_name: str) -> Tuple[str, str]:
    """
    Best-effort extraction of subject and session from file names.
    - subject_id: from zip stem or csv name
    - session_name: from csv stem
    """
    # Example: Subject01.zip -> Subject01
    subject_id = zip_path.stem

    # session name: csv without extension
    session_name = Path(csv_name).stem

    return subject_id, session_name


def crew_id_for_subject(subject_id: str) -> Optional[str]:
    """
    Optional: derive crew ID if subject_id contains a number.
    If subject number is available and your rule is:
    crew i = subjects (2i-1, 2i)
    then crew_id = (subject_num + 1)//2
    """
    m = re.search(r"(\d+)", subject_id)
    if not m:
        return None
    subj_num = int(m.group(1))
    crew = (subj_num + 1) // 2
    return f"Crew{crew:02d}"


def update_subject_metadata(
    subject_dir: Path,
    subject_id: str,
    crew_id: Optional[str],
    sampling_rate: float,
    sessions_summary: List[Dict],
) -> None:
    """
    Save metadata.json in processed/<subject>/.
    """
    meta_path = subject_dir / "metadata.json"
    payload = {
        "subjectID": subject_id,
        "crewID": crew_id,
        "sampling_rate": sampling_rate,
        "conversion_applied": True,
        "ingestion_timestamp": utc_now_iso(),
        "channels_present_union": sorted({c for s in sessions_summary for c in s.get("channels_present", [])}),
        "sessions": sessions_summary,
        "manual_fixes": [],  # keep as list for future notes
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_ingest_log(log_path: Path, row: SessionLogRow) -> None:
    """
    Append to ingest_log.csv (create with header if not exists).
    """
    is_new = not log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(row).keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(asdict(row))


def ingest_zip(zip_path: Path, output_root: Path, sampling_rate: float, ingest_log_path: Path) -> None:
    import zipfile

    subject_sessions = []

    with zipfile.ZipFile(zip_path, "r") as z:
        csv_files = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_files:
            print(f"[WARN] No CSV files found in {zip_path.name}")
            return

        for csv_name in csv_files:
            file_bytes = z.read(csv_name)

            # 1) Read robustly
            df = robust_read_csv(file_bytes)

            # 2) Standardize columns
            df.columns = [standardize_colname(c) for c in df.columns]
            df = drop_unnamed(df)

            # 3) Ensure TimeSecs
            df, time_inferred, duration_seconds = ensure_timesecs(df, sampling_rate)

            # 4) Fix Event labels
            df, event_counts, event_notes = coerce_event(df)

            # 5) ADC conversions
            df, applied_cols = apply_adc_conversions(df)

            # Derive IDs/names
            subject_id, session_name = extract_subject_and_session(zip_path, csv_name)
            crew_id = crew_id_for_subject(subject_id)

            # Output path
            subject_dir = output_root / subject_id
            subject_dir.mkdir(parents=True, exist_ok=True)

            # Save parquet
            out_file = subject_dir / f"{session_name}_clean.parquet"
            df.to_parquet(out_file, index=False)

            # Session summary for metadata
            session_summary = {
                "session_name": session_name,
                "source_zip": zip_path.name,
                "source_csv": csv_name,
                "duration_seconds": duration_seconds,
                "num_samples": int(len(df)),
                "channels_present": list(df.columns),
                "time_inferred": bool(time_inferred),
                "event_counts": event_counts,
                "adc_conversions_applied_to": applied_cols,
            }
            subject_sessions.append(session_summary)

            # Log row
            log_row = SessionLogRow(
                ingestion_timestamp=utc_now_iso(),
                zip_file=zip_path.name,
                csv_file=csv_name,
                subject_id=subject_id,
                crew_id=crew_id,
                session_name=session_name,
                sampling_rate_hz=sampling_rate,
                time_inferred=time_inferred,
                duration_seconds=duration_seconds,
                num_samples=int(len(df)),
                columns="|".join(df.columns),
                event_counts=json.dumps(event_counts, sort_keys=True),
                notes="; ".join([n for n in [event_notes] if n]),
            )
            append_ingest_log(ingest_log_path, log_row)

            print(f"[OK] {subject_id} / {session_name}: saved {out_file}")

    # Write/update metadata.json for this subject (union across sessions in this zip)
    # Note: if multiple zips correspond to same subject, this will overwrite with latest run's sessions from this zip.
    subject_id = zip_path.stem
    crew_id = crew_id_for_subject(subject_id)
    subject_dir = output_root / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    update_subject_metadata(
        subject_dir=subject_dir,
        subject_id=subject_id,
        crew_id=crew_id,
        sampling_rate=sampling_rate,
        sessions_summary=subject_sessions,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest NASA subject zip files and output cleaned parquet sessions.")
    parser.add_argument("--input", required=True, help="Input directory containing .zip files (e.g., ./raw)")
    parser.add_argument("--output", required=True, help="Output directory for processed files (e.g., ./processed)")
    parser.add_argument("--sampling-rate", type=float, default=DEFAULT_SAMPLING_RATE, help="Sampling rate Hz (default 256)")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    zip_files = sorted(input_dir.glob("*.zip"))
    if not zip_files:
        print(f"[ERROR] No .zip files found in: {input_dir}", file=sys.stderr)
        return 3

    ingest_log_path = output_root / "ingest_log.csv"

    print(f"Input:  {input_dir}")
    print(f"Output: {output_root}")
    print(f"Found {len(zip_files)} zip files.")

    for zp in zip_files:
        try:
            print(f"\n--- Ingesting: {zp.name} ---")
            ingest_zip(zp, output_root, args.sampling_rate, ingest_log_path)
        except Exception as e:
            # Log failure too (so runs are auditable)
            fail_row = SessionLogRow(
                ingestion_timestamp=utc_now_iso(),
                zip_file=zp.name,
                csv_file="",
                subject_id=zp.stem,
                crew_id=crew_id_for_subject(zp.stem),
                session_name="",
                sampling_rate_hz=args.sampling_rate,
                time_inferred=False,
                duration_seconds=None,
                num_samples=0,
                columns="",
                event_counts="{}",
                notes=f"FAILED: {repr(e)}",
            )
            append_ingest_log(ingest_log_path, fail_row)
            print(f"[FAIL] {zp.name}: {e}", file=sys.stderr)

    print("\nDone.")
    print(f"Ingest log: {ingest_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
