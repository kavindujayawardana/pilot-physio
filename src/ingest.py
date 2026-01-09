#!/usr/bin/env python3
"""
Data ingestion script.

Reads subject-level ZIP archives from the NASA Flight Crew Physiological
Data for Crew State Monitoring dataset, standardizes session files,
cleans labels, applies ADC conversions, and writes cleaned session-level
Parquet files.

IMPORTANT:
- Explicitly ignores macOS metadata files (._*) and __MACOSX directories
- Prevents junk files from ever entering processed/
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zipfile
import re

import pandas as pd


# =========================
# Configuration
# =========================

DEFAULT_SAMPLING_RATE = 256.0

ADC_CONVERSIONS = {
    "ECG": 0.012215,
    "R": 0.2384186,
    "GSR": 0.2384186,
}

INVALID_EVENT_TO_ZERO = {3, 4}


# =========================
# Utilities
# =========================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def standardize_column(name: str) -> str:
    name = name.strip()
    name = name.replace("-", "_")
    name = re.sub(r"\s+", "_", name)
    return name


def robust_read_csv(data: bytes) -> pd.DataFrame:
    encodings = ["utf-8", "latin-1"]
    seps = [",", ";", "\t", "|"]

    for enc in encodings:
        for sep in seps:
            try:
                return pd.read_csv(BytesIO(data), sep=sep, engine="python", encoding=enc)
            except Exception:
                pass

    raise RuntimeError("Failed to read CSV with all fallback delimiters/encodings.")


def drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if not c.lower().startswith("unnamed")]
    return df[cols]


def ensure_timesecs(df: pd.DataFrame, fs: float) -> Tuple[pd.DataFrame, bool, float]:
    if "TimeSecs" in df.columns:
        t = pd.to_numeric(df["TimeSecs"], errors="coerce")
        df["TimeSecs"] = t
        duration = float(t.iloc[-1] - t.iloc[0]) if len(t) > 1 else 0.0
        return df, False, duration

    n = len(df)
    df["TimeSecs"] = [i / fs for i in range(n)]
    duration = (n - 1) / fs if n > 1 else 0.0
    return df, True, duration


def clean_event(df: pd.DataFrame) -> Dict[int, int]:
    if "Event" not in df.columns:
        df["Event"] = 0
        return {0: len(df)}

    df["Event"] = pd.to_numeric(df["Event"], errors="coerce").fillna(0).astype(int)
    df.loc[df["Event"].isin(INVALID_EVENT_TO_ZERO), "Event"] = 0
    return df["Event"].value_counts().to_dict()


def apply_adc_conversions(df: pd.DataFrame) -> List[str]:
    applied = []
    for col, factor in ADC_CONVERSIONS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") * factor
            applied.append(col)
    return applied


def extract_subject_and_session(zip_path: Path, csv_path: str) -> Tuple[str, str]:
    subject_id = zip_path.stem
    session_name = Path(csv_path).name
    session_name = Path(session_name).stem
    return subject_id, session_name


def crew_id_from_subject(subject_id: str) -> Optional[str]:
    m = re.search(r"(\d+)", subject_id)
    if not m:
        return None
    subj = int(m.group(1))
    crew = (subj + 1) // 2
    return f"Crew{crew:02d}"


# =========================
# Logging
# =========================

@dataclass
class IngestLogRow:
    timestamp: str
    zip_file: str
    csv_file: str
    subject_id: str
    crew_id: Optional[str]
    session_name: str
    duration_seconds: float
    num_samples: int
    event_counts: str
    time_inferred: bool
    notes: str


def append_ingest_log(path: Path, row: IngestLogRow) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(row).keys())
        if new:
            writer.writeheader()
        writer.writerow(asdict(row))


# =========================
# Core ingestion
# =========================

def ingest_zip(zip_path: Path, out_root: Path, fs: float, log_path: Path) -> None:
    subject_sessions = []

    with zipfile.ZipFile(zip_path, "r") as z:
        # -------- LAYER 1 FIX: ignore macOS junk --------
        csv_files = [
            name for name in z.namelist()
            if name.lower().endswith(".csv")
            and "__macosx" not in name.lower()
            and not Path(name).name.startswith("._")
        ]

        if not csv_files:
            print(f"[WARN] No valid CSV files in {zip_path.name}")
            return

        for csv_name in csv_files:
            data = z.read(csv_name)

            df = robust_read_csv(data)
            df.columns = [standardize_column(c) for c in df.columns]
            df = drop_unnamed(df)

            df, time_inferred, duration = ensure_timesecs(df, fs)
            event_counts = clean_event(df)
            applied = apply_adc_conversions(df)

            subject_id, session_name = extract_subject_and_session(zip_path, csv_name)
            crew_id = crew_id_from_subject(subject_id)

            subject_dir = out_root / subject_id
            subject_dir.mkdir(parents=True, exist_ok=True)

            out_file = subject_dir / f"{session_name}_clean.parquet"
            df.to_parquet(out_file, index=False)

            append_ingest_log(
                log_path,
                IngestLogRow(
                    timestamp=utc_now(),
                    zip_file=zip_path.name,
                    csv_file=csv_name,
                    subject_id=subject_id,
                    crew_id=crew_id,
                    session_name=session_name,
                    duration_seconds=duration,
                    num_samples=len(df),
                    event_counts=json.dumps(event_counts),
                    time_inferred=time_inferred,
                    notes=f"ADC applied to: {applied}" if applied else "",
                ),
            )

            subject_sessions.append({
                "session_name": session_name,
                "source_csv": csv_name,
                "duration_seconds": duration,
                "num_samples": len(df),
                "event_counts": event_counts,
                "adc_converted": applied,
            })

            print(f"[OK] {subject_id} / {session_name}")

    # Save metadata.json
    subject_id = zip_path.stem
    subject_dir = out_root / subject_id
    meta = {
        "subjectID": subject_id,
        "crewID": crew_id_from_subject(subject_id),
        "sampling_rate": fs,
        "ingestion_timestamp": utc_now(),
        "sessions": subject_sessions,
    }
    (subject_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


# =========================
# Entry point
# =========================

def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest raw NASA subject ZIP files.")
    parser.add_argument("--input", required=True, help="Raw ZIP directory")
    parser.add_argument("--output", required=True, help="Processed output directory")
    parser.add_argument("--fs", type=float, default=DEFAULT_SAMPLING_RATE)
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(input_dir.glob("*.zip"))
    if not zip_files:
        print("[ERROR] No ZIP files found.")
        return 1

    log_path = output_dir / "ingest_log.csv"

    for zp in zip_files:
        print(f"\n[INGEST] {zp.name}")
        ingest_zip(zp, output_dir, args.fs, log_path)

    print("\nIngestion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
