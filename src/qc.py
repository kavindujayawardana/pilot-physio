#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from statsmodels.tsa.stattools import acf

try:
    from jinja2 import Template
except ImportError:
    Template = None


DEFAULT_FS = 256.0


@dataclass
class QCSummaryRow:
    subject_id: str
    session_name: str
    file_path: str

    # Time diagnostics
    mean_dt: float
    median_dt: float
    expected_dt: float
    num_gaps_gt_2samples: int
    first_gap_time: Optional[float]

    # Channel diagnostics (high-level)
    channels_present: str
    num_channels: int
    flatline_channels: str
    noisy_channels: str

    # Event diagnostics
    event_counts_json: str
    event_total_duration_json: str

    # Simple flags
    flag_time_gaps: bool
    flag_flatline: bool
    flag_noisy: bool


def find_subject_session(parquet_path: Path) -> Tuple[str, str]:
    # processed/<subject>/<session>_clean.parquet
    subject_id = parquet_path.parent.name
    session_name = parquet_path.stem.replace("_clean", "")
    return subject_id, session_name


def pick_channel(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    # Case-insensitive match
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def pick_representative_eeg(df: pd.DataFrame) -> Optional[str]:
    # Heuristic: pick first column that looks like EEG channel name
    # If you have a known list, replace this with your real channel list.
    ignore = {"timesecs", "event"}
    for c in df.columns:
        if c.lower() in ignore:
            continue
        # EEG often has short labels; but dataset-specific.
        # We'll just pick the first non-TimeSecs/Event if nothing else is known.
        return c
    return None


def compute_time_diagnostics(df: pd.DataFrame, fs: float) -> Tuple[float, float, float, int, Optional[float]]:
    expected_dt = 1.0 / fs
    if "TimeSecs" not in df.columns:
        # If ingestion guaranteed it, this is unusual.
        return np.nan, np.nan, expected_dt, 0, None

    dt = df["TimeSecs"].diff().dropna()
    if len(dt) == 0:
        return np.nan, np.nan, expected_dt, 0, None

    mean_dt = float(dt.mean())
    median_dt = float(dt.median())

    gaps = dt[dt > (2.0 * expected_dt)]
    num_gaps = int(len(gaps))
    first_gap_time = float(df["TimeSecs"].iloc[gaps.index[0]]) if num_gaps > 0 else None
    return mean_dt, median_dt, expected_dt, num_gaps, first_gap_time


def compute_channel_stats(df: pd.DataFrame) -> pd.DataFrame:
    ignore = {"TimeSecs", "Event"}
    rows = []
    for col in df.columns:
        if col in ignore:
            continue
        x = pd.to_numeric(df[col], errors="coerce").astype(float)
        x = x.dropna()
        if len(x) == 0:
            rows.append({"channel": col, "mean": np.nan, "std": np.nan, "pct_zero": np.nan, "kurtosis": np.nan})
            continue

        pct_zero = float((x == 0).mean() * 100.0)
        rows.append({
            "channel": col,
            "mean": float(x.mean()),
            "std": float(x.std()),
            "pct_zero": pct_zero,
            "kurtosis": float(kurtosis(x.values, fisher=False, nan_policy="omit")),
        })
    return pd.DataFrame(rows)


def flag_channels(stats: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # Simple beginner thresholds. You can tune later.
    flat = []
    noisy = []

    # Flatline: std nearly zero or almost always zero
    for _, r in stats.iterrows():
        ch = r["channel"]
        if pd.notna(r["std"]) and r["std"] < 1e-6:
            flat.append(ch)
        elif pd.notna(r["pct_zero"]) and r["pct_zero"] > 95:
            flat.append(ch)

    # Noisy: very large std compared to median std across channels
    stds = stats["std"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(stds) > 0:
        med = float(stds.median())
        for _, r in stats.iterrows():
            ch = r["channel"]
            if pd.notna(r["std"]) and med > 0 and r["std"] > (10.0 * med):
                noisy.append(ch)

    return flat, noisy


def compute_event_stats(df: pd.DataFrame, fs: float) -> Tuple[Dict[int, int], Dict[int, float]]:
    if "Event" not in df.columns:
        return {0: int(len(df))}, {0: float(len(df) / fs)}

    ev = pd.to_numeric(df["Event"], errors="coerce").fillna(0).astype(int)
    counts = ev.value_counts().to_dict()
    counts = {int(k): int(v) for k, v in counts.items()}

    # durations: contiguous segments
    dt = 1.0 / fs
    total_dur = {k: 0.0 for k in counts.keys()}

    # find runs
    vals = ev.values
    if len(vals) == 0:
        return counts, total_dur

    start = 0
    for i in range(1, len(vals)):
        if vals[i] != vals[i - 1]:
            label = int(vals[i - 1])
            seg_len = i - start
            total_dur[label] = total_dur.get(label, 0.0) + seg_len * dt
            start = i

    # last segment
    label = int(vals[-1])
    seg_len = len(vals) - start
    total_dur[label] = total_dur.get(label, 0.0) + seg_len * dt

    # ensure python floats
    total_dur = {int(k): float(v) for k, v in total_dur.items()}
    return counts, total_dur


def plot_snippet_with_events(
    df: pd.DataFrame,
    subject_id: str,
    session_name: str,
    out_png: Path,
    fs: float,
    snippet_start_s: float = 0.0,
    snippet_len_s: float = 10.0,
) -> None:
    if "TimeSecs" not in df.columns:
        return

    t0 = snippet_start_s
    t1 = snippet_start_s + snippet_len_s
    snippet = df[(df["TimeSecs"] >= t0) & (df["TimeSecs"] <= t1)].copy()
    if len(snippet) == 0:
        return

    # Pick channels
    eeg_ch = pick_representative_eeg(snippet)
    ecg_ch = pick_channel(snippet, ["ECG"])
    gsr_ch = pick_channel(snippet, ["GSR"])

    plt.figure()
    plt.title(f"{subject_id} - {session_name} (10s snippet with events)")
    plt.xlabel("Time (s)")

    # Plot available channels (scaled separately by default; simple overlay)
    # For clarity you can also normalize each signal later.
    if eeg_ch is not None:
        plt.plot(snippet["TimeSecs"], pd.to_numeric(snippet[eeg_ch], errors="coerce"), label=f"EEG: {eeg_ch}")
    if ecg_ch is not None:
        plt.plot(snippet["TimeSecs"], pd.to_numeric(snippet[ecg_ch], errors="coerce"), label="ECG")
    if gsr_ch is not None:
        plt.plot(snippet["TimeSecs"], pd.to_numeric(snippet[gsr_ch], errors="coerce"), label="GSR")

    # Event overlay: vertical lines where event changes
    if "Event" in snippet.columns:
        ev = pd.to_numeric(snippet["Event"], errors="coerce").fillna(0).astype(int).values
        times = snippet["TimeSecs"].values
        for i in range(1, len(ev)):
            if ev[i] != ev[i - 1]:
                plt.axvline(times[i], linestyle="--", linewidth=1)
        # annotate start event
        plt.text(times[0], plt.ylim()[1], f"Event start: {int(ev[0])}", va="top")

    plt.legend(loc="best")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_acf(
    series: pd.Series,
    title: str,
    out_png: Path,
    fs: float,
    max_lag_s: float = 10.0,
) -> None:
    x = pd.to_numeric(series, errors="coerce").astype(float).dropna().values
    if len(x) == 0:
        return

    nlags = int(max_lag_s * fs)
    nlags = min(nlags, len(x) - 1) if len(x) > 1 else 0
    if nlags <= 1:
        return

    vals = acf(x, nlags=nlags, fft=True)
    lags_s = np.arange(len(vals)) / fs

    plt.figure()
    plt.title(title)
    plt.xlabel("Lag (s)")
    plt.ylabel("ACF")
    plt.plot(lags_s, vals)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def make_html_report(
    subject_id: str,
    session_name: str,
    parquet_path: Path,
    time_diag: Tuple[float, float, float, int, Optional[float]],
    channel_stats: pd.DataFrame,
    flat: List[str],
    noisy: List[str],
    event_counts: Dict[int, int],
    event_durations: Dict[int, float],
    snippet_png: Optional[Path],
    acf_ecg_png: Optional[Path],
    acf_eeg_png: Optional[Path],
    out_html: Path,
) -> None:
    # Minimal HTML (works even without jinja2)
    chan_table_html = channel_stats.to_html(index=False, float_format=lambda x: f"{x:.6g}")

    mean_dt, median_dt, expected_dt, num_gaps, first_gap = time_diag

    html = f"""
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>QC Report - {subject_id} {session_name}</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        code {{ background: #f4f4f4; padding: 2px 4px; }}
        .section {{ margin-top: 20px; }}
        img {{ max-width: 100%; border: 1px solid #ddd; padding: 4px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 6px; }}
        th {{ background: #f2f2f2; }}
      </style>
    </head>
    <body>
      <h1>QC Report</h1>
      <p><b>Subject:</b> {subject_id}<br/>
         <b>Session:</b> {session_name}<br/>
         <b>File:</b> <code>{parquet_path}</code></p>

      <div class="section">
        <h2>Time Diagnostics</h2>
        <ul>
          <li>Expected dt: {expected_dt:.8f} s</li>
          <li>Mean dt: {mean_dt:.8f} s</li>
          <li>Median dt: {median_dt:.8f} s</li>
          <li>Gaps > 2 samples: {num_gaps} {f"(first gap at {first_gap:.3f}s)" if first_gap is not None else ""}</li>
        </ul>
      </div>

      <div class="section">
        <h2>Channel Statistics</h2>
        <p><b>Flatline channels:</b> {", ".join(flat) if flat else "None"}</p>
        <p><b>Noisy channels:</b> {", ".join(noisy) if noisy else "None"}</p>
        {chan_table_html}
      </div>

      <div class="section">
        <h2>Event Statistics</h2>
        <p><b>Counts:</b> {json.dumps(event_counts, sort_keys=True)}</p>
        <p><b>Total durations (s):</b> {json.dumps(event_durations, sort_keys=True)}</p>
      </div>

      <div class="section">
        <h2>Plots</h2>
        <h3>10-second snippet with events</h3>
        {"<img src='" + snippet_png.as_posix() + "'/>" if snippet_png else "<p>Not available</p>"}

        <h3>ACF (ECG)</h3>
        {"<img src='" + acf_ecg_png.as_posix() + "'/>" if acf_ecg_png else "<p>Not available</p>"}

        <h3>ACF (EEG representative channel)</h3>
        {"<img src='" + acf_eeg_png.as_posix() + "'/>" if acf_eeg_png else "<p>Not available</p>"}
      </div>

    </body>
    </html>
    """

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="QC diagnostics for processed session parquet files.")
    parser.add_argument("--processed", required=True, help="Processed directory (e.g., ./processed)")
    parser.add_argument("--out", default="qc_report", help="QC report output directory (default qc_report)")
    parser.add_argument("--fs", type=float, default=DEFAULT_FS, help="Sampling rate Hz (default 256)")
    args = parser.parse_args()

    processed_dir = Path(args.processed).resolve()
    qc_out_dir = Path(args.out).resolve()
    plots_dir = qc_out_dir / "plots"
    qc_out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(processed_dir.glob("*/*_clean.parquet"))
    if not parquet_files:
        print(f"[ERROR] No session parquet files found under: {processed_dir}")
        return 2

    summary_rows: List[QCSummaryRow] = []

    for pq in parquet_files:
        subject_id, session_name = find_subject_session(pq)
        print(f"[QC] {subject_id} / {session_name}")

        df = pd.read_parquet(pq)

        # ---- Compute diagnostics ----
        time_diag = compute_time_diagnostics(df, args.fs)
        chan_stats = compute_channel_stats(df)
        flat, noisy = flag_channels(chan_stats)
        ev_counts, ev_durs = compute_event_stats(df, args.fs)

        # ---- Plots ----
        # Use relative paths from HTML to plots (store under qc_report/plots)
        snippet_png = plots_dir / f"{subject_id}_{session_name}_snippet.png"
        acf_ecg_png = plots_dir / f"{subject_id}_{session_name}_acf_ecg.png"
        acf_eeg_png = plots_dir / f"{subject_id}_{session_name}_acf_eeg.png"

        plot_snippet_with_events(df, subject_id, session_name, snippet_png, args.fs)

        ecg_ch = pick_channel(df, ["ECG"])
        eeg_ch = pick_representative_eeg(df)

        if ecg_ch is not None:
            plot_acf(df[ecg_ch], f"{subject_id} - {session_name} ACF (ECG)", acf_ecg_png, args.fs)
        else:
            acf_ecg_png = None

        if eeg_ch is not None:
            plot_acf(df[eeg_ch], f"{subject_id} - {session_name} ACF (EEG: {eeg_ch})", acf_eeg_png, args.fs)
        else:
            acf_eeg_png = None

        # ---- HTML report ----
        out_html = qc_out_dir / f"{subject_id}_{session_name}_qc.html"

        # In HTML, image paths should be relative (so opening HTML works easily)
        rel_snip = Path("plots") / snippet_png.name if snippet_png.exists() else None
        rel_ecg = Path("plots") / acf_ecg_png.name if acf_ecg_png is not None and (plots_dir / acf_ecg_png.name).exists() else None
        rel_eeg = Path("plots") / acf_eeg_png.name if acf_eeg_png is not None and (plots_dir / acf_eeg_png.name).exists() else None

        make_html_report(
            subject_id, session_name, pq,
            time_diag, chan_stats, flat, noisy,
            ev_counts, ev_durs,
            rel_snip, rel_ecg, rel_eeg,
            out_html
        )

        # Optional: session CSV stats (nice for debugging)
        session_csv = qc_out_dir / f"{subject_id}_{session_name}_qc.csv"
        chan_stats.to_csv(session_csv, index=False)

        # ---- Summary row ----
        mean_dt, median_dt, expected_dt, num_gaps, first_gap = time_diag
        row = QCSummaryRow(
            subject_id=subject_id,
            session_name=session_name,
            file_path=str(pq),

            mean_dt=mean_dt,
            median_dt=median_dt,
            expected_dt=expected_dt,
            num_gaps_gt_2samples=num_gaps,
            first_gap_time=first_gap,

            channels_present="|".join(df.columns),
            num_channels=int(len(df.columns)),
            flatline_channels="|".join(flat),
            noisy_channels="|".join(noisy),

            event_counts_json=json.dumps(ev_counts, sort_keys=True),
            event_total_duration_json=json.dumps(ev_durs, sort_keys=True),

            flag_time_gaps=(num_gaps > 0),
            flag_flatline=(len(flat) > 0),
            flag_noisy=(len(noisy) > 0),
        )
        summary_rows.append(row)

    # Write aggregated summary CSV
    summary_df = pd.DataFrame([asdict(r) for r in summary_rows])
    summary_csv = Path("qc_summary.csv").resolve()
    summary_df.to_csv(summary_csv, index=False)

    print("\nDone.")
    print(f"Per-session reports: {qc_out_dir}")
    print(f"Aggregated summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
