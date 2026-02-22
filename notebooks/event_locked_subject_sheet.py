#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

FS_DEFAULT = 256


def bandpower(psd: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs <= high)
    if mask.sum() < 2:
        return np.nan
    return float(np.trapezoid(psd[mask], freqs[mask]))


def sliding_band_logpower(eeg: np.ndarray, fs: int, win_s: float = 1.0):
    """1s Welch sliding band powers (log10), updated each sample."""
    win = int(win_s * fs)
    if len(eeg) < win + 1:
        return None

    eps = 1e-12
    delta, theta, alpha, beta = [], [], [], []

    for i in range(0, len(eeg) - win):
        seg = eeg[i : i + win]
        f, psd = welch(seg, fs=fs, nperseg=win)
        delta.append(bandpower(psd, f, 1, 4))
        theta.append(bandpower(psd, f, 4, 8))
        alpha.append(bandpower(psd, f, 8, 13))
        beta.append(bandpower(psd, f, 13, 30))

    delta = np.log10(np.array(delta) + eps)
    theta = np.log10(np.array(theta) + eps)
    alpha = np.log10(np.array(alpha) + eps)
    beta  = np.log10(np.array(beta)  + eps)

    return delta, theta, alpha, beta


def safe_col(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns


def list_subjects(processed_root: Path) -> list[str]:
    subs = []
    for p in processed_root.iterdir():
        if p.is_dir() and p.name.isdigit():
            subs.append(p.name)
    return sorted(subs, key=lambda x: int(x))


def main():
    ap = argparse.ArgumentParser(description="Make 1 big PNG per subject with all transition panels.")
    ap.add_argument("--processed", default="./processed", help="processed/ folder")
    ap.add_argument("--out", default="./results/event_locked_subject_sheets", help="output folder")
    ap.add_argument("--fs", type=int, default=FS_DEFAULT)
    ap.add_argument("--pre", type=float, default=10.0, help="seconds before transition")
    ap.add_argument("--post", type=float, default=30.0, help="seconds after transition")
    ap.add_argument("--eeg-channel", default="EEG_FP1", help="EEG channel for band power panel")
    ap.add_argument("--max-transitions", type=int, default=0,
                    help="0 = ALL transitions. Otherwise cap total transitions per subject.")
    args = ap.parse_args()

    processed_root = Path(args.processed)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    pre_n = int(args.pre * args.fs)
    post_n = int(args.post * args.fs)

    # Collect all session files grouped by subject
    session_files = sorted(processed_root.glob("*/*_clean.parquet"))
    by_subject: dict[str, list[Path]] = {}
    for pq in session_files:
        subject = pq.parent.name
        session = pq.stem.replace("_clean", "")
        if session.startswith("._") or pq.name.startswith("._"):
            continue
        by_subject.setdefault(subject, []).append(pq)

    if not by_subject:
        raise SystemExit(f"No */*_clean.parquet found under {processed_root}")

    # One image per subject
    for subject, files in by_subject.items():
        panels = []  # each panel is dict with data to plot
        # Build a list of transitions across all sessions
        for pq in sorted(files):
            session = pq.stem.replace("_clean", "")
            df = pd.read_parquet(pq)

            if "Event" not in df.columns:
                continue

            # Clean Event deterministically
            df["Event"] = (
                pd.to_numeric(df["Event"], errors="coerce")
                  .replace([np.inf, -np.inf], np.nan)
                  .fillna(0)
                  .astype(int)
            )

            # Find change points
            change_idx = df.index[df["Event"].diff().fillna(0) != 0].to_numpy()
            if len(change_idx) == 0:
                continue

            # Columns we want
            has_ecg = safe_col(df, "ECG")
            has_gsr = safe_col(df, "GSR")
            has_r = safe_col(df, "R")
            has_eeg = safe_col(df, args.eeg_channel)

            if not (has_ecg or has_gsr or has_r or has_eeg):
                continue

            for idx in change_idx:
                idx = int(idx)
                start = max(0, idx - pre_n)
                end = min(len(df), idx + post_n)

                seg = df.iloc[start:end].copy()
                if len(seg) < (pre_n + 10):  # too short to be useful
                    continue

                # time axis centered at transition
                t = np.arange(len(seg)) / args.fs - args.pre

                before = int(df["Event"].iloc[idx - 1]) if idx - 1 >= 0 else int(df["Event"].iloc[idx])
                after = int(df["Event"].iloc[idx])

                # raw series (keep as numpy for plotting)
                ecg = seg["ECG"].to_numpy() if has_ecg else None
                gsr = seg["GSR"].to_numpy() if has_gsr else None
                rr  = seg["R"].to_numpy()   if has_r   else None

                bands = None
                if has_eeg:
                    eeg = pd.to_numeric(seg[args.eeg_channel], errors="coerce").to_numpy()
                    eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)
                    bands = sliding_band_logpower(eeg, fs=args.fs, win_s=1.0)

                panels.append({
                    "session": session,
                    "idx": idx,
                    "before": before,
                    "after": after,
                    "t": t,
                    "ecg": ecg,
                    "gsr": gsr,
                    "r": rr,
                    "bands": bands,  # (delta,theta,alpha,beta) or None
                })

        if not panels:
            print(f"[WARN] Subject {subject}: no transitions/plots found. Skipping.")
            continue

        # Optional cap across subject
        if args.max_transitions and len(panels) > args.max_transitions:
            panels = panels[: args.max_transitions]

        # Layout: each transition row has 4 columns (ECG, GSR, R, EEG-bands)
        n = len(panels)
        ncols = 4
        nrows = n

        # Figure sizing (tuneable): each row ~1.6 inches tall
        fig_h = max(6, nrows * 1.6)
        fig_w = 16
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.suptitle(
            f"Subject {subject} — Event transitions (−{args.pre:.0f}s to +{args.post:.0f}s) | EEG={args.eeg_channel}",
            fontsize=12,
            y=0.995,
        )

        for i, p in enumerate(panels):
            row = i

            # ---- ECG
            ax = plt.subplot(nrows, ncols, row * ncols + 1)
            if p["ecg"] is not None:
                ax.plot(p["t"], p["ecg"], linewidth=0.6)
            ax.axvline(0, color="red", linewidth=0.8)
            ax.set_title(f"{p['session']} idx={p['idx']}  {p['before']}→{p['after']}", fontsize=8)
            ax.set_ylabel("ECG", fontsize=8)
            if row != nrows - 1:
                ax.set_xticklabels([])

            # ---- GSR
            ax = plt.subplot(nrows, ncols, row * ncols + 2)
            if p["gsr"] is not None:
                ax.plot(p["t"], p["gsr"], linewidth=0.6)
            ax.axvline(0, color="red", linewidth=0.8)
            ax.set_ylabel("GSR", fontsize=8)
            if row != nrows - 1:
                ax.set_xticklabels([])

            # ---- Respiration
            ax = plt.subplot(nrows, ncols, row * ncols + 3)
            if p["r"] is not None:
                ax.plot(p["t"], p["r"], linewidth=0.6)
            ax.axvline(0, color="red", linewidth=0.8)
            ax.set_ylabel("R", fontsize=8)
            if row != nrows - 1:
                ax.set_xticklabels([])

            # ---- EEG bands
            ax = plt.subplot(nrows, ncols, row * ncols + 4)
            if p["bands"] is not None:
                delta, theta, alpha, beta = p["bands"]
                tb = p["t"][: len(alpha)]
                ax.plot(tb, delta, linewidth=0.6, label="δ")
                ax.plot(tb, theta, linewidth=0.6, label="θ")
                ax.plot(tb, alpha, linewidth=0.6, label="α")
                ax.plot(tb, beta,  linewidth=0.6, label="β")
                ax.axvline(0, color="red", linewidth=0.8)
                if row == 0:
                    ax.legend(fontsize=7, ncol=4, loc="upper right", frameon=False)
                ax.set_ylabel("log10(P)", fontsize=8)
            else:
                ax.text(0.02, 0.5, "EEG missing", transform=ax.transAxes, fontsize=8)
                ax.axvline(0, color="red", linewidth=0.8)

            if row == nrows - 1:
                ax.set_xlabel("Time (s)", fontsize=8)
            else:
                ax.set_xticklabels([])

            # cosmetic: smaller ticks
            for a in fig.axes[-4:]:
                a.tick_params(labelsize=7)

        out_png = out_root / f"subject_{subject}_transitions.png"
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(out_png, dpi=150)
        plt.close(fig)

        print(f"[OK] Saved {out_png} (transitions plotted: {len(panels)})")


if __name__ == "__main__":
    main()