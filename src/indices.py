#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def safe_std(x: pd.Series, eps: float = 1e-8) -> float:
    sd = float(x.std(ddof=0))
    if not np.isfinite(sd) or sd < eps:
        return eps
    return sd


def zscore(series: pd.Series, mu: float, sd: float) -> pd.Series:
    sd = sd if (sd is not None and sd > 0) else 1.0
    out = (series.astype(float) - float(mu)) / float(sd)
    # keep NaNs if series has NaNs; don’t fill here
    return out


def benchmark_mask(df: pd.DataFrame) -> pd.Series:
    """
    Benchmark-only mask:
    - BenchmarkTask == True (preferred)
      OR SessionType in {CA,DA,SS} if BenchmarkTask missing.
    - Exclude transitions from stats if IsTransition exists (best practice).
    - Exclude bad windows from stats if IsBadWindow exists (best practice).
    """
    if "BenchmarkTask" in df.columns:
        m = df["BenchmarkTask"].astype(bool)
    elif "SessionType" in df.columns:
        m = df["SessionType"].astype(str).str.upper().isin(["CA", "DA", "SS"])
    else:
        # if we have no clue, treat all as benchmark (not ideal, but avoids crash)
        m = pd.Series(True, index=df.index)

    if "IsTransition" in df.columns:
        m = m & (~df["IsTransition"].astype(bool))

    if "IsBadWindow" in df.columns:
        m = m & (~df["IsBadWindow"].astype(bool))

    return m


@dataclass
class IndexSpec:
    name: str
    description: str
    features: list[str]
    combination: str


def load_indices_spec(spec_path: Path) -> dict[str, IndexSpec]:
    spec_raw = read_json(spec_path)
    out: dict[str, IndexSpec] = {}
    for name, cfg in spec_raw.items():
        out[name] = IndexSpec(
            name=name,
            description=str(cfg.get("description", "")),
            features=list(cfg.get("features", [])),
            combination=str(cfg.get("combination", "")),
        )
    return out


def compute_subject_feature_norms(df: pd.DataFrame, feats: list[str]) -> dict[str, dict[str, float]]:
    """
    Compute per-feature mean/std from benchmark rows only.
    """
    m = benchmark_mask(df)
    ref = df.loc[m]

    norms: dict[str, dict[str, float]] = {}
    for f in feats:
        if f not in df.columns:
            continue
        s = pd.to_numeric(ref[f], errors="coerce")
        mu = float(s.mean(skipna=True))
        sd = safe_std(s.dropna())
        norms[f] = {"mean": mu, "std": sd}
    return norms


def compute_indices(df: pd.DataFrame, spec: dict[str, IndexSpec]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Returns:
      - df with SAI / CAI added
      - stats dict for transparency (benchmark-only means/stds)
    """
    # collect all required feature names
    required_feats: list[str] = []
    for idx in spec.values():
        for f in idx.features:
            if f not in required_feats:
                required_feats.append(f)

    # sanity check required columns exist
    missing = [f for f in required_feats if f not in df.columns]
    if missing:
        raise ValueError(
            "Missing required feature columns for indices: "
            + ", ".join(missing)
            + "\n(Ensure Step 7 features extraction produced these columns.)"
        )

    # compute per-subject benchmark-only norms for those features
    feat_norms = compute_subject_feature_norms(df, required_feats)

    # z-score each required feature for all rows (using benchmark-only stats)
    z_cols: dict[str, pd.Series] = {}
    for f in required_feats:
        mu = feat_norms[f]["mean"]
        sd = feat_norms[f]["std"]
        z_cols[f] = zscore(pd.to_numeric(df[f], errors="coerce"), mu, sd)

    # build raw indices
    out = df.copy()

    # SAI: z-score sum of listed features
    if "SAI" in spec:
        sai_feats = spec["SAI"].features
        sai_raw = None
        for f in sai_feats:
            sai_raw = z_cols[f] if sai_raw is None else (sai_raw + z_cols[f])
        out["SAI_raw"] = sai_raw

    # CAI: z(theta_beta_ratio) - z(frontal_mean_alpha)
    if "CAI" in spec:
        cai_feats = spec["CAI"].features
        if len(cai_feats) != 2:
            raise ValueError("CAI spec must contain exactly 2 features: [theta_beta_ratio, frontal_mean_alpha]")
        out["CAI_raw"] = z_cols[cai_feats[0]] - z_cols[cai_feats[1]]

    # optionally standardize index itself to mean 0, std 1 on benchmark rows
    m = benchmark_mask(out)

    idx_stats: dict[str, Any] = {
        "created_utc": utc_now(),
        "benchmark_filter": {
            "used_BenchmarkTask_if_present": "BenchmarkTask" in out.columns,
            "excluded_IsTransition_if_present": "IsTransition" in out.columns,
            "excluded_IsBadWindow_if_present": "IsBadWindow" in out.columns,
        },
        "feature_norms_benchmark_only": feat_norms,
        "index_norms_benchmark_only": {},
    }

    for idx_name in ["SAI", "CAI"]:
        raw_col = f"{idx_name}_raw"
        if raw_col not in out.columns:
            continue

        ref = pd.to_numeric(out.loc[m, raw_col], errors="coerce")
        mu = float(ref.mean(skipna=True))
        sd = safe_std(ref.dropna())

        idx_stats["index_norms_benchmark_only"][idx_name] = {"mean": mu, "std": sd}

        out[idx_name] = zscore(pd.to_numeric(out[raw_col], errors="coerce"), mu, sd)

    return out, idx_stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute SAI/CAI anxiety indices from extracted features (benchmark-only norms).")
    ap.add_argument("--features", required=True, help="Path to features/ (contains <subject>/features.parquet)")
    ap.add_argument("--spec", default="features/indices_spec.json", help="Path to indices_spec.json")
    ap.add_argument("--out", default=None, help="Output root. Default: same as --features")
    ap.add_argument(
        "--write-stats",
        action="store_true",
        help="Write per-subject indices_stats.json next to outputs (recommended, but do NOT commit).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite features.parquet with indices columns (NOT recommended). Default writes features_with_indices.parquet",
    )
    args = ap.parse_args()

    features_root = Path(args.features).resolve()
    spec_path = Path(args.spec).resolve()
    out_root = Path(args.out).resolve() if args.out else features_root

    if not spec_path.exists():
        print(f"[ERROR] Spec file not found: {spec_path}")
        return 1

    spec = load_indices_spec(spec_path)

    # find subject feature files
    feat_files = sorted(features_root.glob("*/features.parquet"))
    if not feat_files:
        print(f"[ERROR] No features.parquet found under {features_root}")
        return 1

    for fp in feat_files:
        subject_id = fp.parent.name
        df = pd.read_parquet(fp)

        # compute
        out_df, stats = compute_indices(df, spec)

        subj_out_dir = out_root / subject_id
        subj_out_dir.mkdir(parents=True, exist_ok=True)

        if args.overwrite:
            out_path = subj_out_dir / "features.parquet"
        else:
            out_path = subj_out_dir / "features_with_indices.parquet"

        out_df.to_parquet(out_path, index=False)

        if args.write_stats:
            stats_path = subj_out_dir / "indices_stats.json"
            write_json(stats_path, stats)

        print(f"[OK] {subject_id}: wrote {out_path.name} (rows={len(out_df)}, cols={out_df.shape[1]})")

    print("\nDone indices.")
    if not args.overwrite:
        print("Tip for modeling: use features_with_indices.parquet and filter out transitions (IsTransition==False) + bad windows if needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())