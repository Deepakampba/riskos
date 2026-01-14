from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from riskos.pipeline import run_pd_trend_and_ear


def make_demo_df(seed: int = 42) -> pd.DataFrame:
    """Small synthetic dataset just to prove the pipeline works end-to-end."""
    np.random.seed(seed)

    quarters = [
        "2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4",
        "2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4",
    ]
    n = len(quarters)

    base_pd = 0.02
    pd_values = [base_pd]
    for _ in range(1, n):
        next_pd = pd_values[-1] + np.random.normal(0.002, 0.001)
        pd_values.append(max(0.0, float(next_pd)))

    df = pd.DataFrame(
        {
            "as_of_quarter": quarters,
            "Sector": "Manufacture",
            "PD_T": pd_values,
            "Exposure": np.random.normal(1_000_000, 150_000, n).round(2),
        }
    )
    return df


def read_input_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {path}")
    return df


def write_outputs(df_final: pd.DataFrame, summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df_path = out_dir / "df_final.csv"
    summary_path = out_dir / "summary.json"

    df_final.to_csv(df_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:")
    print(f" - {df_path}")
    print(f" - {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run RiskOS PD trend + Exposure-at-Risk pipeline on demo data or an input CSV."
    )

    p.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input CSV. If omitted, runs on synthetic demo data.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="outputs",
        help="Output directory (default: outputs).",
    )

    p.add_argument(
        "--slope-th",
        type=float,
        default=0.0015,
        help="Slope threshold for deterioration flag (PD per quarter). Default 0.0015 (~15 bps/q).",
    )
    p.add_argument(
        "--seed-prev-pd-1",
        type=float,
        default=0.019,
        help="Seed PD for T-1 before first row (default: 0.019). Use 'nan' to disable seeding.",
    )
    p.add_argument(
        "--seed-prev-pd-2",
        type=float,
        default=0.018,
        help="Seed PD for T-2 before first row (default: 0.018). Use 'nan' to disable seeding.",
    )

    # Demo-only
    p.add_argument(
        "--demo-seed",
        type=int,
        default=42,
        help="Random seed for demo dataframe generation (default: 42).",
    )

    return p


def _nan_to_none(x: float) -> Optional[float]:
    # Allows: --seed-prev-pd-1 nan
    return None if (isinstance(x, float) and np.isnan(x)) else float(x)

def print_headline(summary: dict) -> None:
    pd_s = summary.get("pd_trend", {})
    ear = summary.get("ear", {})

    tier = pd_s.get("tier", "NA")
    slope_bps = pd_s.get("slope_bps_per_q", float("nan"))
    flags = pd_s.get("flags", 0)
    n_q = pd_s.get("n_quarters", 0)
    latest_q = pd_s.get("latest_quarter", "NA")
    latest_flag = pd_s.get("latest_flag", 0)

    total_exp = ear.get("total_exposure", 0.0)
    ear_exp = ear.get("exposure_at_risk", 0.0)
    ear_pct = ear.get("exposure_at_risk_pct", 0.0)

    print("\nHeadline:")
    print(
        f"Tier={tier} | slope={slope_bps:.1f} bps/q | "
        f"EaR={ear_pct:.1f}% ({ear_exp:,.0f} / {total_exp:,.0f}) | "
        f"flagged_quarters={flags}/{n_q} | latest={latest_q} flag={latest_flag}"
    )

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve() if args.input else None
    out_dir = Path(args.out).expanduser().resolve()

    seed_prev_pd_1 = _nan_to_none(args.seed_prev_pd_1)
    seed_prev_pd_2 = _nan_to_none(args.seed_prev_pd_2)

    if input_path is None:
        df = make_demo_df(seed=args.demo_seed)
        print(f"Running DEMO mode (demo-seed={args.demo_seed})")
    else:
        df = read_input_csv(input_path)
        print(f"Running CSV mode (input={input_path})")

    df_final, summary = run_pd_trend_and_ear(
        df,
        slope_th=args.slope_th,
        seed_prev_pd_1=seed_prev_pd_1,
        seed_prev_pd_2=seed_prev_pd_2,
    )

    write_outputs(df_final, summary, out_dir)
    print_headline(summary)

if __name__ == "__main__":
    main()