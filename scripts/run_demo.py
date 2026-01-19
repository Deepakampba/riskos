from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from riskos.pipeline import run_pd_trend_and_ear, run_portfolio_pd_trend_and_ear
from riskos.validation import DataContract, normalize_and_aggregate, validate_single_series


def make_demo_df(seed: int = 42) -> pd.DataFrame:
    """Synthetic multi-sector dataset just to test portfolio mode end-to-end."""
    rng = np.random.default_rng(seed)

    quarters = [
        "2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4",
        "2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4",
    ]
    n = len(quarters)
    sectors = {
        "Manufacture": {"base_pd": 0.020, "drift": 0.0020, "vol": 0.0010,"exp_mu":1_000_000,"exp_sd":150_000},
        "Retail":      {"base_pd": 0.015, "drift": 0.0010, "vol": 0.0008,"exp_mu":800_000,"exp_sd":120_000},
        "Tech":      {"base_pd": 0.010, "drift": 0.0004, "vol": 0.0006,"exp_mu":1_200_000,"exp_sd":180_000},
        "RealEstate":      {"base_pd": 0.025, "drift": 0.0025, "vol": 0.0012,"exp_mu":1_500_000,"exp_sd":220_000}
    }
    frames = []
    for sector, cfg in sectors.items():
        pd_values = [float(cfg["base_pd"])]

        for _ in range(1,n):
            step = float(rng.normal(cfg["drift"],cfg["vol"]))
            next_pd = max(0.0, pd_values[-1] + step)
            pd_values.append(next_pd)
        exposure = rng.normal(cfg["exp_mu"], cfg["exp_sd"], n)
        exposure = np.maximum(0.0, exposure).round(2)  # keep non-negative
        frames.append(
             pd.DataFrame(
                {
                    "as_of_quarter": quarters,
                    "Sector": sector,
                    "PD_T": pd_values,
                    "Exposure": exposure,
                }
             )
        )     

    df = pd.concat(frames, ignore_index=True)
      # Optional: inject a deliberate duplicate row to prove normalize_and_aggregate works
    # (Same Sector + Quarter appears twice)
    dup = df.iloc[[0]].copy()
    dup["Exposure"] = (dup["Exposure"] * 0.25).round(2)
    df = pd.concat([df, dup], ignore_index=True)
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

    p.add_argument(
        "--portfolio", 
        action="store_true",
        help="Run per-group portfolio mode"
    )
    p.add_argument(
        "--group-col", 
        default="Sector",
        help="Column to group by in portfolio mode"
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
        if not args.portfolio:
            df = df[df[args.group_col].eq(df[args.group_col].iloc[0])].copy()
        print(f"Running DEMO mode (demo-seed={args.demo_seed})")
    else:
        df = read_input_csv(input_path)
        print(f"Running CSV mode (input={input_path})")

    contract = DataContract(group_col=args.group_col)
    print("RAW DF COLUMNS:", df.columns.tolist())
   
    df=normalize_and_aggregate(df,contract=contract)
    
    if not args.portfolio:
        validate_single_series(df,contract=contract)

    if args.portfolio:
        df_final, summary = run_portfolio_pd_trend_and_ear(
        df,
        group_col=args.group_col,
        slope_th=args.slope_th,
        seed_prev_pd_1=seed_prev_pd_1,
        seed_prev_pd_2=seed_prev_pd_2,
    )
    else:
        df_final, summary = run_pd_trend_and_ear(
        df,
        slope_th=args.slope_th,
        seed_prev_pd_1=seed_prev_pd_1,
        seed_prev_pd_2=seed_prev_pd_2,
    )

    write_outputs(df_final, summary, out_dir)
    
    if args.portfolio:
        print("\nPortfolio headline:")
        p = summary["portfolio"]
        print(
            f"Groups={p['n_groups']} | TotalExp={p['total_exposure']:,.0f} | "
            f"EaR={p['exposure_at_risk_pct']:.1f}% ({p['exposure_at_risk']:,.0f} / {p['total_exposure']:,.0f})"
        )
    else:
        print_headline(summary)

if __name__ == "__main__":
    main()