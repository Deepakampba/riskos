from __future__ import annotations
import argparse
import ast
import json
import sys
import os
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, Dict
from riskos.doctor import doctor_csv, print_doctor_report
from riskos.sample_data import make_sample_df, write_sample_csv
import numpy as np
import pandas as pd
from uuid import uuid4
from riskos.pipeline import run_pd_trend_and_ear, run_portfolio_pd_trend_and_ear
from riskos.validation import DataContract, normalize_and_aggregate, validate_single_series


def write_meta(
    out_dir: Path,
    *,
    status: str,
    error: Optional[str] = None,
    error_type: Optional[str] = None,
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    missing_columns: Optional[list[str]] = None,
    required_columns: Optional[list[str]] = None,
    input_source: Optional[str] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"status": status}
    if run_id:
        meta["run_id"] = run_id
    if started_at:
        meta["started_at"] = started_at
    if finished_at:
        meta["finished_at"] = finished_at
    if error:
        meta["error"] = error
    if error_type:
        meta["error_type"] = error_type
    if missing_columns:
        meta["missing_columns"] = missing_columns
    if required_columns:
        meta["required_columns"] = required_columns
    if input_source:
        meta["input_source"] = input_source
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def print_top_risk_groups(summary: dict, *, group_col: str, top_n: int = 5) -> None:
    by_group = summary.get("by_group")
    if not by_group:
        print("\nTop Risk Groups: (no by_group data found in summary)")
        return

    df_bg = pd.DataFrame(by_group)

    # Be defensive: ensure expected columns exist
    required_cols = [group_col, "exposure_at_risk_pct", "slope_bps_per_q", "latest_flag", "tier"]
    missing = [c for c in required_cols if c not in df_bg.columns]
    if missing:
        print(f"\nTop Risk Groups: missing columns in by_group summary: {missing}")
        print("Available:", df_bg.columns.tolist())
        return

    df_bg_sorted = df_bg.sort_values(
        ["exposure_at_risk_pct", "slope_bps_per_q", "latest_flag"],
        ascending=[False, False, False],
        kind="mergesort",
    )

    print(f"\nTop Risk Groups (ranked by EaR% -> slope -> latest_flag) | group_col={group_col}")

    for i, row in enumerate(df_bg_sorted.head(top_n).itertuples(index=False), start=1):
        group_value = getattr(row, group_col)  # row.Sector if group_col="Sector"
        # row.group is the name of the group value (e.g., "Retail")
        print(
            f"{i}. {str(group_value):12s} | "
            f"EaR={row.exposure_at_risk_pct:5.1f}% | "
            f"slope={row.slope_bps_per_q:5.1f} bps/q | "
            f"latest_flag={int(row.latest_flag)} | "
            f"tier={row.tier}"
        )


def make_demo_df(seed: int = 42) -> pd.DataFrame:
    """Synthetic multi-sector dataset just to test portfolio mode end-to-end."""
    rng = np.random.default_rng(seed)

    quarters = [
        "2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4",
        "2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4",
    ]
    n = len(quarters)
    sectors = {
        "Manufacture": {"base_pd": 0.020, "drift": 0.0020, "vol": 0.0010, "exp_mu": 1_000_000, "exp_sd": 150_000},
        "Retail": {"base_pd": 0.015, "drift": 0.0010, "vol": 0.0008, "exp_mu": 800_000, "exp_sd": 120_000},
        "Tech": {"base_pd": 0.010, "drift": 0.0004, "vol": 0.0006, "exp_mu": 1_200_000, "exp_sd": 180_000},
        "RealEstate": {"base_pd": 0.025, "drift": 0.0025, "vol": 0.0012, "exp_mu": 1_500_000, "exp_sd": 220_000},
    }
    frames = []
    for sector, cfg in sectors.items():
        pd_values = [float(cfg["base_pd"])]

        for _ in range(1, n):
            step = float(rng.normal(cfg["drift"], cfg["vol"]))
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

def _utc_now_compact() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _get_git_sha_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "nogit"
    except Exception:
        return "nogit"

def _get_riskos_version() -> str:
    # If you set version in pyproject.toml this will work.
    try:
        from importlib.metadata import version
        return version("riskos")
    except Exception:
        return "unknown"

def build_run_metadata(*, args: Any, input_source: str) -> Dict[str, Any]:
    return {
        "run_id": getattr(args, "run_id", None) or str(uuid4())[:8],
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "portfolio" if getattr(args, "portfolio", False) else "single",
        "command": getattr(args, "command", "run"),
        "input_source": input_source,
        "out_base": str(getattr(args, "out", "")),
        "group_col": getattr(args, "group_col", None),
        "pd_col": getattr(args, "pd_col", "PD_T"),
        "quarter_col": getattr(args, "quarter_col", "as_of_quarter"),
        "exposure_col": getattr(args, "exposure_col", "Exposure"),
        "slope_th": getattr(args, "slope_th", None),
        "seed_prev_pd_1": getattr(args, "seed_prev_pd_1", None),
        "seed_prev_pd_2": getattr(args, "seed_prev_pd_2", None),
        "demo_seed": getattr(args, "demo_seed", None),
        "git_sha": _get_git_sha_short(),
        "riskos_version": _get_riskos_version(),
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
    }

def build_report_text(
    summary: dict,
    *,
    group_col: str,
    top_n: int = 5,
    run_info: Optional[dict] = None,
) -> str:
    lines: list[str] = []
    runtime_local = "NA"
    if run_info:
        runtime_local = str(run_info.get("runtime_local", "NA"))

    title = "RiskOS Report"
    width = 80
    header = title
    if runtime_local:
        if len(title) + len(runtime_local) + 1 <= width:
            header = title.ljust(width - len(runtime_local)) + runtime_local
        else:
            header = f"{title} | {runtime_local}"
    lines.append(header)
    lines.append("=" * len(header))

    if run_info:
        lines.append("")
        lines.append("Run Metadata")
        lines.append("------------")
        mode = run_info.get("mode", "NA")
        input_source = run_info.get("input_source", "NA")
        slope_th = run_info.get("slope_th", "NA")
        seed_prev_pd_1 = run_info.get("seed_prev_pd_1", "NA")
        seed_prev_pd_2 = run_info.get("seed_prev_pd_2", "NA")
        lines.append(f"Mode: {mode}")
        lines.append(f"Input source: {input_source}")
        lines.append(f"Slope threshold: {slope_th}")
        lines.append(f"Seed PDs: prev_pd_1={seed_prev_pd_1} | prev_pd_2={seed_prev_pd_2}")

    if "portfolio" in summary:
        p = summary["portfolio"]
        lines.append("")
        lines.append("Portfolio Summary")
        lines.append("-----------------")
        lines.append(f"Group column: {group_col}")
        lines.append(f"Groups: {p.get('n_groups', 0)}")
        total_exp = p.get("total_exposure", 0.0)
        ear = p.get("exposure_at_risk", 0.0)
        ear_pct = p.get("exposure_at_risk_pct", 0.0)
        lines.append(f"Exposure at Risk: {ear:,.0f} / {total_exp:,.0f} ({ear_pct:.1f}%)")

        by_group = summary.get("by_group") or []
        if by_group:
            df_bg = pd.DataFrame(by_group)
            sort_cols = ["exposure_at_risk_pct", "slope_bps_per_q", "latest_flag"]
            missing = [c for c in sort_cols if c not in df_bg.columns]
            if missing:
                lines.append("")
                lines.append(f"Top risk groups: missing columns: {missing}")
            else:
                df_bg = df_bg.sort_values(
                    by=sort_cols,
                    ascending=[False, False, False],
                    kind="mergesort",
                )
                lines.append("")
                lines.append(f"Top {top_n} Risk Groups (by EaR% -> slope -> latest_flag)")
                lines.append(
                    f"{group_col:12s} | {'EaR%':>5s} | {'EaR':>9s} | {'TotalExp':>9s} | "
                    f"{'slope bps/q':>10s} | {'latest flag':>11s} | tier"
                )
                for row in df_bg.head(top_n).itertuples(index=False):
                    group_value = getattr(row, group_col)
                    lines.append(
                        f"{str(group_value):12s} | "
                        f"{row.exposure_at_risk_pct:5.1f} | "
                        f"{row.exposure_at_risk:9,.0f} | "
                        f"{row.total_exposure:9,.0f} | "
                        f"{row.slope_bps_per_q:10.1f} | "
                        f"{int(row.latest_flag):11d} | "
                        f"{row.tier}"
                    )
            
                flagged = df_bg[df_bg["latest_flag"] == 1][group_col].astype(str).tolist()
                lines.append("")
                
                lines.append(f"Groups with latest_flag=1: {', '.join(flagged)}") if flagged else lines.append("Groups with latest_flag=1: none")
        else:
            lines.append("")
            lines.append("Top risk groups: none")
    else:
        pd_s = summary.get("pd_trend", {})
        ear_s = summary.get("ear", {})
        lines.append("")
        lines.append("Single-Series Summary")
        lines.append("---------------------")
        lines.append(f"Tier: {pd_s.get('tier', 'NA')}")
        lines.append(f"Slope (bps/q): {pd_s.get('slope_bps_per_q', 0.0):.1f}")
        lines.append(
            f"Flags: {pd_s.get('flags', 0)} / {pd_s.get('n_quarters', 0)}"
        )
        lines.append(f"Latest quarter: {pd_s.get('latest_quarter', 'NA')}")
        lines.append(f"Latest flag: {pd_s.get('latest_flag', 0)}")
        total_exp = ear_s.get("total_exposure", 0.0)
        ear = ear_s.get("exposure_at_risk", 0.0)
        ear_pct = ear_s.get("exposure_at_risk_pct", 0.0)
        lines.append(f"Exposure at Risk: {ear:,.0f} / {total_exp:,.0f} ({ear_pct:.1f}%)")

    if "portfolio" in summary:
        p = summary["portfolio"]
        ear_pct = p.get("exposure_at_risk_pct", 0.0)
        top_names: list[str] = []
        top_ear_share_pct: Optional[float] = None
        by_group = summary.get("by_group") or []
        if by_group:
            df_bg = pd.DataFrame(by_group)
            if (
                group_col in df_bg.columns
                and "exposure_at_risk" in df_bg.columns
                and "exposure_at_risk_pct" in df_bg.columns
            ):
                df_bg = df_bg.sort_values(
                    by="exposure_at_risk",
                    ascending=False,
                    kind="mergesort",
                )
                df_bg = df_bg[df_bg["exposure_at_risk_pct"] >= 20.0]
                top_names = df_bg[group_col].astype(str).tolist()
                portfolio_ear = float(p.get("exposure_at_risk", 0.0))
                top_ear = float(df_bg["exposure_at_risk"].sum()) if not df_bg.empty else 0.0
                if portfolio_ear:
                    top_ear_share_pct = top_ear / portfolio_ear * 100.0
        lines.append("")
        if top_names:
            top_list = ", ".join(top_names)
            share_text = (
                f" ({top_ear_share_pct:.1f}% of portfolio EaR)"
                if top_ear_share_pct is not None
                else ""
            )
            lines.append(
                f"Portfolio EaR is {ear_pct:.1f}% driven mainly by top sectors contributing: "
                f"{top_list}{share_text}."
            )
        else:
            lines.append(
                f"Portfolio EaR is {ear_pct:.1f}% driven mainly by top sectors contributing."
            )

    lines.append("")
    return "\n".join(lines)


def write_outputs(
    df_final: pd.DataFrame,
    summary: dict,
    out_dir: Path,
    *,
    group_col: Optional[str] = None,
    run_info: Optional[dict] = None,
    meta: Optional[dict] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    df_path = out_dir / "df_final.csv"
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.txt"
    wrote = [df_path, summary_path]
    wrote.append(meta_path)
    wrote.append(report_path)

    df_final.to_csv(df_path, index=False)
    # save meta.json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    # put meta into summary.json too (nice for one-file machine read)
    summary_with_meta = dict(summary)
    summary_with_meta["meta"] = meta
    with open(meta_path,"w",encoding="utf-8") as f:
        json.dump(summary_with_meta, f, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # ---- NEW: portfolio outputs ----
    if "by_group" in summary:
        by_group_df = pd.DataFrame(summary["by_group"])
        by_group_csv = out_dir / "by_group.csv"
        by_group_json = out_dir / "by_group.json"

        # Sort like a risk report: highest EaR%, then steepest slope, then latest_flag
        sort_cols = ["exposure_at_risk_pct", "slope_bps_per_q", "latest_flag"]
        for c in sort_cols:
            if c not in by_group_df.columns:
                raise KeyError(
                    (
                        "by_group is missing expected column: "
                        f"{c}. Available: {by_group_df.columns.tolist()}"
                    )
                )
        by_group_df = by_group_df.sort_values(
            by=sort_cols,
            ascending=[False, False, False],
            kind="mergesort",
        )
        by_group_df.to_csv(by_group_csv, index=False)
        with open(by_group_json, "w", encoding="utf-8") as f:
            json.dump(summary["by_group"], f, indent=2)
        wrote.extend([by_group_csv, by_group_json])

    if group_col is None:
        group_col = summary.get("portfolio", {}).get("group_col", "Group")
    report_text = build_report_text(
        summary,
        group_col=group_col,
        top_n=5,
        run_info=run_info,
    )
    report_path.write_text(report_text, encoding="utf-8")
    wrote.append(report_path)

    print("Wrote:")
    for path in wrote:
        print(f" - {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run RiskOS PD trend + Exposure-at-Risk pipeline on demo data or an input CSV."
    )
    p.add_argument("--run-id", default=None, help="Optional run id tag (default: auto).")
    p.add_argument("--tag", default=None, help="Optional free-form tag appended to output folder.")
    p.add_argument(
        "--no-run-subdir",
        action="store_true",
        help="Write outputs directly into --out (no versioned subfolder).",
    )
    subparsers = p.add_subparsers(dest="command")
    run_p = subparsers.add_parser(
        "run",
        help="Run the pipeline and write outputs to disk.",
       
    )
    run_p.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input CSV. If omitted, runs on synthetic demo data.",
    )
    run_p.add_argument(
        "--out",
        type=str,
        default="outputs",
        help="Output directory (default: outputs).",
    )

    run_p.add_argument(
        "--slope-th",
        type=float,
        default=0.0015,
        help="Slope threshold for deterioration flag (PD per quarter). Default 0.0015 (~15 bps/q).",
    )
    run_p.add_argument(
        "--seed-prev-pd-1",
        type=float,
        default=0.019,
        help="Seed PD for T-1 before first row (default: 0.019). Use 'nan' to disable seeding.",
    )
    run_p.add_argument(
        "--seed-prev-pd-2",
        type=float,
        default=0.018,
        help="Seed PD for T-2 before first row (default: 0.018). Use 'nan' to disable seeding.",
    )

    # Demo-only
    run_p.add_argument(
        "--demo-seed",
        type=int,
        default=42,
        help="Random seed for demo dataframe generation (default: 42).",
    )

    run_p.add_argument(
        "--portfolio",
        action="store_true",
        help="Run per-group portfolio mode",
    )
    run_p.add_argument(
        "--group-col",
        default="Sector",
        help="Column to group by in portfolio mode",
    )
    doctor_p=subparsers.add_parser(
        "doctor",
        help="Inspect an input CSV and report data-quality issues."
        )
    doctor_p.add_argument(
        "--input",
          required=True,
            help="Path to input CSV"
            )
    doctor_p.add_argument(
        "--group-col",
          default="Sector"
          )
    doctor_p.add_argument(
        "--quarter-col",
          default="as_of_quarter"
          )
    doctor_p.add_argument(
        "--pd-col",
          default="PD_T"
          )
    doctor_p.add_argument(
        "--exposure-col",
          default="Exposure"
          )
    doctor_p.add_argument(
        "--allow-pd-gt1", 
        action="store_true", 
        help="Allow PD > 1 (if PD is in percent)"
        )
    sample_p = subparsers.add_parser("make-sample",
                                     help="Generate sample CSV (Single or multi-sector)."
                                     )
    sample_p.add_argument(
        "--mode",
          choices=["single", "portfolio"],
            default="portfolio"
            )
    sample_p.add_argument(
        "--out",
          required=True,
            help="Output CSV path (e.g. data/sample.csv)")
    sample_p.add_argument(
        "--seed",
          type=int, 
          default=42
          )
    sample_p.add_argument(
        "--n-quarters",
          type=int,
            default=8
            )
    sample_p.add_argument(
        "--start-quarter",
          type=str,
            default="2023-Q1"
            )
    sample_p.add_argument(
        "--no-duplicate",
          action="store_true",
            help="Do not inject a duplicate row")
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


def _extract_missing_columns(message: str) -> Optional[list[str]]:
    prefix = "Missing required columns:"
    if not message.startswith(prefix):
        return None
    raw = message.split(":", 1)[1].strip()
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None
    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
        return parsed
    return None


def main(argv: Optional[list[str]] = None) -> None:
    
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0].startswith("-"):
        argv = ["run", *argv]
    args = parser.parse_args(argv)
    if args.command == "doctor":
        
        contract = DataContract(
            quarter_col=args.quarter_col,
            group_col=args.group_col,
            pd_col=args.pd_col,
            exposure_col=args.exposure_col,

        )
        rep=doctor_csv(Path(args.input), contract=contract, allow_pd_gt1=args.allow_pd_gt1)
        print_doctor_report(rep)
       # optional exit code behavior (nice for CI later) 
       # OK -> 0, WARN -> 0, FAIL -> 2
        if rep.get("status") == "FAIL":
         raise SystemExit(2)
        raise SystemExit(0)
    if args.command == "make-sample":
        df=make_sample_df(
            mode=args.mode,
            seed=args.seed,
            n_quarters=args.n_quarters,
            start_quarter=args.start_quarter,
            include_duplicate=not args.no_duplicate,
        )
        out = write_sample_csv(df, Path(args.out))
        print(f"Wrote sample CSV: {out}")
    
        raise SystemExit(0)
    run_id = uuid4().hex
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    input_path = Path(args.input).expanduser().resolve() if args.input else None
    out_dir = Path(args.out).expanduser().resolve()
    contract = DataContract(group_col=args.group_col)
    required_columns = [
        contract.quarter_col,
        contract.group_col,
        contract.pd_col,
        contract.exposure_col,
    ]

    seed_prev_pd_1 = _nan_to_none(args.seed_prev_pd_1)
    seed_prev_pd_2 = _nan_to_none(args.seed_prev_pd_2)

    if input_path is None:
        df = make_demo_df(seed=args.demo_seed)
        if not args.portfolio:
            df = df[df[args.group_col].eq(df[args.group_col].iloc[0])].copy()
        print(f"Running DEMO mode (demo-seed={args.demo_seed})")
        input_source = f"DEMO (seed={args.demo_seed})"
    else:
        input_source = str(input_path)
        try:
            df = read_input_csv(input_path)
        except Exception as exc:
            message = f"Input failed: {exc}"
            print(message)
            write_meta(
                out_dir,
                status="FAILED",
                error=message,
                error_type="input",
                run_id=run_id,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                required_columns=required_columns,
                input_source=input_source,
            )
            raise SystemExit(1) from exc
        print(f"Running CSV mode (input={input_path})")

    print("RAW DF COLUMNS:", df.columns.tolist())
    meta = build_run_metadata(args=args, input_source=input_source)

    base_out = Path(args.out).expanduser().resolve()

    if args.no_run_subdir:
        out_dir = base_out
    else:
        ts = _utc_now_compact()
        sha = meta["git_sha"]
        tag = f"_{args.tag}" if args.tag else ""
        demo = ""
        if input_path is None:
            demo = f"_demo{args.demo_seed}"
        mode = "_portfolio" if args.portfolio else "_single"
        folder = f"run_{ts}_{sha}{demo}{mode}{tag}"
        out_dir = base_out / folder
    try:
        df = normalize_and_aggregate(df, contract=contract)
        if not args.portfolio:
            validate_single_series(df, contract=contract)
    except Exception as exc:  # Contract/data validation errors
        message = f"Contract validation failed: {exc}"
        print(message)
        missing_columns = _extract_missing_columns(str(exc))
        write_meta(
            out_dir,
            status="FAILED",
            error=message,
            error_type="validation",
            run_id=run_id,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            missing_columns=missing_columns,
            required_columns=required_columns,
            input_source=input_source,
        )
        raise SystemExit(2) from exc

    try:
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
    except Exception as exc:
        message = f"Pipeline failed: {exc}"
        print(message)
        write_meta(
            out_dir,
            status="FAILED",
            error=message,
            error_type="pipeline",
            run_id=run_id,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            required_columns=required_columns,
            input_source=input_source,
        )
        raise SystemExit(1) from exc

    run_info = {
        "runtime_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "portfolio" if args.portfolio else "single",
        "input_source": input_source,
        "slope_th": args.slope_th,
        "seed_prev_pd_1": seed_prev_pd_1,
        "seed_prev_pd_2": seed_prev_pd_2,
    }
    try:
        write_outputs(
            df_final,
            summary,
            out_dir,
            group_col=args.group_col,
            run_info=run_info,
        )
        write_meta(
            out_dir,
            status="SUCCESS",
            run_id=run_id,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            required_columns=required_columns,
            input_source=input_source,
        )
    except Exception as exc:
        message = f"Output failed: {exc}"
        print(message)
        try:
            write_meta(
                out_dir,
                status="FAILED",
                error=message,
                error_type="output",
                run_id=run_id,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                required_columns=required_columns,
                input_source=input_source,
            )
        except Exception:
            pass
        raise SystemExit(1) from exc

    if args.portfolio:
        print("\nPortfolio headline:")
        p = summary["portfolio"]
        print(
            f"Groups={p['n_groups']} | TotalExp={p['total_exposure']:,.0f} | "
            f"EaR={p['exposure_at_risk_pct']:.1f}% ({p['exposure_at_risk']:,.0f} / {p['total_exposure']:,.0f})"
        )
        print_top_risk_groups(summary, group_col=args.group_col, top_n=3)
    else:
        print_headline(summary)


if __name__ == "__main__":
    main()
