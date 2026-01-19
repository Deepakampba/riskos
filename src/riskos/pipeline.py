from __future__ import annotations

from typing import Dict, Tuple, List
import pandas as pd

from .signals.pd_trend import compute_pd_trend_signals
from .metrics.exposure_at_risk import compute_exposure_at_risk


def run_pd_trend_and_ear(
    df: pd.DataFrame,
    *,
    pd_col: str = "PD_T",
    quarter_col: str = "as_of_quarter",
    sector_col: str = "Sector",
    exposure_col: str = "Exposure",
    slope_th: float = 0.0015,
    seed_prev_pd_1: float | None = None,
    seed_prev_pd_2: float | None = None,
    flag_col: str = "deterioration_flag",
) -> Tuple[pd.DataFrame, Dict]:
    df_sig, trend_summary = compute_pd_trend_signals(
        df,
        pd_col=pd_col,
        quarter_col=quarter_col,
        sector_col=sector_col,
        exposure_col=exposure_col,
        slope_th=slope_th,
        seed_prev_pd_1=seed_prev_pd_1,
        seed_prev_pd_2=seed_prev_pd_2,
    )

    df_final, ear_summary = compute_exposure_at_risk(
        df_sig,
        exposure_col=exposure_col,
        flag_col=flag_col,
        add_row_level=True,
    )

    summary = {"pd_trend": trend_summary, "ear": ear_summary}
    return df_final, summary


def run_portfolio_pd_trend_and_ear(
    df: pd.DataFrame,
    *,
    group_col: str = "Sector",
    pd_col: str = "PD_T",
    quarter_col: str = "as_of_quarter",
    exposure_col: str = "Exposure",
    slope_th: float = 0.0015,
    seed_prev_pd_1: float | None = None,
    seed_prev_pd_2: float | None = None,
    flag_col: str = "deterioration_flag",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run PD trend + EaR for each segment (e.g., Sector), then aggregate to portfolio.

    Returns:
      df_portfolio: concatenated enriched rows for all groups
      summary: {
        "portfolio": {...},
        "by_group": [ {group, tier, slope_bps_per_q, ear_pct, ...}, ... ]
      }
    """
    if group_col not in df.columns:
        raise ValueError(f"Missing group_col '{group_col}' in input df")

    by_group: List[Dict] = []
    frames: List[pd.DataFrame] = []

    # group_keys=False preserves original index values; we reset later anyway
    for g, df_g in df.groupby(group_col, sort=True):
        df_g = df_g.copy().reset_index(drop=True)

        df_final_g, summary_g = run_pd_trend_and_ear(
            df_g,
            pd_col=pd_col,
            quarter_col=quarter_col,
            sector_col=group_col,      # reuse same column name (Sector) conceptually
            exposure_col=exposure_col,
            slope_th=slope_th,
            seed_prev_pd_1=seed_prev_pd_1,
            seed_prev_pd_2=seed_prev_pd_2,
            flag_col=flag_col,
        )

        frames.append(df_final_g)

        pd_s = summary_g["pd_trend"]
        ear_s = summary_g["ear"]

        by_group.append(
            {
                group_col: g,
                "tier": pd_s.get("tier"),
                "slope_bps_per_q": pd_s.get("slope_bps_per_q"),
                "flags": pd_s.get("flags"),
                "n_quarters": pd_s.get("n_quarters"),
                "latest_quarter": pd_s.get("latest_quarter"),
                "latest_flag": pd_s.get("latest_flag"),
                "total_exposure": ear_s.get("total_exposure"),
                "exposure_at_risk": ear_s.get("exposure_at_risk"),
                "exposure_at_risk_pct": ear_s.get("exposure_at_risk_pct"),
            }
        )

    df_portfolio = pd.concat(frames, ignore_index=True) if frames else df.copy()

    # Portfolio rollup from concatenated rows
    total_exposure = float(df_portfolio[exposure_col].sum()) if exposure_col in df_portfolio.columns else 0.0
    total_ear = float(df_portfolio.get("flagged_exposure", pd.Series([0.0])).sum()) if len(df_portfolio) else 0.0
    total_ear_pct = (total_ear / total_exposure * 100.0) if total_exposure else 0.0

    by_group_df = pd.DataFrame(by_group)
    # Rank “worst” by Exposure-at-Risk (absolute) by default
    top_risk = (
        by_group_df.sort_values("exposure_at_risk", ascending=False)
        .head(5)
        .to_dict(orient="records")
        if not by_group_df.empty
        else []
    )

    summary = {
        "portfolio": {
            "group_col": group_col,
            "n_groups": int(by_group_df.shape[0]),
            "total_exposure": total_exposure,
            "exposure_at_risk": total_ear,
            "exposure_at_risk_pct": total_ear_pct,
            "top_risk_groups": top_risk,
        },
        "by_group": by_group,
    }

    return df_portfolio, summary