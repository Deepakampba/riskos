from __future__ import annotations

from typing import Dict, Tuple
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
    """
    Pipeline:
      1) PD trend signals + deterioration flag
      2) Exposure-at-Risk summary

    Returns:
      df_final, summary
    """
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