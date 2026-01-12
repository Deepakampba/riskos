from __future__ import annotations

from typing import Tuple, Dict
import numpy as np
import pandas as pd


def pd_trend_tier(slope: float) -> str:
    """Convert slope (PD per quarter) into a human-readable tier."""
    slope_bps = slope * 10000  # 0.001 => 10 bps per quarter
    if slope_bps < 5:
        return "Stable"
    elif slope_bps < 15:
        return "Mild deterioration"
    elif slope_bps < 30:
        return "Moderate deterioration"
    else:
        return "Severe deterioration"


def compute_pd_trend_signals(
    df: pd.DataFrame,
    *,
    pd_col: str = "PD_T",
    quarter_col: str = "as_of_quarter",
    sector_col: str = "Sector",
    exposure_col: str = "Exposure",
    slope_th: float = 0.0015,
    seed_prev_pd_1: float | None = None,  # PD at T-1 before first row
    seed_prev_pd_2: float | None = None,  # PD at T-2 before first row
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute PD trend slope, delta, acceleration and a deterioration flag.
    Flag rule: (slope > slope_th) & (dPD_1Q > 0) & (accel >= 0)

    Returns:
      df_out: enriched dataframe
      summary: dict with slope/intercept/tier/flags/latest status
    """
    required = {quarter_col, pd_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_out = df.copy()

    def _quarter_key(q: str) -> tuple[int, int]:
        y, qn = q.split("-Q")
        return int(y), int(qn)

    df_out = (
        df_out.sort_values(by=quarter_col, key=lambda s: s.map(_quarter_key))
             .reset_index(drop=True)
    )

    # Lag features
    df_out["PD_T_1"] = df_out[pd_col].shift(1)
    df_out["PD_T_2"] = df_out[pd_col].shift(2)

    # Seed first row with pre-history (if provided)
    if seed_prev_pd_1 is not None:
        df_out.loc[0, "PD_T_1"] = float(seed_prev_pd_1)
    if seed_prev_pd_2 is not None:
        df_out.loc[0, "PD_T_2"] = float(seed_prev_pd_2)

    # Also seed second row's PD_T_2, so its "T-2" points to the pre-history PD_T_1
    if seed_prev_pd_1 is not None and len(df_out) > 1:
        df_out.loc[1, "PD_T_2"] = float(seed_prev_pd_1)

    # Time index
    df_out["t"] = np.arange(len(df_out), dtype=float)

    # Linear trend fit
    slope, intercept = np.polyfit(df_out["t"], df_out[pd_col].astype(float), 1)

    # Deltas and acceleration
    df_out["dPD_1Q"] = df_out[pd_col] - df_out["PD_T_1"]
    df_out["dPD_prev"] = df_out["PD_T_1"] - df_out["PD_T_2"]
    df_out["accel"] = df_out["dPD_1Q"] - df_out["dPD_prev"]

    # Flag logic
    df_out["reason_slope"] = slope > slope_th
    df_out["reason_direction"] = df_out["dPD_1Q"] > 0
    df_out["reason_accel"] = df_out["accel"] >= 0

    df_out["deterioration_flag"] = (
        df_out["reason_slope"]
        & df_out["reason_direction"]
        & df_out["reason_accel"]
    ).astype(int)

    # Human reason
    slope_bps = slope * 10000
    base_reason = f"Trend↑ ({slope_bps:.1f} bps/q) + ΔPD>0 + accel≥0"
    df_out["flag_reason"] = np.where(df_out["deterioration_flag"].eq(1), base_reason, "No flag")

    # Summary
    tier = pd_trend_tier(slope)
    flags = int(df_out["deterioration_flag"].sum())
    n_q = int(len(df_out))

    summary = {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_bps_per_q": float(slope_bps),
        "tier": tier,
        "flags": flags,
        "n_quarters": n_q,
        "latest_quarter": df_out[quarter_col].iloc[-1] if n_q else None,
        "latest_flag": int(df_out["deterioration_flag"].iloc[-1]) if n_q else 0,
        "slope_threshold": float(slope_th),
    }

    # Column order
    preferred = [
        quarter_col,
        sector_col if sector_col in df_out.columns else None,
        pd_col,
        "PD_T_1",
        "PD_T_2",
        exposure_col if exposure_col in df_out.columns else None,
        "t",
        "dPD_1Q",
        "dPD_prev",
        "accel",
        "deterioration_flag",
        "flag_reason",
    ]
    preferred = [c for c in preferred if c is not None and c in df_out.columns]
    remaining = [c for c in df_out.columns if c not in preferred]
    df_out = df_out[preferred + remaining]

    return df_out, summary