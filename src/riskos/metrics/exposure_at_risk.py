from __future__ import annotations
from typing import Dict, Tuple
import pandas as pd
import numpy as np


def validate_inputs(
    df_out: pd.DataFrame,
    *,
    exposure_col: str = "Exposure",
    flag_col: str = "deterioration_flag",
    el_col: str = "EL",
    require_el: bool = False,
) -> None:
    required = {exposure_col, flag_col}
    if require_el:
        required.add(el_col)

    missing = required - set(df_out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not pd.api.types.is_numeric_dtype(df_out[exposure_col]):
        raise TypeError(f"{exposure_col} must be numeric")
    if (df_out[exposure_col] < 0).any():
        raise ValueError(f"{exposure_col} contains negative values")
    if df_out[exposure_col].isna().any():
        raise ValueError(f"{exposure_col} contains NaN values")

    valid_flags = {0, 1}
    unique_flags = set(df_out[flag_col].dropna().unique().tolist())
    if not unique_flags.issubset(valid_flags):
        raise ValueError(f"{flag_col} must contain only 0/1. Found: {unique_flags}")
    if df_out[flag_col].isna().any():
        raise ValueError(f"{flag_col} contains NaN values")

    if require_el:
        if not pd.api.types.is_numeric_dtype(df_out[el_col]):
            raise TypeError(f"{el_col} must be numeric")
        if (df_out[el_col] < 0).any():
            raise ValueError(f"{el_col} contains negative values")
        if df_out[el_col].isna().any():
            raise ValueError(f"{el_col} contains NaN values")


def compute_exposure_at_risk(
    df_out: pd.DataFrame,
    *,
    exposure_col: str = "Exposure",
    flag_col: str = "deterioration_flag",
    add_row_level: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Computes:
      total_exposure
      exposure_at_risk (sum exposure where flag==1)
      exposure_at_risk_pct
    Optionally adds row-level column: flagged_exposure
    """
    validate_inputs(df_out, exposure_col=exposure_col, flag_col=flag_col,require_el=False)

    total_exposure = float(df_out[exposure_col].sum())
    exposure_at_risk = float(df_out.loc[df_out[flag_col] == 1, exposure_col].sum())
    exposure_at_risk_pct = (exposure_at_risk / total_exposure * 100.0) if total_exposure else 0.0

    if add_row_level:
        df_out = df_out.copy()
        df_out["flagged_exposure"] = np.where(df_out[flag_col] == 1, df_out[exposure_col], 0.0)

    summary = {
        "total_exposure": total_exposure,
        "exposure_at_risk": exposure_at_risk,
        "exposure_at_risk_pct": exposure_at_risk_pct,
        "ear_exposure_col": exposure_col,
        "ear_flag_col": flag_col,
    }
    return df_out, summary