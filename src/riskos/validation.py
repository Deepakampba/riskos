from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


_QUARTER_RE = re.compile(r"^\d{4}-Q[1-4]$")


@dataclass(frozen=True)
class DataContract:
    quarter_col: str = "as_of_quarter"
    group_col: str = "Sector"
    pd_col: str = "PD_T"
    exposure_col: str = "Exposure"


def validate_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def validate_no_missing(df: pd.DataFrame, cols: Iterable[str]) -> None:
    cols = list(cols)
    if df[cols].isna().any().any():
        bad = df[cols].isna().sum()
        bad = bad[bad > 0].to_dict()
        raise ValueError(f"Missing (NaN) values found in: {bad}")


def validate_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise TypeError(f"Column '{c}' must be numeric. Got dtype={df[c].dtype}")


def validate_numeric_nonneg(df: pd.DataFrame, cols: Iterable[str]) -> None:
    validate_numeric(df, cols)
    for c in cols:
        if (df[c] < 0).any():
            n_bad = int((df[c] < 0).sum())
            raise ValueError(f"Column '{c}' contains {n_bad} negative values.")


def validate_pd_bounds(df: pd.DataFrame, pd_col: str, *, allow_gt1: bool = False) -> None:
    """
    PD is usually in [0,1]. If your PD is in percent (0-100), set allow_gt1=True
    and we won't enforce the upper bound.
    """
    if (df[pd_col] < 0).any():
        raise ValueError(f"Column '{pd_col}' contains negative PD values.")
    if not allow_gt1 and (df[pd_col] > 1).any():
        max_pd = float(df[pd_col].max())
        raise ValueError(
            f"Column '{pd_col}' has values > 1 (max={max_pd}). "
            f"If your PD is in percent, convert to decimals or set allow_gt1=True."
        )


def validate_quarter_format(series: pd.Series, *, col_name: str = "as_of_quarter") -> None:
    s = series.astype(str)
    bad_mask = ~s.map(lambda x: bool(_QUARTER_RE.match(x)))
    if bad_mask.any():
        bad_vals = sorted(s[bad_mask].unique().tolist())[:10]
        raise ValueError(
            f"Invalid quarter format in '{col_name}'. Expected 'YYYY-Qn' (n=1..4). "
            f"Examples of bad values: {bad_vals}"
        )


def quarter_key(q: str) -> tuple[int, int]:
    """Sort key for 'YYYY-Qn'."""
    y, qn = q.split("-Q")
    return int(y), int(qn)


def normalize_and_aggregate(
    df: pd.DataFrame,
    *,
    contract: DataContract = DataContract(),
    pd_weighted_by_exposure: bool = True,
) -> pd.DataFrame:
    
    """
    Bank-style normalization:
    - Validates schema and types
    - Aggregates duplicates per (group, quarter)
      Exposure = sum
      PD_T = exposure-weighted average (default) OR simple mean
    - Sorts within each group by quarter
    """
    quarter_col = contract.quarter_col
    group_col = contract.group_col
    pd_col = contract.pd_col
    exposure_col = contract.exposure_col

    validate_required_columns(df, [quarter_col, group_col, pd_col, exposure_col])
    
    # Basic null checks
    validate_no_missing(df, [quarter_col, group_col, pd_col, exposure_col])

    # Quarter format
    validate_quarter_format(df[quarter_col], col_name=quarter_col)

    # Numeric checks
    validate_numeric_nonneg(df, [exposure_col])
    validate_numeric(df, [pd_col])
    validate_pd_bounds(df, pd_col, allow_gt1=False)

    df2 = df.copy()

    # Ensure clean types
    df2[quarter_col] = df2[quarter_col].astype(str)
    df2[group_col] = df2[group_col].astype(str)
    df2[pd_col] = df2[pd_col].astype(float)
    df2[exposure_col] = df2[exposure_col].astype(float)

    # Aggregate duplicates per group+quarter
    keys = [group_col, quarter_col]
    if df2.duplicated(keys).any():
        if pd_weighted_by_exposure:
            def _agg_group(g: pd.DataFrame) -> pd.Series:
                exp = float(g[exposure_col].sum())
                if exp>0:
                    pd_wavg = float(np.average(g[pd_col].to_numpy(float),weights=g[exposure_col].to_numpy(float)))
                else:
                    pd_wavg=float(g[pd_col].mean())
                return pd.Series({exposure_col:exp, pd_col:pd_wavg})
            df2 = (
                df2.groupby(keys)
                   .apply(_agg_group, include_groups=False)
                   .reset_index()
            )
                       
        else:
            df2 = (
                df2.groupby(keys, as_index=False)
                   .agg(
                       **{
                           exposure_col: (exposure_col, "sum"),
                           pd_col: (pd_col, "mean"),
                       }
                   )
            )

    # Sort properly by quarter within group
    df2 = df2.sort_values(
        by=[group_col, quarter_col],
        key=lambda s: s.map(quarter_key) if s.name == quarter_col else s
    ).reset_index(drop=True)
    
    return df2
    

def validate_single_series(
    df: pd.DataFrame,
    *,
    contract: DataContract = DataContract(),
) -> None:
    """
    Extra guard for single-series mode: the dataframe should contain exactly one group.
    """
    group_col = contract.group_col
    unique_groups = df[group_col].nunique(dropna=False)
    if unique_groups != 1:
        raise ValueError(
            f"Single-series mode expects exactly 1 group in '{group_col}', "
            f"but found {unique_groups}. Use --portfolio mode instead."
        )