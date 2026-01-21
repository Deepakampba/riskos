# src/riskos/sample_data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

Mode = Literal["single", "portfolio"]


def generate_quarters(start: str = "2023-Q1", n: int = 8) -> List[str]:
    y_s, q_s = start.split("-Q")
    y, q = int(y_s), int(q_s)
    out = []
    for _ in range(n):
        out.append(f"{y:04d}-Q{q}")
        q += 1
        if q == 5:
            q = 1
            y += 1
    return out


def make_sample_df(
    *,
    mode: Mode = "portfolio",
    seed: int = 42,
    n_quarters: int = 8,
    start_quarter: str = "2023-Q1",
    include_duplicate: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quarters = generate_quarters(start_quarter, n_quarters)

    sectors: Dict[str, Dict[str, float]] = {
        "Manufacture": {"base_pd": 0.020, "drift": 0.0020, "vol": 0.0010, "exp_mu": 1_000_000, "exp_sd": 150_000},
        "Retail":      {"base_pd": 0.015, "drift": 0.0010, "vol": 0.0008, "exp_mu":   800_000, "exp_sd": 120_000},
        "Tech":        {"base_pd": 0.010, "drift": 0.0004, "vol": 0.0006, "exp_mu": 1_200_000, "exp_sd": 180_000},
        "RealEstate":  {"base_pd": 0.025, "drift": 0.0025, "vol": 0.0012, "exp_mu": 1_500_000, "exp_sd": 220_000},
    }

    if mode == "single":
        sectors = {"Manufacture": sectors["Manufacture"]}

    frames = []
    for sector, cfg in sectors.items():
        pd_values = [float(cfg["base_pd"])]
        for _ in range(1, n_quarters):
            step = float(rng.normal(cfg["drift"], cfg["vol"]))
            pd_values.append(max(0.0, pd_values[-1] + step))

        exposure = rng.normal(cfg["exp_mu"], cfg["exp_sd"], n_quarters)
        exposure = np.maximum(0.0, exposure).round(2)

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

    if include_duplicate:
        dup = df.iloc[[0]].copy()
        dup["Exposure"] = (dup["Exposure"] * 0.25).round(2)
        df = pd.concat([df, dup], ignore_index=True)

    return df


def write_sample_csv(df: pd.DataFrame, out_path: Path) -> Path:
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path