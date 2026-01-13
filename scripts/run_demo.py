from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from riskos.pipeline import run_pd_trend_and_ear


def make_demo_df() -> pd.DataFrame:
    """Small synthetic dataset just to prove the pipeline works end-to-end."""
    np.random.seed(42)

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


def main() -> None:
    df = make_demo_df()

    df_final, summary = run_pd_trend_and_ear(
        df,
        slope_th=0.0015,
        seed_prev_pd_1=0.019,  # previous quarter before 2023-Q1
        seed_prev_pd_2=0.018,  # two quarters before 2023-Q1
    )

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save enriched dataframe
    df_final.to_csv(out_dir / "df_final.csv", index=False)

    # 2) Save summary json
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:")
    print(" - outputs/df_final.csv")
    print(" - outputs/summary.json")


if __name__ == "__main__":
    main()