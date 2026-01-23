from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from riskos.pipeline import run_portfolio_pd_trend_and_ear
from riskos.validation import DataContract, normalize_and_aggregate


def test_golden_portfolio_summary_stable() -> None:
    # Fixed input (checked into repo)
    csv_path = Path(__file__).parent / "golden" / "sample_multi_sector.csv"
    df = pd.read_csv(csv_path)

    # Normalize like the CLI does (bank-grade input handling)
    contract = DataContract(
        quarter_col="as_of_quarter",
        group_col="Sector",
        pd_col="PD_T",
        exposure_col="Exposure",
    )
    df = normalize_and_aggregate(df, contract=contract)

    # Run portfolio pipeline
    df_final, summary = run_portfolio_pd_trend_and_ear(df, group_col="Sector",slope_th=0.0015,seed_prev_pd_1=0.019,seed_prev_pd_2=0.018,)

    # --- Stable assertions ---
    assert "portfolio" in summary
    p = summary["portfolio"]

    assert p["n_groups"] == 4
    assert p["group_col"] == "Sector"

    # These should be stable for a fixed CSV; allow tiny float noise anyway
    assert p["total_exposure"] == pytest.approx(36449369.58, rel=0, abs=1e-6)
    assert p["exposure_at_risk_pct"] == pytest.approx((p["exposure_at_risk"] / p["total_exposure"]) * 100.0, rel=0, abs=1e-6)

    # by_group should have 4 rows and all expected sectors
    by_group = summary.get("by_group")
    assert by_group is not None
    assert len(by_group) == 4

    sectors = {row["Sector"] for row in by_group}
    assert sectors == {"Manufacture", "Retail", "Tech", "RealEstate"}

    # df_final sanity (pipeline should enrich)
    assert not df_final.empty
    assert "deterioration_flag" in df_final.columns