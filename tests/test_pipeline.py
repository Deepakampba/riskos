import pandas as pd

from riskos.pipeline import run_pd_trend_and_ear, run_portfolio_pd_trend_and_ear


def test_run_pd_trend_and_ear_returns_expected_keys():
    df = pd.DataFrame(
        {
            "as_of_quarter": ["2023-Q1", "2023-Q2", "2023-Q3"],
            "Sector": ["A", "A", "A"],
            "PD_T": [0.02, 0.025, 0.03],
            "Exposure": [100.0, 200.0, 300.0],
        }
    )

    df_final, summary = run_pd_trend_and_ear(df, slope_th=0.0015)

    assert "pd_trend" in summary
    assert "ear" in summary

    # dataframe has deterioration_flag and ear columns (depending on your add_row_level)
    assert "deterioration_flag" in df_final.columns


def test_run_portfolio_pd_trend_and_ear_has_portfolio_and_by_group():
    df = pd.DataFrame(
        {
            "as_of_quarter": ["2023-Q1", "2023-Q2", "2023-Q1", "2023-Q2"],
            "Sector": ["A", "A", "B", "B"],
            "PD_T": [0.02, 0.03, 0.01, 0.015],
            "Exposure": [100.0, 200.0, 300.0, 400.0],
        }
    )

    df_final, summary = run_portfolio_pd_trend_and_ear(df, group_col="Sector", slope_th=0.0015)

    assert "portfolio" in summary
    assert "by_group" in summary
    assert summary["portfolio"]["n_groups"] == 2