import pandas as pd
import pytest

from riskos.validation import DataContract, normalize_and_aggregate


def test_normalize_and_aggregate_sums_exposure_and_wavg_pd():
    # Duplicate key: (Sector="A", as_of_quarter="2023-Q1")
    # Exposure sums: 100 + 300 = 400
    # PD weighted avg: (0.02*100 + 0.04*300)/400 = 0.035
    df = pd.DataFrame(
        {
            "as_of_quarter": ["2023-Q1", "2023-Q1", "2023-Q2"],
            "Sector": ["A", "A", "A"],
            "PD_T": [0.02, 0.04, 0.05],
            "Exposure": [100.0, 300.0, 200.0],
        }
    )
    out = normalize_and_aggregate(df, contract=DataContract())

    # After aggregation, there should be only 2 rows (Q1 aggregated + Q2)
    assert len(out) == 2

    q1 = out[out["as_of_quarter"] == "2023-Q1"].iloc[0]
    assert q1["Exposure"] == pytest.approx(400.0)
    assert q1["PD_T"] == pytest.approx(0.035)


def test_normalize_and_aggregate_rejects_bad_quarter_format():
    df = pd.DataFrame(
        {
            "as_of_quarter": ["2024Q2"],  # bad
            "Sector": ["A"],
            "PD_T": [0.02],
            "Exposure": [100.0],
        }
    )
    with pytest.raises(ValueError) as e:
        normalize_and_aggregate(df, contract=DataContract())
    assert "Invalid quarter format" in str(e.value)


def test_normalize_and_aggregate_missing_required_columns():
    df = pd.DataFrame(
        {
            "as_of_quarter": ["2023-Q1"],
            "Sector": ["A"],
            "PD_T": [0.02],
            # Exposure missing
        }
    )
    with pytest.raises(ValueError) as e:
        normalize_and_aggregate(df, contract=DataContract())
    assert "Missing required columns" in str(e.value)
    assert "Exposure" in str(e.value)