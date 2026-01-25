import pandas as pd
import pytest

from riskos.metrics.exposure_at_risk import compute_exposure_at_risk


def test_ear_none_flagged():
    df = pd.DataFrame({
        "Exposure": [100, 200, 300],
        "deterioration_flag": [0, 0, 0],
    })
    df2, out = compute_exposure_at_risk(df, exposure_col="Exposure", flag_col="deterioration_flag")
    assert out["total_exposure"] == pytest.approx(600.0)
    assert out["exposure_at_risk"] == pytest.approx(0.0)
    assert out["exposure_at_risk_pct"] == pytest.approx(0.0)
    assert "flagged_exposure" in df2.columns
    assert df2["flagged_exposure"].sum() == pytest.approx(0.0)


def test_ear_all_flagged():
    df = pd.DataFrame({
        "Exposure": [100, 200],
        "deterioration_flag": [1, 1],
    })
    df2, out = compute_exposure_at_risk(df, exposure_col="Exposure", flag_col="deterioration_flag")
    assert out["total_exposure"] == pytest.approx(300.0)
    assert out["exposure_at_risk"] == pytest.approx(300.0)
    assert out["exposure_at_risk_pct"] == pytest.approx(100.0)
    assert df2["flagged_exposure"].sum() == pytest.approx(300.0)


def test_ear_some_flagged():
    df = pd.DataFrame({
        "Exposure": [100, 200, 300],
        "deterioration_flag": [1, 0, 1],
    })
    df2, out = compute_exposure_at_risk(df, exposure_col="Exposure", flag_col="deterioration_flag")
    assert out["total_exposure"] == pytest.approx(600.0)
    assert out["exposure_at_risk"] == pytest.approx(400.0)
    assert out["exposure_at_risk_pct"] == pytest.approx(400.0 / 600.0 * 100.0)
    assert df2["flagged_exposure"].tolist() == [100.0, 0.0, 300.0]


def test_ear_zero_total_exposure():
    df = pd.DataFrame({
        "Exposure": [0, 0],
        "deterioration_flag": [1, 0],
    })
    df2, out = compute_exposure_at_risk(df, exposure_col="Exposure", flag_col="deterioration_flag")
    assert out["total_exposure"] == pytest.approx(0.0)
    assert out["exposure_at_risk"] == pytest.approx(0.0)
    assert out["exposure_at_risk_pct"] == pytest.approx(0.0)
    assert df2["flagged_exposure"].sum() == pytest.approx(0.0)