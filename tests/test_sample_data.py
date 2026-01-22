from riskos.sample_data import make_sample_df


def test_make_sample_df_portfolio_mode_has_multiple_sectors():
    df = make_sample_df(mode="portfolio", seed=1, n_quarters=4)
    assert df["Sector"].nunique() >= 2
    assert len(df) >= 4  # at least one sector * quarters


def test_make_sample_df_single_mode_has_one_sector():
    df = make_sample_df(mode="single", seed=1, n_quarters=4)
    assert df["Sector"].nunique() == 1