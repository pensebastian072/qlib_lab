"""Wave-3: calendar sanity, FRED lookahead, funding shift, netliq arithmetic."""
import numpy as np
import pandas as pd

from qlib_lab import funding, macro_calendar


def test_fomc_counts_sane():
    idx = macro_calendar.fomc_dates()
    counts = pd.Series(idx.year).value_counts()
    for year in range(2011, 2027):
        assert 7 <= counts.get(year, 0) <= 9, f"{year}: {counts.get(year)}"


def test_fomc_2024_matches_published():
    idx = macro_calendar.fomc_dates()
    got = [str(d.date()) for d in idx if d.year == 2024]
    assert got == ["2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
                   "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18"]


def test_nfp_first_fridays():
    nfp = macro_calendar.nfp_dates("2026-01-01", "2026-06-30")
    assert all(d.dayofweek == 4 for d in nfp)
    assert all(d.day <= 7 for d in nfp)
    assert len(nfp) == 6


def test_days_to_next_fomc_no_negative_and_capped():
    idx = pd.bdate_range("2026-01-02", "2026-03-31")
    d = macro_calendar.days_to_next_fomc(idx)
    valid = d.dropna()
    assert (valid >= 0).all() and (valid <= 45).all()
    # the decision day itself counts 0
    assert d.loc[pd.Timestamp("2026-01-28")] == 0


def test_funding_features_shift_one_day():
    dates = pd.date_range("2025-01-01", periods=200, freq="D")
    hist = pd.DataFrame({
        "date": np.repeat(dates, 3),
        "ts_ms": range(600),
        "rate": 0.0001,
    })
    feats = funding.daily_features(hist)
    # day t's own sum must NOT be visible at t (shifted to t+1)
    assert pd.isna(feats.loc[dates[0], "funding_1d"])
    assert feats.loc[dates[1], "funding_1d"] == 3 * 0.0001


def test_netliq_arithmetic(monkeypatch, tmp_path):
    from qlib_lab import fred_liquidity as fl
    days = pd.bdate_range("2024-01-01", periods=10)
    walcl = pd.Series(7_000_000.0, index=days)   # millions
    tga = pd.Series(800_000.0, index=days)       # millions
    rrp = pd.Series(100.0, index=days)           # billions

    def fake_fetch(series_id, timeout=30):
        return {"WALCL": walcl, "WDTGAL": tga, "RRPONTSYD": rrp}[series_id]

    monkeypatch.setattr(fl, "fetch_series", fake_fetch)
    # TGA is sourced from the Treasury DTS since 2026-08-31; stub it out so this
    # stays an offline arithmetic test of the WDTGAL fallback path. Without
    # this the call reaches the live DTS and the "800bn" below is not the TGA
    # actually used. Daily-DTS arithmetic is covered in test_fred_liquidity.py.
    monkeypatch.setattr(fl, "fetch_dts_tga", lambda *a, **k: None)
    df = fl.build_netliq()
    # 7000bn - 800bn - 100bn = 6100bn (after unit normalization + lag shifts)
    assert abs(df["netliq"].dropna().iloc[-1] - 6100.0) < 1e-6
    assert (df["tga_source"] == "wdtgal_weekly_fallback").all()
