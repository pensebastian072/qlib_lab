"""The reconstructed history must agree with the live workbook, or it is a second
opinion rather than a record. Synthetic frames -- no qlib, no network.
"""
import numpy as np
import pandas as pd
import pytest

from qlib_lab import asset_frame as af, market_state as ms, state_panel as sp


def _panel(n=400, seed=0, assets=None):
    rng = np.random.default_rng(seed)
    assets = assets or ms.COT_ASSETS
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {a: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for a in assets}, index=idx)


def test_rolling_absorption_matches_the_published_construction():
    """The history and the workbook must come from ONE implementation. Two copies of
    this eigen block drifted by 5e-4 on 2026-08-24 -- small enough to be explained away
    as a regime change, which is exactly the danger."""
    px = _panel()
    out = sp.rolling_structure(px, window=60)
    last = out.dropna().index[-1]
    i = px.index.get_loc(last)
    w = px.pct_change().iloc[i - 59:i + 1].dropna(axis=1, how="all")
    want, want_top1, _ = ms.absorption_from_corr(w.corr())
    assert float(out.loc[last, "absorption"]) == pytest.approx(want, rel=1e-12)
    assert float(out.loc[last, "top1_share"]) == pytest.approx(want_top1, rel=1e-12)


def test_rolling_window_is_inclusive_of_its_own_session():
    """An exclusive window silently shifts every reading one session into the past."""
    px = _panel(n=200)
    out = sp.rolling_structure(px, window=60)
    i = 150
    date = px.index[i]
    w_incl = px.pct_change().iloc[i - 59:i + 1].dropna(axis=1, how="all")
    w_excl = px.pct_change().iloc[i - 60:i].dropna(axis=1, how="all")
    got = float(out.loc[date, "absorption"])
    assert got == pytest.approx(ms.absorption_from_corr(w_incl.corr())[0], rel=1e-12)
    assert got != pytest.approx(ms.absorption_from_corr(w_excl.corr())[0], rel=1e-9)


def test_breadth_history_excludes_the_same_names_the_live_reading_does():
    """Counting the VIX family inverts breadth exactly when it matters. The exclusion
    list is imported from asset_frame, so this test fails if the two ever diverge."""
    px = _panel(n=300, assets=["SPY", "QQQ", "IWM", "VIX", "EURUSD", "BIL"])
    out = sp.rolling_breadth(px)
    kept = [c for c in px.columns if c not in af.BREADTH_EXCLUDE]
    assert set(kept) == {"SPY", "QQQ", "IWM"}
    assert out["breadth_n"].max() == 3
    assert (out["breadth_above_200"].dropna() <= 1.0).all()


def test_breadth_only_counts_an_instrument_once_its_slow_ema_is_warm():
    """A young series otherwise votes against an EMA seeded by its own first price."""
    px = _panel(n=300, assets=["SPY", "QQQ"])
    px.loc[px.index[:250], "QQQ"] = np.nan          # QQQ has 50 sessions of history
    out = sp.rolling_breadth(px)
    assert out["breadth_n"].iloc[-1] == 1           # QQQ not warm yet, SPY is


def test_story_history_uses_the_same_signed_rule_as_the_sheet():
    px = _panel(n=400, assets=["BTC", "GOLD", "SPY", "DXY", "COPPER", "TLT",
                               "VIX", "HYG", "EEM", "OIL", "US10Y"])
    out = sp.rolling_stories(px)
    col = "story_corr_BTC_IS_DIGITAL_GOLD"
    state = "story_state_BTC_IS_DIGITAL_GOLD"
    thr = next(s["thr"] for s in ms.STORIES if s["id"] == "BTC_IS_DIGITAL_GOLD")
    last = out[col].dropna().index[-1]
    assert out.loc[last, state] == ms._story_state(float(out.loc[last, col]), thr)


def test_panel_never_carries_a_reading_backwards_in_time():
    """Every join is forward-fill from a date already public. A backward fill would put
    Friday's report into Wednesday's row and nothing downstream would notice."""
    idx = pd.bdate_range("2026-01-01", periods=10)
    weekly = pd.DataFrame({
        "asset": ["SPY"] * 2,
        "effective_date": [idx[2], idx[7]],
        "idx52": [0.9, 0.1], "z156": [1.0, -1.0], "chg4w": [0.01, -0.01],
    })
    from qlib_lab import cot
    daily = cot.daily_panel(weekly, idx, cols=("idx52",))["SPY"]
    assert pd.isna(daily["idx52"].iloc[0])          # nothing known before the first row
    assert daily["idx52"].iloc[2] == 0.9
    assert daily["idx52"].iloc[6] == 0.9            # still the old report
    assert daily["idx52"].iloc[7] == 0.1
