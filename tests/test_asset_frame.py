"""Asset framework: indicator maths, per-layer verdicts, and the honesty guarantees.

Synthetic series throughout -- no qlib, so a broken store cannot mask a logic bug.
"""
import numpy as np
import pandas as pd
import pytest

from qlib_lab import asset_frame as af


def _series(vals, start="2020-01-01"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)))


def _trending(n=400, slope=0.4, base=100.0):
    return _series(base + slope * np.arange(n))


# ------------------------------------------------------------------ indicators

def test_ema_tracks_a_constant_series():
    assert float(af.ema(_series([5.0] * 100), 21).iloc[-1]) == pytest.approx(5.0)


def test_rsi_saturates_on_a_monotonic_series():
    assert float(af.rsi(_trending()).iloc[-1]) > 95      # only up-days
    assert float(af.rsi(_trending(slope=-0.4)).iloc[-1]) < 5


def test_rsi_midrange_on_alternating_series():
    alt = _series([100 + (1 if i % 2 else -1) for i in range(200)])
    assert 30 < float(af.rsi(alt).iloc[-1]) < 70


def test_macd_positive_in_an_uptrend_negative_in_a_downtrend():
    assert float(af.macd(_trending())["macd"].iloc[-1]) > 0
    assert float(af.macd(_trending(slope=-0.4))["macd"].iloc[-1]) < 0


# --------------------------------------------------------------------- layers

def test_trend_bullish_above_stacked_emas():
    out = af._trend(_trending())
    assert out["verdict"] == af.BULL
    assert out["stacked"] is True
    assert out["dist_ema200_pct"] > 0


def test_trend_bearish_in_a_downtrend():
    assert af._trend(_trending(slope=-0.2, base=300.0))["verdict"] == af.BEAR


def test_trend_no_data_on_short_history():
    assert af._trend(_series([1.0] * 50))["verdict"] == af.NO_DATA


def test_momentum_degrades_to_neutral_when_overbought():
    """An expanding positive MACD at RSI>=80 is accelerating AND stretched; calling
    that plainly bullish hides the more useful half."""
    out = af._momentum(_trending())
    assert out["rsi"] >= 80
    assert out["verdict"] == af.NEUTRAL
    assert "overbought" in out["detail"]


def test_volume_no_data_for_index_products():
    """VIX-family carry volume 0.0 in this store. That is a MISSING measurement, and
    must never read as 'low participation' (bearish)."""
    px = _trending(n=200)
    out = af._volume(pd.Series(0.0, index=px.index), px)
    assert out["verdict"] == af.NO_DATA
    assert af._volume(None, px)["verdict"] == af.NO_DATA


def test_volume_bullish_on_expansion_with_up_day_confirmation():
    px = _trending(n=200)
    vol = pd.Series(1000.0, index=px.index)
    vol.iloc[-5:] = 5000.0
    assert af._volume(vol, px)["verdict"] == af.BULL


def test_breadth_reads_the_panel_not_one_asset():
    up = pd.DataFrame({f"A{i}": _trending() for i in range(10)})
    assert af._breadth(up)["verdict"] == af.BULL
    assert af._breadth(up)["pct_above_200"] == pytest.approx(1.0)
    down = pd.DataFrame({f"A{i}": _trending(slope=-0.2, base=300.0) for i in range(10)})
    assert af._breadth(down)["verdict"] == af.BEAR


def test_rel_strength_benchmark_is_not_compared_to_itself():
    spy = _trending()
    assert af._rel_strength(spy, spy, "SPY")["detail"] == "is the benchmark"


def test_rel_strength_detects_outperformance():
    spy = _trending(slope=0.1)
    fast = _trending(slope=0.5)
    assert af._rel_strength(fast, spy, "X")["verdict"] == af.BULL
    assert af._rel_strength(spy, fast, "X")["verdict"] == af.BEAR


# ------------------------------------------------------- positioning semantics

def test_crowded_long_reads_as_vulnerable_not_bullish():
    """COT crowding is a RISK reading. A 52-week-max long is fragile, so the layer
    must not report it as bullish confirmation."""
    out = af._positioning("GOLD", {"GOLD": {"idx52": 1.00, "z156": 1.2, "chg4w": 0.06}}, {})
    assert out["verdict"] == af.BEAR
    assert "vulnerable" in out["detail"]


def test_crowded_short_reads_as_the_other_side():
    out = af._positioning("QQQ", {"QQQ": {"idx52": 0.10, "z156": -1.7, "chg4w": -0.05}}, {})
    assert out["verdict"] == af.BULL


def test_positioning_no_data_without_a_contract():
    assert af._positioning("XLU", {}, {})["verdict"] == af.NO_DATA


def test_funding_extreme_folds_into_btc_positioning():
    fund = {"z156": 3.08, "funding_1d_annualized_pct": 11.3, "idx52w": 0.95,
            "read": "crowded_longs"}
    out = af._positioning("BTC", {}, fund)
    assert out["verdict"] == af.BEAR
    assert "funding" in out["detail"]


# ------------------------------------------------------------------- macro

def test_macro_uses_liquidity_and_skips_self_correlation():
    """`trend` exists only on the liquidity FLAG, not the parquet frame -- reading it
    off the frame silently left this layer neutral for every asset."""
    spy = _trending()
    liq = {"netliq_chg13w": -139.5, "netliq_chg13w_z": -0.71}
    out = af._macro(spy, spy, spy, liq, "SPY")
    assert out["verdict"] == af.BEAR
    assert "corr SPY" not in out["detail"]          # no self-correlation of +1.00
    assert "net liquidity falling" in out["detail"]
    rising = af._macro(spy, _trending(slope=0.1), spy, {"netliq_chg13w": 50.0}, "X")
    assert rising["verdict"] == af.BULL


def _noise(n=200, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return _series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n) * scale)))


def _linked(bench, sign=1.0, seed=1):
    """A price series whose RETURNS correlate `sign`-ly with the benchmark's."""
    rng = np.random.default_rng(seed)
    r = bench.pct_change().fillna(0.0) * sign + rng.normal(0, 0.002, len(bench))
    return _series((100 * (1 + r).cumprod()).to_numpy())


def test_macro_signs_the_liquidity_impulse_by_each_assets_own_co_movement():
    """The impulse is market-wide; the co-movement is not. Using only the impulse
    printed one identical verdict for all 89 names -- it measured each asset's link to
    the risk complex and then discarded it."""
    spy = _noise(seed=7)
    with_risk, against_risk = _linked(spy, +1.0), _linked(spy, -1.0)
    drain = {"netliq_chg13w": -139.7, "netliq_chg13w_z": -0.71}

    a = af._macro(with_risk, spy, None, drain, "A")
    b = af._macro(against_risk, spy, None, drain, "B")
    assert a["verdict"] == af.BEAR and b["verdict"] == af.BULL   # same tape, opposite rows
    assert "fighting" in a["detail"] and "helping" in b["detail"]

    pump = {"netliq_chg13w": 88.0, "netliq_chg13w_z": 1.40}
    assert af._macro(with_risk, spy, None, pump, "A")["verdict"] == af.BULL
    assert af._macro(against_risk, spy, None, pump, "B")["verdict"] == af.BEAR


def test_macro_is_neutral_when_the_asset_is_unlinked_to_the_risk_complex():
    spy = _noise(seed=7)
    indep = _noise(seed=99)
    out = af._macro(indep, spy, None, {"netliq_chg13w": -139.7,
                                       "netliq_chg13w_z": -0.71}, "X")
    assert abs(float(out["detail"].split("corr SPY ")[1].split(" ")[0])) < af.MACRO_CORR_MIN
    assert out["verdict"] == af.NEUTRAL
    assert "does not reach it" in out["detail"]


def test_macro_treats_a_weak_liquidity_change_as_drift_not_an_impulse():
    spy = _noise(seed=7)
    out = af._macro(_linked(spy), spy, None,
                    {"netliq_chg13w": -4.0, "netliq_chg13w_z": -0.11}, "A")
    assert out["verdict"] == af.NEUTRAL
    assert "drift, not an impulse" in out["detail"]


def test_macro_never_calls_a_direction_it_cannot_sign():
    """No overlap with SPY -> the impulse cannot be routed to this asset. Neutral and
    said out loud, never the market-wide verdict borrowed as if it were per-asset."""
    spy = _noise(seed=7)
    other = _series(np.linspace(100, 120, 50), start="2031-01-01")   # disjoint dates
    out = af._macro(other, spy, None, {"netliq_chg13w": -139.7,
                                       "netliq_chg13w_z": -0.71}, "X")
    assert out["verdict"] == af.NEUTRAL
    assert "unmeasurable" in out["detail"]


# ------------------------------------------------------------------ honesty

def test_unmeasurable_layers_are_no_data_not_guessed():
    """The on-chain half of the framework is not on this box. It must say so rather
    than be filled with a proxy that reads like the real measurement."""
    ctx = {"close": {"X": _trending(), "SPY": _trending()}, "volume": {},
           "cot": {}, "funding": {}, "liq": {}, "flows": {},
           "breadth": {"verdict": af.NEUTRAL, "detail": "-"}}
    f = af.asset_frame("X", ctx)
    assert f["valuation"] == af.NO_DATA
    assert f["fundamentals"] == af.NO_DATA
    assert "MVRV" in f["valuation_detail"]
    assert "hashrate" in f["fundamentals_detail"]


def test_every_layer_is_reported_for_every_asset():
    ctx = {"close": {"X": _trending(), "SPY": _trending()}, "volume": {},
           "cot": {}, "funding": {}, "liq": {}, "flows": {},
           "breadth": {"verdict": af.NEUTRAL, "detail": "-"}}
    f = af.asset_frame("X", ctx)
    for key, _q in af.LAYERS:
        assert key in f and f[key] in (af.BULL, af.BEAR, af.NEUTRAL, af.NO_DATA)
        assert f[f"{key}_detail"]
    assert f["n_bullish"] + f["n_bearish"] + f["n_neutral"] + f["n_no_data"] == len(af.LAYERS)


def test_inverse_instruments_are_flagged_so_the_sheet_is_not_read_backwards():
    ctx = {"close": {"VIX": _trending(), "SPY": _trending()}, "volume": {},
           "cot": {}, "funding": {}, "liq": {}, "flows": {},
           "breadth": {"verdict": af.NEUTRAL, "detail": "-"}}
    f = af.asset_frame("VIX", ctx)
    assert f["inverse"] is True
    assert "risk-OFF" in f["reads"]


def test_frame_never_raises_on_a_missing_asset():
    ctx = {"close": {"SPY": _trending()}, "volume": {}, "cot": {}, "funding": {},
           "liq": {}, "flows": {}, "breadth": {"verdict": af.NEUTRAL, "detail": "-"}}
    assert "error" in af.asset_frame("NOPE", ctx)


# ------------------------------------------------------- cross-universe coverage

def test_qlib_universe_is_a_superset_of_every_tracked_universe():
    """Every lab on this box tracks an overlapping macro set; alpaca addresses them by
    ETF ticker. Once aliases resolve, nothing is tracked anywhere that the framework
    does not cover -- if this fails, an asset is being tracked with no framework row."""
    from qlib_lab import config
    tracked = set().union(*af.UNIVERSES.values())
    assert tracked - set(config.UNIVERSE) == set()


def test_aliases_resolve_etf_tickers_to_logical_names():
    for ticker, logical in (("GLD", "GOLD"), ("USO", "OIL"), ("CPER", "COPPER"),
                            ("BTCUSD", "BTC"), ("BND", "AGG")):
        assert af.ALIASES[ticker] == logical
    assert "COPPER" in af.ALPACA_22 and "CPER" not in af.ALPACA_22
    assert "AGG" in af.ALPACA_CORE and "BND" not in af.ALPACA_CORE


def test_universes_for_reports_membership_and_qlib_only():
    assert "macro22" in af.universes_for("SPY")
    assert "cot" in af.universes_for("BTC")
    assert af.universes_for("XLU") == "qlib_only"


def test_breadth_excludes_inverse_and_fx_from_the_denominator():
    """VIX-family rise above their EMAs exactly when participation collapses, so
    counting them INFLATES breadth in a selloff. FX pair signs are a naming
    convention, US10Y is a yield, BIL is cash."""
    n = 400
    panel = pd.DataFrame({
        "SPY": _trending(n), "QQQ": _trending(n), "IWM": _trending(n),
        "VIX": _trending(n, slope=-0.2, base=300.0),      # would drag breadth down
        "EURUSD": _trending(n, slope=-0.2, base=300.0),
        "US10Y": _trending(n, slope=-0.2, base=300.0),
    })
    out = af._breadth(panel)
    assert out["pct_above_200"] == pytest.approx(1.0)   # only the 3 equity names count
    assert "3 direction-comparable" in out["detail"]
    assert "3 inverse/FX/rate names excluded" in out["detail"]


def test_flows_resolves_the_etf_ticker_alias():
    """etf_flows.json is keyed by ETF ticker (GLD) while the framework uses the logical
    name (GOLD). Missing this told the reader the box could not see data it had."""
    flows = {"GLD": {"z": 2.0, "flow_20d": 1e9, "snapshots": 60}, "_since": "2026-07-13"}
    out = af._flows("GOLD", flows)
    assert out["verdict"] == af.BULL
    assert out["flow_z"] == 2.0


def test_flows_no_data_message_names_the_ticker_it_looked_for():
    out = af._flows("COPPER", {"_since": "2026-07-13"})
    assert out["verdict"] == af.NO_DATA
    assert "CPER" in out["detail"]


def test_asset_table_survives_one_bad_instrument(monkeypatch):
    """A bare comprehension let one bad asset delete both sheets for all 89."""
    ctx = {"close": pd.DataFrame({"SPY": _trending(), "X": _trending()}),
           "volume": {}, "cot": {}, "funding": {}, "liq": {}, "flows": {},
           "breadth": {"verdict": af.NEUTRAL, "detail": "-"}}
    real = af.asset_frame

    def flaky(name, c):
        if name == "X":
            raise RuntimeError("synthetic")
        return real(name, c)

    monkeypatch.setattr(af, "asset_frame", flaky)
    out = af.asset_table(ctx)
    assert list(out["asset"]) == ["SPY"]


def test_macro_drops_the_dxy_term_when_absent():
    """Falling back to `px` printed corr DXY +1.00 as if it were a dollar reading."""
    spy = _trending()
    out = af._macro(_trending(slope=0.2), spy, None, {"netliq_chg13w": -10.0}, "X")
    assert "DXY" not in out["detail"]


# --------------------------------------------------------------- narrative layer

def _story(sid, state, assets):
    return {"id": sid, "claim": "c", "pair": "A/B", "implies": "corr(A,B) > +0.40",
            "corr_fast": 0.5, "corr_slow": 0.4, "state": state, "state_slow": state,
            "assets": assets}


def test_narrative_verdict_comes_from_the_registered_stories():
    asof = pd.Timestamp("2026-08-24")
    intact = af._narrative("BTC", [_story("S1", "HOLDING", ("BTC",))], asof)
    assert intact["verdict"] == af.BULL
    broke = af._narrative("BTC", [_story("S1", "HOLDING", ("BTC",)),
                                  _story("S2", "BROKEN", ("BTC",))], asof)
    assert broke["verdict"] == af.BEAR
    assert "fragility reading, not a price call" in broke["detail"]
    weak = af._narrative("BTC", [_story("S1", "WEAK", ("BTC",))], asof)
    assert weak["verdict"] == af.NEUTRAL


def test_narrative_keeps_the_event_countdown_and_says_when_nothing_applies():
    """The old catalyst row was NEUTRAL for all 89 names forever. Replacing it must not
    silently drop the FOMC countdown it did carry."""
    out = af._narrative("XLU", [_story("S1", "HOLDING", ("BTC",))],
                        pd.Timestamp("2026-08-24"))
    assert out["verdict"] == af.NEUTRAL
    assert "No registered story references this instrument" in out["detail"]
    assert "FOMC" in out["detail"]


def test_narrative_replaced_catalyst_in_the_layer_list():
    keys = [k for k, _ in af.LAYERS]
    assert "narrative" in keys and "catalyst" not in keys
    assert len(af.LAYERS) == 12


def test_flows_no_data_reason_is_the_rows_own_note():
    """"no issuer feed for this fund" and "history has not accumulated" are different
    facts; the generic sentence hid which one applied."""
    flows = {"QQQ": {"z": None, "flow_20d": None, "snapshots": 0,
                     "note": "Invesco: the documented CSV download URLs return HTML"}}
    out = af._flows("QQQ", flows)
    assert out["verdict"] == af.NO_DATA
    assert "Invesco" in out["detail"]


def test_flows_detail_carries_the_source_and_its_resolution():
    flows = {"SPY": {"z": 2.1, "flow_20d": 4.2e9, "snapshots": 40, "source": "ssga",
                     "noise_usd": 5_300_000}}
    out = af._flows("SPY", flows)
    assert out["verdict"] == af.BULL
    assert "ssga" in out["detail"] and "5,300,000" in out["detail"]
