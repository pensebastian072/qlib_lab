"""Vol desk: BS estimate sanity + decision-matrix branches."""
from qlib_lab import telegram_notify as tn
from qlib_lab import vol_desk as vd


def _mag(magnitude_pct=0.30, spot=750.0):
    return {"sigma_1d_pct": 1.0, "expected_move_usd": round(spot * 0.01, 2),
            "expected_move_pct": 1.0, "magnitude_pct": magnitude_pct,
            "proxy": "vix_implied", "note": "test"}


def _vrp(pct, spot=750.0, vix=16.0, magnitude_pct=0.30):
    return {"spot": spot, "vix": vix, "vrp": 50.0, "vrp_pct": pct,
            "data_through": "2026-07-14", "magnitude": _mag(magnitude_pct, spot)}


def _shift(erupt=False, calm5=True, calm21=True, stale=False):
    return {"stale": stale, "horizons": {
        "5d": {"calm": calm5, "p_eruption": 0.3, "eruption_predicted": erupt},
        "21d": {"calm": calm21, "p_eruption": 0.2, "eruption_predicted": erupt}}}


def test_bs_put_and_delta_sane():
    p = vd.bs_put(750, 720, 0.16, 35 / 365)
    assert 0 < p < 30
    d = vd.put_delta(750, 720, 0.16, 35 / 365)
    assert -0.5 < d < 0
    k = vd.strike_for_put_delta(750, 0.16, 35 / 365)
    assert 650 < k < 750
    assert abs(vd.put_delta(750, k, 0.16, 35 / 365) - vd.SHORT_PUT_DELTA) < 0.02


def test_straddle_above_intrinsic():
    s = vd.bs_straddle(750, 750, 0.16, 35 / 365)
    assert s > 0
    assert s > vd.bs_put(750, 750, 0.16, 35 / 365)


def test_matrix_sell_when_rich_and_calm():
    t = vd.build_ticket(_vrp(0.85), _shift(erupt=False, calm5=True))
    assert t["action"] == "SELL_PUT_SPREAD"
    st = t["structure"]
    assert st["long_strike"] == st["short_strike"] - vd.SPREAD_WIDTH
    assert st["est_credit"] > 0
    assert st["max_risk_usd"] < vd.SPREAD_WIDTH * 100


def test_matrix_long_vol_when_eruption_and_cheap():
    t = vd.build_ticket(_vrp(0.30), _shift(erupt=True, calm21=True))
    assert t["action"] == "LONG_VOL"
    assert t["structure"]["type"] == "straddle"


def test_matrix_no_trade_when_eruption_but_rich():
    t = vd.build_ticket(_vrp(0.90), _shift(erupt=True, calm21=True))
    assert t["action"] == "NO_TRADE"
    assert "too rich" in t["reason"]


def test_matrix_no_trade_when_cheap_and_quiet():
    t = vd.build_ticket(_vrp(0.20), _shift(erupt=False))
    assert t["action"] == "NO_TRADE"


def test_stale_shift_suspends_everything():
    t = vd.build_ticket(_vrp(0.95), _shift(stale=True))
    assert t["action"] == "NO_TRADE"
    assert "stale" in t["reason"]


def test_vrp_forward_tilt_buckets():
    rich = vd.vrp_forward_tilt(0.80)
    cheap = vd.vrp_forward_tilt(0.20)
    assert rich["bucket"] == "rich"
    assert cheap["bucket"] == "cheap"
    # rich insurance -> higher historical forward return (H07 split)
    assert rich["expected_fwd_ret_1m"] > cheap["expected_fwd_ret_1m"]


def test_magnitude_and_tilt_shown_on_every_ticket():
    t = vd.build_ticket(_vrp(0.20), _shift(erupt=False))  # NO_TRADE branch
    assert t["magnitude"]["proxy"] == "vix_implied"
    assert t["vrp_forward"]["bucket"] == "cheap"


def test_big_move_regime_vetoes_selling_premium():
    # rich + calm would normally SELL_PUT_SPREAD, but a high magnitude
    # percentile (big-move regime) must block selling premium (B07 lesson).
    t = vd.build_ticket(_vrp(0.85, magnitude_pct=0.90), _shift(erupt=False, calm5=True))
    assert t["action"] == "NO_TRADE"
    assert "big-move regime" in t["reason"]


def test_sell_still_fires_when_magnitude_low():
    t = vd.build_ticket(_vrp(0.85, magnitude_pct=0.30), _shift(erupt=False, calm5=True))
    assert t["action"] == "SELL_PUT_SPREAD"


def test_telegram_format_never_raises():
    t = vd.build_ticket(_vrp(0.85, magnitude_pct=0.30), _shift(erupt=False, calm5=True))
    msg = tn.format_ticket(t)
    assert isinstance(msg, str)
    assert "VOL DESK" in msg
    assert "advisory paper ticket" in msg
