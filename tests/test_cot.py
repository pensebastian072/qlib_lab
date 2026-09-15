"""COT: lookahead rule, sign flips, per-instrument alignment."""
from datetime import timedelta

import numpy as np
import pandas as pd

from qlib_lab import cot


def _hist(asset="GOLD", n=60, start="2024-01-02"):
    dates = pd.date_range(start, periods=n, freq="7D")  # Tuesdays-ish
    return pd.DataFrame({
        "report_date": dates, "asset": asset,
        "net_pct": np.linspace(-0.2, 0.2, n), "oi": 1e5,
    })


def test_effective_date_release_lag():
    feats = cot.weekly_features(_hist())
    lag = (feats["effective_date"] - feats["report_date"]).dt.days
    assert (lag == cot.RELEASE_LAG_DAYS).all()


def test_daily_panel_no_lookahead():
    feats = cot.weekly_features(_hist())
    first_report = feats["report_date"].min()
    sessions = pd.bdate_range(first_report, periods=15)
    panel = cot.daily_panel(feats, sessions, cols=("net_pct",))["GOLD"]
    # sessions BEFORE report_date + lag must be NaN — the report isn't out yet
    cutoff = first_report + timedelta(days=cot.RELEASE_LAG_DAYS)
    assert panel.loc[panel.index < cutoff, "net_pct"].isna().all()
    assert panel.loc[panel.index >= cutoff, "net_pct"].notna().any()


def test_normalize_sign_flip():
    df = pd.DataFrame({
        "code": ["097741", "088691"],  # JPY (flip), GOLD (no flip)
        "report_date": ["2026-07-07", "2026-07-07"],
        "nc_long": [100.0, 100.0], "nc_short": [40.0, 40.0], "oi": [1000.0, 1000.0],
    })
    out = cot._normalize(df).set_index("asset")
    assert out.loc["USDJPY", "net_pct"] == -0.06  # flipped
    assert out.loc["GOLD", "net_pct"] == 0.06


def test_per_instrument_features_alignment(tmp_path, monkeypatch):
    hist = _hist(asset="GOLD", n=60)
    p = tmp_path / "cot.parquet"
    hist.to_parquet(p, index=False)
    monkeypatch.setattr(cot, "COT_PARQUET", p)
    dates = pd.bdate_range("2025-06-02", periods=10)
    idx = pd.MultiIndex.from_product([["GOLD", "XLK"], dates],
                                     names=["instrument", "datetime"])
    out = cot.per_instrument_features(idx)
    assert out is not None
    assert out.loc["GOLD"]["cot_z156"].notna().any()
    assert out.loc["XLK"].isna().all().all()  # unmapped stays NaN
