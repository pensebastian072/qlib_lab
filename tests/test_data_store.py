"""Sanity: the .bin store loads and expression math matches hand calc."""
import numpy as np
import pandas as pd
import pytest

from qlib_lab import config


@pytest.fixture(scope="module")
def qlib_init():
    import qlib
    from qlib.constant import REG_US

    qlib.init(provider_uri=str(config.QLIB_DATA_DIR), region=REG_US)
    return qlib


def test_features_load_and_match_csv(qlib_init):
    from qlib.data import D

    df = D.features(["SPY"], ["$close", "Mean($close, 20)", "Ref($close, 1)"])
    assert not df.empty
    got = df.droplevel(0)

    csv = pd.read_csv(config.CSV_DIR / "SPY.csv", parse_dates=["date"]).set_index("date")
    joined = got.join(csv["close"], how="inner").dropna()
    assert len(joined) > 2000
    # $close == csv adjusted close
    np.testing.assert_allclose(joined["$close"], joined["close"], rtol=1e-5)
    # Mean($close,20) == rolling mean hand calc
    hand = csv["close"].rolling(20).mean()
    j2 = got["Mean($close, 20)"].to_frame("qlib").join(hand.rename("hand")).dropna()
    np.testing.assert_allclose(j2["qlib"], j2["hand"], rtol=1e-5)
    # Ref($close,1) == yesterday's close (no lookahead)
    j3 = got["Ref($close, 1)"].to_frame("qlib").join(csv["close"].shift(1).rename("hand")).dropna()
    np.testing.assert_allclose(j3["qlib"], j3["hand"], rtol=1e-5)


def test_universe_present(qlib_init):
    from qlib.data import D

    instruments = D.list_instruments(D.instruments("all"), as_list=True)
    kept = [a for a in config.ASSETS if a.lower() in [i.lower() for i in instruments]]
    assert len(kept) >= 15, f"only {kept}"


# ------------------------------------------------- merge_with_stored (2026-08-24)

import pandas as pd
import pytest

from qlib_lab import data_fetch as dfm


def _frame(dates, close, factor=1.0):
    idx = pd.to_datetime(dates)
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": [0.0] * len(idx),
                         "factor": [factor] * len(idx)}, index=idx)


def _write(path, df):
    out = df.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.to_csv(path, index=False)


def test_merge_unions_and_fresh_wins(tmp_path):
    p = tmp_path / "X.csv"
    _write(p, _frame(["2024-01-01", "2024-01-02"], [1.0, 2.0]))
    fresh = _frame(["2024-01-02", "2024-01-03"], [99.0, 3.0])
    m = dfm.merge_with_stored(p, fresh)
    assert list(m["close"]) == [1.0, 99.0, 3.0]      # old kept, fresh wins on overlap


def test_merge_returns_fresh_when_no_stored_file(tmp_path):
    fresh = _frame(["2024-01-01"], [1.0])
    assert dfm.merge_with_stored(tmp_path / "none.csv", fresh) is fresh


def test_merge_skips_write_on_corrupt_stored(tmp_path):
    """Must NOT fail open into the destructive overwrite it exists to prevent."""
    p = tmp_path / "X.csv"
    p.write_text("this is not,a valid\ncsv at all\x00\x00")
    assert dfm.merge_with_stored(p, _frame(["2024-01-01"], [1.0])) is None


def test_merge_rebases_stored_rows_onto_fresh_vintage(tmp_path):
    """Yahoo re-adjusts history on ex-div; without rebasing the series splices two
    bases and shows a spurious return jump at the seam."""
    p = tmp_path / "X.csv"
    _write(p, _frame(["2024-01-01", "2024-01-02"], [10.0, 20.0], factor=0.5))
    fresh = _frame(["2024-01-02", "2024-01-03"], [20.0, 30.0], factor=1.0)
    m = dfm.merge_with_stored(p, fresh)
    # k = 1.0/0.5 = 2 applied to rows STRICTLY BEFORE the first common date
    assert m.loc[pd.Timestamp("2024-01-01"), "close"] == pytest.approx(20.0)
    assert m.loc[pd.Timestamp("2024-01-01"), "factor"] == pytest.approx(1.0)
    # rows at/after the seam come from fresh untouched
    assert m.loc[pd.Timestamp("2024-01-02"), "close"] == pytest.approx(20.0)


def test_merge_leaves_prices_alone_when_vintage_unchanged(tmp_path):
    p = tmp_path / "X.csv"
    _write(p, _frame(["2024-01-01", "2024-01-02"], [10.0, 20.0], factor=1.0))
    fresh = _frame(["2024-01-02", "2024-01-03"], [20.0, 30.0], factor=1.0)
    m = dfm.merge_with_stored(p, fresh)
    assert m.loc[pd.Timestamp("2024-01-01"), "close"] == pytest.approx(10.0)


def test_cboe_index_shape_matches_yahoo_contract(monkeypatch):
    csv = ("DATE,OPEN,HIGH,LOW,CLOSE\n"
           "2011-01-03,0.0,0.0,0.0,0.0\n"      # pre-inception placeholder
           "2024-01-02,12.0,13.0,11.0,12.5\n")

    class _R:
        def read(self): return csv.encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(dfm.urllib.request, "urlopen", lambda *a, **k: _R())
    out = dfm.fetch_cboe_index("VIX9D")
    assert list(out.columns) == ["open", "high", "low", "close", "volume", "adjclose"]
    assert len(out) == 1                       # zero-close placeholder dropped
    assert out["adjclose"].iloc[0] == out["close"].iloc[0]
    assert out["volume"].iloc[0] == 0.0
