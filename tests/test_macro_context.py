"""Macro-context features: alignment and no-lookahead."""
import numpy as np
import pandas as pd

from qlib_lab import macro_context


def _fake_rets(n=200, cols=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(rng.normal(0, 0.01, (n, cols)), index=idx,
                        columns=[f"A{i}" for i in range(cols)])


def test_absorption_no_lookahead():
    rets = _fake_rets()
    a1 = macro_context.absorption_series(rets, window=60)
    # perturb the FUTURE beyond t — values at/before t must not change
    rets2 = rets.copy()
    rets2.iloc[150:] = rets2.iloc[150:] * 10
    a2 = macro_context.absorption_series(rets2, window=60)
    pd.testing.assert_series_equal(a1.iloc[:150], a2.iloc[:150])


def test_absorption_bounds():
    a = macro_context.absorption_series(_fake_rets(), window=60).dropna()
    assert len(a) > 100
    assert ((a > 0) & (a <= 1)).all()


def test_augment_alignment():
    idx = pd.MultiIndex.from_product(
        [["SPY", "QQQ"], pd.bdate_range("2024-01-01", periods=5)],
        names=["instrument", "datetime"])
    feat = pd.DataFrame({"f1": range(10)}, index=idx)
    ctx = pd.DataFrame({"ctx_x": [10, 20, 30, 40, 50]},
                       index=pd.bdate_range("2024-01-01", periods=5))
    out = macro_context.augment(feat, ctx)
    assert list(out.columns) == ["f1", "ctx_x"]
    # same date -> same ctx value for every instrument
    spy = out.loc["SPY"]["ctx_x"]
    qqq = out.loc["QQQ"]["ctx_x"]
    assert (spy.values == qqq.values).all()
    assert spy.iloc[0] == 10 and spy.iloc[-1] == 50
