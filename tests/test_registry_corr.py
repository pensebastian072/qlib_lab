"""Registry redundancy maths: effective dimension, decorrelation, families.

Synthetic matrices with known answers throughout -- if these drift, every "how many
independent signals do we really have" claim drifts with them.
"""
import numpy as np
import pandas as pd
import pytest

from qlib_lab import market_state as ms, registry_corr as rc


def _corr(vals, names):
    return pd.DataFrame(vals, index=names, columns=names)


def test_effective_dim_counts_independent_signals():
    """Identity -> every column is its own signal. All-ones -> one signal wearing n
    different names, which is exactly the failure the registry matrix looks for."""
    names = list("abcdefgh")
    ident = rc.effective_dim(_corr(np.eye(8), names))
    assert ident["participation_ratio"] == pytest.approx(8.0, rel=1e-6)
    assert ident["n_for_90pct"] == 8

    ones = rc.effective_dim(_corr(np.ones((8, 8)), names))
    assert ones["participation_ratio"] == pytest.approx(1.0, rel=1e-6)
    assert ones["n_for_90pct"] == 1
    assert ones["absorption"] == pytest.approx(1.0)


def test_effective_dim_sees_two_blocks_as_two_signals():
    m = np.zeros((6, 6))
    m[:3, :3] = 1.0
    m[3:, 3:] = 1.0
    d = rc.effective_dim(_corr(m, list("abcdef")))
    assert d["participation_ratio"] == pytest.approx(2.0, rel=1e-6)
    assert d["n_for_90pct"] == 2


def test_effective_dim_refuses_an_incomplete_matrix():
    m = np.eye(4)
    m[0, 3] = m[3, 0] = np.nan
    assert rc.effective_dim(_corr(m, list("abcd")))["participation_ratio"] is None
    assert rc.effective_dim(pd.DataFrame())["participation_ratio"] is None


def test_decorrelation_time_recovers_a_known_persistence():
    """An AR(1) with phi=0.9 has autocorrelation 0.9^k, which crosses 1/e near lag 10.
    This number is the denominator for every claim made on the panel."""
    rng = np.random.default_rng(0)
    n, phi = 4000, 0.9
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    s = pd.Series(x, index=pd.bdate_range("2010-01-01", periods=n))
    assert 8 <= rc.decorrelation_time(s) <= 13

    white = pd.Series(rng.normal(size=n), index=s.index)
    assert rc.decorrelation_time(white) <= 2
    assert rc.decorrelation_time(pd.Series([1.0] * 100)) is None   # constant


def test_decorrelation_time_reports_the_cap_not_a_measurement():
    """A trending level never decorrelates. Returning the search cap is fine; the caller
    has to KNOW it is a cap, which is why the report names those readings."""
    s = pd.Series(np.arange(2000.0), index=pd.bdate_range("2010-01-01", periods=2000))
    assert rc.decorrelation_time(s, max_lag=250) == 250.0


def test_corr_matrix_is_complete_case_and_says_what_it_used():
    """Pairwise cells come from different samples and can make the eigenvalues negative.
    Complete-case costs rows -- so the rows it kept are reported."""
    idx = pd.bdate_range("2020-01-01", periods=600)
    df = pd.DataFrame({"a": np.linspace(0, 1, 600), "b": np.linspace(1, 0, 600)},
                      index=idx)
    df.loc[idx[:300], "c"] = np.nan
    df["c"] = pd.Series(np.linspace(0, 1, 600), index=idx).where(
        pd.Series(range(600), index=idx) >= 300)
    c, meta = rc.corr_matrix(df, mode="level")
    assert meta["rows_used"] == 300
    assert not c.isna().any().any()
    assert rc.effective_dim(c)["participation_ratio"] is not None


def test_corr_matrix_drops_a_constant_column_instead_of_dividing_by_zero():
    idx = pd.bdate_range("2020-01-01", periods=400)
    df = pd.DataFrame({"a": np.arange(400.0), "b": np.arange(400.0) * -1,
                       "flat": np.ones(400)}, index=idx)
    c, _ = rc.corr_matrix(df, mode="level")
    assert "flat" not in c.columns


def test_families_group_a_reading_with_its_own_threshold():
    """The registry's most common redundancy: a condition and the number it thresholds,
    or a story state and the correlation it came from."""
    names = ["vrp", "vrp_pct", "cond_vol_rich", "netliq", "absorption"]
    m = np.eye(5)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        m[i, j] = m[j, i] = 0.95
    fam = rc.families(_corr(m, names))
    lab = dict(zip(fam["reading"], fam["family"]))
    assert lab["vrp"] == lab["vrp_pct"] == lab["cond_vol_rich"]
    assert lab["netliq"] != lab["vrp"] and lab["absorption"] != lab["vrp"]


def test_story_states_encode_ordinally_and_no_data_stays_missing():
    """NO_DATA must not become a zero -- zero is WEAK, an actual reading."""
    idx = pd.bdate_range("2026-01-01", periods=4)
    panel = pd.DataFrame({
        "story_state_X": [ms.HOLDING, ms.WEAK, ms.BROKEN, ms.NO_DATA],
        "vol_vrp_pct": [0.1, 0.2, 0.3, 0.4],
        "cot_idx52_SPY": [0.9, 0.9, 0.9, 0.9],
        "cot_n_crowded": [1, 2, 3, 4],
    }, index=idx)
    out = rc.numeric_registry(panel, per_asset=False)
    assert list(out["story_state_X"][:3]) == [1.0, 0.0, -1.0]
    assert pd.isna(out["story_state_X"].iloc[3])
    assert "cot_idx52_SPY" not in out.columns          # per-asset excluded
    assert "cot_n_crowded" in out.columns
    assert "cot_idx52_SPY" in rc.numeric_registry(panel, per_asset=True).columns
