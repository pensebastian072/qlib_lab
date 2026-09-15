"""Daily-TGA (Treasury DTS) contract for the netliq plumbing.

Guards the two traps that make a naive DTS reader fail SILENTLY rather than
loudly: the string "null" in the balance column, and the two historical row
renames that would leave a 12-year hole before 2022.
"""
import json

import pandas as pd
import pytest

from qlib_lab import fred_liquidity as fl


def _fake_urlopen(payload):
    """Stand in for urllib.request.urlopen, matching test_data_store's idiom."""
    class _R:
        def read(self): return json.dumps(payload).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False
    return lambda *a, **k: _R()


# --- trap 2: the literal string "null" ------------------------------------

@pytest.mark.parametrize("raw", [None, "", "null", "NULL", "None", ".", "  "])
def test_dts_num_rejects_the_string_null(raw):
    assert fl._dts_num(raw) is None


@pytest.mark.parametrize("raw,want", [("950804", 950804.0), (" 12.5 ", 12.5), (0, 0.0)])
def test_dts_num_parses_real_values(raw, want):
    assert fl._dts_num(raw) == want


# --- trap 1: three schema eras, resolved by priority ----------------------

def test_all_three_eras_resolve_to_one_series_per_date(monkeypatch):
    rows = [
        # 2010 era: value in close_today_bal
        {"record_date": "2010-01-05", "account_type": "Federal Reserve Account",
         "open_today_bal": "171889", "close_today_bal": "163428"},
        {"record_date": "2010-01-05", "account_type": "Tax and Loan Note Accounts (Table V)",
         "open_today_bal": "1938", "close_today_bal": "1934"},
        # brief 2022 era: renamed row, still close_today_bal
        {"record_date": "2022-04-05", "account_type": "Treasury General Account (TGA)",
         "open_today_bal": "579607", "close_today_bal": "542176"},
        # current era: close is the STRING "null", value in open_today_bal
        {"record_date": "2026-08-28", "account_type": "Treasury General Account (TGA) Opening Balance",
         "open_today_bal": "950804", "close_today_bal": "null"},
        {"record_date": "2026-08-28", "account_type": "Treasury General Account (TGA) Closing Balance",
         "open_today_bal": "971343", "close_today_bal": "null"},
    ]
    monkeypatch.setattr(fl.urllib.request, "urlopen", _fake_urlopen({"data": rows}))
    s = fl.fetch_dts_tga()

    assert list(s.index) == [pd.Timestamp("2010-01-05"),
                             pd.Timestamp("2022-04-05"),
                             pd.Timestamp("2026-08-28")]
    assert s.loc["2010-01-05"] == 163428.0      # close_today_bal, not open
    assert s.loc["2022-04-05"] == 542176.0      # close_today_bal, not open
    assert s.loc["2026-08-28"] == 971343.0      # Closing row's OPEN column


def test_closing_row_wins_over_opening_row_on_the_same_date(monkeypatch):
    """Both rows carry a number; only the Closing row is the closing balance."""
    rows = [
        {"record_date": "2026-08-28", "account_type": "Treasury General Account (TGA) Opening Balance",
         "open_today_bal": "950804", "close_today_bal": "null"},
        {"record_date": "2026-08-28", "account_type": "Treasury General Account (TGA) Closing Balance",
         "open_today_bal": "971343", "close_today_bal": "null"},
    ]
    monkeypatch.setattr(fl.urllib.request, "urlopen", _fake_urlopen({"data": rows}))
    assert fl.fetch_dts_tga().loc["2026-08-28"] == 971343.0


def test_unknown_schema_returns_none_not_an_empty_series(monkeypatch):
    """A fourth rename must fail LOUD (None -> weekly fallback), never silently
    hand back an empty/partial series that would ffill into a frozen TGA."""
    rows = [{"record_date": "2027-01-04", "account_type": "Some Brand New Label",
             "open_today_bal": "1", "close_today_bal": "2"}]
    monkeypatch.setattr(fl.urllib.request, "urlopen", _fake_urlopen({"data": rows}))
    assert fl.fetch_dts_tga() is None


def test_fetch_failure_returns_none(monkeypatch):
    def _boom(*a, **k):
        raise OSError("no route to host")
    monkeypatch.setattr(fl.urllib.request, "urlopen", _boom)
    assert fl.fetch_dts_tga() is None


# --- the daily series must actually be daily -------------------------------

def test_dts_resolves_every_business_day_not_just_wednesdays(monkeypatch):
    """The whole point of the swap: weekly WDTGAL -> daily DTS."""
    days = pd.bdate_range("2026-08-17", "2026-08-28")
    rows = [{"record_date": str(d.date()),
             "account_type": "Treasury General Account (TGA) Closing Balance",
             "open_today_bal": str(900000 + i * 1000), "close_today_bal": "null"}
            for i, d in enumerate(days)]
    monkeypatch.setattr(fl.urllib.request, "urlopen", _fake_urlopen({"data": rows}))
    s = fl.fetch_dts_tga()
    assert len(s) == len(days)
    assert s.index.dayofweek.max() <= 4


# --- cross-check -----------------------------------------------------------

def test_cross_check_reports_the_gap_in_bn():
    idx = pd.DatetimeIndex(["2026-08-19", "2026-08-26"])
    dts = pd.Series([900000.0, 950000.0], index=idx)          # $mm
    wtr = pd.Series([900000.0, 948000.0], index=idx)          # $mm
    cc = fl.tga_cross_check(dts, wtr)
    assert cc["n"] == 2
    assert cc["max_abs_diff_bn"] == pytest.approx(2.0)        # 2000 $mm -> $2bn
    assert cc["worst_date"] == "2026-08-26"


def test_cross_check_handles_a_missing_source():
    assert fl.tga_cross_check(None, pd.Series(dtype=float)) is None


def test_cross_check_no_overlap_is_reported_not_crashed():
    dts = pd.Series([1.0], index=pd.DatetimeIndex(["2026-08-19"]))
    wtr = pd.Series([1.0], index=pd.DatetimeIndex(["2020-01-01"]))
    assert fl.tga_cross_check(dts, wtr)["n"] == 0


# --- fail-safe: a dead DTS must not kill netliq ----------------------------

def _weekly(name, start, periods, value):
    idx = pd.date_range(start, periods=periods, freq="W-WED")
    return pd.Series([value] * periods, index=idx, name=name)


def test_dead_dts_falls_back_to_weekly_wdtgal(monkeypatch):
    monkeypatch.setattr(fl, "fetch_dts_tga", lambda *a, **k: None)
    monkeypatch.setattr(fl, "fetch_series", lambda sid, **k: {
        "WALCL": _weekly("WALCL", "2024-01-03", 200, 7_000_000.0),
        "WDTGAL": _weekly("WDTGAL", "2024-01-03", 200, 700_000.0),
        "RRPONTSYD": _weekly("RRPONTSYD", "2024-01-03", 200, 500.0),
    }[sid])
    df = fl.build_netliq()
    assert df is not None and not df.empty
    assert (df["tga_source"] == "wdtgal_weekly_fallback").all()
    # netliq($bn) = 7000 - 700 - 500
    assert df["netliq"].iloc[-1] == pytest.approx(5800.0)


def test_dts_is_primary_when_available(monkeypatch):
    daily = pd.Series(
        [700_000.0] * 500,
        index=pd.bdate_range("2024-01-02", periods=500), name="TGA_DTS")
    monkeypatch.setattr(fl, "fetch_dts_tga", lambda *a, **k: daily)
    monkeypatch.setattr(fl, "fetch_series", lambda sid, **k: {
        "WALCL": _weekly("WALCL", "2024-01-03", 200, 7_000_000.0),
        "WDTGAL": _weekly("WDTGAL", "2024-01-03", 200, 690_000.0),
        "RRPONTSYD": _weekly("RRPONTSYD", "2024-01-03", 200, 500.0),
    }[sid])
    df = fl.build_netliq()
    assert (df["tga_source"] == "dts_daily").all()
    assert df["tga"].iloc[-1] == pytest.approx(700.0)         # DTS, not WDTGAL
    assert df["tga_weekly"].iloc[-1] == pytest.approx(690.0)  # kept for comparison
    assert "tga_crosscheck" in df.attrs


def test_units_are_dollars_bn_not_mm(monkeypatch):
    """A missed /1000 on the DTS would inflate the TGA leg 1000x and invert netliq."""
    daily = pd.Series([950_804.0] * 500,
                      index=pd.bdate_range("2024-01-02", periods=500), name="TGA_DTS")
    monkeypatch.setattr(fl, "fetch_dts_tga", lambda *a, **k: daily)
    monkeypatch.setattr(fl, "fetch_series", lambda sid, **k: {
        "WALCL": _weekly("WALCL", "2024-01-03", 200, 6_600_000.0),
        "WDTGAL": _weekly("WDTGAL", "2024-01-03", 200, 950_804.0),
        "RRPONTSYD": _weekly("RRPONTSYD", "2024-01-03", 200, 12.0),
    }[sid])
    df = fl.build_netliq()
    assert df["tga"].iloc[-1] == pytest.approx(950.804)
    assert 4000.0 < df["netliq"].iloc[-1] < 7000.0
