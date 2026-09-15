"""Fail-safe contract: readers never raise, degrade to neutral."""
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from qlib_lab import config, etf_flows, publish


def test_read_state_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FLAG_PATH", tmp_path / "nope.json")
    st = publish.read_state()
    assert st["stale"] is True and st["enforce"] == "no" and st["assets"] == {}


def test_read_state_corrupt(tmp_path, monkeypatch):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    monkeypatch.setattr(config, "FLAG_PATH", p)
    assert publish.read_state()["stale"] is True


def test_read_state_stale(tmp_path, monkeypatch):
    p = tmp_path / "old.json"
    old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    p.write_text(json.dumps({"as_of": old, "stale": False, "assets": {"SPY": {}}}))
    monkeypatch.setattr(config, "FLAG_PATH", p)
    st = publish.read_state()
    assert st["stale"] is True


def test_etf_read_state_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ETF_FLOWS_PATH", tmp_path / "nope.json")
    st = etf_flows.read_state()
    assert st["stale"] is True and st["rows"] == []


def test_flow_math():
    # 3 snapshots for SPY: +1M shares day2, -2M day3 at $750
    hist = pd.DataFrame(
        {
            "date": ["2026-07-10", "2026-07-11", "2026-07-12"],
            "ticker": ["SPY"] * 3,
            "shares": [1_000_000_000.0, 1_001_000_000.0, 999_000_000.0],
            "price": [740.0, 745.0, 750.0],
            "source": ["ssga"] * 3,
        }
    )
    rows = etf_flows.compute_flows(hist)
    spy = next(r for r in rows if r["ticker"] == "SPY")
    assert spy["flow_1d"] == (999_000_000 - 1_001_000_000) * 750.0  # -1.5B
    assert spy["flow_5d"] is None  # not enough history yet
    assert spy["note"] is None and spy["snapshots"] == 3


def test_flow_accumulating_first_snapshot():
    hist = pd.DataFrame(
        {"date": ["2026-07-12"], "ticker": ["QQQ"], "shares": [4e8], "price": [713.0],
         "source": ["ssga"]}
    )
    rows = etf_flows.compute_flows(hist)
    qqq = next(r for r in rows if r["ticker"] == "QQQ")
    assert qqq["flow_1d"] is None and qqq["note"] == "accumulating"


# ------------------------------------------------- issuer-sourced flows (2026-08-25)

def test_yahoo_era_rows_are_never_differenced_against_issuer_rows():
    """Yahoo `sharesOutstanding` for an ETF is a static cached field -- 19 snapshots
    held one distinct value. Differencing across the source change would print one
    enormous fictional creation on the first issuer day."""
    hist = pd.DataFrame({
        "date": ["2026-08-20", "2026-08-21", "2026-08-24", "2026-08-25"],
        "ticker": ["SPY"] * 4,
        # the two legacy rows carry the frozen Yahoo count, then the real one appears
        "shares": [9.5e8, 9.5e8, 1.059e9, 1.060e9],
        "price": [760.0, 761.0, 763.45, 764.0],
        "source": [None, None, "ssga", "ssga"],
        "noise_usd": [None, None, 5.3e6, 5.3e6],
    })
    rows = etf_flows.compute_flows(hist)
    spy = next(r for r in rows if r["ticker"] == "SPY")
    assert spy["snapshots"] == 2                     # legacy rows excluded entirely
    assert spy["flow_1d"] == (1.060e9 - 1.059e9) * 764.0
    assert spy["legacy_rows_ignored"] == 2


def test_a_fund_with_no_issuer_feed_says_so_instead_of_reporting_zero():
    hist = pd.DataFrame({"date": ["2026-08-25"], "ticker": ["SPY"], "shares": [1.0e9],
                         "price": [763.45], "source": ["ssga"], "noise_usd": [5.3e6]})
    rows = {r["ticker"]: r for r in etf_flows.compute_flows(hist)}
    assert rows["QQQ"]["flow_1d"] is None and rows["QQQ"]["snapshots"] == 0
    assert "Invesco" in rows["QQQ"]["note"]
    assert rows["USO"]["note"] and rows["CPER"]["note"]


def test_a_move_under_the_sources_own_rounding_is_not_called_a_flow():
    """SSGA rounds NAV to a cent, so ~$5m/day on SPY is arithmetic, not a creation."""
    hist = pd.DataFrame({
        "date": [f"2026-08-{d:02d}" for d in (10, 11, 12)],
        "ticker": ["SPY"] * 3,
        "shares": [1.000000e9, 1.000000e9, 1.000001e9],   # 1000 shares ~ $0.76m
        "price": [763.45] * 3,
        "source": ["ssga"] * 3,
        "noise_usd": [5.3e6] * 3,
    })
    spy = next(r for r in etf_flows.compute_flows(hist) if r["ticker"] == "SPY")
    assert spy["note"] == "flow_1d below this source's resolution"
    assert spy["noise_usd"] == 5300000


def test_history_starts_at_the_source_switch_not_at_the_yahoo_install(tmp_path, monkeypatch):
    """`history_since` must not claim the five weeks of un-differenceable Yahoo rows."""
    parquet = tmp_path / "etf_shares.parquet"
    pd.DataFrame({
        "date": ["2026-07-13", "2026-08-25"], "ticker": ["SPY", "SPY"],
        "shares": [9.5e8, 1.059e9], "price": [740.0, 763.45],
        "source": [None, "ssga"], "noise_usd": [None, 5.3e6],
    }).to_parquet(parquet, index=False)
    monkeypatch.setattr(config, "ETF_SHARES_PARQUET", parquet)
    monkeypatch.setattr(config, "ETF_FLOWS_PATH", tmp_path / "etf_flows.json")
    etf_flows.publish([], ok=True)
    state = json.loads((tmp_path / "etf_flows.json").read_text())
    assert state["history_since"] == "2026-08-25"
    assert state["legacy_history_since"] == "2026-07-13"
    assert "QQQ" in state["uncovered"]


def test_issuer_snapshot_is_keyed_by_the_issuers_own_as_of_date(tmp_path, monkeypatch):
    """A re-run before the daily NAV posts must not create a second date holding the
    same numbers, which would difference into a zero flow on a day that never was."""
    parquet = tmp_path / "etf_shares.parquet"
    monkeypatch.setattr(config, "ETF_SHARES_PARQUET", parquet)
    snap = {"SPY": {"shares": 1.059e9, "price": 763.45, "source": "ssga",
                    "noise_usd": 5.3e6, "as_of": "2026-08-24"}}
    etf_flows.append_history(snap)
    hist = etf_flows.append_history(snap)          # same issuer date, run twice
    assert list(hist["date"]) == ["2026-08-24"]


def test_issuer_fetchers_return_nothing_rather_than_raising(monkeypatch):
    from qlib_lab import issuer_shares

    def boom(*a, **k):
        raise TimeoutError("proxy ate it")

    monkeypatch.setattr(issuer_shares, "_get", boom)
    assert issuer_shares.fetch_ssga() == {}
    assert issuer_shares.fetch_ishares() == {}
    assert issuer_shares.fetch_all() == {}


def test_shares_come_from_the_published_assets_over_nav_identity(monkeypatch):
    from qlib_lab import issuer_shares
    ssga = {"data": {"funds": {"etfs": {"datas": [
        {"fundTicker": "SPY\u00ae", "nav": ["$763.45", 763.45],
         "aum": ["$808,670.04 M", 808670.04], "asOfDate": ["Aug 24 2026", "2026-08-24"]},
        {"fundTicker": "BROKEN", "nav": ["n/a", 0.0], "aum": ["n/a", 0.0],
         "asOfDate": ["", ""]},
    ]}}}}
    ishares = {"239454": {"localExchangeTicker": "TLT",
                          "navAmount": {"r": 82.537462}, "navAmountAsOf": {"r": 20260824},
                          "totalNetAssets": {"r": 47153651938.65}}}

    monkeypatch.setattr(issuer_shares, "_get",
                        lambda url, timeout: json.dumps(
                            ssga if "ssga" in url else ishares).encode())
    spy = issuer_shares.fetch_ssga()["SPY"]
    assert round(spy["shares"]) == round(808670.04e6 / 763.45)
    assert spy["as_of"] == "2026-08-24" and spy["source"] == "ssga"
    assert "BROKEN" not in issuer_shares.fetch_ssga()      # nav 0 is dropped, not inf

    tlt = issuer_shares.fetch_ishares()["TLT"]
    assert round(tlt["shares"]) == round(47153651938.65 / 82.537462)
    assert tlt["as_of"] == "2026-08-24"
    # SSGA's cent-rounded NAV is a real dollar noise floor; iShares' is one share
    assert spy["noise_usd"] > 1e6 > tlt["noise_usd"]
