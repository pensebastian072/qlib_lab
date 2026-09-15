"""Real ETF shares outstanding, read from the issuers.

WHY THIS EXISTS. `etf_flows` used yfinance/Yahoo `sharesOutstanding`, which for an
ETF is a STATIC cached field: 19 daily snapshots of 19 funds produced exactly ONE
distinct value per fund, so `delta(shares) x price` was identically zero and always
would be. That is a structural dead end, not a short history -- waiting three months
buys 60 copies of the same constant.

Shares outstanding is recovered from the identity every ETF publishes daily:

    shares = total net assets / NAV per share

Two providers cover 20 of the 23 tracked funds:

  SSGA fund-finder JSON   SPY, GLD, BIL and the 11 Select Sector SPDRs
  iShares screener JSON   IWM, EEM, TLT, TIP, HYG, LQD

QQQ (Invesco), USO and CPER (USCF) have no machine-readable public endpoint that
answers here -- their documented download URLs return the HTML page -- so they report
NOTHING rather than a fabricated number, exactly like the NO_DATA layers in
`asset_frame`.

PRECISION IS PART OF THE MEASUREMENT and is carried per row as `noise_usd`. SSGA
rounds NAV to a cent and AUM to $0.01M, so the implied share count carries an error of
about `shares * 0.005 / nav`, i.e. a dollar noise floor near `aum * 0.005 / nav`. For
SPY that is roughly $5-6m: a real creation day (commonly $100m+) clears it easily, a
quiet day does not and must not be read as flow. iShares publishes full-precision NAV
and total net assets, so its floor is about one share.

Fails safe like every other reader here: a provider that errors, times out or changes
shape contributes no rows and logs at INFO. It never raises.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger("qlib_lab.issuer_shares")

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) qlib_lab/1.0"

SSGA_URL = ("https://www.ssga.com/bin/v1/ssmp/fund/fundfinder"
            "?country=us&language=en&role=intermediary&product=etfs&ui=fund-finder")
ISHARES_URL = ("https://www.ishares.com/us/product-screener/product-screener-v3.1.jsn"
               "?dcrPath=/templatedata/config/product-screener-v3/data/en/us-ishares/"
               "ishares-product-screener-backend-config&siteEntryPassthrough=true")

# Funds whose shares outstanding cannot be read from a public endpoint today. Listed
# so the gap is visible instead of looking like a fetch that quietly returned nothing.
UNCOVERED = {
    "QQQ": "Invesco: the documented CSV download URLs return the HTML page",
    "USO": "USCF: no machine-readable endpoint found",
    "CPER": "USCF: no machine-readable endpoint found",
}

# SSGA rounds NAV to $0.01 and AUM to $0.01M; the NAV rounding dominates.
SSGA_NAV_ROUNDING = 0.005


def _get(url: str, timeout: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _row(ticker, shares, nav, aum_usd, as_of, source, noise_usd) -> dict:
    return {"ticker": ticker, "shares": float(shares), "nav": float(nav),
            "aum_usd": float(aum_usd), "as_of": as_of, "source": source,
            "noise_usd": float(noise_usd)}


def fetch_ssga(timeout: int = 30) -> dict[str, dict]:
    """SPDR funds. `aum` is in $ MILLIONS and `nav` is a 2dp dollar value."""
    try:
        raw = json.loads(_get(SSGA_URL, timeout))
        funds = raw["data"]["funds"]["etfs"]["datas"]
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as e:
        logger.info(f"ssga fetch failed: {e}")
        return {}
    out: dict[str, dict] = {}
    for f in funds:
        try:
            # the registered-mark suffix is part of the published ticker string
            ticker = str(f.get("fundTicker", "")).replace("®", "").strip()
            nav = float(f["nav"][1])
            aum = float(f["aum"][1]) * 1e6
            as_of = str(f["asOfDate"][1])
            if not ticker or nav <= 0 or aum <= 0:
                continue
            shares = aum / nav
            # error in shares ~ shares * (nav rounding / nav); in DOLLARS that is
            # that error times nav, i.e. simply shares * the rounding step
            out[ticker] = _row(ticker, shares, nav, aum, as_of, "ssga",
                               shares * SSGA_NAV_ROUNDING)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return out


def fetch_ishares(timeout: int = 40) -> dict[str, dict]:
    """iShares funds. `totalNetAssets` and `navAmount` both carry full precision in
    their `r` (raw) field; the `d` field is the rounded display string."""
    try:
        raw = json.loads(_get(ISHARES_URL, timeout))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        logger.info(f"ishares fetch failed: {e}")
        return {}
    out: dict[str, dict] = {}
    for f in (raw.values() if isinstance(raw, dict) else []):
        try:
            ticker = str(f.get("localExchangeTicker", "")).strip()
            nav = float(f["navAmount"]["r"])
            tna = float(f["totalNetAssets"]["r"])
            as_of = _iso_date(f.get("navAmountAsOf", {}).get("r"))
            if not ticker or nav <= 0 or tna <= 0:
                continue
            shares = tna / nav
            # full precision published: one share is the honest floor, not zero
            out[ticker] = _row(ticker, shares, nav, tna, as_of, "ishares", nav)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return out


def _iso_date(v) -> str | None:
    """iShares dates arrive as the integer 20260824."""
    s = str(v or "")
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 and s.isdigit() else None


def fetch_all(tickers: list[str] | None = None, timeout: int = 40) -> dict[str, dict]:
    """Merged issuer snapshot. iShares wins a tie only because it is more precise;
    no fund is published by both."""
    rows = {**fetch_ssga(timeout), **fetch_ishares(timeout)}
    if tickers is None:
        return rows
    return {t: rows[t] for t in tickers if t in rows}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = fetch_all()
    print(f"{len(rows)} funds from ssga+ishares")
    for t in ("SPY", "TLT", "XLF", "GLD"):
        if t in rows:
            r = rows[t]
            print(f"  {t:5s} shares={r['shares']:>15,.0f} nav={r['nav']:>9.4f} "
                  f"as_of={r['as_of']} noise=${r['noise_usd']:,.0f} ({r['source']})")
    for t, why in UNCOVERED.items():
        print(f"  {t:5s} NO ISSUER FEED - {why}")


if __name__ == "__main__":
    main()
