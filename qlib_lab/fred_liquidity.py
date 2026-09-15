"""Fed net-liquidity plumbing — the macro driver prices can't see.

    netliq = WALCL (Fed total assets, weekly Wed)         [FRED]
           - TGA   (Treasury General Account, DAILY)      [Treasury DTS]
           - RRPONTSYD (overnight reverse repo, daily)    [FRED]

Rising net liquidity = reserves pushed into the system; falling = drained.

TGA SOURCE (changed 2026-08-31). TGA is the most volatile leg of netliq and was
the only one read at WEEKLY resolution, forward-filled across the week, so $30bn+
intraweek swings were either missed or booked into the wrong week. The Treasury's
own Daily Treasury Statement publishes the same balance DAILY with no API key, so
it is now the primary source. The weekly FRED series is still fetched and is BOTH
the cross-check (`tga_cross_check`) and the fail-safe fallback.

That weekly series is now WDTGAL, not WTREGEN. Caught by the cross-check itself:
against the DTS, WTREGEN showed a median |gap| of $12.9bn and a $215.8bn worst
case on 2025-04-16 (tax week). Reason — WTREGEN is titled "...General Account:
WEEK AVERAGE", while WALCL is a "Wednesday Level". So the old netliq subtracted a
weekly MEAN TGA from a Wednesday-level balance sheet, an apples-to-oranges mix
that predates the DTS work. WDTGAL is the Wednesday LEVEL of the same account and
lines up with both WALCL and the DTS.

LOOKAHEAD RULE: H.4.1 (WALCL/WDTGAL) is published Thursday ~16:30 ET for the
Wednesday as-of date -> effective_date = obs_date + 2 business days. RRP is
next-day -> +1 business day. The DTS posts ~16:00 ET the next business day, and
PreOpenVolDesk runs 08:55 ET, so the DTS is +1 business day too. Never join on
the observation date.

Key: env FRED_API_KEY, else secrets/fred_api_key.txt (gitignored). The key is
never logged. The DTS needs no key. All fetches fail safe; a dead FRED leaves
the last parquet and a stale-marked flag. Advisory only.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger("qlib_lab.fred_liquidity")

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
SERIES = {
    "WALCL": {"lag_bd": 2},      # Wednesday LEVEL, published Thursday
    "WDTGAL": {"lag_bd": 2},     # Wednesday LEVEL of the TGA (NOT WTREGEN, which is
                                 # a week AVERAGE) -> cross-check + fallback only
    "RRPONTSYD": {"lag_bd": 1},  # daily, published next business day
}
START = "2010-01-01"

DTS_URL = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
           "/v1/accounting/dts/operating_cash_balance")
DTS_LAG_BD = 1
DTS_PAGE_SIZE = 10000
DTS_MAX_PAGES = 12           # ~4 rows/day since 2010 fits in 2; 12 is a runaway guard

# TWO TRAPS, both verified against the live API on 2026-08-31.
#
# 1. The DTS has renamed the TGA row twice, and each era ALSO moved the closing
#    balance into a different column:
#       2010-01 .. 2022-04   'Federal Reserve Account'                    close_today_bal
#       ~2022-04 (weeks)     'Treasury General Account (TGA)'             close_today_bal
#       2022-04-18 .. now    'Treasury General Account (TGA) Closing ...' open_today_bal
#    Filtering on today's label alone returns NOTHING before 2022 — a silent
#    12-year hole. Resolved by PRIORITY per record_date below, never by a
#    hardcoded changeover date, so a fourth rename surfaces as a coverage GAP
#    (which `tga_cross_check` catches) instead of a silently stale series.
#
# 2. In the current era `close_today_bal` is the literal STRING "null", not JSON
#    null, on every row — the real closing balance sits in `open_today_bal` of
#    the "Closing Balance" row (checked: each day's Closing == next day's
#    Opening). `pd.to_numeric(..., errors="coerce")` would coerce that to NaN,
#    ffill it, and leave the TGA frozen forever. `_dts_num` handles it.
DTS_TGA_ROWS = (
    ("Treasury General Account (TGA) Closing Balance", "open_today_bal"),
    ("Treasury General Account (TGA)", "close_today_bal"),
    ("Federal Reserve Account", "close_today_bal"),
)
LIQ_PARQUET = config.DATA_DIR / "fred_liquidity.parquet"
LIQ_FLAG = config.FLAGS_DIR / "liquidity_state.json"
KEY_FILE = config.BASE_DIR / "secrets" / "fred_api_key.txt"


def _api_key() -> str | None:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if key:
        return key
    try:
        return KEY_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def fetch_series(series_id: str, timeout: int = 30) -> pd.Series | None:
    """One FRED series (index=obs date) or None. Never raises, never logs the key."""
    key = _api_key()
    if not key:
        logger.info("no FRED api key (env FRED_API_KEY or secrets/fred_api_key.txt)")
        return None
    params = urllib.parse.urlencode({
        "series_id": series_id, "api_key": key, "file_type": "json",
        "observation_start": START,
    })
    try:
        with urllib.request.urlopen(f"{FRED_URL}?{params}", timeout=timeout) as r:
            raw = json.loads(r.read())
        obs = raw["observations"]
    except (urllib.error.URLError, TimeoutError, OSError,
            json.JSONDecodeError, KeyError) as e:
        logger.info(f"fred fetch failed {series_id}: {type(e).__name__}")
        return None
    idx, vals = [], []
    for o in obs:
        v = o.get("value")
        if v in (None, "", "."):
            continue
        idx.append(pd.Timestamp(o["date"]))
        vals.append(float(v))
    if not vals:
        return None
    return pd.Series(vals, index=pd.DatetimeIndex(idx), name=series_id).sort_index()


def _dts_num(raw) -> float | None:
    """Parse one DTS balance cell.

    The API emits the literal string "null" (not JSON null) in columns it is not
    using for that row, so this must reject it explicitly — see trap 2 above.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if text.lower() in ("", "null", "none", "."):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fetch_dts_tga(timeout: int = 30) -> pd.Series | None:
    """Daily TGA closing balance in $mm, indexed by OBSERVATION date. No API key.

    Returns None on any failure (network, schema change, empty) so the caller can
    fall back to the weekly FRED series. Never raises.
    """
    rows: list[dict] = []
    for page in range(1, DTS_MAX_PAGES + 1):
        params = urllib.parse.urlencode({
            "filter": f"record_date:gte:{START}",
            "fields": "record_date,account_type,open_today_bal,close_today_bal",
            "sort": "record_date",
            "page[size]": DTS_PAGE_SIZE,
            "page[number]": page,
        })
        try:
            with urllib.request.urlopen(f"{DTS_URL}?{params}", timeout=timeout) as r:
                batch = json.loads(r.read())["data"]
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError, KeyError, TypeError) as e:
            logger.info(f"dts fetch failed (page {page}): {type(e).__name__}")
            return None
        rows.extend(batch)
        if len(batch) < DTS_PAGE_SIZE:
            break
    else:
        logger.info(f"dts fetch hit the {DTS_MAX_PAGES}-page guard; using what we have")

    # Resolve one balance per date by row priority, not by a changeover date.
    best: dict[pd.Timestamp, tuple[int, float]] = {}
    for row in rows:
        label = row.get("account_type")
        for rank, (want, column) in enumerate(DTS_TGA_ROWS):
            if label != want:
                continue
            value = _dts_num(row.get(column))
            if value is not None:
                ts = pd.Timestamp(row["record_date"])
                if ts not in best or rank < best[ts][0]:
                    best[ts] = (rank, value)
            break

    if not best:
        logger.info("dts: no row matched any known TGA label — schema changed?")
        return None
    return pd.Series({ts: v for ts, (_, v) in best.items()},
                     name="TGA_DTS").sort_index()


def tga_cross_check(dts_mm: pd.Series | None,
                    wtregen_mm: pd.Series | None) -> dict | None:
    """Compare the two TGA sources on the dates BOTH observe.

    Both are the same balance in $mm, so a material gap means one of them is
    being read wrong. Compared on raw OBSERVATION dates (pre-lag, pre-ffill) so
    this measures the sources, not the forward-fill. Reported, never smoothed.
    """
    if dts_mm is None or wtregen_mm is None:
        return None
    both = dts_mm.index.intersection(wtregen_mm.index)
    if len(both) == 0:
        return {"n": 0, "note": "no overlapping observation dates"}
    diff = (dts_mm.reindex(both) - wtregen_mm.reindex(both)).abs() / 1000.0
    worst = diff.idxmax()
    return {
        "n": int(len(both)),
        "median_abs_diff_bn": round(float(diff.median()), 3),
        "max_abs_diff_bn": round(float(diff.max()), 3),
        "worst_date": str(worst.date()),
    }


def build_netliq() -> pd.DataFrame | None:
    """Daily effective-dated frame: walcl, tga, rrp, netliq (+ change z-scores)."""
    series = {}
    raw_obs = {}
    for sid, meta in SERIES.items():
        s = fetch_series(sid)
        if s is None:
            return None
        raw_obs[sid] = s.copy()          # pre-lag, for the cross-check
        # shift to the date the market could actually know the number
        s.index = s.index + pd.offsets.BDay(meta["lag_bd"])
        series[sid] = s[~s.index.duplicated(keep="last")]

    # Daily DTS is primary; weekly WDTGAL is the cross-check and the fallback.
    dts_raw = fetch_dts_tga()
    crosscheck = tga_cross_check(dts_raw, raw_obs.get("WDTGAL"))
    if dts_raw is None:
        tga_source = "wdtgal_weekly_fallback"
        logger.info("dts unavailable — falling back to weekly WDTGAL for TGA")
        dts = None
    else:
        tga_source = "dts_daily"
        dts = dts_raw.copy()
        dts.index = dts.index + pd.offsets.BDay(DTS_LAG_BD)
        dts = dts[~dts.index.duplicated(keep="last")]

    spans = list(series.values()) + ([dts] if dts is not None else [])
    days = pd.bdate_range(min(s.index.min() for s in spans),
                          max(s.index.max() for s in spans))
    # Units: WALCL millions, WDTGAL millions, DTS millions, RRPONTSYD billions.
    # Normalize everything to $bn (FRED verified 2026-07-14, DTS 2026-08-31).
    tga_weekly = series["WDTGAL"].reindex(days).ffill() / 1000.0
    df = pd.DataFrame({
        "walcl": series["WALCL"].reindex(days).ffill() / 1000.0,
        "rrp": series["RRPONTSYD"].reindex(days).ffill().fillna(0.0),
        "tga_weekly": tga_weekly,
    })
    if dts is not None:
        # Cover any stretch the DTS does not reach with the weekly number rather
        # than dropping those dates out of the history entirely.
        df["tga"] = (dts.reindex(days).ffill() / 1000.0).fillna(tga_weekly)
    else:
        df["tga"] = tga_weekly
    df["tga_source"] = tga_source
    df["netliq"] = df["walcl"] - df["tga"] - df["rrp"]
    chg4 = df["netliq"].diff(20)     # ~4 weeks of business days
    chg13 = df["netliq"].diff(65)    # ~13 weeks
    df["netliq_chg4w_z"] = (chg4 - chg4.rolling(756).mean()) / chg4.rolling(756).std()
    df["netliq_chg13w_z"] = (chg13 - chg13.rolling(756).mean()) / chg13.rolling(756).std()
    df["netliq_chg13w"] = chg13
    out = df.dropna(subset=["netliq"])
    # attrs are in-memory only (parquet drops them); publish() is called with the
    # live frame in main(), so the cross-check reaches the flag on the happy path.
    out.attrs["tga_crosscheck"] = crosscheck
    return out


def publish(df: pd.DataFrame | None, ok: bool) -> str:
    state = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "stale": not ok,
        "method": "netliq($bn) = WALCL - TGA - RRP, release-lagged (+2bd/+1bd), "
                  "TGA daily from Treasury DTS",
        "note": None if ok else "FRED fetch failed — last stored history shown",
    }
    if df is not None and not df.empty:
        last = df.iloc[-1]
        state.update({
            "data_through": str(df.index[-1].date()),
            "tga_source": str(last["tga_source"]) if "tga_source" in df.columns else None,
            "tga_crosscheck": df.attrs.get("tga_crosscheck"),
            "netliq_bn": round(float(last["netliq"]), 1),
            "walcl_bn": round(float(last["walcl"]), 1),
            "tga_bn": round(float(last["tga"]), 1),
            "rrp_bn": round(float(last["rrp"]), 1),
            "chg13w_bn": None if pd.isna(last["netliq_chg13w"]) else round(float(last["netliq_chg13w"]), 1),
            "chg13w_z": None if pd.isna(last["netliq_chg13w_z"]) else round(float(last["netliq_chg13w_z"]), 2),
            "trend": ("rising" if last["netliq_chg13w"] > 0 else "falling")
                     if not pd.isna(last["netliq_chg13w"]) else None,
        })
    tmp = LIQ_FLAG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(LIQ_FLAG)
    return str(LIQ_FLAG)


def read_state() -> dict:
    fb = {"as_of": None, "stale": True, "note": "fallback-neutral"}
    try:
        raw = json.loads(LIQ_FLAG.read_text())
    except Exception:  # noqa: BLE001
        return fb
    try:
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(raw["as_of"])).total_seconds() / 86400.0
        if age > 9:
            raw["stale"] = True
    except Exception:  # noqa: BLE001
        raw["stale"] = True
    return raw


def daily_frame() -> pd.DataFrame | None:
    """Stored history for feature building (None if never fetched)."""
    if LIQ_PARQUET.exists():
        try:
            return pd.read_parquet(LIQ_PARQUET)
        except Exception:  # noqa: BLE001
            return None
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = build_netliq()
    if df is None:
        path = publish(daily_frame(), ok=False)
        print(f"FRED fetch FAILED -> stale flag {path}")
        raise SystemExit(0)  # fail-safe, daily job continues
    df.to_parquet(LIQ_PARQUET)
    path = publish(df, ok=True)
    last = df.iloc[-1]
    print(f"netliq history: {len(df)} days ({df.index[0].date()}..{df.index[-1].date()})")
    print(f"latest: netliq ${last['netliq']:.0f}bn = walcl ${last['walcl']:.0f}bn "
          f"- tga ${last['tga']:.0f}bn - rrp ${last['rrp']:.0f}bn | "
          f"13w chg ${last['netliq_chg13w']:+.0f}bn (z={last['netliq_chg13w_z']:+.2f})")
    print(f"tga source: {last['tga_source']}")
    cc = df.attrs.get("tga_crosscheck")
    if cc and cc.get("n"):
        print(f"tga cross-check vs WDTGAL: n={cc['n']} "
              f"median |diff| ${cc['median_abs_diff_bn']}bn, "
              f"max ${cc['max_abs_diff_bn']}bn on {cc['worst_date']}")
    else:
        print("tga cross-check: UNAVAILABLE")
    print(f"flag -> {path}")


if __name__ == "__main__":
    main()
