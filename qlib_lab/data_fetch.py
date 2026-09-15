"""Fetch OHLCV history and build the qlib .bin store.

Yahoo v8 chart API (period1/period2 + interval=1d — range=max silently returns
MONTHLY bars; hard-won macro_gpu_lab lesson) → per-symbol CSVs in qlib's dump
format → vendored scripts/dump_bin.py → qlib_data/us_data.

Qlib price convention: stored prices are ADJUSTED, with factor = adjclose/close
(adjusted/raw) kept as its own field; volume is divided by factor (mirrors qlib's
own yahoo collector). Every asset is reindexed to SPY's session calendar so the
dump_bin calendar union is exactly the NYSE session set (BTC weekends dropped,
FX holiday extras dropped). Rows an asset is missing stay absent from its CSV —
qlib NaN-fills against the global calendar and LightGBM handles NaN natively.

Every network call fails safe to None; a dropped asset never kills the build.
"""
from __future__ import annotations

import io
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from . import config

logger = logging.getLogger("qlib_lab.data_fetch")

_UA = {"User-Agent": "Mozilla/5.0 qlib_lab/1.0"}
YAHOO_CHART = (
    "https://query1.finance.yahoo.com/v8/finance/chart/"
    "{sym}?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit"
)


def fetch_yahoo_ohlcv(sym: str, timeout: int = 15,
                      p1_epoch: int | None = None) -> pd.DataFrame | None:
    """Daily OHLCV+adjclose frame (index=naive date) or None. Never raises.

    p1_epoch overrides the default DAILY_YEARS window start (deep-history use)."""
    p2 = int(time.time())
    p1 = p1_epoch if p1_epoch is not None else p2 - 3600 * 24 * 365 * config.DAILY_YEARS
    url = YAHOO_CHART.format(sym=sym, p1=p1, p2=p2)
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        logger.info(f"yahoo fetch failed {sym}: {e}")
        return None
    try:
        result = raw["chart"]["result"][0]
        ts = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    except (KeyError, IndexError, TypeError):
        return None
    if not ts:
        return None
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_localize(None).normalize()
    df = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=idx,
    )
    df["adjclose"] = adj if adj is not None else df["close"]
    df = df.dropna(subset=["close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df if not df.empty else None


def to_qlib_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply qlib's adjusted-price convention: prices *= factor, volume /= factor."""
    out = df.copy()
    out["adjclose"] = out["adjclose"].fillna(out["close"])
    factor = (out["adjclose"] / out["close"]).fillna(1.0)
    for col in ("open", "high", "low", "close"):
        out[col] = out[col] * factor
    vol = out["volume"].fillna(0.0)
    out["volume"] = (vol / factor.replace(0.0, 1.0)).round(4)
    out["factor"] = factor
    return out[["open", "high", "low", "close", "volume", "factor"]]


# Yahoo stopped serving history for these two: as of 2026-08-24 ^VIX9D and ^VIX3M
# each return ONE row where ^VIX returns 2512. CBOE publishes them free and complete.
# This is the PUBLIC cdn.cboe.com daily_prices file -- NOT the metered LiveVol API in
# options_desk. It costs nothing and spends no data points (global rule 6).
# Deliberately scoped to the two broken symbols: VIX/VVIX/SKEW still come from Yahoo,
# because swapping a working series would silently change VRP and every percentile
# built on it.
CBOE_INDICES = {"VIX9D", "VIX3M"}
CBOE_HISTORY = ("https://cdn.cboe.com/api/global/us_indices/daily_prices/"
                "{name}_History.csv")


def fetch_cboe_index(name: str, timeout: int = 30) -> pd.DataFrame | None:
    """Daily OHLC for a CBOE index, shaped like fetch_yahoo_ohlcv. Never raises.

    Indices have no volume and no split/dividend adjustment, so volume=0 and
    adjclose=close -- which is exactly what the Yahoo path already produced for
    these symbols (factor 1.0), so the stored CSV shape is unchanged.
    """
    url = CBOE_HISTORY.format(name=name)
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.info(f"cboe fetch failed {name}: {e}")
        return None
    try:
        df = pd.read_csv(io.StringIO(raw))
        df.columns = [c.strip().upper() for c in df.columns]
        df.index = pd.to_datetime(df["DATE"]).dt.normalize()
        out = pd.DataFrame({
            "open": pd.to_numeric(df["OPEN"], errors="coerce"),
            "high": pd.to_numeric(df["HIGH"], errors="coerce"),
            "low": pd.to_numeric(df["LOW"], errors="coerce"),
            "close": pd.to_numeric(df["CLOSE"], errors="coerce"),
            "volume": 0.0,
        }, index=df.index)
    except (KeyError, ValueError, TypeError) as e:
        logger.info(f"cboe parse failed {name}: {e}")
        return None
    out["adjclose"] = out["close"]
    out = out.dropna(subset=["close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    # CBOE serves 0.0 placeholders for the years before a series went live
    out = out[out["close"] > 0]
    return out if not out.empty else None


def merge_with_stored(path, fresh: pd.DataFrame) -> pd.DataFrame:
    """Union fresh rows over stored history, fresh winning on conflict.

    Yahoo intermittently serves a SHORT or GAPPED history for a symbol. Observed
    2026-08-24: ^VIX9D and ^VIX3M each returned ONE row where ^VIX returned 2512, and
    earlier they returned a history missing 2026-07-18..08-18 entirely. A plain
    to_csv overwrite lets a single bad response delete history permanently -- that is
    exactly how the vol surface lost a month and vol_desk started publishing a
    term_ratio built from VIX9D and VIX3M on DIFFERENT dates.

    Merging makes the store monotonic: a bad response can only add, never remove.
    MIN_ASSET_ROWS already rejects the 1-row case, but it cannot catch a response
    that is long enough to pass yet still missing the middle.

    This mirrors cot.py / funding.py / fred_liquidity.py, which have always merged
    against their stored parquet. data_fetch was the one module that did not.
    """
    if not path.exists():
        return fresh

    # Norton transiently locks freshly-written files on this box (see
    # run_market_state.ps1's retry loop). Returning `fresh` on a read failure would
    # silently restore the destructive overwrite this function exists to prevent, so
    # retry, then give up by returning None -- the caller SKIPS the write and leaves
    # the stored file intact. Never fail open into data loss.
    old = None
    for attempt in range(5):
        try:
            old = pd.read_csv(path)
            old.index = pd.to_datetime(old["date"])
            old = old.drop(columns=["date"])
            break
        except (OSError, PermissionError) as e:
            if attempt == 4:
                logger.info(f"stored csv locked, SKIPPING write ({path.name}): {e}")
                return None
            time.sleep(0.3)
        except Exception as e:  # noqa: BLE001  (corrupt/truncated file)
            logger.info(f"stored csv unreadable, SKIPPING write ({path.name}): {e}")
            return None
    if old is None or old.empty:
        return fresh

    # Rebase stored rows onto the FRESH adjustment vintage before merging.
    # Yahoo re-adjusts the whole history on every ex-dividend, so `factor`
    # (=adjclose/close) for a given past date drifts between fetches. Under the old
    # overwrite the file always held one vintage. Merging without rebasing freezes
    # every row that falls out of the DAILY_YEARS window at its vintage-of-the-day,
    # producing a series spliced from many bases -- a spurious ~0.3% return jump per
    # dividend at each seam, permanent and never self-healing. Because `factor` is
    # stored per row, the correction is exact: scale old rows by the ratio of the two
    # vintages at the earliest overlapping date.
    common = old.index.intersection(fresh.index)
    if len(common) and {"factor"} <= set(old.columns) & set(fresh.columns):
        d0 = common.min()
        f_old, f_new = float(old.at[d0, "factor"]), float(fresh.at[d0, "factor"])
        if f_old > 0 and f_new > 0:
            k = f_new / f_old
            if abs(k - 1.0) > 1e-9:
                pre = old.index < d0
                for col in ("open", "high", "low", "close", "factor"):
                    if col in old.columns:
                        old.loc[pre, col] = old.loc[pre, col] * k
                if "volume" in old.columns:
                    old.loc[pre, "volume"] = old.loc[pre, "volume"] / k
                logger.info(f"{path.name}: rebased {int(pre.sum())} stored rows "
                            f"onto the fresh adjustment vintage (k={k:.6f})")

    merged = pd.concat([old, fresh])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


def build_csvs() -> dict:
    """Fetch every universe asset, align to SPY sessions, write csv/<name>.csv."""
    anchor_sym = config.UNIVERSE[config.CALENDAR_ANCHOR]
    anchor = fetch_yahoo_ohlcv(anchor_sym)
    if anchor is None or anchor.shape[0] < config.MIN_ASSET_ROWS:
        return {"ok": False, "note": "anchor (SPY) fetch failed — aborting build"}

    # The session set must come from the MERGED anchor, not the fresh fetch. Stored
    # CSVs predate the current rolling DAILY_YEARS window, so filtering merged frames
    # against a fresh-only session set either truncates real history or -- if the
    # filter is skipped -- lets a symbol whose stored history starts EARLIER than
    # SPY's current window inject dates that are not sessions at all. That already
    # happened: VIX9D/VIX3M carried 2016-08-22..24 while SPY's window starts 08-25,
    # and dump_bin's calendar union grew to 2515 days against SPY's 2512, leaving
    # three phantom sessions where every other instrument is NaN.
    anchor_path = config.CSV_DIR / f"{config.CALENDAR_ANCHOR}.csv"
    anchor_merged = merge_with_stored(anchor_path, to_qlib_frame(anchor))
    if anchor_merged is None:
        return {"ok": False, "note": "anchor csv unreadable — aborting build"}
    sessions = anchor_merged.index

    provenance: dict[str, dict] = {}
    written = 0
    for name, sym in config.UNIVERSE.items():
        if name == config.CALENDAR_ANCHOR:
            df = None            # already fetched, merged and framed above
        elif name in CBOE_INDICES:
            # CBOE first (Yahoo serves 1 row for these); Yahoo only as a fallback.
            # Explicit `is None` -- `or` on a DataFrame raises "truth value is ambiguous".
            df = fetch_cboe_index(name)
            if df is None:
                df = fetch_yahoo_ohlcv(sym)
        else:
            df = fetch_yahoo_ohlcv(sym)
        meta = {"rows": 0, "start": None, "end": None, "kept": False}
        path = config.CSV_DIR / f"{name}.csv"

        if name == config.CALENDAR_ANCHOR:
            q, fetched = anchor_merged, int(anchor.shape[0])
        else:
            if df is not None:
                df = df[df.index.isin(sessions)]
            if df is None or df.shape[0] < config.MIN_ASSET_ROWS:
                provenance[name] = meta
                logger.info(f"{name}: dropped ({0 if df is None else df.shape[0]} rows)")
                continue
            fetched = int(df.shape[0])
            q = merge_with_stored(path, to_qlib_frame(df))
            if q is None:          # stored file locked/corrupt -> leave it alone
                provenance[name] = meta
                continue

        # Re-filter AFTER the merge: stored rows were never session-checked, and a
        # symbol whose stored history starts before the anchor's would otherwise add
        # dates that are not sessions (the 2016-08-22..24 phantom-calendar bug).
        q = q[q.index.isin(sessions)]
        if q.empty:
            provenance[name] = meta
            logger.info(f"{name}: dropped (no rows on the session calendar)")
            continue
        out = q.copy()
        out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
        out.to_csv(path, index=False)
        meta.update(
            rows=int(out.shape[0]),
            fetched=fetched,
            start=out["date"].iloc[0],
            end=out["date"].iloc[-1],
            kept=True,
        )
        if fetched < meta["rows"]:
            logger.info(f"{name}: fetch returned {fetched} rows, kept {meta['rows']} "
                        f"after merge with stored history")
        provenance[name] = meta
        written += 1

    return {
        "ok": written >= 10,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_written": written,
        "dropped": [a for a, m in provenance.items() if not m["kept"]],
        "provenance": provenance,
    }


def dump_bin_all() -> None:
    """Convert csv/ to the qlib .bin store (full rebuild, idempotent)."""
    sys.path.insert(0, str(config.BASE_DIR / "scripts"))
    from dump_bin import DumpDataAll  # vendored from microsoft/qlib

    DumpDataAll(
        data_path=str(config.CSV_DIR),
        qlib_dir=str(config.QLIB_DATA_DIR),
        include_fields="open,high,low,close,volume,factor",
        date_field_name="date",
        file_suffix=".csv",
        max_workers=4,
    ).dump()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    meta = build_csvs()
    (config.DATA_DIR / "fetch_meta.json").write_text(json.dumps(meta, indent=2))
    if not meta.get("ok"):
        print(f"FETCH FAILED: {meta.get('note', meta.get('dropped'))}")
        raise SystemExit(1)
    print(f"csvs written: {meta['n_written']}/{len(config.ASSETS)}  dropped: {meta['dropped']}")
    dump_bin_all()
    cal = config.QLIB_DATA_DIR / "calendars" / "day.txt"
    n_days = len(cal.read_text().splitlines()) if cal.exists() else 0
    print(f"qlib .bin store rebuilt: {config.QLIB_DATA_DIR}  calendar days: {n_days}")


if __name__ == "__main__":
    main()
