"""CFTC Commitments-of-Traders positioning — a data dimension prices don't have.

Legacy futures-only report, weekly (as-of Tuesday, published Friday ~15:30 ET).
Sources, both keyless + fail-safe:
  primary   Socrata API  https://publicreporting.cftc.gov/resource/6dca-aqww.json
  fallback  annual zips  https://www.cftc.gov/files/dea/history/deacot{YYYY}.zip

Features per mapped asset:
  net_pct   (noncommercial long - short) / open interest
  z156      z-score of net_pct over trailing 156 weeks (>=52 to emit)
  idx52     percentile of net_pct within trailing 52 weeks (COT index)

LOOKAHEAD RULE: a report as-of Tuesday is only usable from the following
Monday — every weekly row gets effective_date = report_date + 6 calendar days
before any daily alignment. Never join on report_date directly.

JPY/CAD futures quote the foreign currency, so their positioning sign is
FLIPPED to align with the USDJPY/USDCAD quote direction used everywhere else.

Outputs: data/cot_history.parquet (full weekly history, backfilled ~2006+),
journal/flags/cot_state.json (atomic, fail-safe reader). Advisory only.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger("qlib_lab.cot")

_UA = {"User-Agent": "Mozilla/5.0 qlib_lab/1.0"}
SOCRATA_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
ZIP_URL = "https://www.cftc.gov/files/dea/history/deacot{year}.zip"
BACKFILL_START_YEAR = 2006
RELEASE_LAG_DAYS = 6      # Tuesday as-of -> usable the following Monday
Z_WEEKS = 156
IDX_WEEKS = 52

# CFTC legacy contract-market code -> (asset name, sign)
COT_MARKETS = {
    "13874A": ("SPY", 1),      # E-mini S&P 500
    "209742": ("QQQ", 1),      # E-mini Nasdaq-100
    "239742": ("IWM", 1),      # E-mini Russell 2000
    "043602": ("TLT", 1),      # 10-Year T-Note (duration proxy)
    "088691": ("GOLD", 1),     # Gold COMEX
    "085692": ("COPPER", 1),   # Copper COMEX
    "067651": ("OIL", 1),      # WTI NYMEX
    "099741": ("EURUSD", 1),   # Euro FX
    "097741": ("USDJPY", -1),  # JPY futures (flip to USDJPY direction)
    "096742": ("GBPUSD", 1),   # British Pound
    "090741": ("USDCAD", -1),  # CAD futures (flip to USDCAD direction)
    "1170E1": ("VIX", 1),      # VIX futures
    "133741": ("BTC", 1),      # CME Bitcoin
}


# ── fetchers (fail-safe -> None) ─────────────────────────────────────

def fetch_socrata(timeout: int = 30) -> pd.DataFrame | None:
    codes = ",".join(f"'{c}'" for c in COT_MARKETS)
    params = {
        "$select": ("cftc_contract_market_code,report_date_as_yyyy_mm_dd,"
                    "noncomm_positions_long_all,noncomm_positions_short_all,"
                    "open_interest_all"),
        "$where": (f"cftc_contract_market_code in({codes}) AND "
                   f"report_date_as_yyyy_mm_dd >= '{BACKFILL_START_YEAR}-01-01'"),
        "$order": "report_date_as_yyyy_mm_dd",
        "$limit": "60000",
    }
    url = SOCRATA_URL + "?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            rows = json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        logger.info(f"socrata fetch failed: {e}")
        return None
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "cftc_contract_market_code": "code",
        "report_date_as_yyyy_mm_dd": "report_date",
        "noncomm_positions_long_all": "nc_long",
        "noncomm_positions_short_all": "nc_short",
        "open_interest_all": "oi",
    })
    return _normalize(df)


def fetch_zips(years: range | None = None, timeout: int = 60) -> pd.DataFrame | None:
    """Annual deacot zips (legacy combined report file). Slow path."""
    years = years or range(BACKFILL_START_YEAR, datetime.now().year + 1)
    frames = []
    for year in years:
        url = ZIP_URL.format(year=year)
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                blob = r.read()
            zf = zipfile.ZipFile(io.BytesIO(blob))
            name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
            text = zf.read(name).decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            logger.info(f"zip fetch failed {year}: {e}")
            continue
        rdr = csv.reader(io.StringIO(text))
        header = next(rdr, None)
        if not header:
            continue
        idx = {h.strip().lower(): i for i, h in enumerate(header)}

        def col(row, *names):
            for n in names:
                i = idx.get(n)
                if i is not None and i < len(row):
                    return row[i]
            return None

        for row in rdr:
            code = (col(row, "cftc contract market code") or "").strip()
            if code not in COT_MARKETS:
                continue
            frames.append({
                "code": code,
                "report_date": col(row, "as of date in form yyyy-mm-dd",
                                   "report date as yyyy-mm-dd"),
                "nc_long": col(row, "noncommercial positions-long (all)"),
                "nc_short": col(row, "noncommercial positions-short (all)"),
                "oi": col(row, "open interest (all)"),
            })
    if not frames:
        return None
    return _normalize(pd.DataFrame(frames))


def _normalize(df: pd.DataFrame) -> pd.DataFrame | None:
    df = df.dropna(subset=["code", "report_date"])
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.normalize()
    for c in ("nc_long", "nc_short", "oi"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["report_date", "nc_long", "nc_short", "oi"])
    df = df[df["oi"] > 0]
    df["asset"] = df["code"].map({k: v[0] for k, v in COT_MARKETS.items()})
    sign = df["code"].map({k: v[1] for k, v in COT_MARKETS.items()})
    df["net_pct"] = sign * (df["nc_long"] - df["nc_short"]) / df["oi"]
    out = (df[["report_date", "asset", "net_pct", "oi"]]
           .drop_duplicates(subset=["report_date", "asset"], keep="last")
           .sort_values(["asset", "report_date"]).reset_index(drop=True))
    return out if not out.empty else None


# ── history + features ───────────────────────────────────────────────

COT_PARQUET = config.DATA_DIR / "cot_history.parquet"
COT_FLAG = config.FLAGS_DIR / "cot_state.json"


def update_history() -> pd.DataFrame | None:
    """Fetch (socrata -> zip fallback), merge with stored history, persist."""
    fresh = fetch_socrata()
    if fresh is None:
        logger.info("socrata failed; trying annual zips")
        fresh = fetch_zips()
    if fresh is None and COT_PARQUET.exists():
        return pd.read_parquet(COT_PARQUET)
    if fresh is None:
        return None
    if COT_PARQUET.exists():
        old = pd.read_parquet(COT_PARQUET)
        fresh = (pd.concat([old, fresh])
                 .drop_duplicates(subset=["report_date", "asset"], keep="last")
                 .sort_values(["asset", "report_date"]).reset_index(drop=True))
    fresh.to_parquet(COT_PARQUET, index=False)
    return fresh


def weekly_features(hist: pd.DataFrame) -> pd.DataFrame:
    """Per (asset, effective_date): net_pct, z156, idx52, wow change."""
    rows = []
    for asset, g in hist.groupby("asset"):
        g = g.sort_values("report_date")
        net = g["net_pct"].reset_index(drop=True)
        z = (net - net.rolling(Z_WEEKS, min_periods=52).mean()) / \
            net.rolling(Z_WEEKS, min_periods=52).std()
        idx = net.rolling(IDX_WEEKS, min_periods=26).rank(pct=True)
        wow4 = net.diff(4)
        eff = g["report_date"].reset_index(drop=True) + timedelta(days=RELEASE_LAG_DAYS)
        rows.append(pd.DataFrame({
            "asset": asset, "report_date": g["report_date"].values,
            "effective_date": eff.values, "net_pct": net.values,
            "z156": z.values, "idx52": idx.values, "chg4w": wow4.values,
        }))
    return pd.concat(rows, ignore_index=True)


def daily_panel(feats: pd.DataFrame, sessions: pd.DatetimeIndex,
                cols=("z156", "idx52")) -> dict[str, pd.DataFrame]:
    """{asset: DataFrame(sessions x cols)} — ffilled from effective dates."""
    out = {}
    for asset, g in feats.groupby("asset"):
        g = g.sort_values("effective_date").set_index("effective_date")
        g = g[~g.index.duplicated(keep="last")]
        out[asset] = g[list(cols)].reindex(
            g.index.union(sessions)).ffill().reindex(sessions)
    return out


def per_instrument_features(index: pd.MultiIndex) -> pd.DataFrame | None:
    """COT columns aligned to an (instrument, datetime) feature index.

    Mapped instruments get cot_z156/cot_idx52 (release-lagged, ffilled);
    everything else stays NaN (LightGBM handles missing natively). Fail-safe:
    no stored history -> None (caller skips the merge).
    """
    if not COT_PARQUET.exists():
        return None
    try:
        feats = weekly_features(pd.read_parquet(COT_PARQUET))
    except Exception as e:  # noqa: BLE001
        logger.info(f"cot history unreadable: {e}")
        return None
    sessions = pd.DatetimeIndex(index.get_level_values(1).unique()).sort_values()
    panels = daily_panel(feats, sessions, cols=("z156", "idx52"))
    out = pd.DataFrame(index=index, columns=["cot_z156", "cot_idx52"], dtype=float)
    instruments = index.get_level_values(0)
    dates = index.get_level_values(1)
    for asset, p in panels.items():
        mask = instruments == asset
        if not mask.any():
            continue
        out.loc[mask, "cot_z156"] = p["z156"].reindex(dates[mask]).values
        out.loc[mask, "cot_idx52"] = p["idx52"].reindex(dates[mask]).values
    return out


# ── flag publish + fail-safe reader ──────────────────────────────────

def publish(feats: pd.DataFrame | None, ok: bool) -> str:
    rows = []
    report_date = None
    if feats is not None and not feats.empty:
        latest = feats.sort_values("report_date").groupby("asset").tail(1)
        report_date = str(latest["report_date"].max().date())
        for _, r in latest.iterrows():
            rows.append({
                "asset": r["asset"],
                "net_pct_oi": None if pd.isna(r["net_pct"]) else round(float(r["net_pct"]), 4),
                "z156": None if pd.isna(r["z156"]) else round(float(r["z156"]), 2),
                "idx52": None if pd.isna(r["idx52"]) else round(float(r["idx52"]), 2),
                "chg4w": None if pd.isna(r["chg4w"]) else round(float(r["chg4w"]), 4),
            })
        rows.sort(key=lambda r: -abs(r["z156"] or 0))
    state = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "stale": not ok,
        "report_date": report_date,
        "method": "CFTC legacy futures-only, noncommercial net %OI; "
                  f"usable report_date+{RELEASE_LAG_DAYS}d",
        "rows": rows,
        "note": None if ok else "fetch failed — last stored history shown",
    }
    tmp = COT_FLAG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(COT_FLAG)
    return str(COT_FLAG)


def read_state() -> dict:
    """Fail-safe reader. Never raises; neutral on missing/stale."""
    fb = {"as_of": None, "stale": True, "rows": [],
          "note": "fallback-neutral (flag missing/stale/corrupt)"}
    try:
        raw = json.loads(COT_FLAG.read_text())
    except Exception:  # noqa: BLE001
        return fb
    try:
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(raw["as_of"])).total_seconds() / 86400.0
        if age > 9:  # weekly cadence: >9 days = missed a release
            raw["stale"] = True
    except Exception:  # noqa: BLE001
        raw["stale"] = True
    return raw


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    hist = update_history()
    if hist is None:
        path = publish(None, ok=False)
        print(f"COT fetch FAILED (both sources) -> stale flag {path}")
        raise SystemExit(0)  # fail-safe, not fatal: daily job continues
    feats = weekly_features(hist)
    path = publish(feats, ok=True)
    n_weeks = hist.groupby("asset")["report_date"].count()
    print(f"cot history: {len(hist)} rows, {hist['asset'].nunique()} assets, "
          f"{hist['report_date'].min().date()}..{hist['report_date'].max().date()}")
    print(f"weeks per asset: min={int(n_weeks.min())} max={int(n_weeks.max())}")
    print(f"flag -> {path}")
    latest = feats.sort_values("report_date").groupby("asset").tail(1)
    for _, r in latest.sort_values("z156", key=lambda s: -s.abs()).head(6).iterrows():
        print(f"  {r['asset']:7s} net%OI={r['net_pct']:+.3f} z156="
              f"{r['z156'] if not pd.isna(r['z156']) else float('nan'):+.2f} "
              f"idx52={r['idx52'] if not pd.isna(r['idx52']) else float('nan'):.2f}")


if __name__ == "__main__":
    main()
