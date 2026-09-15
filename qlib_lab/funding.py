"""BTC perp funding rates — daily crowding gauge for leveraged longs.

Funding = what perp longs pay shorts; persistent extremes mean crowded leverage
(crypto-carry literature: Schmeling-Schrimpf-Todorov). Reachability on this box
(probed 2026-07-14): Binance 451 geo-block, Bybit 403, OKX OK but only ~3
months of public history, **Deribit OK with hourly history back to 2019** —
so Deribit BTC-PERPETUAL is the source (hourly interest_1h, month-windowed
pagination, incremental updates), summed to a daily funding total.

A day's funding is fully known at end of day -> features are shift(1): usable
the NEXT day, no lookahead. Features: z156d + 52w percentile. Outputs:
data/funding_history.parquet + journal/flags/funding_state.json. Advisory only.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger("qlib_lab.funding")

DERIBIT_URL = "https://www.deribit.com/api/v2/public/get_funding_rate_history"
INST = "BTC-PERPETUAL"
FUND_PARQUET = config.DATA_DIR / "funding_history.parquet"
FUND_FLAG = config.FLAGS_DIR / "funding_state.json"
HISTORY_START_MS = 1554076800000  # 2019-04-01 — around perp funding launch
WINDOW_MS = 30 * 86400000         # month-sized request windows


def _get(url: str, timeout: int = 20) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 qlib_lab/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        logger.info(f"deribit fetch failed: {type(e).__name__}")
        return None


def fetch_range(start_ms: int, end_ms: int) -> pd.DataFrame | None:
    """Hourly funding records in [start_ms, end_ms), windowed. Partial-safe."""
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        win_end = min(cursor + WINDOW_MS, end_ms)
        params = urllib.parse.urlencode({
            "instrument_name": INST,
            "start_timestamp": str(cursor),
            "end_timestamp": str(win_end),
        })
        body = _get(f"{DERIBIT_URL}?{params}")
        if body is None:
            break  # keep what we have (partial ok)
        for rec in body.get("result") or []:
            rows.append({"ts_ms": int(rec["timestamp"]),
                         "rate": float(rec.get("interest_1h") or 0.0)})
        cursor = win_end
        time.sleep(0.15)
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates(subset="ts_ms").sort_values("ts_ms")
    df["date"] = (pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
                  .dt.tz_localize(None).dt.normalize())
    return df[["date", "ts_ms", "rate"]].reset_index(drop=True)


def update_history() -> pd.DataFrame | None:
    """Incremental: continue from the last stored timestamp."""
    old = pd.read_parquet(FUND_PARQUET) if FUND_PARQUET.exists() else None
    start = (int(old["ts_ms"].max()) + 1) if old is not None and not old.empty \
        else HISTORY_START_MS
    fresh = fetch_range(start, int(time.time() * 1000))
    if fresh is None:
        return old
    merged = fresh if old is None else (
        pd.concat([old, fresh]).drop_duplicates(subset="ts_ms")
        .sort_values("ts_ms").reset_index(drop=True))
    merged.to_parquet(FUND_PARQUET, index=False)
    return merged


def daily_features(hist: pd.DataFrame) -> pd.DataFrame:
    """Daily funding sum + z156d + idx52w, usable from the NEXT day (shift 1)."""
    daily = hist.groupby("date")["rate"].sum().rename("funding_1d")
    df = daily.to_frame()
    df["z156"] = ((df["funding_1d"] - df["funding_1d"].rolling(156, min_periods=60).mean())
                  / df["funding_1d"].rolling(156, min_periods=60).std())
    df["idx52w"] = df["funding_1d"].rolling(364, min_periods=120).rank(pct=True)
    return df.shift(1)  # day t's sum known at end of t -> usable t+1


def publish(feats: pd.DataFrame | None, ok: bool) -> str:
    state = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "stale": not ok,
        "inst": INST,
        "method": "Deribit hourly funding summed daily; usable next day; z over 156d",
        "note": None if ok else "Deribit fetch failed — last stored history shown",
    }
    if feats is not None and not feats.empty:
        valid = feats.dropna(subset=["funding_1d"])
        if not valid.empty:
            last = valid.iloc[-1]
            state.update({
                "data_through": str(valid.index[-1].date()),
                "funding_1d": round(float(last["funding_1d"]), 6),
                "funding_1d_annualized_pct": round(float(last["funding_1d"]) * 365 * 100, 1),
                "z156": None if pd.isna(last["z156"]) else round(float(last["z156"]), 2),
                "idx52w": None if pd.isna(last["idx52w"]) else round(float(last["idx52w"]), 2),
                "read": ("crowded_longs" if not pd.isna(last["z156"]) and last["z156"] > 1.5
                         else "crowded_shorts" if not pd.isna(last["z156"]) and last["z156"] < -1.5
                         else "normal"),
            })
    tmp = FUND_FLAG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(FUND_FLAG)
    return str(FUND_FLAG)


def read_state() -> dict:
    fb = {"as_of": None, "stale": True, "note": "fallback-neutral"}
    try:
        raw = json.loads(FUND_FLAG.read_text())
    except Exception:  # noqa: BLE001
        return fb
    try:
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(raw["as_of"])).total_seconds() / 86400.0
        if age > 3:
            raw["stale"] = True
    except Exception:  # noqa: BLE001
        raw["stale"] = True
    return raw


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    hist = update_history()
    if hist is None or hist.empty:
        path = publish(None, ok=False)
        print(f"funding fetch FAILED -> stale flag {path}")
        raise SystemExit(0)
    feats = daily_features(hist)
    path = publish(feats, ok=True)
    print(f"funding history: {len(hist)} hourly records, "
          f"{hist['date'].min().date()}..{hist['date'].max().date()}")
    valid = feats.dropna(subset=["funding_1d"])
    last = valid.iloc[-1]
    z = last["z156"]
    print(f"latest daily funding {last['funding_1d']*100:.4f}% "
          f"(z156={z if pd.isna(z) else round(float(z),2)}) flag -> {path}")


if __name__ == "__main__":
    main()
