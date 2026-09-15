"""ETF institutional-flow tracker.

    flow_$ ~= delta(sharesOutstanding) x price

ETF shares outstanding only change through authorized-participant creations and
redemptions -- institutional primary-market activity -- so the day-over-day delta
times price is a dollar creation/redemption estimate.

SOURCE CHANGED 2026-08-25. This used Yahoo `sharesOutstanding`, which for an ETF is a
STATIC cached field: 19 snapshots over five weeks held exactly ONE distinct value per
fund, so every flow was identically zero and always would be. Shares now come from the
issuers via `issuer_shares` (total net assets / NAV), covering 20 of the 23 tracked
funds; QQQ, USO and CPER have no public machine-readable feed and report no flow rather
than a fake one.

THE LEGACY ROWS ARE NOT COMPARABLE. `etf_shares.parquet` still holds the Yahoo-era
snapshots (no `source` column). Differencing across the source change would print one
enormous fictional creation on the first issuer day, so `compute_flows` reads ONLY rows
carrying an issuer source and reports how many legacy rows it ignored. History restarts
at the switch -- honestly, from a field that actually moves.

Every row carries `noise_usd`, the dollar resolution of its own source (SSGA rounds NAV
to a cent, which is ~$5m/day on SPY; iShares publishes full precision). A flow smaller
than that number is rounding, not a creation, and the flag says so.

Everything fails safe -- a failed snapshot leaves history untouched and writes a
stale-marked flag; a reader never sees an exception.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pandas as pd

from . import config, issuer_shares

logger = logging.getLogger("qlib_lab.etf_flows")

Z_WINDOW = 60          # snapshots used for the flow z-score
ISSUER_SOURCES = ("ssga", "ishares")


def fetch_snapshot(timeout: int = 40) -> dict[str, dict] | None:
    """{ticker: {shares, price, source, noise_usd, as_of}} for ETF_MAP, or None.

    `price` is the fund's own NAV, not the last trade: the share count is derived from
    NAV in the first place, so pricing the delta at anything else mixes two valuations.
    Never raises.
    """
    rows = issuer_shares.fetch_all(timeout=timeout)
    out = {t: {"shares": r["shares"], "price": r["nav"], "source": r["source"],
               "noise_usd": r["noise_usd"], "as_of": r["as_of"]}
           for t, r in rows.items() if t in config.ETF_MAP}
    if not out:
        logger.info("no issuer rows returned")
        return None
    missing = sorted(set(config.ETF_MAP) - set(out))
    if missing:
        logger.info(f"no issuer feed for: {', '.join(missing)}")
    return out


def append_history(snap: dict[str, dict]) -> pd.DataFrame:
    """Append today's snapshot to the parquet history (one row per date+ticker)."""
    # the ISSUER's as-of date, not the clock: a re-run on a weekend or before the
    # daily NAV posts would otherwise write a second date holding the same numbers and
    # then difference them into a flow of zero on a day that never happened
    fallback = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new = pd.DataFrame(
        [{"date": v.get("as_of") or fallback, "ticker": t, "shares": v["shares"],
          "price": v["price"], "source": v.get("source"),
          "noise_usd": v.get("noise_usd")}
         for t, v in snap.items()]
    )
    if config.ETF_SHARES_PARQUET.exists():
        hist = pd.read_parquet(config.ETF_SHARES_PARQUET)
        hist = pd.concat([hist, new], ignore_index=True)
    else:
        hist = new
    hist = hist.drop_duplicates(subset=["date", "ticker"], keep="last").sort_values(["ticker", "date"])
    hist.to_parquet(config.ETF_SHARES_PARQUET, index=False)
    return hist


def compute_flows(hist: pd.DataFrame) -> list[dict]:
    """Per-ticker flow rows from the accumulated snapshot history."""
    rows: list[dict] = []
    if "source" not in hist.columns:
        hist = hist.assign(source=None)
    legacy = int((~hist["source"].isin(ISSUER_SOURCES)).sum())
    hist = hist[hist["source"].isin(ISSUER_SOURCES)]
    for ticker, label in config.ETF_MAP.items():
        g = hist[hist["ticker"] == ticker].sort_values("date")
        if g.empty:
            rows.append({"ticker": ticker, "label": label, "price": None,
                         "shares": None, "snapshots": 0,
                         "flow_1d": None, "flow_5d": None, "flow_20d": None,
                         "z": None, "noise_usd": None, "source": None,
                         "note": issuer_shares.UNCOVERED.get(
                             ticker, "no issuer snapshot yet")})
            continue
        last = g.iloc[-1]
        row = {
            "ticker": ticker,
            "label": label,
            "price": round(float(last["price"]), 2),
            "shares": float(last["shares"]),
            "snapshots": int(len(g)),
            "flow_1d": None, "flow_5d": None, "flow_20d": None, "z": None,
            "noise_usd": (None if "noise_usd" not in g.columns
                          or pd.isna(last.get("noise_usd"))
                          else round(float(last["noise_usd"]))),
            "source": last.get("source"),
            "note": None,
        }
        if len(g) < 2:
            row["note"] = "accumulating"
            rows.append(row)
            continue

        def flow_back(k: int) -> float | None:
            if len(g) <= k:
                return None
            ref = g.iloc[-1 - k]
            return float((last["shares"] - ref["shares"]) * last["price"])

        row["flow_1d"] = flow_back(1)
        row["flow_5d"] = flow_back(5)
        row["flow_20d"] = flow_back(20)
        d1 = (g["shares"].diff() * g["price"]).dropna().tail(Z_WINDOW)
        if len(d1) >= 10 and float(d1.std()) > 0:
            row["z"] = round(float((d1.iloc[-1] - d1.mean()) / d1.std()), 2)
        # a move under the source's own rounding step is arithmetic, not a creation
        if (row["noise_usd"] and row["flow_1d"] is not None
                and abs(row["flow_1d"]) < row["noise_usd"]):
            row["note"] = "flow_1d below this source's resolution"
        rows.append(row)
    rows.sort(key=lambda r: abs(r["flow_1d"] or 0), reverse=True)
    if legacy:
        for r in rows:
            r["legacy_rows_ignored"] = legacy
    return rows


NEUTRAL_FALLBACK = {"as_of": None, "stale": True, "rows": [],
                    "note": "fallback-neutral (flag missing/stale/corrupt)"}


def publish(rows: list[dict], ok: bool) -> str:
    state = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "stale": not ok,
        "method": ("delta(shares outstanding) x NAV, shares = total net assets / NAV "
                   "read from the issuer (SSGA / iShares). Yahoo-era rows are excluded: "
                   "that field never moved."),
        "history_since": None,
        "rows": rows,
        "note": None if ok else "snapshot failed — showing last known history",
    }
    if config.ETF_SHARES_PARQUET.exists():
        try:
            hist = pd.read_parquet(config.ETF_SHARES_PARQUET)
            # the Yahoo-era rows go back to 2026-07-13 but are not differenceable
            # against issuer rows, so reporting THEIR start date as the history the
            # flows are computed from would overstate the record by five weeks
            iss = (hist[hist["source"].isin(ISSUER_SOURCES)]
                   if "source" in hist.columns else hist.iloc[0:0])
            state["history_since"] = None if iss.empty else str(iss["date"].min())
            state["legacy_history_since"] = (str(hist["date"].min())
                                             if not hist.empty else None)
            state["uncovered"] = issuer_shares.UNCOVERED
        except Exception:  # noqa: BLE001
            pass
    tmp = config.ETF_FLOWS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(config.ETF_FLOWS_PATH)
    return str(config.ETF_FLOWS_PATH)


def read_state() -> dict:
    """Fail-safe reader. Never raises; neutral on missing/stale."""
    fb = dict(NEUTRAL_FALLBACK)
    try:
        raw = json.loads(config.ETF_FLOWS_PATH.read_text())
    except Exception:  # noqa: BLE001
        return fb
    try:
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(raw["as_of"])).total_seconds() / 86400.0
        if age > config.FLAG_STALE_DAYS:
            raw["stale"] = True
    except Exception:  # noqa: BLE001
        raw["stale"] = True
    return raw


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    snap = fetch_snapshot()
    if snap is None:
        hist = (pd.read_parquet(config.ETF_SHARES_PARQUET)
                if config.ETF_SHARES_PARQUET.exists() else pd.DataFrame(
                    columns=["date", "ticker", "shares", "price"]))
        path = publish(compute_flows(hist) if not hist.empty else [], ok=False)
        print(f"snapshot FAILED — stale flag -> {path}")
        return
    hist = append_history(snap)
    rows = compute_flows(hist)
    path = publish(rows, ok=True)
    print(f"snapshot ok: {len(snap)}/{len(config.ETF_MAP)} tickers -> {path}")
    for r in rows[:8]:
        px = "n/a" if r["price"] is None else f"{r['price']:,.2f}"
        flow = "n/a" if r["flow_1d"] is None else f"{r['flow_1d']:,.0f}"
        print(f"  {r['ticker']:5s} nav={px:>10} snaps={r['snapshots']:>3} "
              f"src={str(r['source']):8s} flow1d={flow:>16} note={r['note']}")


if __name__ == "__main__":
    main()
