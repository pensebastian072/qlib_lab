"""Wave-3 pre-registered battery — option surface, event drift, funding, liquidity.

Fixed parameters, costed non-overlapping blocks, canonical gate. Cumulative
honest deflation: **N_TRIALS = 42** (34 prior session trials + 8 here).

  V1 ts_contango_timer   Simon-Campasano — long SPY while VIX<VIX3M (contango),
                         flat in backwardation; 21d blocks
  V2 backwardation_riskoff  VIX>VIX3M -> long TLT/short SPY, else long SPY; 21d
  V3 skew_follow         CBOE SKEW z156>+1 -> long SPY 21d, else flat
  V4 vvix_stress         VVIX z156>+1.5 -> flat, else long SPY; 5d blocks
  E1 pre_fomc_drift      Lucca-Moench — long SPY close(T-1)->close(T) on each
                         scheduled FOMC decision day (2011+; event-based)
  E2 nfp_day             long SPY close(T-1)->close(T) on NFP first-Fridays
  F1 funding_extreme_fade  daily BTC funding z156 >+1.5 -> short BTC 5d,
                         <-1.5 -> long BTC 5d (Deribit history 2019+)
  L1 netliq_trend        13w Fed net-liquidity change >0 -> long SPY 21d, else flat

Event studies (E1/E2) also report the non-event-day control mean + welch t —
the anomaly claim is the DIFFERENCE, not the raw mean.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config, funding, macro_calendar
from .data_fetch import fetch_yahoo_ohlcv
from .experiments import COST_PER_SIDE, _cost, _fwd_ret, _signal_pnls, evaluate_gate
from .fred_liquidity import daily_frame as liq_frame

logger = logging.getLogger("qlib_lab.experiments_wave3")

N_TRIALS = 42
P1_1995 = 788918400


def deep_panel(names: list[str]) -> pd.DataFrame:
    series = {}
    for name in names:
        sym = config.UNIVERSE.get(name)
        if not sym:
            continue
        df = fetch_yahoo_ohlcv(sym, p1_epoch=P1_1995)
        if df is not None and df.shape[0] > 250:
            series[name] = df["adjclose"].fillna(df["close"])
    return pd.DataFrame(series).sort_index()


def _welch(a: np.ndarray, b: np.ndarray) -> float:
    return float((a.mean() - b.mean())
                 / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def run() -> dict:
    px = deep_panel(["SPY", "VIX", "VIX3M", "SKEW", "VVIX", "TLT"])
    # keep US SESSION rows only — mixing BTC's 7-day calendar into this panel
    # would turn every "21-row" horizon into ~15 trading days (caught 2026-07-14)
    px = px.loc[px["SPY"].notna()]
    btc = deep_panel(["BTC"]).get("BTC")   # own 7-day calendar, used only by F1
    print(f"panel: {px.shape[0]} sessions x {px.shape[1]} assets "
          f"({px.index.min().date()}..{px.index.max().date()})")
    spy = px["SPY"]
    fwd_spy21 = _fwd_ret(spy, 21)
    fwd_spy5 = _fwd_ret(spy, 5)
    out: dict[str, dict] = {}

    def add(name, pnls, note, window_start=None, extra=None):
        pnls = [p for p in pnls if not pd.isna(p)]
        v = evaluate_gate(np.asarray(pnls, dtype=float), n_trials=N_TRIALS)
        v["note"] = note
        v["n"] = len(pnls)
        v["mean_pnl"] = round(float(np.mean(pnls)), 5) if pnls else None
        if window_start is not None:
            v["window_start"] = str(window_start)
        if extra:
            v.update(extra)
        out[name] = v
        dsr = (v.get("deflated_sharpe") or {}).get("ratio")
        print(f"{name:24s} n={v['n']:4d} passes={v['passes']} dsr={dsr} "
              f"pbo={v.get('pbo')} pf={v.get('profit_factor')} mean={v['mean_pnl']}")

    # ── V1/V2: VIX term structure ────────────────────────────────────
    if "VIX3M" in px.columns:
        ts = px["VIX"] / px["VIX3M"]
        first = ts.first_valid_index()
        start = px.index.get_loc(first) + 1
        contango = (ts < 1.0).astype(float)
        add("V1_ts_contango_timer", _signal_pnls(contango, fwd_spy21, 21, start),
            "long SPY in contango, flat in backwardation", first.date())

        fwd_tlt21 = _fwd_ret(px["TLT"], 21)
        pnls, prev = [], 0.0
        for i in range(start, len(px) - 21, 21):
            t = px.index[i]
            v = ts.loc[t]
            if pd.isna(v) or pd.isna(fwd_spy21.loc[t]):
                continue
            if v > 1.0 and not pd.isna(fwd_tlt21.loc[t]):
                pnl, pos = 0.5 * float(fwd_tlt21.loc[t]) - 0.5 * float(fwd_spy21.loc[t]), -1.0
            else:
                pnl, pos = float(fwd_spy21.loc[t]), 1.0
            pnls.append(pnl - _cost(prev, pos))
            prev = pos
        add("V2_backwardation_riskoff", pnls,
            "backwardation -> long TLT/short SPY, else long SPY", first.date())

    # ── V3: SKEW ─────────────────────────────────────────────────────
    if "SKEW" in px.columns:
        sk = px["SKEW"]
        z = (sk - sk.rolling(156).mean()) / sk.rolling(156).std()
        first = z.first_valid_index()
        sig = (z > 1.0).astype(float)
        add("V3_skew_follow", _signal_pnls(sig, fwd_spy21, 21,
                                           px.index.get_loc(first) + 1),
            "SKEW z156>+1 -> long SPY, else flat", first.date())

    # ── V4: VVIX stress ──────────────────────────────────────────────
    if "VVIX" in px.columns:
        vv = px["VVIX"]
        z = (vv - vv.rolling(156).mean()) / vv.rolling(156).std()
        first = z.first_valid_index()
        sig = (z <= 1.5).astype(float)   # long unless vol-of-vol stressed
        add("V4_vvix_stress", _signal_pnls(sig, fwd_spy5, 5,
                                           px.index.get_loc(first) + 1),
            "flat when VVIX z156>+1.5, else long SPY (5d)", first.date())

    # ── E1/E2: scheduled-event drift (event-based, with control) ─────
    ret1d = spy / spy.shift(1) - 1   # close(T-1)->close(T) measured AT T

    def event_study(name, event_dates, note):
        ev = px.index.intersection(event_dates)
        ev_ret = ret1d.reindex(ev).dropna()
        non = ret1d.drop(ev, errors="ignore").dropna()
        pnls = [float(r) - 2 * COST_PER_SIDE for r in ev_ret]
        add(name, pnls, note, ev.min().date() if len(ev) else None, extra={
            "event_day_mean": round(float(ev_ret.mean()), 5) if len(ev_ret) else None,
            "non_event_mean": round(float(non.mean()), 5),
            "welch_t_vs_control": round(_welch(ev_ret.values, non.values), 2)
            if len(ev_ret) > 10 else None,
        })

    event_study("E1_pre_fomc_drift", macro_calendar.fomc_dates(),
                "long SPY close(T-1)->close(T), scheduled FOMC decisions")
    event_study("E2_nfp_day",
                macro_calendar.nfp_dates(str(px.index.min().date()),
                                         str(px.index.max().date())),
                "long SPY close(T-1)->close(T), NFP first-Fridays")

    # ── F1: BTC funding extreme fade (7-day BTC calendar) ────────────
    if funding.FUND_PARQUET.exists() and btc is not None:
        feats = funding.daily_features(pd.read_parquet(funding.FUND_PARQUET))
        fwd_btc5 = _fwd_ret(btc, 5)
        z = feats["z156"].reindex(btc.index).ffill(limit=3)
        pnls, prev = [], 0.0
        first = z.first_valid_index()
        if first is not None:
            for i in range(btc.index.get_loc(first), len(btc) - 5, 5):
                t = btc.index[i]
                zt, r = z.loc[t], fwd_btc5.loc[t]
                if pd.isna(zt) or pd.isna(r):
                    continue
                pos = -1.0 if zt > 1.5 else (1.0 if zt < -1.5 else 0.0)
                pnls.append(pos * float(r) - _cost(prev, pos))
                prev = pos
            add("F1_funding_extreme_fade", pnls,
                "fade BTC funding extremes |z156|>1.5, 5d", first.date())
    else:
        out["F1_funding_extreme_fade"] = {"passes": False, "n": 0,
                                          "note": "no funding history available"}

    # ── L1: Fed net-liquidity trend ──────────────────────────────────
    liq = liq_frame()
    if liq is not None and "netliq_chg13w" in liq.columns:
        chg = liq["netliq_chg13w"].reindex(spy.index).ffill(limit=5)
        first = chg.first_valid_index()
        sig = (chg > 0).astype(float)
        add("L1_netliq_trend", _signal_pnls(sig, fwd_spy21, 21,
                                            spy.index.get_loc(first) + 1),
            "long SPY while 13w net liquidity rising, else flat", first.date())
    else:
        out["L1_netliq_trend"] = {"passes": False, "n": 0,
                                  "note": "no FRED history available"}

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = config.SCORECARDS_DIR / f"experiments_w3_{stamp}.json"
    path.write_text(json.dumps({
        "as_of": datetime.now(timezone.utc).isoformat(),
        "n_trials": N_TRIALS,
        "battery": out,
    }, indent=2, default=str))
    print(f"scorecard -> {path}")
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()


if __name__ == "__main__":
    main()
