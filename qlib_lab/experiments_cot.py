"""Pre-registered COT battery — positioning data, 2006-2026, canonical gate.

Two hypotheses, parameters fixed before results (contrarian thresholds and the
4-week window are the standard choices in the COT literature — Briese, "The
Commitments of Traders Bible"; Wang 2003 on trader-position predictability):

  C1 cot_extreme_fade   speculator crowding mean-reverts: net-position z156
                        < -1.5 -> LONG the asset, > +1.5 -> SHORT, 21d hold,
                        pooled across mapped assets (one PnL per rebalance
                        with >=1 active leg)
  C2 cot_flow_momentum  positioning FLOW leads price: sign of the 4-week
                        change in net %OI -> direction, 21d hold, pooled

Cumulative honest deflation: N_TRIALS=34 (32 prior session trials + these 2).
Prices come straight from Yahoo deep history (p1_epoch=2005) so the test spans
2006-2026 — two full regimes, not just the post-2016 bull.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config, cot
from .data_fetch import fetch_yahoo_ohlcv
from .experiments import COST_PER_SIDE, evaluate_gate

logger = logging.getLogger("qlib_lab.experiments_cot")

N_TRIALS = 34
P1_2005 = 1104537600  # 2005-01-01 UTC
Z_ENTER = 1.5
HOLD = 21


def deep_close_panel(assets: list[str]) -> pd.DataFrame:
    """Daily close panel 2005+ for the COT-mapped assets, from Yahoo."""
    series = {}
    for name in assets:
        sym = config.UNIVERSE.get(name)
        if not sym:
            continue
        df = fetch_yahoo_ohlcv(sym, p1_epoch=P1_2005)
        if df is not None and df.shape[0] > 500:
            series[name] = df["adjclose"].fillna(df["close"])
    return pd.DataFrame(series).sort_index()


def pooled_pnls(px: pd.DataFrame, sig_daily: dict[str, pd.DataFrame],
                col: str, rule) -> list[float]:
    """One pooled costed PnL per 21d rebalance; rule(v) -> position in {-1,0,1}."""
    fwd = px.shift(-HOLD) / px - 1
    dates = px.index
    pnls: list[float] = []
    for i in range(0, len(dates) - HOLD, HOLD):
        t = dates[i]
        legs = []
        for asset, feats in sig_daily.items():
            if asset not in px.columns or t not in feats.index:
                continue
            v = feats.at[t, col]
            r = fwd.at[t, asset] if asset in fwd.columns else np.nan
            if pd.isna(v) or pd.isna(r):
                continue
            pos = rule(float(v))
            if pos:
                legs.append(pos * float(r) - 2 * COST_PER_SIDE)
        if legs:
            pnls.append(float(np.mean(legs)))
    return pnls


def run() -> dict:
    hist = (pd.read_parquet(cot.COT_PARQUET) if cot.COT_PARQUET.exists()
            else cot.update_history())
    if hist is None or hist.empty:
        return {"error": "no COT history available"}
    feats = cot.weekly_features(hist)
    assets = sorted(feats["asset"].unique())
    px = deep_close_panel(assets)
    print(f"price panel: {px.shape[0]} sessions x {px.shape[1]} assets "
          f"({px.index.min().date()}..{px.index.max().date()})")
    sig_daily = cot.daily_panel(feats, px.index, cols=("z156", "chg4w"))

    out = {}
    for name, col, rule, note in [
        ("C1_cot_extreme_fade", "z156",
         lambda z: -1.0 if z > Z_ENTER else (1.0 if z < -Z_ENTER else 0.0),
         "fade speculator crowding at |z156|>1.5"),
        ("C2_cot_flow_momentum", "chg4w",
         lambda c: float(np.sign(c)),
         "follow 4-week positioning flow"),
    ]:
        pnls = pooled_pnls(px, sig_daily, col, rule)
        v = evaluate_gate(np.asarray(pnls, dtype=float), n_trials=N_TRIALS)
        v["note"] = note
        v["n"] = len(pnls)
        v["mean_pnl"] = round(float(np.mean(pnls)), 5) if pnls else None
        out[name] = v
        dsr = (v.get("deflated_sharpe") or {}).get("ratio")
        print(f"{name:22s} n={len(pnls):4d} passes={v['passes']} dsr={dsr} "
              f"pbo={v.get('pbo')} pf={v.get('profit_factor')} mean={v['mean_pnl']}")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = config.SCORECARDS_DIR / f"experiments_cot_{stamp}.json"
    path.write_text(json.dumps({
        "as_of": datetime.now(timezone.utc).isoformat(),
        "n_trials": N_TRIALS,
        "window": [str(px.index.min().date()), str(px.index.max().date())],
        "battery": out,
    }, indent=2, default=str))
    print(f"scorecard -> {path}")
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()


if __name__ == "__main__":
    main()
