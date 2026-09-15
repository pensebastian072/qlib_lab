"""Regime honesty check: do the 2016-2026 PF>1.3 candidates replicate pre-2016?

Re-runs the frozen battery rules (NO parameter changes) on 1995-2015 Yahoo
history — a window containing two bear markets (2000-02, 2008) the training
decade never saw. Primary metric is REPLICATION (PF, mean, sign agreement with
the 2016-2026 result), not gate passage. Assets start when they start (GLD
2004, HYG 2007, CPER 2011); each candidate runs on its available window and
reports it. Also re-runs the VRP conditional split (ctx_vrp keep/drop call).

Output: journal/scorecards/deep_history_<date>.json + printed table.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qlib_lab import config  # noqa: E402
from qlib_lab.data_fetch import fetch_yahoo_ohlcv  # noqa: E402
from qlib_lab.experiments import (COST_PER_SIDE, _cost, _fwd_ret,  # noqa: E402
                                  _signal_pnls, evaluate_gate)
from qlib_lab.macro_context import RISK_BASKET, absorption_series  # noqa: E402

P1_1995 = 788918400          # 1995-01-01 UTC
CUTOFF = "2015-12-31"
# 2016-2026 reference PFs from the wave-1/2 scorecards (for sign agreement)
RECENT_PF = {"H07_vrp_spy": 5.02, "H08_absorption_shift": 1.40,
             "H10_copper_gold_cycl": 1.36, "H13_turn_of_month": 1.37,
             "H15_dxy_gold": 1.30, "W03_vol_managed_spy": 2.18,
             "W04_faber_sma": 1.72, "W09_halloween": 1.69}

PANEL_NAMES = ["SPY", "VIX", "QQQ", "IWM", "DIA", "MDY", "EFA", "EEM", "TLT",
               "LQD", "TIP", "HYG", "GOLD", "OIL", "COPPER", "DXY", "BIL",
               "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]


def deep_panel() -> pd.DataFrame:
    series = {}
    for name in PANEL_NAMES:
        sym = config.UNIVERSE.get(name)
        if not sym:
            continue
        df = fetch_yahoo_ohlcv(sym, p1_epoch=P1_1995)
        if df is not None and df.shape[0] > 250:
            series[name] = df["adjclose"].fillna(df["close"])
    px = pd.DataFrame(series).sort_index()
    return px.loc[:CUTOFF]


def run_candidates(px: pd.DataFrame) -> dict:
    out = {}
    spy = px["SPY"]
    spy_r = spy.pct_change()
    fwd_spy21 = _fwd_ret(spy, 21)

    def add(name, pnls, window_start):
        pnls = [p for p in pnls if not pd.isna(p)]
        v = evaluate_gate(np.asarray(pnls, dtype=float), n_trials=32)
        pf = v.get("profit_factor")
        recent = RECENT_PF.get(name)
        if len(pnls) < 30:
            verdict = "insufficient-history"
        elif pf is not None and pf > 1.0 and recent and recent > 1.0:
            verdict = "replicates"
        else:
            verdict = "regime-specific"
        out[name] = {"n": len(pnls), "pf": pf,
                     "mean_pnl": round(float(np.mean(pnls)), 5) if pnls else None,
                     "pf_2016_2026": recent, "window_start": str(window_start),
                     "verdict": verdict, "gate": {"passes": v["passes"],
                                                  "pbo": v.get("pbo")}}

    # H07 VRP -> long SPY when VRP above trailing median
    rv = spy_r.rolling(20).std() * np.sqrt(252) * 100
    vrp = px["VIX"] ** 2 - rv ** 2
    sig = (vrp > vrp.rolling(252).median()).astype(float)
    add("H07_vrp_spy", _signal_pnls(sig, fwd_spy21, 21, 253), px.index[253].date())

    # VRP conditional split (ctx_vrp keep/drop)
    rows = [(bool(sig.iloc[i]), float(fwd_spy21.iloc[i]))
            for i in range(253, len(spy) - 21, 21) if not pd.isna(fwd_spy21.iloc[i])]
    hi = np.array([r for s, r in rows if s])
    lo = np.array([r for s, r in rows if not s])
    t = ((hi.mean() - lo.mean())
         / np.sqrt(hi.var(ddof=1) / len(hi) + lo.var(ddof=1) / len(lo)))
    out["VRP_split"] = {"hi_mean": round(float(hi.mean()), 5),
                        "lo_mean": round(float(lo.mean()), 5),
                        "welch_t": round(float(t), 2),
                        "n_hi": len(hi), "n_lo": len(lo)}

    # H08 absorption spike -> long TLT/short SPY, else long SPY
    basket = [c for c in RISK_BASKET if c in px.columns]
    ar = absorption_series(px[basket].pct_change(), window=60)
    d_ar = ar.rolling(15).mean() - ar.rolling(60).mean()
    spike = d_ar > d_ar.rolling(252).std()
    fwd_tlt21 = _fwd_ret(px["TLT"], 21)
    pnls, prev = [], 0.0
    first = ar.first_valid_index()
    start_i = max(313, px.index.get_loc(first) + 253) if first is not None else 313
    for i in range(start_i, len(px) - 21, 21):
        tt = px.index[i]
        if pd.isna(spike.loc[tt]) or pd.isna(fwd_spy21.loc[tt]):
            continue
        if bool(spike.loc[tt]) and not pd.isna(fwd_tlt21.loc[tt]):
            pnl, pos = 0.5 * float(fwd_tlt21.loc[tt]) - 0.5 * float(fwd_spy21.loc[tt]), -1.0
        else:
            pnl, pos = float(fwd_spy21.loc[tt]), 1.0
        pnls.append(pnl - _cost(prev, pos))
        prev = pos
    add("H08_absorption_shift", pnls, first.date() if first is not None else "n/a")

    # H09-style leader->follower reruns
    for name, pair, follower, look, hold in [
        ("H10_copper_gold_cycl", ("COPPER", "GOLD"), "IWM", 20, 21),
        ("H15_dxy_gold", ("DXY", None), "GOLD", 20, 21),
    ]:
        a, b = pair
        if a not in px.columns or (b and b not in px.columns) or follower not in px.columns:
            continue
        series = np.log(px[a] / px[b]) if b else np.log(px[a])
        sig = np.sign(series - series.shift(look))
        if name == "H15_dxy_gold":
            sig = -sig
        first = series.first_valid_index()
        add(name, _signal_pnls(sig, _fwd_ret(px[follower], hold), hold, look + 1),
            first.date() if first is not None else "n/a")

    # H13 turn-of-month
    month = pd.Series(px.index.month, index=px.index)
    is_last = month != month.shift(-1)
    fwd4 = spy.shift(-4) / spy - 1
    pnls = [float(fwd4.loc[tt]) - 2 * COST_PER_SIDE
            for tt in px.index[is_last.values] if not pd.isna(fwd4.loc[tt])]
    add("H13_turn_of_month", pnls, px.index[0].date())

    # W03 vol-managed SPY
    var20 = spy_r.rolling(20).var()
    w = (var20.rolling(252).median() / var20).clip(upper=2.0)
    pnls, prev = [], 0.0
    for i in range(273, len(spy) - 21, 21):
        tt = spy.index[i]
        wi, r = w.loc[tt], fwd_spy21.loc[tt]
        if pd.isna(wi) or pd.isna(r):
            continue
        pnls.append(float(wi) * float(r) - _cost(prev, float(wi)))
        prev = float(wi)
    add("W03_vol_managed_spy", pnls, spy.index[273].date())

    # W04 Faber 10m SMA over the available ETF book
    etfs = [c for c in px.columns if c not in ("VIX", "DXY")]
    epx = px[etfs]
    fwd21 = _fwd_ret(epx, 21)
    sma = epx.rolling(210).mean()
    above = epx > sma
    pnls, prev_w = [], pd.Series(dtype=float)
    for i in range(211, len(epx) - 21, 21):
        tt = epx.index[i]
        sel = above.loc[tt]
        names = sel[sel].index
        r = fwd21.loc[tt].reindex(names).dropna()
        wvec = (pd.Series(1.0 / len(names), index=names)
                if len(names) else pd.Series(dtype=float))
        pnls.append((float(r.mean()) if len(r) else 0.0) - _cost(prev_w, wvec))
        prev_w = wvec
    add("W04_faber_sma", pnls, epx.index[211].date())

    # W09 Halloween
    sig = pd.Series(np.where(px.index.month.isin([11, 12, 1, 2, 3, 4]), 1.0, 0.0),
                    index=px.index)
    add("W09_halloween", _signal_pnls(sig, fwd_spy21, 21, 1), px.index[0].date())

    return out


def main():
    px = deep_panel()
    print(f"deep panel: {px.shape[0]} sessions x {px.shape[1]} assets "
          f"({px.index.min().date()}..{px.index.max().date()})")
    res = run_candidates(px)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = config.SCORECARDS_DIR / f"deep_history_{stamp}.json"
    path.write_text(json.dumps({
        "as_of": datetime.now(timezone.utc).isoformat(),
        "window": [str(px.index.min().date()), str(px.index.max().date())],
        "results": res,
    }, indent=2, default=str))
    print(f"scorecard -> {path}\n")
    print(f"{'candidate':22s} {'n':>4s} {'pf<2016':>8s} {'pf 16-26':>9s} {'mean':>9s} verdict")
    for name, r in res.items():
        if name == "VRP_split":
            continue
        print(f"{name:22s} {r['n']:4d} {r['pf'] if r['pf'] is not None else 'n/a':>8} "
              f"{r['pf_2016_2026']:>9} {r['mean_pnl']:>9} {r['verdict']}")
    v = res["VRP_split"]
    print(f"\nVRP split pre-2016: hi={v['hi_mean']:+.4f} (n={v['n_hi']}) "
          f"lo={v['lo_mean']:+.4f} (n={v['n_lo']}) welch_t={v['welch_t']}")


if __name__ == "__main__":
    main()
