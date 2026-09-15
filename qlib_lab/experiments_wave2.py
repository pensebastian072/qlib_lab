"""Wave-2 pre-registered battery — mechanisms wave 1 didn't touch.

Same contract as experiments.py (fixed parameters, pooled non-overlapping PnLs,
fractional costs, canonical gate). N_TRIALS now counts EVERYTHING tried on this
dataset this session: 4 ML variant-horizons + 15 wave-1 + 11 wave-2 ≈ 30,
rounded up to 32. The bar rises with every idea — that is the scientific cost
of exploration, paid in the deflation term instead of hidden.

  W01 overnight_spy    Lou-Polk-Skouras 2019 — hold SPY close->open only, daily costs
  W02 tug_of_war       Lou et al — XS rank of 60d (overnight - intraday) spread, L/S top8, 21d
  W03 vol_managed_spy  Moreira-Muir 2017 — SPY weight = trailing-median var / var20, cap 2x, 21d
  W04 faber_sma        Faber 2007 GTAA — each ETF long iff close > SMA210, pooled, 21d
  W05 volume_shock     Gervais-Kaniel-Mingelgrin 2001 — log-volume z60 > 2 -> long 5d, event-pooled
  W06 skew_xs          Amaya et al 2015 — 60d realized skew, long low / short high, 21d
  W07 amihud_illiq     Amihud 2002 — 20d mean |ret|/dollar-vol, long illiquid / short liquid, 21d
  W08 range_breakout   range compression (5d avg (H-L)/C in bottom decile of 252d)
                       -> follow sign(mom20), 5d, event-pooled
  W09 halloween        Bouman-Jacobsen 2002 — long SPY Nov-Apr, flat May-Oct, monthly
  W10 pca_residual     Avellaneda-Lee 2010 (lite) — 3-PC factor model on 60d returns,
                       fade 20d residual z beyond +/-1.5, hold 5d, pooled
  W11 corr_cond_mom    Daniel-Moskowitz 2016 spirit — H02 momentum ONLY when avg
                       pairwise corr < trailing 252d median (crash regime filter)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config
from .experiments import (COST_PER_SIDE, TOPK, _cost, _fwd_ret,
                          _ls_portfolio_pnls, _signal_pnls, evaluate_gate)
from .macro_context import RISK_BASKET, _avg_pairwise_corr

logger = logging.getLogger("qlib_lab.experiments_wave2")

N_TRIALS = 32


def panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(close, open, volume) wide panels from the .bin store."""
    from qlib.data import D

    df = D.features(config.ASSETS, ["$close", "$open", "$volume"])
    out = []
    for col in ["$close", "$open", "$volume"]:
        w = df[col].unstack(level=0)
        w.columns = [str(c) for c in w.columns]
        out.append(w.sort_index())
    return tuple(out)


def run_wave2(px: pd.DataFrame, opn: pd.DataFrame, vol: pd.DataFrame) -> dict[str, dict]:
    etfs = [a for a in config.MODEL_UNIVERSE if a in px.columns]
    epx, eopn, evol = px[etfs], opn[etfs], vol[etfs]
    rets = epx.pct_change()
    fwd21, fwd5 = _fwd_ret(epx, 21), _fwd_ret(epx, 5)
    out: dict[str, dict] = {}

    def add(name: str, pnls: list[float], note: str):
        v = evaluate_gate(np.asarray(pnls, dtype=float), n_trials=N_TRIALS)
        v["note"] = note
        v["n"] = len(pnls)
        v["mean_pnl"] = round(float(np.mean(pnls)), 5) if pnls else None
        out[name] = v
        logger.info(f"{name:20s} n={len(pnls):4d} passes={v['passes']} "
                    f"dsr={(v.get('deflated_sharpe') or {}).get('ratio')} "
                    f"pbo={v.get('pbo')} pf={v.get('profit_factor')}")

    spy_c, spy_o = px["SPY"], opn["SPY"]

    # W01 overnight-only SPY: daily close->open, 2 trades/day, 21-session blocks
    on = (spy_o / spy_c.shift(1) - 1) - 2 * COST_PER_SIDE
    blocks = [float(on.iloc[i:i + 21].sum()) for i in range(1, len(on) - 21, 21)]
    add("W01_overnight_spy", blocks, "overnight-only SPY incl daily costs")

    # W02 tug-of-war: XS rank of 60d mean(overnight - intraday)
    on_all = eopn / epx.shift(1) - 1
    intra = epx / eopn - 1
    tug = (on_all - intra).rolling(60).mean()
    add("W02_tug_of_war", _ls_portfolio_pnls(tug, fwd21, 21, 61),
        "overnight-vs-intraday tug of war")

    # W03 vol-managed SPY: w = trailing-252d-median var / var20, capped at 2
    var20 = spy_c.pct_change().rolling(20).var()
    w = (var20.rolling(252).median() / var20).clip(upper=2.0)
    fwd_spy21 = _fwd_ret(spy_c, 21)
    pnls, prev = [], 0.0
    for i in range(273, len(spy_c) - 21, 21):
        t = spy_c.index[i]
        wi, r = w.loc[t], fwd_spy21.loc[t]
        if pd.isna(wi) or pd.isna(r):
            continue
        pnls.append(float(wi) * float(r) - _cost(prev, float(wi)))
        prev = float(wi)
    add("W03_vol_managed_spy", pnls, "Moreira-Muir volatility-managed SPY")

    # W04 Faber 10-month SMA: long each ETF above SMA210, equal-weight, pooled
    sma = epx.rolling(210).mean()
    above = (epx > sma)
    pnls, prev_w = [], pd.Series(dtype=float)
    for i in range(211, len(epx) - 21, 21):
        t = epx.index[i]
        sel = above.loc[t]
        names = sel[sel].index
        r = fwd21.loc[t].reindex(names).dropna()
        wvec = (pd.Series(1.0 / len(names), index=names)
                if len(names) else pd.Series(dtype=float))
        pnl = float(r.mean()) if len(r) else 0.0
        pnls.append(pnl - _cost(prev_w, wvec))
        prev_w = wvec
    add("W04_faber_sma", pnls, "Faber 10m SMA timing, pooled ETF book")

    # W05 volume shock: log-volume z60 > 2 -> long 5 sessions, event-pooled per date
    lv = np.log(evol.replace(0, np.nan))
    vz = (lv - lv.rolling(60).mean()) / lv.rolling(60).std()
    pnls = []
    for i in range(61, len(epx) - 5, 5):
        t = epx.index[i]
        hits = vz.loc[t][vz.loc[t] > 2].index
        r = fwd5.loc[t].reindex(hits).dropna()
        if len(r):
            pnls.append(float(r.mean()) - 2 * COST_PER_SIDE)
    add("W05_volume_shock", pnls, "Gervais high-volume-return premium")

    # W06 realized skew XS: long LOW skew / short HIGH skew
    skew = rets.rolling(60).skew()
    add("W06_skew_xs", _ls_portfolio_pnls(skew, fwd21, 21, 61, invert=True),
        "Amaya realized-skew premium")

    # W07 Amihud illiquidity: long high illiq / short low
    dollar = (evol * epx).replace(0, np.nan)
    illiq = (rets.abs() / dollar).rolling(20).mean()
    add("W07_amihud_illiq", _ls_portfolio_pnls(illiq, fwd21, 21, 21),
        "Amihud illiquidity premium")

    # W08 range compression -> momentum continuation
    from qlib.data import D

    hi = D.features(etfs, ["$high"])["$high"].unstack(level=0)
    lo_ = D.features(etfs, ["$low"])["$low"].unstack(level=0)
    hi.columns = [str(c) for c in hi.columns]
    lo_.columns = [str(c) for c in lo_.columns]
    rng = ((hi - lo_) / epx).rolling(5).mean()
    rng_pct = rng.rolling(252).rank(pct=True)
    mom20 = epx.pct_change(20)
    pnls = []
    for i in range(253, len(epx) - 5, 5):
        t = epx.index[i]
        quiet = rng_pct.loc[t][rng_pct.loc[t] < 0.10].index
        if not len(quiet):
            continue
        sgn = np.sign(mom20.loc[t].reindex(quiet))
        r = fwd5.loc[t].reindex(quiet)
        ok = r.notna() & (sgn != 0)
        if ok.sum():
            pnls.append(float((sgn[ok] * r[ok]).mean()) - 2 * COST_PER_SIDE)
    add("W08_range_breakout", pnls, "range compression -> trend continuation")

    # W09 Halloween: long SPY Nov-Apr, flat May-Oct (monthly blocks)
    sig = pd.Series(np.where(spy_c.index.month.isin([11, 12, 1, 2, 3, 4]), 1.0, 0.0),
                    index=spy_c.index)
    add("W09_halloween", _signal_pnls(sig, fwd_spy21, 21, 1),
        "Bouman-Jacobsen Halloween effect")

    # W10 PCA residual reversion (Avellaneda-Lee lite)
    pnls = []
    vals = rets.values
    idx = rets.index
    for i in range(80, len(rets) - 5, 5):
        win = vals[i - 60:i]
        mask = ~np.isnan(win).any(axis=0)
        if mask.sum() < 20:
            continue
        x = win[:, mask]
        x = (x - x.mean(0)) / (x.std(0) + 1e-12)
        cov = np.cov(x, rowvar=False)
        ev, evec = np.linalg.eigh(cov)
        f = x @ evec[:, -3:]                       # 3 leading factors
        beta, *_ = np.linalg.lstsq(f, x, rcond=None)
        resid = x - f @ beta
        cum20 = resid[-20:].sum(0)
        z = (cum20 - resid.sum(0).mean()) / (resid.std(0).mean() * np.sqrt(20) + 1e-12)
        names = np.array(etfs)[mask]
        stretched = np.abs(z) > 1.5
        if not stretched.any():
            continue
        side = -np.sign(z[stretched])
        r = fwd5.loc[idx[i]].reindex(names[stretched])
        ok = r.notna().values
        if ok.sum():
            pnls.append(float((side[ok] * r.values[ok]).mean()) - 2 * COST_PER_SIDE)
    add("W10_pca_residual", pnls, "PCA factor-residual reversion (stat-arb lite)")

    # W11 corr-conditioned momentum: H02 only in low-correlation regimes
    basket = [c for c in RISK_BASKET if c in px.columns]
    avg_corr = _avg_pairwise_corr(px[basket].pct_change())
    low_corr = avg_corr < avg_corr.rolling(252).median()
    mom = epx.shift(21) / epx.shift(252) - 1
    pnls, prev_w = [], pd.Series(dtype=float)
    for i in range(253, len(epx) - 21, 21):
        t = epx.index[i]
        lc = low_corr.loc[t]
        if pd.isna(lc) or not bool(lc):
            # high-corr/unknown regime: flat (and pay to unwind if held)
            if len(prev_w):
                pnls.append(-_cost(prev_w, pd.Series(dtype=float)))
                prev_w = pd.Series(dtype=float)
            continue
        s = mom.loc[t].dropna()
        if len(s) < 2 * TOPK + 1:
            continue
        ranked = s.sort_values(ascending=False)
        longs, shorts = ranked.index[:TOPK], ranked.index[-TOPK:]
        wvec = pd.concat([pd.Series(0.5 / TOPK, index=longs),
                          pd.Series(-0.5 / TOPK, index=shorts)])
        r = fwd21.loc[t]
        top, bot = r.reindex(longs).dropna(), r.reindex(shorts).dropna()
        if len(top) and len(bot):
            pnls.append(0.5 * float(top.mean()) - 0.5 * float(bot.mean())
                        - _cost(prev_w, wvec))
            prev_w = wvec
    add("W11_corr_cond_mom", pnls, "momentum gated to low-correlation regimes")

    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import qlib
    from qlib.constant import REG_US

    qlib.init(provider_uri=str(config.QLIB_DATA_DIR), region=REG_US)
    px, opn, vol = panels()
    print(f"panel: {px.shape[0]} sessions x {px.shape[1]} assets")
    res = run_wave2(px, opn, vol)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = config.SCORECARDS_DIR / f"experiments_w2_{stamp}.json"
    path.write_text(json.dumps({
        "as_of": datetime.now(timezone.utc).isoformat(),
        "n_trials": N_TRIALS,
        "battery": res,
    }, indent=2, default=str))
    print(f"scorecard -> {path}")
    print(f"\n{'hypothesis':22s} {'n':>4s} {'pass':>5s} {'dsr':>9s} {'pbo':>6s} {'pf':>7s} {'mean':>8s}")
    for name, v in res.items():
        dsr = (v.get("deflated_sharpe") or {}).get("ratio")
        print(f"{name:22s} {v['n']:4d} {str(v['passes']):>5s} "
              f"{dsr if dsr is not None else 'n/a':>9} "
              f"{v.get('pbo') if v.get('pbo') is not None else 'n/a':>6} "
              f"{v.get('profit_factor') if v.get('profit_factor') is not None else 'n/a':>7} "
              f"{v.get('mean_pnl'):>8}")


if __name__ == "__main__":
    main()
