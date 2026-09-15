"""Pre-registered anomaly battery — the scientific way out of the Alpha158 box.

Every hypothesis below is FIXED before looking at results: a published anomaly,
its canonical parameters, and one implementation. No tuning loops, no picking
the best lookback after the fact. The whole battery counts in the Deflated
Sharpe n_trials (N_TRIALS below includes the session's earlier ML variants),
so a survivor has to clear a much higher bar than a lone backtest — that is
the point. A survivor still stays SHADOW (promotion is a human env flip).

Harness contract (same ethos as pipeline.py):
  - one PnL per rebalance/event, NON-overlapping (step = holding period),
  - portfolio-level pooling (never per-asset pseudo-replication),
  - costs as a fraction of notional per side, charged on position change,
  - gate = macro_gpu_lab.validate.evaluate_gate, never re-ported.

Hypotheses (paper, fixed params):
  H01 tsmom_12_1        Moskowitz-Ooi-Pedersen 2012 — sign(12m-1m mom), all ETFs, 21d rebal
  H02 xs_mom_12_1       Jegadeesh-Titman 1993 — rank 12-1 mom, top8/bot8, 21d
  H03 xs_reversal_5d    Lehmann 1990 — rank 5d return, LONG losers/SHORT winners, 5d
  H04 low_vol           Ang et al 2006 — rank vol60, long low8/short high8, 21d
  H05 bab_beta          Frazzini-Pedersen 2014 (lite) — rank beta60 vs SPY, long low8/short high8, 21d
  H06 high_52w          George-Hwang 2004 — rank close/252d-max, top8/bot8, 21d
  H07 vrp_spy           Bollerslev-Tauchen-Zhou 2009 — VIX^2 - RV20^2 > trailing-252d median -> long SPY 21d, else flat
  H08 absorption_shift  Kritzman et al 2011 — AR(15d mean) - AR(60d mean) > +1 trailing sigma -> long TLT/short SPY 21d, else long SPY
  H09 credit_leads      HYG/LQD 20d mom sign -> SPY direction, 5d
  H10 copper_gold_cycl  COPPER/GOLD 20d mom sign -> IWM direction, 21d
  H11 semis_lead_qqq    SMH-vs-SPY 10d rel-strength sign -> QQQ direction, 5d
  H12 transports_dow    IYT 20d mom sign -> XLI direction, 21d (Dow theory)
  H13 turn_of_month     Lakonishok-Smidt 1988 — long SPY last session..+3, else flat
  H14 pair_z_reversion  config.SPREADS-style macro pairs, |z120|>2 -> fade, hold 21d, pooled
  H15 dxy_gold          dollar factor — DXY 20d mom < 0 -> long GOLD, else short GOLD, 21d
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config
from .macro_context import RISK_BASKET, absorption_series

logger = logging.getLogger("qlib_lab.experiments")

sys.path.insert(0, str(config.MACRO_GPU_LAB_DIR))
from macro_gpu_lab.validate import evaluate_gate  # noqa: E402  (canonical gate)

COST_PER_SIDE = 0.0005
# Honest deflation: 15 battery strategies + the 4 LightGBM variant-horizon
# trials already run on this dataset this session, rounded up.
N_TRIALS = 20
TOPK = 8

PAIRS = {
    "copper_gold": ("COPPER", "GOLD"),
    "qqq_tlt":     ("QQQ", "TLT"),
    "gold_dxy":    ("GOLD", "DXY"),
    "spy_hyg":     ("SPY", "HYG"),
    "tip_tlt":     ("TIP", "TLT"),
    "lqd_hyg":     ("LQD", "HYG"),
    "eem_spy":     ("EEM", "SPY"),
    "xlf_spy":     ("XLF", "SPY"),
    "iwm_spy":     ("IWM", "SPY"),
}


# ── data ─────────────────────────────────────────────────────────────

def close_panel() -> pd.DataFrame:
    from qlib.data import D

    df = D.features(config.ASSETS, ["$close"])
    wide = df["$close"].unstack(level=0)
    wide.columns = [str(c) for c in wide.columns]
    return wide.sort_index()


def _fwd_ret(px: pd.DataFrame | pd.Series, h: int) -> pd.DataFrame | pd.Series:
    return px.shift(-h) / px - 1


def _cost(prev_w: pd.Series | float, w: pd.Series | float) -> float:
    """Cost = per-side rate x total turnover (fraction of gross notional)."""
    if np.isscalar(w):
        turn = abs(float(w) - float(prev_w))
        return COST_PER_SIDE * turn
    joined = pd.concat([prev_w, w], axis=1).fillna(0.0)
    return COST_PER_SIDE * float((joined.iloc[:, 1] - joined.iloc[:, 0]).abs().sum())


def _ls_portfolio_pnls(score: pd.DataFrame, fwd: pd.DataFrame, hold: int,
                       start: int, invert: bool = False) -> list[float]:
    """Rank-based long/short top-K portfolio, one pooled PnL per rebalance."""
    dates = score.index
    pnls: list[float] = []
    prev_w = pd.Series(dtype=float)
    for i in range(start, len(dates) - hold, hold):
        t = dates[i]
        s = score.loc[t].dropna()
        if invert:
            s = -s
        if len(s) < 2 * TOPK + 1:
            continue
        r = fwd.loc[t]
        ranked = s.sort_values(ascending=False)
        longs, shorts = ranked.index[:TOPK], ranked.index[-TOPK:]
        w = pd.concat([pd.Series(0.5 / TOPK, index=longs),
                       pd.Series(-0.5 / TOPK, index=shorts)])
        top, bot = r.reindex(longs).dropna(), r.reindex(shorts).dropna()
        if not (len(top) and len(bot)):
            continue
        gross = 0.5 * float(top.mean()) - 0.5 * float(bot.mean())
        pnls.append(gross - _cost(prev_w, w))
        prev_w = w
    return pnls


def _signal_pnls(sig: pd.Series, fwd: pd.Series, hold: int, start: int) -> list[float]:
    """Single-asset directional strategy: position in {-1,0,+1}, one PnL per block."""
    dates = sig.index
    pnls: list[float] = []
    prev = 0.0
    for i in range(start, len(dates) - hold, hold):
        t = dates[i]
        pos = sig.loc[t]
        r = fwd.loc[t]
        if pd.isna(pos) or pd.isna(r):
            continue
        pnls.append(float(pos) * float(r) - _cost(prev, float(pos)))
        prev = float(pos)
    return pnls


# ── the battery ──────────────────────────────────────────────────────

def run_battery(px: pd.DataFrame) -> dict[str, dict]:
    etfs = [a for a in config.MODEL_UNIVERSE if a in px.columns]
    epx = px[etfs]
    rets = epx.pct_change()
    out: dict[str, dict] = {}

    def add(name: str, pnls: list[float], note: str):
        v = evaluate_gate(np.asarray(pnls, dtype=float), n_trials=N_TRIALS)
        v["note"] = note
        v["n"] = len(pnls)
        mean = float(np.mean(pnls)) if pnls else None
        v["mean_pnl"] = None if mean is None else round(mean, 5)
        out[name] = v
        logger.info(f"{name:18s} n={len(pnls):4d} passes={v['passes']} "
                    f"dsr={(v.get('deflated_sharpe') or {}).get('ratio')} "
                    f"pbo={v.get('pbo')} pf={v.get('profit_factor')}")

    # H01 TSMOM: sign of 12-1 momentum, every ETF, equal-weight pooled, 21d
    mom = epx.shift(21) / epx.shift(252) - 1
    fwd21, fwd5 = _fwd_ret(epx, 21), _fwd_ret(epx, 5)
    pnls = []
    prev_w = pd.Series(dtype=float)
    for i in range(253, len(epx) - 21, 21):
        t = epx.index[i]
        s = np.sign(mom.loc[t].dropna())
        r = fwd21.loc[t].reindex(s.index)
        ok = r.notna() & (s != 0)
        if ok.sum() < 10:
            continue
        w = (s[ok] / ok.sum())
        pnls.append(float((s[ok] * r[ok]).mean()) - _cost(prev_w, w))
        prev_w = w
    add("H01_tsmom_12_1", pnls, "Moskowitz 2012 time-series momentum")

    # H02 cross-sectional 12-1 momentum
    add("H02_xs_mom_12_1", _ls_portfolio_pnls(mom, fwd21, 21, 253),
        "Jegadeesh-Titman cross-sectional momentum")

    # H03 5d reversal (long losers)
    rev = epx.pct_change(5)
    add("H03_xs_reversal_5d", _ls_portfolio_pnls(rev, fwd5, 5, 60, invert=True),
        "Lehmann short-term reversal")

    # H04 low-vol (long low vol)
    vol60 = rets.rolling(60).std()
    add("H04_low_vol", _ls_portfolio_pnls(vol60, fwd21, 21, 61, invert=True),
        "Ang low-volatility anomaly")

    # H05 BAB-lite (long low beta)
    spy_r = px["SPY"].pct_change()
    cov = rets.rolling(60).cov(spy_r)
    beta = cov.div(spy_r.rolling(60).var(), axis=0)
    add("H05_bab_beta", _ls_portfolio_pnls(beta, fwd21, 21, 61, invert=True),
        "Frazzini-Pedersen betting-against-beta (lite)")

    # H06 52-week-high proximity (long nearest high)
    prox = epx / epx.rolling(252).max()
    add("H06_high_52w", _ls_portfolio_pnls(prox, fwd21, 21, 253),
        "George-Hwang 52-week high")

    # H07 variance risk premium -> SPY
    if "VIX" in px.columns:
        rv = spy_r.rolling(20).std() * np.sqrt(252) * 100
        vrp = px["VIX"] ** 2 - rv ** 2
        sig = (vrp > vrp.rolling(252).median()).astype(float)
        add("H07_vrp_spy", _signal_pnls(sig, _fwd_ret(px["SPY"], 21), 21, 253),
            "Bollerslev variance risk premium")

    # H08 absorption spike -> risk-off (long TLT - short SPY), else long SPY
    basket = [c for c in RISK_BASKET if c in px.columns]
    ar = absorption_series(px[basket].pct_change(), window=60)
    d_ar = ar.rolling(15).mean() - ar.rolling(60).mean()
    thr = d_ar.rolling(252).std()
    spike = d_ar > thr
    fwd_spy21 = _fwd_ret(px["SPY"], 21)
    fwd_tlt21 = _fwd_ret(px["TLT"], 21)
    pnls = []
    prev = 0.0
    for i in range(313, len(px) - 21, 21):
        t = px.index[i]
        if pd.isna(spike.loc[t]) or pd.isna(fwd_spy21.loc[t]):
            continue
        if bool(spike.loc[t]) and not pd.isna(fwd_tlt21.loc[t]):
            pnl = 0.5 * float(fwd_tlt21.loc[t]) - 0.5 * float(fwd_spy21.loc[t])
            pos = -1.0
        else:
            pnl = float(fwd_spy21.loc[t])
            pos = 1.0
        pnls.append(pnl - _cost(prev, pos))
        prev = pos
    add("H08_absorption_shift", pnls, "Kritzman absorption-ratio spike")

    # H09-H12, H15: leader momentum sign -> follower direction
    for name, leader_expr, follower, look, hold, note in [
        ("H09_credit_leads", ("HYG", "LQD"), "SPY", 20, 5, "credit leads equity"),
        ("H10_copper_gold_cycl", ("COPPER", "GOLD"), "IWM", 20, 21, "copper/gold leads cyclicals"),
        ("H11_semis_lead_qqq", ("SMH", "SPY"), "QQQ", 10, 5, "semis lead tech"),
        ("H12_transports_dow", ("IYT", None), "XLI", 20, 21, "Dow theory transports"),
        ("H15_dxy_gold", ("DXY", None), "GOLD", 20, 21, "dollar momentum vs gold (inverted)"),
    ]:
        a, b = leader_expr
        if a not in px.columns or (b and b not in px.columns) or follower not in px.columns:
            continue
        series = np.log(px[a] / px[b]) if b else np.log(px[a])
        sig = np.sign(series - series.shift(look))
        if name == "H15_dxy_gold":
            sig = -sig
        add(name, _signal_pnls(sig, _fwd_ret(px[follower], hold), hold, look + 1), note)

    # H13 turn-of-month: long SPY last session..+3 each month, else flat
    idx = px.index
    month = pd.Series(idx.month, index=idx)
    is_last = month != month.shift(-1)          # last session of its month
    fwd4 = px["SPY"].shift(-4) / px["SPY"] - 1  # hold 4 sessions
    pnls = [float(fwd4.loc[t]) - 2 * COST_PER_SIDE
            for t in idx[is_last.values] if not pd.isna(fwd4.loc[t])]
    add("H13_turn_of_month", pnls, "Lakonishok-Smidt turn-of-month (event-based)")

    # H14 pair z-reversion, pooled across macro pairs
    pnls_by_date: dict = {}
    for pname, (a, b) in PAIRS.items():
        if a not in px.columns or b not in px.columns:
            continue
        spread = np.log(px[a] / px[b])
        z = (spread - spread.rolling(120).mean()) / spread.rolling(120).std()
        fa, fb = _fwd_ret(px[a], 21), _fwd_ret(px[b], 21)
        for i in range(121, len(px) - 21, 21):
            t = px.index[i]
            zt = z.loc[t]
            if pd.isna(zt) or abs(zt) < 2 or pd.isna(fa.loc[t]) or pd.isna(fb.loc[t]):
                continue
            side = -np.sign(zt)  # fade the stretch
            pnl = side * 0.5 * (float(fa.loc[t]) - float(fb.loc[t])) - 2 * COST_PER_SIDE
            pnls_by_date.setdefault(t, []).append(pnl)
    pooled = [float(np.mean(v)) for _, v in sorted(pnls_by_date.items())]
    add("H14_pair_z_reversion", pooled, "macro-pair divergence fade (pooled)")

    return out


# ── lead-lag relationship miner (split-half validated) ───────────────

def mine_lead_lag(px: pd.DataFrame, lags=(1, 2, 3, 5), top_n: int = 25) -> list[dict]:
    """Scan leader->follower lagged correlations with split-half validation.

    Discovery = first half of the sample, validation = second half. A pair is
    reported only if |corr| clears a Bonferroni-ish z threshold in the FIRST
    half AND keeps the same sign with |corr| above a floor in the SECOND half.
    Output is a RESEARCH artifact (hypothesis feedstock), not a trade signal.
    """
    names = [a for a in config.MODEL_UNIVERSE if a in px.columns]
    rets = px[names].pct_change().iloc[1:]
    n = len(rets)
    half = n // 2
    a_, b_ = rets.iloc[:half], rets.iloc[half:]
    n_tests = len(names) * (len(names) - 1) * len(lags)
    # Bonferroni on the discovery half: two-sided z for alpha=0.05/n_tests
    from math import erf, sqrt

    def _p_from_r(r, m):
        z = abs(r) * sqrt(m)
        return 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))

    alpha = 0.05 / n_tests
    found = []
    for lag in lags:
        lead_a = a_.shift(lag).iloc[lag:]
        tgt_a = a_.iloc[lag:]
        lead_b = b_.shift(lag).iloc[lag:]
        tgt_b = b_.iloc[lag:]
        for lead in names:
            la, lb = lead_a[lead], lead_b[lead]
            corr_a = tgt_a.corrwith(la)
            corr_b = tgt_b.corrwith(lb)
            for tgt in names:
                if tgt == lead:
                    continue
                ra, rb = corr_a.get(tgt), corr_b.get(tgt)
                if ra is None or rb is None or pd.isna(ra) or pd.isna(rb):
                    continue
                if _p_from_r(ra, half - lag) < alpha and np.sign(ra) == np.sign(rb) and abs(rb) > 0.05:
                    found.append({"leader": lead, "target": tgt, "lag": lag,
                                  "corr_discovery": round(float(ra), 4),
                                  "corr_validation": round(float(rb), 4)})
    found.sort(key=lambda d: -abs(d["corr_validation"]))
    return found[:top_n], n_tests


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import qlib
    from qlib.constant import REG_US

    qlib.init(provider_uri=str(config.QLIB_DATA_DIR), region=REG_US)
    px = close_panel()
    print(f"panel: {px.shape[0]} sessions x {px.shape[1]} assets")

    battery = run_battery(px)
    leadlag, n_tests = mine_lead_lag(px)

    result = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "n_trials": N_TRIALS,
        "cost_per_side": COST_PER_SIDE,
        "battery": battery,
        "lead_lag": {"n_tests": n_tests, "bonferroni_alpha": 0.05 / n_tests,
                     "survivors": leadlag},
        "note": "pre-registered battery; survivors stay SHADOW pending human review",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = config.SCORECARDS_DIR / f"experiments_{stamp}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    print(f"scorecard -> {path}")

    print(f"\n{'hypothesis':20s} {'n':>4s} {'pass':>5s} {'dsr':>8s} {'pbo':>6s} {'pf':>6s}")
    for name, v in battery.items():
        dsr = (v.get("deflated_sharpe") or {}).get("ratio")
        print(f"{name:20s} {v['n']:4d} {str(v['passes']):>5s} "
              f"{dsr if dsr is not None else 'n/a':>8} {v.get('pbo') if v.get('pbo') is not None else 'n/a':>6} "
              f"{v.get('profit_factor') if v.get('profit_factor') is not None else 'n/a':>6}")
    print(f"\nlead-lag: {len(leadlag)} split-half survivors of {n_tests} tests")
    for r in leadlag[:10]:
        print(f"  {r['leader']:6s} -> {r['target']:6s} lag={r['lag']} "
              f"disc={r['corr_discovery']:+.3f} valid={r['corr_validation']:+.3f}")


if __name__ == "__main__":
    main()
