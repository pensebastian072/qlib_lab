"""Market-wide cross-asset context features, one row per session.

Ports the shared "_market_context" ideas from macro_gpu_lab/features.py (the
macro-brain relationship layer) into this venv — the numpy<2/pandas-2 pins here
make a cross-venv import brittle, and these are FEATURES, not the gate (the
gate itself stays imported from macro_gpu_lab.validate, never re-ported):

  - absorption ratio (Kritzman): share of return variance captured by the top
    n//5 eigenvectors of the 60d correlation matrix of the macro risk basket —
    high = fragile, tightly-coupled market,
  - average pairwise correlation (20d) of the same basket,
  - credit stress: z(120d) of the HYG/LQD log ratio,
  - real-rate / duration signal: z(120d) of the TIP/TLT log ratio,
  - VIX level + 5d change, SPY realized vol(20) and momentum(20),
  - DXY and GOLD momentum(20).

All features at date t use closes through t only (rolling windows, no centering)
so they merge leak-free onto Alpha158 rows keyed by (instrument, date).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Macro risk basket for absorption / pairwise corr (US-session names only).
RISK_BASKET = ["SPY", "QQQ", "IWM", "EEM", "XLF", "TLT", "TIP",
               "GOLD", "OIL", "COPPER", "HYG", "LQD", "BIL"]
ABS_WINDOW = 60
CORR_WINDOW = 20
Z_WINDOW = 120
MOM_WINDOW = 20


def _close_panel(names: list[str]) -> pd.DataFrame:
    """Wide close panel (index=date, one column per logical name) from the .bin store."""
    from qlib.data import D

    df = D.features(names, ["$close"])
    wide = df["$close"].unstack(level=0)
    wide.columns = [str(c) for c in wide.columns]
    return wide.sort_index()


def absorption_series(rets: pd.DataFrame, window: int = ABS_WINDOW) -> pd.Series:
    """Rolling Kritzman absorption ratio (top n//5 eigenvalues / total)."""
    k = max(1, rets.shape[1] // 5)   # macro_gpu_lab ABSORPTION_K_DIVISOR = 5
    out = pd.Series(np.nan, index=rets.index)
    vals = rets.values
    for i in range(window, len(rets)):
        win = vals[i - window:i]
        mask = ~np.isnan(win).any(axis=0)
        if mask.sum() < 5:
            continue
        c = np.corrcoef(win[:, mask], rowvar=False)
        if np.isnan(c).any():
            continue
        ev = np.linalg.eigvalsh(c)
        out.iloc[i] = float(ev[-k:].sum() / ev.sum())
    return out


def _zscore(s: pd.Series, window: int = Z_WINDOW) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - m) / sd.replace(0.0, np.nan)


def build_context() -> pd.DataFrame:
    """Daily context frame indexed by date. NaN-tolerant (LightGBM handles NaN)."""
    names = sorted(set(RISK_BASKET + ["VIX", "DXY", "GOLD", "SPY",
                                      "VIX3M", "SKEW", "VVIX"]))
    px = _close_panel([n for n in names if n in config.ASSETS])
    rets = px[[c for c in RISK_BASKET if c in px.columns]].pct_change()

    ctx = pd.DataFrame(index=px.index)
    ctx["ctx_absorption"] = absorption_series(rets)
    ctx["ctx_avg_corr"] = _avg_pairwise_corr(rets)
    if "HYG" in px and "LQD" in px:
        ctx["ctx_credit_stress"] = _zscore(np.log(px["HYG"] / px["LQD"]))
    if "TIP" in px and "TLT" in px:
        ctx["ctx_curve"] = _zscore(np.log(px["TIP"] / px["TLT"]))
    if "VIX" in px:
        ctx["ctx_vix"] = px["VIX"]
        ctx["ctx_vix_chg5"] = px["VIX"].pct_change(5)
        # variance risk premium (Bollerslev 2009); split-test 2026-07-13 showed
        # +2.0% vs +0.7% 21d SPY blocks (t=1.46, sign stable across halves)
        rv_ann = px["SPY"].pct_change().rolling(20).std() * np.sqrt(252) * 100
        ctx["ctx_vrp"] = px["VIX"] ** 2 - rv_ann ** 2
    spy = px["SPY"].pct_change()
    ctx["ctx_spy_vol20"] = spy.rolling(MOM_WINDOW).std()
    ctx["ctx_spy_mom20"] = px["SPY"].pct_change(MOM_WINDOW)
    if "DXY" in px:
        ctx["ctx_dxy_mom20"] = px["DXY"].pct_change(MOM_WINDOW)
    if "GOLD" in px:
        ctx["ctx_gold_mom20"] = px["GOLD"].pct_change(MOM_WINDOW)

    # option-surface context (wave 3): term-structure slope, tail pricing, vol-of-vol
    if "VIX" in px and "VIX3M" in px:
        ctx["ctx_vix_ts"] = px["VIX"] / px["VIX3M"]   # >1 = backwardation/stress
    if "SKEW" in px:
        ctx["ctx_skew_z"] = _zscore(px["SKEW"], 156)
    if "VVIX" in px:
        ctx["ctx_vvix_z"] = _zscore(px["VVIX"], 156)

    # Fed net liquidity (release-lagged upstream; fail-soft if never fetched)
    try:
        from .fred_liquidity import daily_frame
        liq = daily_frame()
        if liq is not None and "netliq_chg13w_z" in liq.columns:
            ctx["ctx_netliq_chg13w_z"] = liq["netliq_chg13w_z"].reindex(ctx.index).ffill()
    except Exception:  # noqa: BLE001 — context stays price-only
        pass

    # sessions until the next scheduled FOMC decision (capped; NaN off-calendar)
    try:
        from .macro_calendar import days_to_next_fomc
        ctx["ctx_days_to_fomc"] = days_to_next_fomc(ctx.index)
    except Exception:  # noqa: BLE001
        pass
    return ctx


def _avg_pairwise_corr(rets: pd.DataFrame, window: int = CORR_WINDOW) -> pd.Series:
    """Mean off-diagonal correlation over a rolling window (loop — small n)."""
    out = pd.Series(np.nan, index=rets.index)
    vals = rets.values
    for i in range(window, len(rets)):
        win = vals[i - window:i]
        mask = ~np.isnan(win).any(axis=0)
        if mask.sum() < 5:
            continue
        c = np.corrcoef(win[:, mask], rowvar=False)
        iu = np.triu_indices_from(c, k=1)
        out.iloc[i] = float(np.nanmean(c[iu]))
    return out


def augment(feat: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """Merge context columns onto an (instrument, datetime)-indexed feature frame."""
    dates = feat.index.get_level_values(1)
    aligned = ctx.reindex(dates)
    aligned.index = feat.index
    return pd.concat([feat, aligned], axis=1)
