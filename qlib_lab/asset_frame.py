"""Per-asset layered framework: Trend -> Momentum -> Volume -> Breadth -> Relative
strength -> Flows -> Positioning -> Valuation -> Fundamentals -> Macro -> Catalyst -> Risk.

The idea (user's framework): one metric alone is weak; independent layers agreeing is
the actual evidence. So every asset is read through the same ordered layers, each
layer answering one question, and the sheet shows WHICH layers agree rather than
collapsing them into a single number to act on.

DESCRIPTIVE, like the rest of market_state. Nothing here forecasts a return, and the
confluence tally at the end is a COUNT OF AGREEING MEASUREMENTS, not a score, not a
ranking, and not something that has ever been tested against forward returns. If it is
ever joined to forward returns it becomes a model and the PBO/DSR gate applies.

Honest coverage: this box holds daily OHLCV, COT, Deribit funding, FRED liquidity and
an FOMC calendar. It does NOT hold on-chain data (realized price, MVRV, Puell,
exchange balances, whale flows, hashrate, difficulty, hashprice), retail sentiment
indices, short interest, or liquidation data. Those layers report NO_DATA rather than
being filled with a proxy that would read as if it were the real measurement.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config, cot, funding, macro_calendar, market_state as ms

logger = logging.getLogger("qlib_lab.asset_frame")

BULL, BEAR, NEUTRAL, NO_DATA = "bullish", "bearish", "neutral", "NO_DATA"

# Layers in the order the framework reads them. `computable` records whether THIS BOX
# can populate the layer at all -- it is displayed, so a reader can tell "measured and
# neutral" apart from "we cannot see this".
LAYERS = [
    ("trend",        "What is the long-term trend?"),
    ("momentum",     "Is momentum accelerating or decelerating?"),
    ("volume",       "Does money actually support the move?"),
    ("breadth",      "Is the whole market participating?"),
    ("rel_strength", "Where is capital concentrating?"),
    ("flows",        "Is large external capital entering or leaving?"),
    ("positioning",  "Is everybody already on the same side?"),
    ("valuation",    "Cheap or euphoric vs holder cost basis?"),
    ("fundamentals", "Is the underlying network/business stronger?"),
    ("macro",        "Is the financial environment helping or fighting?"),
    ("narrative",    "Does the story this asset trades on still hold?"),
    ("risk",         "What outside force can hijack the trade?"),
]

# Layers this box cannot measure at all, with the metric that would be needed. Kept
# explicit so the gap is visible on the sheet instead of quietly absent.
MISSING = {
    "valuation": "needs MVRV/realized price (on-chain) or P/E-CAPE style fundamentals; "
                 "neither is on this box for a cross-asset universe",
    "fundamentals": "needs hashrate/difficulty/hashprice (crypto) or company financials; "
                    "company_lab covers S&P 500 equities only, not this universe",
}

# Volatility/inverse instruments: a rising price here is risk-OFF, so a "bullish"
# trend row means the opposite of what it means on SPY. Flagged, not re-signed --
# re-signing would make the raw measurement disagree with the chart.
INVERSE = {"VIX", "VIX9D", "VIX3M", "VVIX", "SKEW"}

# Direction-arbitrary or inverted instruments, excluded from the BREADTH denominator
# only (they still get their own rows). See _breadth for why each is here.
BREADTH_EXCLUDE = INVERSE | {"DXY", "EURUSD", "GBPUSD", "USDJPY", "USDCAD",
                             "USDCNH", "US10Y", "BIL"}

# Cross-repo universe membership. Every lab on this box tracks an overlapping set of
# the same macro assets, but alpaca_gpu_lab addresses them by ETF TICKER where
# qlib_lab uses a logical name. Resolving the aliases shows the real picture: the
# 89-name qlib universe is a strict superset of every other universe here -- the only
# name absent anywhere is alpaca's BND, whose equivalent AGG is covered.
ALIASES = {
    "GLD": "GOLD", "USO": "OIL", "CPER": "COPPER", "BTCUSD": "BTC",
    "UUP": "DXY", "VIXY": "VIX", "FXE": "EURUSD", "FXB": "GBPUSD",
    "FXY": "USDJPY", "FXC": "USDCAD", "BND": "AGG",
}

# The 22 macro assets HQ's macro brain and macro_gpu_lab both track (identical sets).
MACRO_22 = {"BIL", "BTC", "COPPER", "DXY", "EEM", "EURUSD", "GBPUSD", "GOLD", "HYG",
            "IWM", "LQD", "OIL", "QQQ", "SPY", "TIP", "TLT", "US10Y", "USDCAD",
            "USDCNH", "USDJPY", "VIX", "XLF"}

# alpaca_gpu_lab's 22-asset intraday bench, alias-resolved from its ETF tickers.
ALPACA_22 = {ALIASES.get(t, t) for t in
             ("BIL", "BTCUSD", "CPER", "EEM", "FXB", "FXC", "FXE", "FXI", "FXY",
              "GLD", "HYG", "IEF", "IWM", "LQD", "QQQ", "SPY", "TIP", "TLT", "USO",
              "UUP", "VIXY", "XLF")}

# alpaca's 4-name intraday core (BND -> AGG).
ALPACA_CORE = {ALIASES.get(t, t) for t in ("SPY", "QQQ", "IWM", "BND")}

UNIVERSES = {
    "macro22": MACRO_22,        # HQ macro brain + macro_gpu_lab (same 22)
    "alpaca22": ALPACA_22,      # alpaca_gpu_lab intraday bench
    "alpaca_core": ALPACA_CORE,
    "cot": set(ms.COT_ASSETS),  # has a CFTC futures contract
}


def universes_for(name: str) -> str:
    """Which tracked universes contain this asset, as a display string."""
    hits = [k for k, s in UNIVERSES.items() if name in s]
    return ",".join(hits) if hits else "qlib_only"


EMA_FAST, EMA_MID, EMA_SLOW = 21, 50, 200
RSI_N = 14
VOL_WINDOW = 20

# Macro-layer parameters. STATED priors, not fitted: no search was run over them and
# nothing here has ever been scored against forward returns. MACRO_CORR_MIN is the
# |corr| below which an asset is treated as unlinked to the risk complex, and
# MACRO_IMPULSE_Z the |z| below which a net-liquidity change is drift, not an impulse.
MACRO_CORR_N = 60
MACRO_CORR_MIN = 0.20
MACRO_IMPULSE_Z = 0.50


# ------------------------------------------------------------------ indicators

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def macd(s: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def rsi(s: pd.Series, n: int = RSI_N) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # No down-days at all makes rs infinite, and up/NaN makes it NaN -- which then
    # silently fails every `rsi >= 80` comparison, so a clean uptrend would never be
    # flagged overbought. Canonical RSI is 100 with no losses, 0 with no gains.
    out = out.mask((dn == 0) & (up > 0), 100.0)
    out = out.mask((up == 0) & (dn > 0), 0.0)
    return out.mask((up == 0) & (dn == 0), 50.0)


def _verdict(bull: bool, bear: bool) -> str:
    if bull and not bear:
        return BULL
    if bear and not bull:
        return BEAR
    return NEUTRAL


# ---------------------------------------------------------------------- layers

def _trend(px: pd.Series) -> dict:
    if len(px) < EMA_SLOW + 5:
        return {"verdict": NO_DATA, "detail": f"needs {EMA_SLOW}+ sessions"}
    last = float(px.iloc[-1])
    e21, e50, e200 = (float(ema(px, n).iloc[-1]) for n in (EMA_FAST, EMA_MID, EMA_SLOW))
    stacked_up = last > e21 > e50 > e200
    stacked_dn = last < e21 < e50 < e200
    above200 = last > e200
    return {
        "verdict": _verdict(above200 and last > e21, (not above200) and last < e21),
        "detail": (f"px {last:,.2f} vs EMA21 {e21:,.2f} / EMA50 {e50:,.2f} / "
                   f"EMA200 {e200:,.2f}; {'stacked bull' if stacked_up else 'stacked bear' if stacked_dn else 'unstacked'}; "
                   f"{(last / e200 - 1) * 100:+.1f}% vs EMA200"),
        "dist_ema200_pct": last / e200 - 1, "stacked": stacked_up,
    }


def _momentum(px: pd.Series) -> dict:
    if len(px) < 60:
        return {"verdict": NO_DATA, "detail": "needs 60+ sessions"}
    m = macd(px)
    h, h_prev = float(m["hist"].iloc[-1]), float(m["hist"].iloc[-2])
    r = float(rsi(px).iloc[-1])
    expanding = abs(h) > abs(h_prev)
    # RSI >=80 with an expanding positive histogram is accelerating AND stretched.
    # Reporting that as plainly "bullish" hides the more useful half, so it degrades
    # to neutral and the extreme is named in the detail.
    hot, cold = r >= 80, r <= 20
    extreme = " [RSI overbought]" if hot else " [RSI oversold]" if cold else ""
    return {
        "verdict": _verdict(h > 0 and expanding and not hot,
                            h < 0 and expanding and not cold),
        "detail": (f"MACD hist {h:+.3f} ({'expanding' if expanding else 'contracting'}), "
                   f"line {float(m['macd'].iloc[-1]):+.3f} vs signal "
                   f"{float(m['signal'].iloc[-1]):+.3f}; RSI14 {r:.1f}{extreme}"),
        "rsi": r, "macd_hist": h,
    }


def _volume(vol: pd.Series | None, px: pd.Series) -> dict:
    # Index products (VIX, VIX9D, SKEW, ...) carry volume 0.0 in this store -- that is
    # absence of a measurement, not low participation, so it must not read as bearish.
    if vol is None or vol.dropna().empty or float(vol.tail(60).sum()) == 0:
        return {"verdict": NO_DATA, "detail": "no volume in store (index product)"}
    v = vol.replace(0, np.nan).dropna()
    if len(v) < 60:
        return {"verdict": NO_DATA, "detail": "needs 60+ volume sessions"}
    recent, base = float(v.tail(5).mean()), float(v.tail(60).mean())
    ratio = recent / base if base else np.nan
    up = px.pct_change().reindex(v.index).tail(VOL_WINDOW) > 0
    # `or 0` does NOT catch NaN (NaN is truthy). With no down-days in the window the
    # mean is NaN, `up_v > NaN` is False, and an all-up-days volume expansion read as
    # BEARISH. Coerce explicitly and treat "no down-day volume" as confirmation.
    def _mean(sel):
        m = v.tail(VOL_WINDOW)[sel].mean()
        return 0.0 if pd.isna(m) else float(m)

    up_v, dn_v = _mean(up), _mean(~up)
    conf = up_v > dn_v
    return {
        "verdict": _verdict(ratio > 1.15 and conf, ratio > 1.15 and not conf),
        "detail": (f"5d vol {ratio:.2f}x its 60d average; up-day volume "
                   f"{'>' if conf else '<='} down-day volume "
                   f"({up_v:,.0f} vs {dn_v:,.0f})"),
        "vol_ratio": ratio,
    }


def _breadth(panel: pd.DataFrame) -> dict:
    """Market-level, identical for every asset: is the rally broad or narrow?

    The denominator must contain only instruments where "price above its EMA" means
    "risk-on". Counting the VIX family inverts the reading exactly when it matters --
    in a selloff they rise above their EMAs and INFLATE breadth while participation
    collapses. FX pairs are worse than noise: their sign is a naming convention
    (USDJPY up is dollar strength, EURUSD up is the opposite), so they contribute an
    arbitrary bit. US10Y is a yield, inverse to the bond price, and BIL is cash.
    Measured impact of excluding them on 2026-08-21: 70%/81% -> 74%/86%.
    """
    if panel.empty or len(panel) < EMA_SLOW + 5:
        return {"verdict": NO_DATA, "detail": "insufficient panel history"}
    keep = [c for c in panel.columns if c not in BREADTH_EXCLUDE]
    dropped = len(panel.columns) - len(keep)
    panel = panel[keep]
    last = panel.iloc[-1]
    e50, e200 = panel.apply(lambda c: ema(c.dropna(), EMA_MID).reindex(panel.index).iloc[-1]), \
        panel.apply(lambda c: ema(c.dropna(), EMA_SLOW).reindex(panel.index).iloc[-1])
    ok = last.notna() & e50.notna() & e200.notna()
    n = int(ok.sum())
    if not n:
        return {"verdict": NO_DATA, "detail": "no comparable instruments"}
    a50 = float((last[ok] > e50[ok]).mean())
    a200 = float((last[ok] > e200[ok]).mean())
    return {
        "verdict": _verdict(a200 > 0.60 and a50 > 0.50, a200 < 0.40 and a50 < 0.50),
        "detail": (f"{a50:.0%} of {n} direction-comparable instruments above EMA50, "
                   f"{a200:.0%} above EMA200 ({dropped} inverse/FX/rate names excluded "
                   f"from the denominator)"),
        "pct_above_200": a200, "pct_above_50": a50,
    }


def _rel_strength(px: pd.Series, bench: pd.Series, name: str, n: int = 63) -> dict:
    if name == "SPY":
        return {"verdict": NEUTRAL, "detail": "is the benchmark"}
    a, b = px.dropna(), bench.dropna()
    common = a.index.intersection(b.index)
    if len(common) < n + 5:
        return {"verdict": NO_DATA, "detail": "insufficient overlap with SPY"}
    a, b = a.reindex(common), b.reindex(common)
    ra = float(a.iloc[-1] / a.iloc[-1 - n] - 1)
    rb = float(b.iloc[-1] / b.iloc[-1 - n] - 1)
    return {
        "verdict": _verdict(ra > rb, ra < rb),
        "detail": f"3m return {ra:+.1%} vs SPY {rb:+.1%} (spread {ra - rb:+.1%})",
        "rs_3m": ra - rb,
    }


def _flows(name: str, flows: dict) -> dict:
    # etf_flows.json is keyed by the ETF TICKER (GLD/USO/CPER) while this framework
    # uses the logical universe name (GOLD/OIL/COPPER). Looking up only the logical
    # name told the reader "this box cannot see flows" for three of the eight focus
    # assets when the data was sitting in the flag file -- a FALSE no-data claim, and
    # the honesty of these labels is exactly what lets this module sit outside the
    # PBO/DSR gate. config.UNIVERSE already holds name -> ticker.
    row = flows.get(name) or flows.get(config.UNIVERSE.get(name, ""))
    if row is None:
        return {"verdict": NO_DATA,
                "detail": f"no ETF-flow row for {name} "
                          f"(ticker {config.UNIVERSE.get(name, '?')}) - not a tracked fund"}
    z, f20 = row.get("z"), row.get("flow_20d")
    if z is None or f20 is None:
        # the row's own note is the specific reason (no issuer feed for this fund vs
        # a history that has not accumulated yet); the generic sentence hides which
        return {"verdict": NO_DATA,
                "detail": (row.get("note")
                           or f"flow history too short ({row.get('snapshots')} "
                              f"snapshots since {flows.get('_since')}); z not yet "
                              f"computable")}
    noise = row.get("noise_usd")
    floor = f", source resolution +/-${noise:,.0f}" if noise else ""
    return {"verdict": _verdict(z > 1, z < -1),
            "detail": (f"20d flow {f20:,.0f} (z {z:+.2f}) from "
                       f"{row.get('source') or 'unknown source'}{floor}"),
            "flow_z": z}


def _positioning(name: str, cot_rows: dict, fund: dict) -> dict:
    bits, verdicts = [], []
    r = cot_rows.get(name)
    if r is not None and r.get("idx52") is not None:
        idx, z = r["idx52"], r.get("z156")
        bits.append(f"COT idx52 {idx:.2f} (z {z:+.2f}), 4w chg {r.get('chg4w'):+.3f}"
                    if z is not None else f"COT idx52 {idx:.2f}")
        # crowded is a RISK reading, not a direction: a crowded long is vulnerable
        verdicts.append(BEAR if idx >= ms.CROWDED else BULL if idx <= ms.UNCROWDED
                        else NEUTRAL)
    if name == "BTC" and fund and fund.get("z156") is not None:
        z = fund["z156"]
        bits.append(f"perp funding {fund.get('funding_1d_annualized_pct')}%/yr "
                    f"(z {z:+.2f}, idx52w {fund.get('idx52w')}) -> {fund.get('read')}")
        verdicts.append(BEAR if z > 1.5 else BULL if z < -1.5 else NEUTRAL)
    if not bits:
        return {"verdict": NO_DATA,
                "detail": "no COT contract and no funding feed for this instrument"}
    v = (BEAR if BEAR in verdicts else BULL if BULL in verdicts else NEUTRAL)
    return {"verdict": v, "detail": "; ".join(bits) + "  [crowded = vulnerable, not bearish price]"}


def _macro(px: pd.Series, spy: pd.Series, dxy: pd.Series, liq: dict, name: str) -> dict:
    """Is the financial environment helping or fighting THIS instrument?

    The liquidity impulse is a MARKET-wide fact, so using it alone made this layer
    print the same verdict for all 89 names -- it measured each asset's co-movement
    with the risk complex and then threw that measurement away. It now spends it: the
    impulse is signed, and the asset's own 60d correlation to SPY says which way that
    impulse lands on it. An asset that moves inverse to the risk complex is helped by
    a liquidity drain, which is the whole point of the row.

    The mapping (impulse sign x co-movement sign) is a STATED prior written here in
    the open, not a fitted one, and it has never been tested against forward returns.
    """
    r = px.pct_change().tail(MACRO_CORR_N)
    bits, corrs = [], {}
    for lbl, other in (("SPY", spy), ("DXY", dxy)):
        if lbl == name or other is None:
            continue                      # a self-correlation of +1.00 says nothing
        o = other.pct_change().reindex(r.index)
        c = r.corr(o)
        if pd.notna(c):
            corrs[lbl] = float(c)
            bits.append(f"corr {lbl} {c:+.2f} ({MACRO_CORR_N}d)")
    # SPY IS the risk complex; skipping the self-correlation above must not be read as
    # "SPY has no measured link to it", and on the sheet an empty term next to a signed
    # verdict looks like a bug.
    if name == "SPY":
        beta = 1.0
        bits.insert(0, "is the risk complex itself (self-correlation not printed)")
    else:
        beta = corrs.get("SPY")

    # `trend` lives only on the liquidity FLAG, not in the parquet frame, so reading
    # liq["trend"] off the frame silently left this layer neutral for every asset.
    # Derive it from the frame instead, which also keeps it date-consistent.
    chg, z = liq.get("netliq_chg13w"), liq.get("netliq_chg13w_z")
    impulse = 0
    if chg is not None and pd.notna(chg):
        strong = True if z is None or pd.isna(z) else abs(float(z)) >= MACRO_IMPULSE_Z
        impulse = (1 if chg > 0 else -1) if strong else 0
        word = "rising" if chg > 0 else "falling"
        bits.append(f"net liquidity {word} ({chg:+,.1f}bn/13w"
                    + (f", z {z:+.2f})" if z is not None and pd.notna(z) else ")")
                    + ("" if impulse else " -- drift, not an impulse"))
    if not bits:
        return {"verdict": NO_DATA, "detail": "no macro context available"}

    if beta is None:
        why = "co-movement with the risk complex unmeasurable, so the impulse cannot be signed for this asset"
        return {"verdict": NEUTRAL, "detail": "; ".join(bits) + f"  [{why}]"}
    linked = 1 if beta >= MACRO_CORR_MIN else -1 if beta <= -MACRO_CORR_MIN else 0
    lean = impulse * linked
    if not impulse:
        why = "no liquidity impulse to transmit"
    elif not linked:
        why = (f"|corr SPY| < {MACRO_CORR_MIN:.2f}: this asset is not moving with or "
               f"against the risk complex, so the impulse does not reach it")
    else:
        moves = "WITH" if linked > 0 else "INVERSE to"
        why = (f"moves {moves} the risk complex and liquidity is "
               f"{'expanding' if impulse > 0 else 'draining'} -> environment "
               f"{'helping' if lean > 0 else 'fighting'} it")
    return {"verdict": _verdict(lean > 0, lean < 0),
            "detail": "; ".join(bits) + f"  [{why}]"}


def _narrative(name: str, stories: list[dict], asof: pd.Timestamp) -> dict:
    """Is the relationship this asset's story implies still holding?

    This replaces the old `catalyst` row, which returned NEUTRAL for all 89 names on
    every date -- an FOMC countdown is market-wide, so it carried no cross-sectional
    information at all. The event countdown is kept in the detail (nothing is lost);
    the VERDICT now comes from stories registered in `market_state.STORIES`: a claim,
    the correlation that claim implies, and whether it currently holds.

    Honest about direction: a BROKEN story is not a price call, it is the same kind of
    FRAGILITY reading as the positioning row -- the asset is being held for a reason
    that is no longer true on the tape. HOLDING means the stated relationship is
    intact, which is confirmation, not a forecast.
    """
    try:
        days = int(macro_calendar.days_to_next_fomc(pd.DatetimeIndex([asof])).iloc[0])
        event = f"{days} sessions to next FOMC"
    except Exception:  # noqa: BLE001
        event = "FOMC calendar unavailable"

    mine = ms.stories_for(name, stories)
    if not mine:
        return {"verdict": NEUTRAL,
                "detail": (f"{event}. No registered story references this instrument, "
                           f"so there is nothing here to check -- earnings, regulation "
                           f"and policy are not tracked on this box either.")}

    def _fmt(st):
        c = st["corr_fast"]
        cs = f"{c:+.2f}" if c is not None else "n/a"
        slow = f"{st['corr_slow']:+.2f}" if st["corr_slow"] is not None else "n/a"
        return (f"\"{st['claim']}\" implies {st['implies']}: {cs} "
                f"({ms.STORY_CORR_FAST}d, {slow} at {ms.STORY_CORR_SLOW}d) -> {st['state']}")

    states = [st["state"] for st in mine]
    broken = ms.BROKEN in states
    intact = all(v == ms.HOLDING for v in states)
    tag = ("  [BROKEN = this asset is held for a reason the tape no longer supports; "
           "a fragility reading, not a price call]" if broken else "")
    return {
        "verdict": _verdict(intact, broken),
        "detail": "; ".join(_fmt(st) for st in mine) + f". {event}{tag}",
    }


def _risk(px: pd.Series) -> dict:
    r = px.pct_change()
    if r.dropna().shape[0] < 260:
        return {"verdict": NO_DATA, "detail": "needs ~1y of returns"}
    rv = (r.rolling(20).std() * np.sqrt(252) * 100).dropna()
    pct = float(ms.trailing_pct(rv).iloc[-1])
    hi = float(px.tail(252).max())
    dd = float(px.iloc[-1]) / hi - 1
    return {
        # high realised vol and a deep drawdown are elevated RISK, hence "bearish"
        # in the risk row -- it is a hazard reading, not a price call
        "verdict": _verdict(pct < 0.35 and dd > -0.10, pct > 0.65 or dd < -0.20),
        "detail": (f"realised vol 20d {float(rv.iloc[-1]):.1f}% ({pct:.0%} of 1y); "
                   f"{dd:+.1%} vs 52w high {hi:,.2f}"),
        "rv_pct": pct, "dd_from_52w_high": dd,
    }


# ------------------------------------------------------------------- assembly

def asset_frame(name: str, ctx: dict) -> dict:
    """All twelve layers for one instrument. Individual LAYERS fail to NO_DATA, but a
    malformed ctx (missing SPY or breadth) can still raise -- asset_table guards it."""
    px = ctx["close"].get(name)
    if px is None or px.dropna().empty:
        return {"asset": name, "error": "no price history"}
    px = px.dropna()
    vol = ctx["volume"].get(name) if ctx.get("volume") is not None else None

    # On an inverse/volatility instrument "bullish trend" means RISING VOLATILITY,
    # i.e. bearish for risk assets. The layers still read mechanically; flagging it
    # stops the sheet being read backwards.
    inverse = name in INVERSE
    out = {"asset": name, "price": float(px.iloc[-1]),
           "as_of": str(px.index[-1].date()), "inverse": inverse,
           "universes": universes_for(name),
           "reads": ("bullish = rising volatility (risk-OFF)" if inverse
                     else "bullish = rising price")}
    layers = {
        "trend": _trend(px),
        "momentum": _momentum(px),
        "volume": _volume(vol, px),
        "breadth": ctx["breadth"],
        "rel_strength": _rel_strength(px, ctx["close"]["SPY"], name),
        "flows": _flows(name, ctx["flows"]),
        "positioning": _positioning(name, ctx["cot"], ctx["funding"]),
        "valuation": {"verdict": NO_DATA, "detail": MISSING["valuation"]},
        "fundamentals": {"verdict": NO_DATA, "detail": MISSING["fundamentals"]},
        # no DXY fallback to `px`: that printed a self-correlation of +1.00 as if it
        # were a real dollar reading. Absent DXY simply drops the term.
        "macro": _macro(px, ctx["close"]["SPY"], ctx["close"].get("DXY"),
                        ctx["liq"], name),
        "narrative": _narrative(name, ctx.get("stories", []), px.index[-1]),
        "risk": _risk(px),
    }
    for key, _q in LAYERS:
        out[key] = layers[key]["verdict"]
        out[f"{key}_detail"] = layers[key]["detail"]

    v = [layers[k]["verdict"] for k, _ in LAYERS]
    out["n_bullish"] = v.count(BULL)
    out["n_bearish"] = v.count(BEAR)
    out["n_neutral"] = v.count(NEUTRAL)
    out["n_no_data"] = v.count(NO_DATA)
    out["confluence"] = f"{out['n_bullish']}B / {out['n_bearish']}Br / {out['n_no_data']}ND"
    return out


def build_context(start: str = ms.START) -> dict:
    """Everything the per-asset frames share, fetched once."""
    import json

    names = sorted(config.UNIVERSE.keys())
    close = ms.close_panel(names, start=ms.WARMUP)
    if close.empty:
        return {}
    today = pd.Timestamp.now().normalize()
    close = close[close.index < today]          # never read a live partial bar

    volume = pd.DataFrame()
    try:
        ms._ensure_qlib()
        from qlib.data import D
        volume = D.features(names, ["$volume"])["$volume"].unstack(level=0)
        volume.columns = [str(c) for c in volume.columns]
        volume.index = pd.to_datetime(volume.index)
        volume = volume[volume.index < today]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"volume unavailable: {e}")

    cot_rows = {r["asset"]: r for r in (cot.read_state().get("rows") or [])}
    fund = funding.read_state()
    liq = ms._last_row(ms.liquidity_history(start))

    flows = {}
    try:
        raw = json.loads((config.FLAGS_DIR / "etf_flows.json").read_text())
        flows = {r["ticker"]: r for r in raw.get("rows", [])}
        flows["_since"] = raw.get("history_since")
    except Exception:  # noqa: BLE001
        pass

    return {"close": close, "volume": volume, "cot": cot_rows, "funding": fund,
            "liq": liq, "flows": flows, "breadth": _breadth(close),
            "stories": ms.story_checks(close)}


def asset_table(ctx: dict | None = None, assets: list[str] | None = None) -> pd.DataFrame:
    ctx = ctx or build_context()
    if not ctx:
        return pd.DataFrame()
    names = assets or sorted(ctx["close"].columns)
    rows = []
    for n in names:
        # per-asset guard: asset_frame CAN raise (an absent SPY/breadth key), and a
        # bare comprehension would let one bad instrument delete both sheets for all 89
        try:
            r = asset_frame(n, ctx)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"framework row failed for {n}: {e}")
            continue
        if "error" not in r:
            rows.append(r)
    return pd.DataFrame(rows)
