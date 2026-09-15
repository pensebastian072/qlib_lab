"""Market State — a DESCRIPTIVE workbook of market conditions. NO FORECAST.

Every predictive model on this box has failed the canonical gate. That verdict is
stable and is not what this module is arguing with. What this module does is keep the
half of each system that was always correct — the measurement — and throw away only
the half that failed — the bet:

    vol desk    VRP percentile, term structure, VVIX, SKEW   (not "sell when >70")
    COT         net %OI, z156, idx52                         (not C1_cot_extreme_fade)
    liquidity   netliq, 13w change, RRP/TGA                  (not L1_netliq_trend)
    macro brain correlations, betas, absorption              (not lead-lag mining)

Because nothing here forecasts a return, the PBO/Deflated-Sharpe gate does not apply:
a thermometer does not need a Sharpe ratio. That is a scope statement, NOT a claim of
validity — see CONDITIONS_NOTE for the one thing in here that could mislead you.

The genuinely new analytic is crowding x structure: COT says WHO is positioned,
correlation says WHAT moves together. Joined, N crowded-but-correlated positions are
reported as ONE bet rather than N diversified ones.

Pure consumer. Reads the qlib .bin store and the three history parquets that cot.py /
funding.py / fred_liquidity.py already persist; re-ports none of their maths. Every
reader is fail-safe (repo hard rule 3): missing/corrupt/stale input yields a neutral
value and a stale marker in the sheet, never an exception.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config, cot, fred_liquidity, funding

logger = logging.getLogger("qlib_lab.market_state")

START = "2016-01-01"          # 10y lookback (see plan: avoids pre-2017 coverage holes)
WARMUP = "2014-01-01"         # lead-in so rolling windows are warm at START
PCT_WINDOW = 252              # trailing-year percentile, matches vol_desk.vrp_snapshot
CORR_WINDOW = 60              # rolling correlation window (sessions)

EXPORT_DIR = config.JOURNAL_DIR / "exports"
EXPORT_PATH = EXPORT_DIR / "market_state.xlsx"
FIRE_LOG = EXPORT_DIR / "conditions_log.jsonl"

VOL_TICKERS = ["SPY", "VIX", "VIX9D", "VIX3M", "VVIX", "SKEW"]
# The 13 CFTC legacy futures-only assets cot.py tracks. All 13 are also in the qlib
# store, which is why the crowding x correlation join can be done in one repo.
COT_ASSETS = ["SPY", "QQQ", "IWM", "TLT", "GOLD", "COPPER", "OIL", "VIX",
              "BTC", "EURUSD", "GBPUSD", "USDJPY", "USDCAD"]

# Assets given a full written framework read-out; the wide sheet covers all 89.
FRAMEWORK_FOCUS = ["BTC", "SPY", "QQQ", "GOLD", "COPPER", "OIL", "TLT", "VIX"]

CROWDED = 0.80                # idx52 at/above this = crowded long
UNCROWDED = 0.20              # idx52 at/below this = crowded short

CONDITIONS_REGISTERED = "2026-08-23"
CONDITIONS_NOTE = (
    "NOT BLIND PRE-REGISTRATION. These thresholds were written on "
    f"{CONDITIONS_REGISTERED}, AFTER the 2026-08 tape had already been looked at. "
    "Any row in the fire log dated before that is BACKFILLED and is not evidence "
    "of anything. The out-of-sample record starts at the registration date and runs "
    "forward. Do not read the backfilled column as a track record."
)

STORIES_REGISTERED = "2026-08-25"
STORY_CORR_FAST = 60          # sessions: the window the state verdict reads
STORY_CORR_SLOW = 252         # sessions: the structural window, reported beside it
STORY_BROKEN = 0.10           # |corr| on the WRONG side of zero before calling it broken
STORIES_NOTE = (
    "Each row is a market STORY people trade on, written next to the relationship that "
    "story implies, and then measured. The thresholds are round conventional numbers "
    "(0.20/0.30/0.40/0.50), not fitted, and no search was run over them -- but the SET "
    f"of stories was chosen on {STORIES_REGISTERED} with the 2026-08 tape already "
    "visible, exactly like CONDITIONS. So the same rule holds: only rows in the log "
    "dated after registration are evidence. What this table is actually for is the "
    "forward record -- it timestamps the session a relationship turns on or off, which "
    "is the thing memory gets wrong afterwards. A HOLDING story is NOT a reason to hold "
    "the asset: this measures whether a stated relationship is intact, and forecasts "
    "nothing."
)

# claim -> implied relationship -> measurement. `thr` carries the DIRECTION: a positive
# threshold means the story needs positive correlation, a negative one negative.
# `assets` is which instruments the story is about, i.e. whose framework row it reaches.
STORIES = [
    {"id": "BTC_IS_DIGITAL_GOLD", "pair": ("BTC", "GOLD"), "thr": 0.40,
     "claim": "BTC is digital gold", "assets": ("BTC", "GOLD")},
    {"id": "BTC_IS_A_RISK_ASSET", "pair": ("BTC", "SPY"), "thr": 0.40,
     "claim": "BTC trades as a risk asset", "assets": ("BTC",)},
    {"id": "DOLLAR_WEIGHS_ON_COMMODITIES", "pair": ("DXY", "COPPER"), "thr": -0.30,
     "claim": "a stronger dollar pushes commodities down", "assets": ("DXY", "COPPER")},
    {"id": "GOLD_IS_A_DOLLAR_TRADE", "pair": ("GOLD", "DXY"), "thr": -0.30,
     "claim": "gold is the other side of the dollar", "assets": ("GOLD", "DXY")},
    {"id": "GOLD_DISLIKES_YIELDS", "pair": ("GOLD", "US10Y"), "thr": -0.30,
     "claim": "gold falls when yields rise (nominal 10y stands in for the real yield "
              "this story is really about)", "assets": ("GOLD", "US10Y")},
    {"id": "BONDS_HEDGE_EQUITIES", "pair": ("TLT", "SPY"), "thr": -0.20,
     "claim": "long bonds hedge equities", "assets": ("TLT", "SPY")},
    {"id": "VOL_IS_THE_EQUITY_HEDGE", "pair": ("VIX", "SPY"), "thr": -0.50,
     "claim": "volatility pays when equities fall", "assets": ("VIX", "SPY")},
    {"id": "CREDIT_TRACKS_EQUITY", "pair": ("HYG", "SPY"), "thr": 0.40,
     "claim": "credit and equity are the same risk trade", "assets": ("HYG", "SPY")},
    {"id": "EM_IS_A_DOLLAR_TRADE", "pair": ("EEM", "DXY"), "thr": -0.30,
     "claim": "emerging markets are short the dollar", "assets": ("EEM", "DXY")},
    {"id": "OIL_IS_A_GROWTH_TRADE", "pair": ("OIL", "SPY"), "thr": 0.30,
     "claim": "oil follows the growth cycle", "assets": ("OIL",)},
]

HOLDING, WEAK, BROKEN, NO_DATA = "HOLDING", "WEAK", "BROKEN", "NO_DATA"


def _story_state(corr: float, thr: float) -> str:
    """Signed against the story's own direction, so one rule covers both signs."""
    if corr is None or pd.isna(corr):
        return NO_DATA
    signed = corr * (1 if thr > 0 else -1)
    if signed >= abs(thr):
        return HOLDING
    if signed <= -STORY_BROKEN:
        return BROKEN            # the relationship has inverted, not merely faded
    return WEAK


def _pair_corr(close: pd.DataFrame, a: str, b: str, n: int) -> float | None:
    if close is None or close.empty or a not in close.columns or b not in close.columns:
        return None
    r = close[[a, b]].pct_change().dropna().tail(n)
    if len(r) < max(20, n // 3):     # a corr off 5 overlapping days is not a measurement
        return None
    c = r[a].corr(r[b])
    return None if pd.isna(c) else float(c)


def story_checks(close: pd.DataFrame) -> list[dict]:
    """Measure every registered story. Never raises; an absent leg reports NO_DATA.

    NOTE the name collision to avoid: `state["narrative"]` is the deterministic PROSE
    paragraph. This table is `state["stories"]`.
    """
    out = []
    for st in STORIES:
        a, b = st["pair"]
        fast = _pair_corr(close, a, b, STORY_CORR_FAST)
        slow = _pair_corr(close, a, b, STORY_CORR_SLOW)
        thr = st["thr"]
        out.append({
            "id": st["id"], "claim": st["claim"], "pair": f"{a}/{b}",
            "implies": f"corr({a},{b}) {'>' if thr > 0 else '<'} {thr:+.2f}",
            "corr_fast": fast, "corr_slow": slow,
            "state": _story_state(fast, thr),
            "state_slow": _story_state(slow, thr),
            "assets": st["assets"],
        })
    return out


def stories_for(name: str, stories: list[dict]) -> list[dict]:
    return [s for s in (stories or []) if name in s.get("assets", ())]


DISCLAIMER = (
    "DESCRIPTIVE ONLY - this workbook measures market state and forecasts nothing. "
    "No number here is a trade signal, a target, or an expected return. Nothing on "
    "this box has cleared the PBO/Deflated-Sharpe gate."
)


# --------------------------------------------------------------------- qlib access

_QLIB_READY = False


def _ensure_qlib() -> None:
    """Initialise qlib once per process. D.features spawns multiprocessing workers
    on Windows, so any caller must sit under an `if __name__ == '__main__'` guard."""
    global _QLIB_READY
    if _QLIB_READY:
        return
    import qlib
    from qlib.constant import REG_US

    qlib.init(provider_uri=str(config.QLIB_DATA_DIR), region=REG_US)
    _QLIB_READY = True


def close_panel(tickers: list[str], start: str = WARMUP) -> pd.DataFrame:
    """Wide daily close frame from the qlib store. Fail-safe -> empty frame."""
    try:
        _ensure_qlib()
        from qlib.data import D

        df = D.features(list(tickers), ["$close"])["$close"].unstack(level=0)
        df.columns = [str(c) for c in df.columns]
        df.index = pd.to_datetime(df.index)
        return df[df.index >= pd.Timestamp(start)].sort_index()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"qlib close panel unavailable: {e}")
        return pd.DataFrame()


def trailing_pct(s: pd.Series, window: int = PCT_WINDOW) -> pd.Series:
    """Trailing-window INCLUSIVE percentile: (window <= current).mean().

    Deliberately not rolling().rank(pct=True) — that averages ties and returns a
    slightly different number. This formula is the one vol_desk.vrp_snapshot uses, so
    the workbook and vol_desk_state.json agree to the digit.
    """
    return s.rolling(window, min_periods=60).apply(
        lambda w: float((w <= w[-1]).mean()), raw=True)


# ------------------------------------------------------------------------ history

def vol_history(start: str = START) -> pd.DataFrame:
    """Daily vol surface + VRP with trailing-year percentiles. Empty on failure."""
    px = close_panel(VOL_TICKERS, start=WARMUP)
    if px.empty or "SPY" not in px or "VIX" not in px:
        return pd.DataFrame()

    # Drop the CURRENT session. Yahoo serves a live partial bar for SPY/VIX the moment
    # the market opens, while the CBOE index files only publish after the close - so a
    # midday run would otherwise compare an intraday VIX against a prior-session VIX3M
    # and report a half-formed bar as "market state". This workbook describes the last
    # COMPLETED session, which is also what the pre-open desk reads.
    today = pd.Timestamp.now().normalize()
    px = px[px.index < today]
    if px.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=px.index)
    for col, tick in (("spy", "SPY"), ("vix", "VIX"), ("vix9d", "VIX9D"),
                      ("vix3m", "VIX3M"), ("vvix", "VVIX"), ("skew", "SKEW")):
        out[col] = px[tick] if tick in px else np.nan

    out["rv20"] = out["spy"].pct_change().rolling(20).std() * np.sqrt(252) * 100
    out["vrp"] = out["vix"] ** 2 - out["rv20"] ** 2

    # VIX9D and VIX3M are 99% covered historically but have not shared a session
    # since 2026-07-17 in this store. vol_desk._vol_surface takes the last non-null
    # of EACH series independently, so the term_ratio it publishes is currently a
    # ratio of two numbers from DIFFERENT dates - which is not a term structure.
    # Here: carry each leg forward a bounded number of sessions, then publish how
    # stale the older leg is so a mismatched reading is visible instead of silent.
    TERM_FFILL = 5
    v9 = out["vix9d"].ffill(limit=TERM_FFILL)
    v3 = out["vix3m"].ffill(limit=TERM_FFILL)
    age9 = out.index.to_series().sub(
        out["vix9d"].dropna().index.to_series().reindex(out.index).ffill())
    age3 = out.index.to_series().sub(
        out["vix3m"].dropna().index.to_series().reindex(out.index).ffill())
    out["term_stale_days"] = np.maximum(age9.dt.days.fillna(-1),
                                        age3.dt.days.fillna(-1))
    out["term_ratio"] = v9 / v3
    out.loc[out["term_stale_days"] < 0, "term_ratio"] = np.nan
    out["structure"] = np.where(out["term_ratio"].isna(), None,
                                np.where(out["term_ratio"] >= 1.0,
                                         "backwardation", "contango"))
    for col in ("vrp", "vix", "skew", "vvix", "term_ratio"):
        out[f"{col}_pct"] = trailing_pct(out[col].dropna()).reindex(out.index)

    return out[out.index >= pd.Timestamp(start)]


def positioning_history(start: str = START) -> pd.DataFrame:
    """COT weekly features (long form). Reuses cot.weekly_features; no re-port."""
    try:
        if not cot.COT_PARQUET.exists():
            return pd.DataFrame()
        feats = cot.weekly_features(pd.read_parquet(cot.COT_PARQUET))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"cot history unreadable: {e}")
        return pd.DataFrame()
    feats["effective_date"] = pd.to_datetime(feats["effective_date"])
    return feats[feats["effective_date"] >= pd.Timestamp(start)].sort_values(
        ["effective_date", "asset"])


def funding_history(start: str = START) -> pd.DataFrame:
    try:
        if not funding.FUND_PARQUET.exists():
            return pd.DataFrame()
        df = funding.daily_features(pd.read_parquet(funding.FUND_PARQUET))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"funding history unreadable: {e}")
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    return df[df.index >= pd.Timestamp(start)]


def liquidity_history(start: str = START) -> pd.DataFrame:
    df = fred_liquidity.daily_frame()
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    return df[df.index >= pd.Timestamp(start)]


# ---------------------------------------------------------------------- structure

def absorption_from_corr(corr: pd.DataFrame) -> tuple[float | None, float | None, int]:
    """Top-k eigenvalue share of a correlation matrix, k = n//4.

    Extracted so `state_panel.rolling_structure` can produce a HISTORY of this number
    that reconciles with the published one to the digit. Two implementations of the same
    eigen block drifted by 5e-4 on 2026-08-24 -- small, but it is exactly the kind of gap
    that later gets explained as a regime change.
    """
    clean = corr.dropna(how="all").dropna(axis=1, how="all")
    if len(clean) < 3 or clean.isna().any().any():
        return None, None, int(len(clean))
    eig = np.sort(np.linalg.eigvalsh(clean.values))[::-1]
    total = float(eig.sum())
    if total <= 0:
        return None, None, int(len(clean))
    k = max(1, len(eig) // 4)
    return float(eig[:k].sum() / total), float(eig[0] / total), int(len(clean))


def structure(start: str = START) -> dict:
    """Correlation matrix, absorption ratio and betas over the COT assets.

    absorption = share of total variance in the top-k eigenvalues of the correlation
    matrix (k = n//4). High absorption means risk is concentrated in few factors, i.e.
    'everything is one trade'. Same construction HQ's macro brain uses; recomputed
    here from qlib daily data so the workbook has no cross-repo dependency.
    """
    px = close_panel(COT_ASSETS + ["DXY"], start=WARMUP)
    # Drop the CURRENT session, exactly as vol_history does. This function had no such
    # guard, so a run during market hours -- which 11:45 is -- built its correlation
    # window on a LIVE PARTIAL bar. The published absorption for 2026-08-24 was 0.71650
    # while the completed session recomputes to 0.71597: a number nobody can reproduce
    # after the close, feeding a condition (ABSORPTION_SPIKE) with a 0.70 threshold.
    px = px[px.index < pd.Timestamp.now().normalize()]
    if px.empty:
        return {"corr": pd.DataFrame(), "absorption": None, "top1_share": None,
                "betas": pd.DataFrame(), "window": CORR_WINDOW, "n_assets": 0}

    rets = px.pct_change()
    rets = rets[rets.index >= pd.Timestamp(start)]
    cols = [c for c in COT_ASSETS if c in rets.columns]
    recent = rets[cols].tail(CORR_WINDOW).dropna(axis=1, how="all")
    corr = recent.corr()

    absorption, top1, _ = absorption_from_corr(corr)

    betas = {}
    for base in ("SPY", "DXY"):
        if base not in rets.columns:
            continue
        b = rets[base].tail(CORR_WINDOW)
        var = float(b.var())
        if not var:
            continue
        betas[base] = {c: float(rets[c].tail(CORR_WINDOW).cov(b) / var)
                       for c in cols if c != base}

    return {"corr": corr, "absorption": absorption, "top1_share": top1,
            "betas": pd.DataFrame(betas), "window": CORR_WINDOW,
            "n_assets": len(clean)}


def crowding_x_structure(pos: pd.DataFrame, struct: dict) -> pd.DataFrame:
    """THE cross-term. COT says who is positioned; correlation says what moves
    together. An asset that is crowded AND highly correlated with the other crowded
    assets is not a diversifying position — it is the same bet wearing another ticker.

    corr_to_crowded is the mean correlation to the OTHER crowded names, so a lone
    crowded asset scores NaN rather than a spurious 1.0.
    """
    if pos.empty:
        return pd.DataFrame()

    latest = pos.sort_values("effective_date").groupby("asset").tail(1)
    corr = struct.get("corr", pd.DataFrame())
    crowded = set(latest.loc[latest["idx52"] >= CROWDED, "asset"]) | \
        set(latest.loc[latest["idx52"] <= UNCROWDED, "asset"])

    rows = []
    for _, r in latest.iterrows():
        asset = r["asset"]
        peers = [a for a in crowded if a != asset and a in corr.columns]
        mean_corr = (float(corr.loc[asset, peers].mean())
                     if asset in corr.index and peers else np.nan)
        idx = r["idx52"]
        side = ("crowded long" if pd.notna(idx) and idx >= CROWDED else
                "crowded short" if pd.notna(idx) and idx <= UNCROWDED else "neutral")
        rows.append({
            "asset": asset, "idx52": idx, "z156": r["z156"],
            "chg4w": r["chg4w"], "side": side,
            "corr_to_crowded": mean_corr,
            "same_bet": bool(side != "neutral" and pd.notna(mean_corr)
                             and abs(mean_corr) >= 0.35),
        })
    return pd.DataFrame(rows).sort_values(
        "idx52", ascending=False, na_position="last").reset_index(drop=True)


# --------------------------------------------------------------------- conditions
#
# Each condition owns the sentence it contributes to the narrative, so the prose is
# assembled from the SAME evaluation that produces the flags and cannot drift from
# them. Tested by test_narrative_matches_fired.

def _f(v, fmt="{:.2f}"):
    return "n/a" if v is None or (isinstance(v, float) and pd.isna(v)) else fmt.format(v)


def _c_short_vol_crowded_cheap(s: dict) -> tuple[bool, str]:
    idx = s["cot_idx"].get("VIX")
    pct = s["vol"].get("vrp_pct")
    if idx is None or pct is None or pd.isna(idx) or pd.isna(pct):
        return False, "inputs unavailable"
    return (idx < 0.25 and pct < 0.25,
            f"VIX COT idx52 {_f(idx)} (<0.25) and VRP pct {_f(pct)} (<0.25)")


def _c_crowded_cluster(s: dict) -> tuple[bool, str]:
    cx = s["crowding"]
    if cx.empty:
        return False, "inputs unavailable"
    hot = cx[(cx["side"] != "neutral") & cx["corr_to_crowded"].notna()]
    members = hot[hot["corr_to_crowded"].abs() >= 0.35]
    if len(members) < 3:
        return False, f"{len(members)} correlated crowded names (need 3)"
    return True, (f"{len(members)} crowded and correlated: "
                  f"{', '.join(members['asset'])} "
                  f"(mean |corr| {_f(members['corr_to_crowded'].abs().mean())})")


def _c_term_backwardation(s: dict) -> tuple[bool, str]:
    r = s["vol"].get("term_ratio")
    if r is None or pd.isna(r):
        return False, "inputs unavailable"
    stale = s["vol"].get("term_stale_days")
    if stale is not None and not pd.isna(stale) and stale > 1:
        # both legs must be effectively current; a ratio across dates is not a
        # term structure and must not be allowed to fire a stress condition
        return False, (f"VIX9D/VIX3M {_f(r, '{:.3f}')} but legs are {int(stale)}d "
                       f"stale - ratio spans different sessions, not evaluated")
    return r >= 1.0, f"VIX9D/VIX3M {_f(r, '{:.3f}')} (>=1.000 = stress)"


def _c_liquidity_shock(s: dict) -> tuple[bool, str]:
    z = s["liq"].get("netliq_chg13w_z")
    if z is None or pd.isna(z):
        return False, "inputs unavailable"
    return z < -2.0, f"netliq 13w change z {_f(z)} (<-2.00)"


def _c_vol_rich(s: dict) -> tuple[bool, str]:
    pct = s["vol"].get("vrp_pct")
    if pct is None or pd.isna(pct):
        return False, "inputs unavailable"
    return pct > 0.80, f"VRP pct {_f(pct)} (>0.80)"


def _c_skew_vs_vix(s: dict) -> tuple[bool, str]:
    sk, vx = s["vol"].get("skew_pct"), s["vol"].get("vix_pct")
    if sk is None or vx is None or pd.isna(sk) or pd.isna(vx):
        return False, "inputs unavailable"
    return (sk > 0.80 and vx < 0.30,
            f"SKEW pct {_f(sk)} (>0.80) while VIX pct {_f(vx)} (<0.30)")


def _c_positioning_unwind(s: dict) -> tuple[bool, str]:
    cx = s["crowding"]
    if cx.empty or cx["chg4w"].isna().all():
        return False, "inputs unavailable"
    worst = cx.loc[cx["chg4w"].abs().idxmax()]
    return (abs(worst["chg4w"]) >= 0.05,
            f"{worst['asset']} net %OI moved {_f(worst['chg4w'], '{:+.3f}')} in 4w "
            f"(|.| >= 0.050)")


def _c_absorption_spike(s: dict) -> tuple[bool, str]:
    a = s["struct"].get("absorption")
    if a is None or pd.isna(a):
        return False, "inputs unavailable"
    return a > 0.70, f"absorption {_f(a)} (>0.70 = risk concentrated in few factors)"


CONDITIONS = [
    {"id": "SHORT_VOL_CROWDED_WHILE_CHEAP", "fn": _c_short_vol_crowded_cheap,
     "rule": "VIX COT idx52 < 0.25 AND VRP pct < 0.25",
     "sentence": "The crowd is short volatility that is already priced cheap, which "
                 "is thin compensation for the position it holds."},
    {"id": "CROWDED_CLUSTER", "fn": _c_crowded_cluster,
     "rule": ">=3 crowded assets with mean |corr| to other crowded names >= 0.35",
     "sentence": "Several crowded positions are moving together, so they are closer "
                 "to one bet than to a diversified set."},
    {"id": "TERM_BACKWARDATION", "fn": _c_term_backwardation,
     "rule": "VIX9D / VIX3M >= 1.0",
     "sentence": "The VIX term structure is in backwardation, which is a stress "
                 "configuration rather than a calm one."},
    {"id": "LIQUIDITY_SHOCK", "fn": _c_liquidity_shock,
     "rule": "netliq 13w change z < -2.0",
     "sentence": "Net liquidity is contracting unusually fast versus its own history."},
    {"id": "VOL_RICH", "fn": _c_vol_rich,
     "rule": "VRP pct > 0.80",
     "sentence": "Insurance is expensive relative to its own trailing year."},
    {"id": "SKEW_VS_VIX_DIVERGENCE", "fn": _c_skew_vs_vix,
     "rule": "SKEW pct > 0.80 AND VIX pct < 0.30",
     "sentence": "Tail hedges are bid while the index itself is calm, so someone is "
                 "paying up for the tail and not the body."},
    {"id": "POSITIONING_UNWIND", "fn": _c_positioning_unwind,
     "rule": "any asset |net %OI change over 4 weeks| >= 0.05",
     "sentence": "At least one crowded position is being unwound quickly."},
    {"id": "ABSORPTION_SPIKE", "fn": _c_absorption_spike,
     "rule": "top-k eigenvalue share > 0.70",
     "sentence": "Cross-asset risk is concentrated in very few factors, so "
                 "diversification is weaker than the position count suggests."},
]


def conditions(state: dict) -> list[dict]:
    """Evaluate every registered condition. Never raises; a broken rule reports
    fired=False with the exception text, so one bad input cannot empty the sheet."""
    out = []
    for c in CONDITIONS:
        try:
            fired, detail = c["fn"](state)
        except Exception as e:  # noqa: BLE001
            fired, detail = False, f"evaluation error: {e}"
        out.append({"id": c["id"], "fired": bool(fired), "detail": detail,
                    "rule": c["rule"], "sentence": c["sentence"]})
    return out


# ---------------------------------------------------------------------- narrative

def narrative(state: dict, fired: list[dict]) -> str:
    """Deterministic template prose built from the SAME evaluated conditions.

    Not generated text. Each fired condition contributes exactly its own registered
    sentence, so the paragraph cannot say anything the flags below it do not.
    """
    vol, liq, cx = state["vol"], state["liq"], state["crowding"]
    lines = []

    pct = vol.get("vrp_pct")
    if pct is not None and not pd.isna(pct):
        band = ("cheap" if pct < 0.25 else "rich" if pct > 0.75 else "middling")
        lines.append(
            f"Volatility is {band}: VRP sits at the {pct * 100:.0f}th percentile of "
            f"its trailing year with VIX at {_f(vol.get('vix'))} and the term "
            f"structure {vol.get('structure') or 'unknown'}.")

    if not cx.empty:
        longs = cx[cx["side"] == "crowded long"]["asset"].tolist()
        shorts = cx[cx["side"] == "crowded short"]["asset"].tolist()
        if longs or shorts:
            parts = []
            if longs:
                parts.append(f"crowded long in {', '.join(longs)}")
            if shorts:
                parts.append(f"crowded short in {', '.join(shorts)}")
            lines.append("Positioning is " + " and ".join(parts) + ".")
        else:
            lines.append("No asset is at a positioning extreme.")

    trend, z = liq.get("trend"), liq.get("netliq_chg13w_z")
    if trend:
        lines.append(f"Net liquidity is {trend} "
                     f"({_f(liq.get('netliq_chg13w'), '{:+.1f}')}bn over 13 weeks, "
                     f"z {_f(z)}).")

    hits = [f for f in fired if f["fired"]]
    if hits:
        lines.append("")
        lines.extend(f["sentence"] for f in hits)
    else:
        lines.append("")
        lines.append("No registered condition is firing today.")

    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


# -------------------------------------------------------------------- state build

def _last_row(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    return {k: v for k, v in df.iloc[-1].to_dict().items()}


def _last_row_asof(df: pd.DataFrame, as_of) -> dict:
    """Last row dated ON OR BEFORE as_of.

    `_last_row` takes the newest row of each frame INDEPENDENTLY, so the sheet could
    carry a liquidity or funding reading from a session after its own header date --
    on 2026-08-24 it printed netliq z -0.7125 and BTC funding z 0.964, both of which
    belong to 2026-08-25. Harmless in direction, but it is the same failure as the
    cross-date term ratio: two readings from two dates presented as one state.
    """
    if df is None or df.empty:
        return {}
    if as_of is None:
        return _last_row(df)
    sub = df[df.index <= pd.Timestamp(as_of)]
    return _last_row(sub if not sub.empty else df)


def build_state(start: str = START) -> dict:
    """Assemble every input into the single dict the sheets and rules read."""
    vol = vol_history(start)
    pos = positioning_history(start)
    fund = funding_history(start)
    liq = liquidity_history(start)
    struct = structure(start)
    # every reading below is taken AS OF this date, not "whatever is newest in its own
    # frame" -- see _last_row_asof
    as_of_ts = vol.index[-1] if not vol.empty else None
    pos_asof = (pos[pos["effective_date"] <= as_of_ts] if as_of_ts is not None else pos)
    latest_cot = (pos_asof.sort_values("effective_date").groupby("asset").tail(1)
                  if not pos_asof.empty else pd.DataFrame())
    cot_idx = (dict(zip(latest_cot["asset"], latest_cot["idx52"]))
               if not latest_cot.empty else {})

    state = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": (str(as_of_ts.date()) if as_of_ts is not None else None),
        "vol": _last_row(vol), "vol_hist": vol,
        "cot_idx": cot_idx, "pos_hist": pos, "pos_latest": latest_cot,
        "funding": _last_row_asof(fund, as_of_ts), "fund_hist": fund,
        "liq": _last_row_asof(liq, as_of_ts), "liq_hist": liq,
        "struct": struct, "crowding": crowding_x_structure(pos_asof, struct),
        "stale": {
            "vol": vol.empty, "cot": pos.empty,
            "funding": fund.empty, "liquidity": liq.empty,
            "structure": struct.get("corr", pd.DataFrame()).empty,
        },
    }
    # Per-asset framework (Trend -> ... -> Risk). Imported lazily so a failure here
    # can never stop the core workbook being written.
    try:
        from . import asset_frame
        ctx = asset_frame.build_context(start)
        state["framework"] = asset_frame.asset_table(ctx) if ctx else pd.DataFrame()
        state["framework_focus"] = [a for a in FRAMEWORK_FOCUS
                                    if a in set(state["framework"].get("asset", []))]
        state["stories"] = ctx.get("stories", []) if ctx else []
    except Exception as e:  # noqa: BLE001
        logger.warning(f"asset framework unavailable: {e}")
        state["framework"] = pd.DataFrame()
        state["framework_focus"] = []
        state["stories"] = []

    state["conditions"] = conditions(state)
    state["narrative"] = narrative(state, state["conditions"])
    return state


def _snapshot(state: dict) -> dict:
    """The numbers behind today's reading, so the log records WHY, not just WHAT.

    A log of fired condition IDs alone cannot answer "was it close?" or "which way is
    this drifting?" -- and those are the only questions a state record is good for.
    Storing the drivers lets a later reader see a condition approach its threshold
    over weeks instead of discovering it the day it flips.

    MEASUREMENTS ONLY. Deliberately no forward return, no realised outcome, no
    scoring of a past reading. Joining this file to future prices would turn the
    workbook into a model and the PBO/DSR gate would apply -- see module docstring.
    """
    vol, liq, st = state.get("vol", {}), state.get("liq", {}), state.get("struct", {})
    fund, cx = state.get("funding", {}), state.get("crowding", pd.DataFrame())

    def _n(v):
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else round(float(v), 6)

    crowded_long, crowded_short = [], []
    if isinstance(cx, pd.DataFrame) and not cx.empty:
        crowded_long = sorted(cx.loc[cx["side"] == "crowded long", "asset"])
        crowded_short = sorted(cx.loc[cx["side"] == "crowded short", "asset"])

    fw = state.get("framework")
    fw_summary = {}
    if isinstance(fw, pd.DataFrame) and not fw.empty:
        fw_summary = {"n_assets": int(len(fw)),
                      "mean_bullish": _n(fw["n_bullish"].mean()),
                      "mean_bearish": _n(fw["n_bearish"].mean()),
                      "mean_no_data": _n(fw["n_no_data"].mean())}

    return {
        "vix": _n(vol.get("vix")), "vrp": _n(vol.get("vrp")),
        "vrp_pct": _n(vol.get("vrp_pct")), "term_ratio": _n(vol.get("term_ratio")),
        "term_stale_days": _n(vol.get("term_stale_days")),
        "vvix": _n(vol.get("vvix")), "skew": _n(vol.get("skew")),
        "skew_pct": _n(vol.get("skew_pct")), "rv20": _n(vol.get("rv20")),
        "netliq_bn": _n(liq.get("netliq")), "netliq_chg13w_bn": _n(liq.get("netliq_chg13w")),
        "netliq_chg13w_z": _n(liq.get("netliq_chg13w_z")), "rrp_bn": _n(liq.get("rrp")),
        "absorption": _n(st.get("absorption")), "top1_share": _n(st.get("top1_share")),
        "btc_funding_z": _n(fund.get("z156")),
        "crowded_long": crowded_long, "crowded_short": crowded_short,
        "framework": fw_summary,
        # the whole point of the story table: a state change gets a DATE attached to it
        # instead of being remembered wrong later. Measurements only, as above.
        "stories": {st["id"]: st["state"] for st in (state.get("stories") or [])},
        "story_corr": {st["id"]: _n(st["corr_fast"])
                       for st in (state.get("stories") or [])},
    }


def append_fire_log(state: dict, path=FIRE_LOG) -> None:
    """Exactly ONE row per session date, rewritten in place on a re-run.

    Idempotent by as_of on purpose. This log is the forward out-of-sample record for
    conditions that were NOT blindly pre-registered, so it is the only evidence they
    will ever have. A plain append would add a duplicate row every time the build runs
    twice in a day (manual rebuild + MarketStateDaily, or any run on a day the tape has
    not moved), silently inflating every fire count in the sheet.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        as_of = state.get("as_of")
        row = {"as_of": as_of,
               "generated_at": state.get("generated_at"),
               "fired": [c["id"] for c in state["conditions"] if c["fired"]],
               **_snapshot(state)}

        rows = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    prev = json.loads(line)
                except ValueError:
                    continue  # skip a torn line rather than losing the whole log
                if prev.get("as_of") != as_of:
                    rows.append(prev)
        rows.append(row)
        rows.sort(key=lambda r: (r.get("as_of") or ""))

        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(path)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"fire log update failed: {e}")
