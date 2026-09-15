"""Every registered reading as a DAILY TIME SERIES, not just today's value.

`market_state` answers "what is the market doing now". Four separate pieces of open
work -- the registry correlation matrix, the regime feasibility read, multiple
lookbacks, and the correlation graph -- all need the same thing first: the same
readings, every session, back as far as the inputs allow. This module builds that once.

Two readings exist only for the latest window and are recomputed here on a rolling
basis, reusing the originals' maths rather than re-porting it:

  * `market_state.structure()` takes `.tail(CORR_WINDOW)` -- absorption and top1_share
    are latest-only. Here the same eigenvalue share is computed on every session's
    trailing window.
  * `asset_frame._breadth()` is likewise latest-only, and carries the exclusion list
    that keeps the VIX family and FX pairs out of the denominator. That list is
    imported, never restated -- if it drifts, breadth history drifts with it.

Everything else is read from the history functions that already exist.

HONESTY. This panel is a reconstruction, and a reconstruction can lie in two ways:

  1. Lookahead. COT is weekly and is forward-filled from `effective_date`
     (report + 6 days), never `report_date`. Funding, liquidity and vol are daily and
     as-published. No column is shifted backwards.
  2. Survivorship of the window. The qlib store keeps a ROLLING 10-year window, so the
     first usable date moves forward every day. A panel built today is not the panel
     that would have been built a year ago, and any claim about 2016 is a claim about
     what the store holds now.

Descriptive, like the rest of `market_state`: nothing here is joined to a forward
return, so the PBO/DSR gate does not apply. That holds only while it stays true.

    .venv\\Scripts\\python.exe -m qlib_lab.state_panel --start 2016-01-01
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from . import asset_frame as af
from . import config, cot, market_state as ms

logger = logging.getLogger("qlib_lab.state_panel")

PANEL_PATH = config.DATA_DIR / "state_panel.parquet"
COVERAGE_PATH = config.JOURNAL_DIR / "exports" / "state_panel_coverage.json"

# The COT assets carry both a positioning reading and a price, which is what the
# crowding x correlation cross-term needs on every date.
STRUCT_ASSETS = ms.COT_ASSETS


# ------------------------------------------------------------------ rolling pieces

def rolling_structure(close: pd.DataFrame, window: int = ms.CORR_WINDOW) -> pd.DataFrame:
    """Absorption and top-1 eigenvalue share on every session's trailing window.

    Same construction as `market_state.structure()` -- k = n//4 of the eigenvalue mass
    -- applied per date instead of once at the end.
    """
    cols = [c for c in STRUCT_ASSETS if c in close.columns]
    rets = close[cols].pct_change()
    out = pd.DataFrame(index=close.index, columns=["absorption", "top1_share",
                                                   "n_struct_assets"], dtype=float)
    for i in range(window, len(rets)):
        # Two alignment traps, both caught by reconciling against the live workbook:
        #   * the window is INCLUSIVE of session i -- `structure()` calls .tail(window)
        #     on data that already contains the last row (exclusive printed 0.7177 vs
        #     the published 0.7165);
        #   * the column selection is dropna(how="all") with a PAIRWISE corr, not
        #     complete-case -- dropping any column with a single gap changes n and moves
        #     absorption in the fourth decimal.
        w = rets.iloc[i - window + 1:i + 1].dropna(axis=1, how="all")
        if w.shape[1] < 3:
            continue
        absorption, top1, n = ms.absorption_from_corr(w.corr())
        if absorption is None:
            continue
        out.iloc[i] = [absorption, top1, n]
    return out


def rolling_breadth(close: pd.DataFrame) -> pd.DataFrame:
    """Share of direction-comparable instruments above EMA50 / EMA200, per session.

    The exclusion list comes from `asset_frame.BREADTH_EXCLUDE`: counting the VIX family
    inverts the reading exactly when it matters, and FX pairs contribute a bit whose sign
    is a naming convention.
    """
    keep = [c for c in close.columns if c not in af.BREADTH_EXCLUDE]
    px = close[keep]
    e50 = px.ewm(span=af.EMA_MID, adjust=False).mean()
    e200 = px.ewm(span=af.EMA_SLOW, adjust=False).mean()
    # only count an instrument once its slow EMA is actually warm, per COLUMN --
    # a young series otherwise votes on an EMA seeded by its own first price
    warm = px.notna().cumsum() >= af.EMA_SLOW
    valid = px.notna() & warm
    n = valid.sum(axis=1)
    above50 = ((px > e50) & valid).sum(axis=1)
    above200 = ((px > e200) & valid).sum(axis=1)
    out = pd.DataFrame(index=close.index)
    out["breadth_above_50"] = (above50 / n).where(n > 0)
    out["breadth_above_200"] = (above200 / n).where(n > 0)
    out["breadth_n"] = n
    return out


def rolling_stories(close: pd.DataFrame) -> pd.DataFrame:
    """The registered STORIES as time series: the fast correlation, the slow one, and
    the resulting state, on every session. `ms._story_state` is the same signed rule the
    workbook prints -- imported, not restated."""
    out = pd.DataFrame(index=close.index)
    for st in ms.STORIES:
        a, b = st["pair"]
        if a not in close.columns or b not in close.columns:
            continue
        ra, rb = close[a].pct_change(), close[b].pct_change()
        fast = ra.rolling(ms.STORY_CORR_FAST, min_periods=20).corr(rb)
        slow = ra.rolling(ms.STORY_CORR_SLOW, min_periods=60).corr(rb)
        out[f"story_corr_{st['id']}"] = fast
        out[f"story_corr_slow_{st['id']}"] = slow
        out[f"story_state_{st['id']}"] = [ms._story_state(c, st["thr"]) for c in fast]
    return out


# ------------------------------------------------------------------ condition replay

def _state_for_date(date, vol_row, liq_row, fund_row, cot_asof, corr, struct_row) -> dict:
    """The minimal state dict `ms.conditions` reads, rebuilt for one past session."""
    cx = ms.crowding_x_structure(cot_asof, {"corr": corr})
    idx = (dict(zip(cot_asof["asset"], cot_asof["idx52"]))
           if not cot_asof.empty else {})
    return {
        "as_of": str(pd.Timestamp(date).date()),
        "vol": vol_row, "liq": liq_row, "funding": fund_row,
        "cot_idx": idx, "crowding": cx,
        "struct": {"absorption": struct_row.get("absorption"),
                   "top1_share": struct_row.get("top1_share"), "corr": corr},
    }


def replay_conditions(close: pd.DataFrame, vol: pd.DataFrame, liq: pd.DataFrame,
                      fund: pd.DataFrame, pos: pd.DataFrame,
                      struct: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the 8 registered CONDITIONS on every session.

    NOT a track record. `CONDITIONS_NOTE` says the thresholds were written on
    2026-08-23 with the August tape visible, so every row before that date is
    BACKFILLED and is evidence of nothing. It is built so the registry matrix can ask
    how much the conditions duplicate each other -- a question about the instruments,
    not about the market.
    """
    cols = [c for c in STRUCT_ASSETS if c in close.columns]
    rets = close[cols].pct_change()
    pos = pos.sort_values("effective_date")
    ids = [c["id"] for c in ms.CONDITIONS]
    out = pd.DataFrame(index=close.index, columns=[f"cond_{i}" for i in ids], dtype=float)

    for i, date in enumerate(close.index):
        if i < ms.CORR_WINDOW or pd.isna(struct["absorption"].iloc[i]):
            continue
        w = rets.iloc[i - ms.CORR_WINDOW + 1:i + 1].dropna(axis=1, how="any")
        if w.shape[1] < 3:
            continue
        corr = w.corr()
        asof = pos[pos["effective_date"] <= date]
        asof = (asof.groupby("asset").tail(1) if not asof.empty else pd.DataFrame())
        state = _state_for_date(
            date,
            vol.loc[date].to_dict() if date in vol.index else {},
            liq.loc[date].to_dict() if date in liq.index else {},
            fund.loc[date].to_dict() if date in fund.index else {},
            asof, corr, struct.iloc[i].to_dict())
        for c in ms.conditions(state):
            out.at[date, f"cond_{c['id']}"] = float(c["fired"])
    return out


# ------------------------------------------------------------------------- assembly

def build_panel(start: str = ms.START, with_conditions: bool = True) -> pd.DataFrame:
    names = sorted(config.UNIVERSE.keys())
    close = ms.close_panel(names, start=ms.WARMUP)
    if close.empty:
        return pd.DataFrame()
    close = close[close.index < pd.Timestamp.now().normalize()]

    vol = ms.vol_history(start=ms.WARMUP)
    liq = ms.liquidity_history(start=ms.WARMUP)
    fund = ms.funding_history(start=ms.WARMUP)
    pos = ms.positioning_history(start="2006-01-01")

    panel = pd.DataFrame(index=close.index)
    for col in ("vix", "vrp", "vrp_pct", "vix_pct", "term_ratio", "term_stale_days",
                "vvix", "skew", "skew_pct", "rv20"):
        if col in vol.columns:
            panel[f"vol_{col}"] = vol[col].reindex(close.index)
    for col in ("netliq", "netliq_chg13w", "netliq_chg13w_z", "rrp", "walcl", "tga"):
        if col in liq.columns:
            panel[f"liq_{col}"] = liq[col].reindex(close.index).ffill()
    for col in ("z156", "idx52w", "funding_1d_annualized_pct"):
        if col in fund.columns:
            panel[f"fund_{col}"] = fund[col].reindex(close.index).ffill()

    struct = rolling_structure(close)
    panel["struct_absorption"] = struct["absorption"]
    panel["struct_top1_share"] = struct["top1_share"]
    panel = panel.join(rolling_breadth(close))

    # COT: ffilled from effective_date (report + 6d), never report_date
    daily = cot.daily_panel(pos, close.index, cols=("idx52", "z156", "chg4w"))
    for asset, g in daily.items():
        for col in ("idx52", "z156", "chg4w"):
            panel[f"cot_{col}_{asset}"] = g[col]
    idx_cols = [c for c in panel.columns if c.startswith("cot_idx52_")]
    if idx_cols:
        # how one-sided the whole COT complex is, in one column
        panel["cot_n_crowded"] = ((panel[idx_cols] >= ms.CROWDED).sum(axis=1)
                                  + (panel[idx_cols] <= ms.UNCROWDED).sum(axis=1))

    panel = panel.join(rolling_stories(close))

    if with_conditions:
        panel = panel.join(replay_conditions(close, vol, liq, fund, pos, struct))

    return panel[panel.index >= pd.Timestamp(start)]


def coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """First and last date each column is actually available, and how much is missing.

    This is the deliverable that decides what any later study may claim: a column that
    starts in 2019 truncates every joint analysis it takes part in.
    """
    rows = []
    for c in panel.columns:
        s = panel[c].dropna()
        rows.append({"column": c, "n": int(len(s)),
                     "first": None if s.empty else str(s.index.min().date()),
                     "last": None if s.empty else str(s.index.max().date()),
                     "pct_null": round(1 - len(s) / max(1, len(panel)), 4)})
    return pd.DataFrame(rows).sort_values(["first", "column"], na_position="last")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=ms.START)
    ap.add_argument("--no-conditions", action="store_true",
                    help="skip the per-session condition replay (much faster)")
    a = ap.parse_args()

    panel = build_panel(a.start, with_conditions=not a.no_conditions)
    if panel.empty:
        print("empty panel - qlib store unavailable?")
        return
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH)
    cov = coverage(panel)
    COVERAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COVERAGE_PATH.write_text(json.dumps(cov.to_dict("records"), indent=1),
                             encoding="utf-8")

    print(f"panel: {panel.shape[0]} sessions x {panel.shape[1]} columns "
          f"({panel.index.min().date()} -> {panel.index.max().date()})")
    print(f"wrote {PANEL_PATH}")
    print("\nwhat truncates the panel (earliest-starting columns last):")
    print(cov.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
