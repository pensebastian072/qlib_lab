"""Point-in-time per-asset score/lean history — the evidence export.

`pipeline.oos_portfolio_pnls` already walks forward correctly but keeps only the
long-short portfolio PnL, discarding the per-asset cross-section. Downstream research
(options_desk) needs the cross-section itself: WHICH asset the model leaned on, on
WHICH date, decided using only data available then.

This module re-runs the identical walk-forward and exports that cross-section. It
reproduces `pipeline.latest_scores` semantics exactly:

  * lean is a CROSS-SECTIONAL RANK: the top-TOPK assets score +1, the bottom-TOPK -1,
    everything else 0. With TOPK=8 over a 76-asset universe only 8 assets can be
    bullish on any date, so the filter is genuinely selective by construction.
  * conviction = |mean(rank_pct) - 0.5| * 2 across horizons.
  * combined lean = sign(sum of per-horizon leans).

No-lookahead guarantees, inherited from the pipeline:
  * refit only every REFIT_EVERY sessions, on labels fully realized before the refit
    date (cutoff = dates[i - horizon]), so a label never overlaps the scored window;
  * a model is only ever applied FORWARD of the data it was fit on.

Research/advisory only. Writes one parquet export; publishes no flag and places no order.

    .venv\\Scripts\\python.exe -m qlib_lab.score_history
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import pandas as pd

from . import config
from .pipeline import REFIT_EVERY, _fit_lgbm, _init_qlib, load_features_and_labels

logger = logging.getLogger("qlib_lab.score_history")

EXPORT_DIR = config.JOURNAL_DIR / "exports"
EXPORT_PATH = EXPORT_DIR / "qlib_score_history.parquet"
META_PATH = EXPORT_DIR / "qlib_score_history.json"


def horizon_scores(feat: pd.DataFrame, label: pd.Series, horizon: int,
                   min_train_days: int | None = None,
                   refit_every: int = REFIT_EVERY) -> pd.DataFrame:
    """Expanding walk-forward; score EVERY session (not just rebalances).

    Returns long rows: date, asset, score, rank_pct, lean.
    """
    min_train_days = config.MIN_TRAIN_DAYS if min_train_days is None else min_train_days
    dates = feat.index.get_level_values(1).unique().sort_values()
    n = len(dates)
    by_date = feat.swaplevel().sort_index()          # (datetime, instrument)

    model = None
    last_fit_i = -10**9
    frames: list[pd.DataFrame] = []
    n_fits = 0

    for i in range(min_train_days, n):
        t = dates[i]
        if i - last_fit_i >= refit_every:
            # Purge: train only on labels fully realized before t (d + horizon <= t).
            cutoff = dates[i - horizon]
            mask = feat.index.get_level_values(1) < cutoff
            x_tr, y_tr = feat[mask], label[mask]
            keep = y_tr.notna()
            x_tr, y_tr = x_tr[keep], y_tr[keep]
            if len(y_tr) < 500:
                continue
            model = _fit_lgbm(x_tr, y_tr)
            last_fit_i = i
            n_fits += 1
        if model is None:
            continue
        if t not in by_date.index.get_level_values(0):
            continue
        x_t = by_date.loc[t]
        if len(x_t) < 2 * config.TOPK + 1:
            continue
        scores = pd.Series(model.predict(x_t.values), index=x_t.index).dropna()
        if scores.empty:
            continue
        ranked = scores.sort_values(ascending=False)
        longs = set(ranked.index[: config.TOPK])
        shorts = set(ranked.index[-config.TOPK:])
        frames.append(pd.DataFrame({
            "date": t,
            "asset": [str(a) for a in scores.index],
            "score": scores.to_numpy(dtype=float),
            "rank_pct": scores.rank(pct=True).to_numpy(dtype=float),
            "lean": [1 if a in longs else (-1 if a in shorts else 0) for a in scores.index],
        }))

    logger.info("horizon %sd: %d fits, %d scored sessions", horizon, n_fits, len(frames))
    if not frames:
        return pd.DataFrame(columns=["date", "asset", "score", "rank_pct", "lean"])
    out = pd.concat(frames, ignore_index=True)
    out.attrs["n_fits"] = n_fits
    return out


def combine(per_horizon: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Merge horizons into one row per (date, asset), matching latest_scores()."""
    merged: pd.DataFrame | None = None
    for horizon, frame in sorted(per_horizon.items()):
        renamed = frame.rename(columns={
            "score": f"score_{horizon}d", "rank_pct": f"rank_pct_{horizon}d",
            "lean": f"lean_{horizon}d"})
        merged = renamed if merged is None else merged.merge(
            renamed, on=["date", "asset"], how="outer")
    if merged is None or merged.empty:
        return pd.DataFrame()
    lean_cols = [c for c in merged.columns if c.startswith("lean_")]
    rank_cols = [c for c in merged.columns if c.startswith("rank_pct_")]
    merged["lean"] = np.sign(merged[lean_cols].sum(axis=1)).astype(int)
    merged["conviction"] = (merged[rank_cols].mean(axis=1) - 0.5).abs() * 2
    merged["conviction"] = merged["conviction"].round(4)
    merged["force"] = merged["asset"].map(lambda a: config.FORCE.get(a, "other"))
    return merged.sort_values(["date", "asset"]).reset_index(drop=True)


def run(min_train_days: int | None = None, refit_every: int = REFIT_EVERY) -> pd.DataFrame:
    _init_qlib()
    feat, labels = load_features_and_labels()
    per_horizon: dict[int, pd.DataFrame] = {}
    fits: dict[str, int] = {}
    for horizon, label in labels.items():
        frame = horizon_scores(feat, label, horizon, min_train_days, refit_every)
        per_horizon[horizon] = frame
        fits[f"{horizon}d"] = int(frame.attrs.get("n_fits", 0))
    combined = combine(per_horizon)
    combined.attrs["fits"] = fits
    return combined


def write(frame: pd.DataFrame, path: Path | None = None) -> str:
    target = path or EXPORT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(frame)),
        "assets": int(frame["asset"].nunique()) if len(frame) else 0,
        "date_start": str(frame["date"].min().date()) if len(frame) else None,
        "date_end": str(frame["date"].max().date()) if len(frame) else None,
        "topk": config.TOPK,
        "universe_size": len(config.MODEL_UNIVERSE),
        "horizons": config.HORIZONS_DAYS,
        "refit_every_sessions": REFIT_EVERY,
        "min_train_days": config.MIN_TRAIN_DAYS,
        "fits": frame.attrs.get("fits", {}),
        "semantics": ("lean = cross-sectional rank: top-TOPK +1, bottom-TOPK -1, else 0; "
                      "conviction = |mean(rank_pct) - 0.5| * 2; combined lean = sign(sum)."),
        "no_lookahead": ("refit every REFIT_EVERY sessions on labels realized before the "
                         "refit date (cutoff = dates[i - horizon]); models applied forward only"),
        "status": "SHADOW research export; no flag published, no order placed",
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return str(target)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Export point-in-time per-asset qlib scores")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--refit-every", type=int, default=REFIT_EVERY)
    parser.add_argument("--min-train-days", type=int, default=None)
    args = parser.parse_args()
    frame = run(args.min_train_days, args.refit_every)
    if frame.empty:
        print("no scores produced (insufficient history)")
        return
    path = write(frame, args.out)
    dates = frame["date"].nunique()
    bullish = int((frame["lean"] == 1).sum())
    print(f"rows={len(frame)} assets={frame['asset'].nunique()} sessions={dates}")
    print(f"bullish rows={bullish} ({bullish / len(frame):.1%} of cross-section)")
    print(f"export -> {path}")


if __name__ == "__main__":
    main()
