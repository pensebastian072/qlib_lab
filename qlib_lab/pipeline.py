"""Alpha158 + LightGBM walk-forward pipeline behind the canonical overfit gate.

Uses qlib for what it is good at — the .bin store and the Alpha158 expression
library (via Alpha158DL feature config + D.features) — and keeps the honest
walk-forward / portfolio-PnL / gate machinery identical in shape to
macro_gpu_lab/models/baseline_rf.py::oos_portfolio_pnls:

  - expanding window, refit every REFIT_EVERY sessions,
  - a gap of `horizon` sessions between train labels and the prediction date
    (train labels may not overlap the OOS window — no peeking),
  - rebalance every `horizon` days into top-TOPK long / bottom-TOPK short of the
    cross-section → ONE non-overlapping PnL per rebalance,
  - per-side cost haircut as a FRACTION of notional (never absolute units),
  - gate = macro_gpu_lab.validate.evaluate_gate (canonical PBO/DSR port,
    n_trials-deflated). Both horizons must pass or the model stays SHADOW.

Label: LABEL(h) = Ref($close,-h)/$close - 1 — forward h-session return from
today's close. Features at t use data through t only (verified in tests).

qrun/MLflow/DatasetH are deliberately NOT used: no server, no mlruns clutter,
and the gate math must stay the canonical one, not qlib's IC reports.
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import datetime, timezone

# lightgbm's sklearn wrapper warns on ndarray-predict after DataFrame-fit; benign
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger("qlib_lab.pipeline")

REFIT_EVERY = 126         # sessions between LightGBM refits (semiannual — the
                          # ~76-ETF panel doubles fit cost; 63 blew the task window)
COST_PER_SIDE = 0.0005    # 5 bps per side per rebalance, fraction of notional

# canonical gate — quant-research-gate skill: reuse, never re-port
sys.path.insert(0, str(config.MACRO_GPU_LAB_DIR))
from macro_gpu_lab.validate import evaluate_gate  # noqa: E402


def _init_qlib() -> None:
    import qlib
    from qlib.constant import REG_US

    qlib.init(provider_uri=str(config.QLIB_DATA_DIR), region=REG_US)


def load_features_and_labels() -> tuple[pd.DataFrame, dict[int, pd.Series]]:
    """Alpha158 feature frame + per-horizon forward-return labels.

    Instruments = config.MODEL_UNIVERSE (US-listed ETF book only; FX/VIX/BTC
    stay context). Returns (features indexed by (instrument, datetime),
    {h: label series}).
    """
    from qlib.contrib.data.loader import Alpha158DL
    from qlib.data import D

    fields, names = Alpha158DL.get_feature_config()
    instruments = [a for a in config.MODEL_UNIVERSE]
    feat = D.features(instruments, fields)
    feat.columns = names

    label_fields = [f"Ref($close, -{h})/$close - 1" for h in config.HORIZONS_DAYS]
    lab = D.features(instruments, label_fields)
    lab.columns = [f"LABEL_{h}" for h in config.HORIZONS_DAYS]
    labels = {h: lab[f"LABEL_{h}"] for h in config.HORIZONS_DAYS}
    return feat, labels


def _fit_lgbm(x: pd.DataFrame, y: pd.Series):
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(**config.LGB_PARAMS)
    model.fit(x.values, y.values)
    return model


def oos_portfolio_pnls(feat: pd.DataFrame, label: pd.Series, horizon: int) -> tuple[list[float], list[str]]:
    """Expanding walk-forward → one long-short portfolio PnL per rebalance."""
    dates = feat.index.get_level_values(1).unique().sort_values()
    n = len(dates)
    pnls: list[float] = []
    stamps: list[str] = []
    model = None
    last_fit_i = -10**9

    by_date = feat.swaplevel().sort_index()          # (datetime, instrument)
    lab_by_date = label.swaplevel().sort_index()

    i = config.MIN_TRAIN_DAYS
    while i < n - horizon:
        t = dates[i]
        if i - last_fit_i >= REFIT_EVERY:
            # train on labels fully realized before t: label date d needs d+h <= t
            cutoff = dates[i - horizon]
            tr_mask = feat.index.get_level_values(1) < cutoff
            x_tr = feat[tr_mask]
            y_tr = label[tr_mask]
            keep = y_tr.notna()
            x_tr, y_tr = x_tr[keep], y_tr[keep]
            if len(y_tr) < 500:
                i += horizon
                continue
            model = _fit_lgbm(x_tr, y_tr)
            last_fit_i = i

        x_t = by_date.loc[t] if t in by_date.index.get_level_values(0) else None
        if model is None or x_t is None or len(x_t) < 2 * config.TOPK + 1:
            i += horizon
            continue
        scores = pd.Series(model.predict(x_t.values), index=x_t.index)
        y_t = lab_by_date.loc[t].reindex(scores.index)
        ranked = scores.dropna().sort_values(ascending=False)
        top = y_t.reindex(ranked.index[: config.TOPK]).dropna()
        bot = y_t.reindex(ranked.index[-config.TOPK:]).dropna()
        if len(top) and len(bot):
            gross = float(top.mean() - bot.mean())
            pnls.append(gross - 2 * COST_PER_SIDE)
            stamps.append(str(t.date()))
        i += horizon

    return pnls, stamps


def latest_scores(feat: pd.DataFrame, labels: dict[int, pd.Series]) -> dict:
    """Fit on the full history per horizon, score the newest session."""
    dates = feat.index.get_level_values(1).unique().sort_values()
    t = dates[-1]
    by_date = feat.swaplevel().sort_index()
    x_t = by_date.loc[t]

    out: dict[str, dict] = {}
    for h, label in labels.items():
        keep = label.notna()
        model = _fit_lgbm(feat[keep], label[keep])
        scores = pd.Series(model.predict(x_t.values), index=x_t.index).dropna()
        rank_pct = scores.rank(pct=True)
        ranked = scores.sort_values(ascending=False)
        longs = set(ranked.index[: config.TOPK])
        shorts = set(ranked.index[-config.TOPK:])
        for asset in scores.index:
            a = out.setdefault(str(asset), {"horizons": {}})
            a["horizons"][f"{h}d"] = {
                "score": round(float(scores[asset]), 6),
                "rank_pct": round(float(rank_pct[asset]), 4),
                "lean": 1 if asset in longs else (-1 if asset in shorts else 0),
            }
    for asset, a in out.items():
        leans = [hh["lean"] for hh in a["horizons"].values()]
        rps = [hh["rank_pct"] for hh in a["horizons"].values()]
        a["lean"] = int(np.sign(sum(leans)))
        a["conviction"] = round(float(abs(np.mean(rps) - 0.5) * 2), 4)
        a["force"] = config.FORCE.get(asset, "other")
    return {"data_through": str(t.date()), "assets": out}


def run_once() -> dict:
    _init_qlib()
    from . import macro_context

    feat, labels = load_features_and_labels()
    ctx = macro_context.build_context()
    augmented = macro_context.augment(feat, ctx)
    # per-instrument CFTC positioning (release-lagged; NaN off the mapped set)
    from . import cot

    cot_feat = cot.per_instrument_features(feat.index)
    if cot_feat is not None:
        augmented = pd.concat([augmented, cot_feat], axis=1)
    variants = {
        "alpha158": feat,
        "alpha158_macro": augmented,
    }
    # honest deflation: every model variant x horizon is a trial
    n_trials = len(variants) * len(config.HORIZONS_DAYS)

    models: dict[str, dict] = {}
    for mname, mfeat in variants.items():
        horizons: dict[str, dict] = {}
        for h in config.HORIZONS_DAYS:
            pnls, stamps = oos_portfolio_pnls(mfeat, labels[h], h)
            verdict = evaluate_gate(np.array(pnls), n_trials=n_trials)
            verdict["first_oos"] = stamps[0] if stamps else None
            verdict["last_oos"] = stamps[-1] if stamps else None
            horizons[f"{h}d"] = verdict
            logger.info(
                f"{mname} h={h}d trades={len(pnls)} passes={verdict['passes']} "
                f"dsr={verdict.get('deflated_sharpe')} pbo={verdict.get('pbo')}"
            )
        models[mname] = {
            "horizons": horizons,
            "promoted": bool(all(v.get("passes") for v in horizons.values())),
        }

    # signals from the macro-augmented variant (advisory either way)
    sig = latest_scores(variants["alpha158_macro"], labels)
    promoted = any(m["promoted"] for m in models.values())
    scorecard = {
        "model": "qlib_lgbm_alpha158_macro",
        "as_of": datetime.now(timezone.utc).isoformat(),
        "data_through": sig["data_through"],
        "universe": sorted({str(a) for a in feat.index.get_level_values(0).unique()}),
        "topk": config.TOPK,
        "cost_per_side": COST_PER_SIDE,
        "refit_every": REFIT_EVERY,
        "n_trials": n_trials,
        "context_features": [c for c in augmented.columns if c not in feat.columns],
        # flag/UI schema: top-level horizons = the augmented variant's verdicts
        "horizons": models["alpha158_macro"]["horizons"],
        "models": models,
        "promoted": bool(promoted),
        "status": "trained",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (config.SCORECARDS_DIR / f"qlib_lgbm_{stamp}.json").write_text(
        json.dumps(scorecard, indent=2, default=str)
    )
    return {"scorecard": scorecard, "signals": sig}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_once()
    from . import publish

    state = publish.build_state(result["scorecard"], result["signals"])
    path = publish.publish(state)
    print(f"flag -> {path}")
    sc = result["scorecard"]
    print(f"promoted={sc['promoted']} (SHADOW until gate clears on every horizon)")
    for hk, v in sc["horizons"].items():
        print(f"  {hk}: passes={v['passes']} n={v.get('n_trades')} "
              f"dsr={v.get('deflated_sharpe')} pbo={v.get('pbo')} pf={v.get('profit_factor')}")


if __name__ == "__main__":
    main()
