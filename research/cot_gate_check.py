"""POST-HOC gate check on the ONE COT cell that looked least dead.

Read this label first: the rule tested here was chosen AFTER seeing the pre-registered
study's output (`cot_predictive_results.json`), so this is not a test, it is an upper
bound. It exists to answer "and if we traded the best-looking thing anyway?" -- and the
honest deflation for that choice is `n_trials = 117`, the number of cells the study
looked at (3 signals x 3 horizons x 13 assets).

Rule (the crowding reading the workbook already prints, turned into a bet):
    on each weekly effective date, SHORT the assets with idx52 >= 0.80 and
    LONG those with idx52 <= 0.20, hold 13 weeks, non-overlapping.

Gate math imported from copper_brain.validate -- the canonical port. Not re-implemented.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, r"C:\Users\<your-user>\copper_brain")

from copper_brain import validate  # noqa: E402
from qlib_lab import cot, market_state as ms  # noqa: E402

HOLD_WEEKS = 13
N_TRIALS = 117            # cells inspected in the pre-registered study


def main() -> None:
    hist = pd.read_parquet(cot.COT_PARQUET)
    feats = cot.weekly_features(hist).dropna(subset=["idx52"])
    px = ms.close_panel(sorted(feats["asset"].unique()), start="2005-01-01")
    px = px[px.index < pd.Timestamp.now().normalize()]

    trades = []
    for asset, g in feats.groupby("asset"):
        if asset not in px.columns:
            continue
        s = px[asset].dropna()
        g = g.sort_values("effective_date")
        # non-overlapping: one entry every HOLD_WEEKS weekly rows
        for _, r in g.iloc[::HOLD_WEEKS].iterrows():
            side = -1 if r["idx52"] >= ms.CROWDED else 1 if r["idx52"] <= ms.UNCROWDED else 0
            if not side:
                continue
            i = s.index.searchsorted(r["effective_date"])
            j = i + HOLD_WEEKS * 5
            if j >= len(s):
                continue
            ret = float(s.iloc[j]) / float(s.iloc[i]) - 1
            trades.append({"asset": asset, "date": s.index[i], "side": side,
                           "pnl": side * ret})

    tr = pd.DataFrame(trades).sort_values("date")
    pnls = tr["pnl"].tolist()
    gate = validate.evaluate_gate(pnls, n_trials=N_TRIALS)
    dsr1 = validate.deflated_sharpe(pnls, n_trials=1)

    out = {
        "label": "POST-HOC, chosen after seeing the study output; not evidence",
        "rule": f"short idx52>={ms.CROWDED}, long idx52<={ms.UNCROWDED}, hold {HOLD_WEEKS}w, non-overlapping",
        "n_trades": len(pnls),
        "n_assets": int(tr["asset"].nunique()),
        "date_range": [str(tr["date"].min().date()), str(tr["date"].max().date())],
        "mean_pnl": float(np.mean(pnls)), "hit_rate": float(np.mean([p > 0 for p in pnls])),
        "profit_factor": validate.profit_factor(pnls),
        "sharpe_per_trade": validate.sharpe(pnls),
        "gate_n_trials_117": gate,
        "deflated_sharpe_if_only_one_trial": dsr1,
    }
    Path(__file__).with_name("cot_gate_results.json").write_text(
        json.dumps(out, indent=1, default=str), encoding="utf-8")

    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("gate_n_trials_117", "deflated_sharpe_if_only_one_trial")},
                     indent=1, default=str))
    print("\ngate (n_trials=117):", json.dumps(gate, indent=1, default=str))
    print("\nDSR if this had been the ONLY thing ever tried:",
          json.dumps(dsr1, indent=1, default=str))


if __name__ == "__main__":
    main()
