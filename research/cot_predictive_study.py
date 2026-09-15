"""Does COT positioning predict, or front-run, price? Pre-registered 2026-08-25.

Read `PREREG_cot_predictive_2026-08-25.md` first -- it fixes the hypotheses, the
directions, the statistics and what would count as a positive result, and it was
written before any of this ran.

This joins COT to FORWARD RETURNS, which is exactly the line `market_state` does not
cross. That is deliberate here: this is a study, not a workbook layer, and nothing it
produces is promoted. Anything that looked tradable would still owe PBO < 0.5 and a
Deflated Sharpe > 1.645 on a proper walk-forward.

Run:  .venv\\Scripts\\python.exe research/cot_predictive_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qlib_lab import cot, market_state as ms  # noqa: E402

HORIZONS = {"1w": 5, "4w": 21, "13w": 63}
SIGNALS = ("idx52", "z156", "chg4w")
OUT = Path(__file__).resolve().parent / "cot_predictive_results.json"


def load_prices(assets: list[str]) -> pd.DataFrame:
    px = ms.close_panel(assets, start="2005-01-01")
    return px[px.index < pd.Timestamp.now().normalize()]


def joined(feats: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """One row per (asset, effective week) carrying the signals and the forward returns.

    The entry price is the first close ON OR AFTER the effective date -- never the
    report date, and never a close before the report was public.
    """
    rows = []
    sessions = px.index
    for asset, g in feats.groupby("asset"):
        if asset not in px.columns:
            continue
        s = px[asset].dropna()
        if s.empty:
            continue
        pos = sessions.searchsorted(pd.DatetimeIndex(g["effective_date"]), side="left")
        for (_, r), i in zip(g.iterrows(), pos):
            if i >= len(sessions):
                continue
            entry_date = sessions[i]
            if entry_date not in s.index:
                continue
            entry = float(s.loc[entry_date])
            row = {"asset": asset, "effective_date": r["effective_date"],
                   "entry_date": entry_date, "entry": entry,
                   **{k: r[k] for k in SIGNALS}}
            j = s.index.searchsorted(entry_date)
            for name, n in HORIZONS.items():
                row[f"fwd_{name}"] = (float(s.iloc[j + n]) / entry - 1
                                      if j + n < len(s) else np.nan)
            # the week BEFORE the report is what H5 needs: did positioning follow price?
            row["prev_1w"] = (entry / float(s.iloc[j - 5]) - 1 if j >= 5 else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def spearman(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 20:
        return np.nan, len(d)
    return float(d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman")), len(d)


def ic_table(df: pd.DataFrame, label: str) -> list[dict]:
    """Per asset, per signal, per horizon. Overlapping rows are used for the IC (rank
    correlation of every observation) and the non-overlapping count is reported beside
    it, because that is what any t-statistic would be entitled to use."""
    out = []
    for asset, g in df.groupby("asset"):
        g = g.sort_values("entry_date")
        for sig in SIGNALS:
            for hname, hdays in HORIZONS.items():
                ic, n = spearman(g[sig], g[f"fwd_{hname}"])
                step = max(1, hdays // 5)          # weekly rows -> non-overlapping step
                ind = g.iloc[::step]
                ic_ind, n_ind = spearman(ind[sig], ind[f"fwd_{hname}"])
                out.append({"window": label, "asset": asset, "signal": sig,
                            "horizon": hname, "ic": ic, "n": n,
                            "ic_nonoverlap": ic_ind, "n_nonoverlap": n_ind})
    return out


def extremes(df: pd.DataFrame, label: str) -> list[dict]:
    """H4: crowded-long vs crowded-short buckets against the unconditional mean."""
    out = []
    for asset, g in df.groupby("asset"):
        for hname in HORIZONS:
            f = g[f"fwd_{hname}"]
            hi = g.loc[g["idx52"] >= ms.CROWDED, f"fwd_{hname}"].dropna()
            lo = g.loc[g["idx52"] <= ms.UNCROWDED, f"fwd_{hname}"].dropna()
            base = f.dropna()
            out.append({
                "window": label, "asset": asset, "horizon": hname,
                "n_crowded_long": int(len(hi)), "n_crowded_short": int(len(lo)),
                "mean_crowded_long": None if hi.empty else float(hi.mean()),
                "mean_crowded_short": None if lo.empty else float(lo.mean()),
                "mean_unconditional": None if base.empty else float(base.mean()),
                "spread_long_minus_short": (None if hi.empty or lo.empty
                                            else float(hi.mean() - lo.mean())),
            })
    return out


def front_running(df: pd.DataFrame, label: str) -> list[dict]:
    """H5, the decisive one: is the weekly change in positioning better explained by the
    week that just happened, or by the week that follows?"""
    out = []
    for asset, g in df.groupby("asset"):
        g = g.sort_values("entry_date").copy()
        g["d_net"] = g["chg4w"].diff()          # week-over-week change in the flow
        back, n_b = spearman(g["chg4w"], g["prev_1w"])
        fwd, n_f = spearman(g["chg4w"], g["fwd_1w"])
        out.append({"window": label, "asset": asset,
                    "corr_chg4w_prev_week": back, "n_prev": n_b,
                    "corr_chg4w_next_week": fwd, "n_next": n_f,
                    "leads": (None if pd.isna(back) or pd.isna(fwd)
                              else bool(abs(fwd) > abs(back)))})
    return out


def summarise(ics: list[dict], window: str) -> list[dict]:
    """The verdict is read from the DISTRIBUTION across assets, never from one cell."""
    df = pd.DataFrame([r for r in ics if r["window"] == window])
    rows = []
    for (sig, h), g in df.groupby(["signal", "horizon"]):
        v = g["ic"].dropna()
        if v.empty:
            continue
        want_neg = sig == "idx52"          # H1 predicts negative, H2/H3 positive
        agree = int((v < 0).sum() if want_neg else (v > 0).sum())
        rows.append({"window": window, "signal": sig, "horizon": h,
                     "n_assets": int(len(v)), "median_ic": float(v.median()),
                     "mean_ic": float(v.mean()),
                     "assets_with_predicted_sign": agree,
                     "predicted_sign": "negative" if want_neg else "positive",
                     "passes_9_of_13": bool(agree >= 9),
                     "passes_median_abs_ic_005": bool(abs(v.median()) >= 0.05)})
    return rows


def main() -> None:
    hist = pd.read_parquet(cot.COT_PARQUET)
    feats = cot.weekly_features(hist)
    assets = sorted(feats["asset"].unique())
    px = load_prices(assets)
    print(f"prices: {px.shape[1]} assets, {px.index.min().date()} -> {px.index.max().date()}")

    full = joined(feats, px)
    cutoff = feats["effective_date"].max() - pd.Timedelta(weeks=52)
    last12 = full[full["effective_date"] >= cutoff]
    print(f"joined rows: full {len(full)}, 12m {len(last12)} (since {cutoff.date()})")

    result = {
        "generated": pd.Timestamp.utcnow().isoformat(),
        "prereg": "research/PREREG_cot_predictive_2026-08-25.md",
        "rows": {"full": int(len(full)), "12m": int(len(last12))},
        "cutoff_12m": str(cutoff.date()),
        "ic": ic_table(full, "full") + ic_table(last12, "12m"),
        "extremes": extremes(full, "full") + extremes(last12, "12m"),
        "front_running": front_running(full, "full") + front_running(last12, "12m"),
    }
    result["summary"] = (summarise(result["ic"], "full")
                         + summarise(result["ic"], "12m"))
    OUT.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")

    print("\n=== H1/H2/H3 across assets (distribution, not a headline) ===")
    s = pd.DataFrame(result["summary"])
    print(s.to_string(index=False))

    print("\n=== H5 front-running: |corr with NEXT week| vs |corr with PRIOR week| ===")
    fr = pd.DataFrame(result["front_running"])
    for w in ("full", "12m"):
        g = fr[fr["window"] == w].dropna(subset=["leads"])
        print(f"{w:4s}  leads price in {int(g['leads'].sum())}/{len(g)} assets; "
              f"median |prev| {g['corr_chg4w_prev_week'].abs().median():.3f} vs "
              f"median |next| {g['corr_chg4w_next_week'].abs().median():.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
