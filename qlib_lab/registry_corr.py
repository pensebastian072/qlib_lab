"""How much independent information does the REGISTRY actually hold?

The framework prints twelve layers, eight conditions and ten stories and then counts how
many agree. That count is only meaningful if the readings are independent, and nothing in
this repo has ever checked. Trend, momentum and relative strength are all price; macro is
a correlation to SPY; several conditions read the same three numbers. "Eight of twelve
agree" may be four facts counted twice each.

This module correlates the registry against ITSELF -- the same eigenvalue-share
construction `market_state.structure()` applies to the market, applied to our own
instruments instead -- and reports the effective number of independent signals.

Two ways to read a correlation here, and they answer different questions, so both are
computed and neither is allowed to stand alone:

  LEVELS  do two readings say the same thing about the state of the market today?
          This is the question the confluence tally depends on. Levels are strongly
          autocorrelated, so the correlation is estimated on very few independent
          observations -- `decorrelation_time` prints how few.
  CHANGES do two readings MOVE together? Immune to the spurious-level problem, but it
          answers a different question, and a pair can be redundant in levels while
          looking independent in changes.

Descriptive. This measures our own instruments, joins nothing to a forward return, and
promotes nothing. `decorrelation_time` and `effective_dim` are also what the regime
feasibility read consumes -- the number of independent EPISODES, not sessions.

    .venv\\Scripts\\python.exe -m qlib_lab.registry_corr
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from . import config, market_state as ms, state_panel as sp

logger = logging.getLogger("qlib_lab.registry_corr")

EXPORT_DIR = config.JOURNAL_DIR / "exports"
RESULT_PATH = EXPORT_DIR / "registry_corr.json"

# ordinal encoding of a story state; NO_DATA stays missing rather than becoming a zero
STORY_CODE = {ms.HOLDING: 1.0, ms.WEAK: 0.0, ms.BROKEN: -1.0}

# families are cut at |corr| >= this, i.e. distance <= 1 - this
FAMILY_CORR = 0.60


def numeric_registry(panel: pd.DataFrame, per_asset: bool = False) -> pd.DataFrame:
    """The registry as numbers. Story states become ordinal, conditions stay binary."""
    keep = []
    for c in panel.columns:
        if c.startswith("story_state_"):
            continue
        if c.startswith("cot_") and not per_asset and c != "cot_n_crowded":
            continue
        if c in ("vol_term_stale_days", "breadth_n", "struct_n_struct_assets"):
            continue          # plumbing, not a reading
        keep.append(c)
    out = panel[keep].copy()
    for c in panel.columns:
        if c.startswith("story_state_"):
            out[c] = panel[c].map(STORY_CODE)
    return out


def decorrelation_time(s: pd.Series, max_lag: int = 500) -> float | None:
    """First lag at which autocorrelation falls below 1/e, in sessions.

    This is the honest denominator for anything computed on this panel: 2,514 sessions of
    a reading that decorrelates in 60 is not 2,514 observations, it is about 42.
    """
    x = s.dropna()
    if len(x) < 60:
        return None
    x = x - x.mean()
    if not float(x.std()):
        return None
    for lag in range(1, min(max_lag, len(x) - 10)):
        ac = float(x.autocorr(lag))
        if not np.isfinite(ac) or ac < np.exp(-1):
            return float(lag)
    return float(max_lag)


def effective_dim(corr: pd.DataFrame) -> dict:
    """How many independent signals a correlation matrix really contains.

    `participation_ratio` = (sum L)^2 / sum(L^2): the number of eigenvalues that
    meaningfully carry variance. `n_for_90pct` is the blunter cousin. `absorption` is the
    SAME top-k share `market_state.structure()` publishes for the market, k = n//4, so the
    two numbers are directly comparable.
    """
    clean = corr.dropna(how="all").dropna(axis=1, how="all")
    clean = clean.loc[clean.index, clean.index] if len(clean) else clean
    if len(clean) < 3 or clean.isna().any().any():
        return {"n_columns": int(len(clean)), "participation_ratio": None,
                "n_for_90pct": None, "absorption": None}
    eig = np.sort(np.linalg.eigvalsh(clean.values))[::-1]
    eig = np.clip(eig, 0, None)
    total = float(eig.sum())
    if total <= 0:
        return {"n_columns": int(len(clean)), "participation_ratio": None,
                "n_for_90pct": None, "absorption": None}
    pr = float(total ** 2 / float((eig ** 2).sum()))
    cum = np.cumsum(eig) / total
    k = max(1, len(eig) // 4)
    return {"n_columns": int(len(clean)),
            "participation_ratio": round(pr, 2),
            "n_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
            "absorption": round(float(eig[:k].sum() / total), 4)}


def corr_matrix(df: pd.DataFrame, mode: str = "level",
                min_overlap: int = 250) -> tuple[pd.DataFrame, dict]:
    """Spearman correlation, on levels or on first differences, COMPLETE CASE.

    Rank correlation because several of these are bounded percentiles and two are binary;
    Pearson on a 0/1 condition series against a percentile is not a number worth printing.

    Complete-case, not pairwise: a pairwise matrix has cells estimated on different
    samples, is not guaranteed positive semi-definite, and its eigenvalues -- which are
    the whole point here -- can come out negative. The cost is that the LATEST-STARTING
    column truncates the window for everybody (BTC funding, 2019), so the window actually
    used is reported next to every number.
    """
    x = df.diff() if mode == "change" else df
    x = x.loc[:, x.notna().sum() >= min_overlap]
    x = x.dropna(axis=0, how="any")
    x = x.loc[:, x.std(numeric_only=True) > 0]
    meta = {"rows_used": int(len(x)),
            "window": ([str(x.index.min().date()), str(x.index.max().date())]
                       if len(x) else None),
            "columns_dropped_constant": int(df.shape[1] - x.shape[1])}
    return x.corr(method="spearman"), meta


def families(corr: pd.DataFrame, threshold: float = FAMILY_CORR) -> pd.DataFrame:
    """Hierarchical (average-linkage) grouping on distance 1 - |corr|.

    Not k-means: the readings are correlated by construction, so a method that assumes
    round clusters in an isotropic space would report shapes the data does not have.
    """
    c = corr.dropna(how="all").dropna(axis=1, how="all")
    c = c.loc[c.index, c.index].fillna(0.0)
    d = 1.0 - c.abs().values
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2.0
    z = linkage(squareform(d, checks=False), method="average")
    labels = fcluster(z, t=1.0 - threshold, criterion="distance")
    order = leaves_list(z)
    return pd.DataFrame({"reading": c.index, "family": labels}).iloc[order].reset_index(drop=True)


def cross_asset_matrix(close: pd.DataFrame, window: int = ms.CORR_WINDOW) -> pd.DataFrame:
    """Part B: the 89-name return-correlation matrix, the input the graph work needs.

    Deliberately a SEPARATE function from `market_state.structure()`, which stays on its
    13 COT assets. `crowding_x_structure`, `ABSORPTION_SPIKE` and the published absorption
    number all read that matrix; widening it in place would silently move a live flag.
    """
    rets = close.pct_change().tail(window)
    rets = rets.loc[:, rets.notna().sum() >= window - 5]
    return rets.corr()


# --------------------------------------------------------------------------- report

def run(panel: pd.DataFrame, close: pd.DataFrame | None = None) -> dict:
    out: dict = {"generated": pd.Timestamp.utcnow().isoformat(),
                 "sessions": int(len(panel)),
                 "span": [str(panel.index.min().date()), str(panel.index.max().date())]}

    for scope, per_asset in (("market_level", False), ("with_per_asset_cot", True)):
        reg = numeric_registry(panel, per_asset=per_asset)
        block = {}
        for mode in ("level", "change"):
            c, meta = corr_matrix(reg, mode=mode)
            block[mode] = {**effective_dim(c), **meta}
            if scope == "market_level" and mode == "level":
                out["families"] = families(c).to_dict("records")
                out["top_redundant_pairs"] = _top_pairs(c, 15)
        out[scope] = block

    reg = numeric_registry(panel, per_asset=False)
    out["decorrelation_days"] = {
        c: decorrelation_time(reg[c]) for c in reg.columns
        if reg[c].notna().sum() >= 250}
    dt = [v for v in out["decorrelation_days"].values() if v]
    if dt:
        span = len(panel)
        capped = sorted(k for k, v in out["decorrelation_days"].items() if v == 500.0)
        out["independent_episodes"] = {
            "median_decorrelation_days": float(np.median(dt)),
            "max_decorrelation_days": float(np.max(dt)),
            "episodes_at_median": round(span / float(np.median(dt)), 1),
            "episodes_at_max": round(span / float(np.max(dt)), 1),
            # a reading that never decorrelates inside 500 sessions has hit the search
            # cap, not a measured horizon -- naming them stops "500" being read as data
            "never_decorrelated_within_500d": capped}

    if close is not None and not close.empty:
        wide = cross_asset_matrix(close)
        out["cross_asset_89"] = effective_dim(wide)
        out["cross_asset_89"]["n_assets"] = int(wide.shape[0])
    return out


def _top_pairs(corr: pd.DataFrame, n: int) -> list[dict]:
    c = corr.abs().where(~np.eye(len(corr), dtype=bool))
    s = c.stack().sort_values(ascending=False)
    seen, rows = set(), []
    for (a, b), v in s.items():
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"a": a, "b": b, "abs_corr": round(float(v), 3)})
        if len(rows) >= n:
            break
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-cross-asset", action="store_true")
    a = ap.parse_args()

    if not sp.PANEL_PATH.exists():
        print(f"no panel at {sp.PANEL_PATH} - run qlib_lab.state_panel first")
        return
    panel = pd.read_parquet(sp.PANEL_PATH)
    close = None
    if not a.no_cross_asset:
        names = sorted(config.UNIVERSE.keys())
        close = ms.close_panel(names, start=ms.WARMUP)
        close = close[close.index < pd.Timestamp.now().normalize()]

    res = run(panel, close)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")

    print(f"registry over {res['sessions']} sessions {res['span'][0]} -> {res['span'][1]}\n")
    for scope in ("market_level", "with_per_asset_cot"):
        for mode in ("level", "change"):
            d = res[scope][mode]
            print(f"{scope:20s} {mode:7s} n={d['n_columns']:>3}  rows={d['rows_used']:>5}  "
                  f"participation_ratio={d['participation_ratio']}  "
                  f"n_for_90pct={d['n_for_90pct']}  absorption={d['absorption']}")
    if "cross_asset_89" in res:
        d = res["cross_asset_89"]
        print(f"\ncross-asset {d['n_assets']} names: participation_ratio="
              f"{d['participation_ratio']}, n_for_90pct={d['n_for_90pct']}, "
              f"absorption={d['absorption']}")
    ep = res.get("independent_episodes")
    if ep:
        print(f"\ndecorrelation: median {ep['median_decorrelation_days']:.0f}d -> "
              f"{ep['episodes_at_median']} independent episodes at the median")
        if ep["never_decorrelated_within_500d"]:
            print(f"  never decorrelated inside the 500d search cap "
                  f"({len(ep['never_decorrelated_within_500d'])}): "
                  f"{', '.join(ep['never_decorrelated_within_500d'][:6])}")
    fam = pd.DataFrame(res["families"])
    print(f"\n{fam['family'].nunique()} families at |corr| >= {FAMILY_CORR} "
          f"across {len(fam)} readings")
    for f, g in fam.groupby("family"):
        if len(g) > 1:
            print(f"  family {f}: {', '.join(g['reading'])}")
    print(f"\nwrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
