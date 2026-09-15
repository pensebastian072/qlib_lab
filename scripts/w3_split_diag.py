"""Wave-3 conditional diagnostics: is the edge in the CONDITION or just beta?

For V1 (term structure) and L1 (net liquidity): compare SPY forward 21d
returns in ON-blocks vs OFF-blocks (welch t) + split-half stability. Same
methodology that validated VRP and exposed H07's beta.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qlib_lab.experiments_wave3 import deep_panel  # noqa: E402
from qlib_lab.fred_liquidity import daily_frame  # noqa: E402


def welch(a, b):
    return float((a.mean() - b.mean())
                 / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def split_report(name, cond: pd.Series, fwd: pd.Series, start_i: int):
    rows = [(bool(cond.iloc[i]), float(fwd.iloc[i]))
            for i in range(start_i, len(cond) - 21, 21)
            if not (pd.isna(cond.iloc[i]) or pd.isna(fwd.iloc[i]))]
    on = np.array([r for c, r in rows if c])
    off = np.array([r for c, r in rows if not c])
    print(f"\n{name}: ON n={len(on)} mean={on.mean():+.4f} | "
          f"OFF n={len(off)} mean={off.mean():+.4f} | t={welch(on, off):.2f}")
    half = len(rows) // 2
    for label, chunk in [("first half", rows[:half]), ("second half", rows[half:])]:
        o = np.array([r for c, r in chunk if c])
        f = np.array([r for c, r in chunk if not c])
        if len(o) > 5 and len(f) > 5:
            print(f"  {label}: ON {o.mean():+.4f} (n={len(o)}) "
                  f"OFF {f.mean():+.4f} (n={len(f)}) t={welch(o, f):.2f}")


def main():
    px = deep_panel(["SPY", "VIX", "VIX3M"])
    px = px.loc[px["SPY"].notna()]
    spy = px["SPY"]
    fwd21 = spy.shift(-21) / spy - 1

    ts = px["VIX"] / px["VIX3M"]
    first = ts.first_valid_index()
    split_report("V1 contango (ON=contango)", (ts < 1.0), fwd21,
                 px.index.get_loc(first) + 1)

    liq = daily_frame()
    chg = liq["netliq_chg13w"].reindex(spy.index).ffill(limit=5)
    first = chg.first_valid_index()
    split_report("L1 netliq (ON=rising)", (chg > 0), fwd21,
                 px.index.get_loc(first) + 1)


if __name__ == "__main__":
    main()
