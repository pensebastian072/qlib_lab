"""H07 follow-up: conditional-mean split. Does VRP-high predict better SPY 21d
blocks than VRP-low? Welch t on non-overlapping blocks + split-half stability."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qlib_lab import config  # noqa: E402


def welch(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return (a.mean() - b.mean()) / np.sqrt(va + vb)


def main():
    # everything qlib lives under the __main__ guard: qlib's D.features spawns
    # multiprocessing workers that re-import this module on Windows
    import qlib
    from qlib.constant import REG_US

    qlib.init(provider_uri=str(config.QLIB_DATA_DIR), region=REG_US)
    from qlib.data import D

    df = D.features(["SPY", "VIX"], ["$close"])["$close"].unstack(level=0)
    df.columns = [str(c) for c in df.columns]
    spy, vix = df["SPY"], df["VIX"]
    rv = spy.pct_change().rolling(20).std() * np.sqrt(252) * 100
    vrp = vix ** 2 - rv ** 2
    sig = vrp > vrp.rolling(252).median()
    fwd = spy.shift(-21) / spy - 1

    rows = [(bool(sig.iloc[i]), float(fwd.iloc[i]))
            for i in range(253, len(spy) - 21, 21) if not pd.isna(fwd.iloc[i])]
    hi = np.array([r for s, r in rows if s])
    lo = np.array([r for s, r in rows if not s])

    print(f"blocks: hi={len(hi)} lo={len(lo)}")
    print(f"mean 21d fwd: VRP-high={hi.mean():+.4f}  VRP-low={lo.mean():+.4f}  "
          f"welch_t={welch(hi, lo):.2f}")
    half = len(rows) // 2
    for name, chunk in [("first half", rows[:half]), ("second half", rows[half:])]:
        h = np.array([r for s, r in chunk if s])
        l = np.array([r for s, r in chunk if not s])
        if len(h) > 5 and len(l) > 5:
            print(f"  {name}: hi={h.mean():+.4f} (n={len(h)})  lo={l.mean():+.4f} "
                  f"(n={len(l)})  t={welch(h, l):.2f}")


if __name__ == "__main__":
    main()
