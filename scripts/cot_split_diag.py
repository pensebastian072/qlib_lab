"""C1 diagnostic (no tuning): does the COT-extreme fade replicate across
independent halves, and is it carried by one asset or broad?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qlib_lab import cot  # noqa: E402
from qlib_lab.experiments_cot import (HOLD, Z_ENTER, COST_PER_SIDE,  # noqa: E402
                                      deep_close_panel)


def main():
    hist = pd.read_parquet(cot.COT_PARQUET)
    feats = cot.weekly_features(hist)
    assets = sorted(feats["asset"].unique())
    px = deep_close_panel(assets)
    sig = cot.daily_panel(feats, px.index, cols=("z156",))
    fwd = px.shift(-HOLD) / px - 1

    rows = []  # (date, asset, pnl)
    for i in range(0, len(px.index) - HOLD, HOLD):
        t = px.index[i]
        for a, f in sig.items():
            if a not in px.columns or t not in f.index:
                continue
            z = f.at[t, "z156"]
            r = fwd.at[t, a] if a in fwd.columns else np.nan
            if pd.isna(z) or pd.isna(r) or abs(z) < Z_ENTER:
                continue
            pos = -1.0 if z > Z_ENTER else 1.0
            rows.append((t, a, pos * float(r) - 2 * COST_PER_SIDE))
    df = pd.DataFrame(rows, columns=["date", "asset", "pnl"])

    def stats(d, label):
        pooled = d.groupby("date")["pnl"].mean()
        pf_num = pooled[pooled > 0].sum()
        pf_den = -pooled[pooled < 0].sum()
        print(f"{label:12s} blocks={len(pooled):4d} mean={pooled.mean():+.5f} "
              f"pf={pf_num / pf_den if pf_den else float('inf'):.3f} "
              f"hit={float((pooled > 0).mean()):.3f}")

    stats(df, "full")
    mid = df["date"].sort_values().iloc[len(df) // 2]
    stats(df[df["date"] < mid], "first half")
    stats(df[df["date"] >= mid], "second half")
    print(f"(split at {mid.date()})")
    print("\nper-asset contribution:")
    per = df.groupby("asset")["pnl"].agg(["count", "sum", "mean"])
    for a, r in per.sort_values("sum", ascending=False).iterrows():
        print(f"  {a:7s} legs={int(r['count']):4d} total={r['sum']:+.3f} mean={r['mean']:+.5f}")
    ex_top = per["sum"].idxmax()
    stats(df[df["asset"] != ex_top], f"ex-{ex_top}")


if __name__ == "__main__":
    main()
