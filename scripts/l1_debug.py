"""Why did L1 only produce 53 blocks?"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qlib_lab.experiments_wave3 import deep_panel  # noqa: E402
from qlib_lab.fred_liquidity import daily_frame  # noqa: E402


def main():
    px = deep_panel(["SPY"])
    spy = px["SPY"]
    liq = daily_frame()
    print("liq index:", liq.index.min(), "..", liq.index.max(), "rows", len(liq))
    print("liq index dtype:", liq.index.dtype)
    chg = liq["netliq_chg13w"].reindex(spy.index).ffill(limit=5)
    print("chg non-na:", chg.notna().sum(), "of", len(chg))
    print("first valid:", chg.first_valid_index(), "last valid:", chg.last_valid_index())
    print("na gaps by year:")
    na = chg[chg.isna()]
    print(pd.Series(na.index.year).value_counts().sort_index().tail(20))


if __name__ == "__main__":
    main()
