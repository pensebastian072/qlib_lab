"""Control: unconditional long-SPY 21d blocks vs the H07 VRP filter."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qlib_lab import config  # noqa: E402

sys.path.insert(0, str(config.MACRO_GPU_LAB_DIR))
from macro_gpu_lab.validate import evaluate_gate  # noqa: E402


def main():
    # qlib work must live under the __main__ guard: D.features spawns
    # multiprocessing workers that re-import this module on Windows
    import qlib
    from qlib.constant import REG_US

    qlib.init(provider_uri=str(config.QLIB_DATA_DIR), region=REG_US)
    from qlib.data import D

    px = D.features(["SPY"], ["$close"])["$close"].droplevel(0)
    fwd = px.shift(-21) / px - 1
    pnls = [float(fwd.iloc[i]) for i in range(253, len(px) - 21, 21) if not pd.isna(fwd.iloc[i])]
    v = evaluate_gate(np.array(pnls), n_trials=1)
    dsr = (v.get("deflated_sharpe") or {}).get("ratio")
    print(f"baseline long-SPY 21d: n={len(pnls)} pf={v.get('profit_factor')} "
          f"dsr={dsr} pbo={v.get('pbo')} mean={np.mean(pnls):.4f} "
          f"hit={np.mean(np.array(pnls) > 0):.3f}")


if __name__ == "__main__":
    main()
