"""Build the Market State workbook.

    .venv\\Scripts\\python.exe scripts\\build_market_state.py [--start 2016-01-01]

Descriptive only - measures market state, forecasts nothing. See qlib_lab/market_state.py.

Everything qlib lives under the __main__ guard: qlib's D.features spawns
multiprocessing workers that re-import this module on Windows.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from qlib_lab import market_state as ms
    from qlib_lab import market_state_xlsx as msx

    ap = argparse.ArgumentParser(description="build the Market State workbook")
    ap.add_argument("--start", default=ms.START, help="history start (YYYY-MM-DD)")
    ap.add_argument("--out", default=None, help="output .xlsx path")
    ap.add_argument("--no-log", action="store_true",
                    help="do not append to the conditions fire log")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    state = ms.build_state(start=args.start)
    if not args.no_log:
        ms.append_fire_log(state)

    out = Path(args.out) if args.out else ms.EXPORT_PATH
    path = msx.build_workbook(state, out)

    stale = [k for k, v in state["stale"].items() if v]
    fired = [c["id"] for c in state["conditions"] if c["fired"]]
    print(f"market state as_of {state.get('as_of')}")
    print(f"  vol rows={len(state['vol_hist'])} cot rows={len(state['pos_hist'])} "
          f"liq rows={len(state['liq_hist'])} fund rows={len(state['fund_hist'])}")
    print(f"  conditions firing: {', '.join(fired) if fired else '(none)'}")
    if stale:
        print(f"  STALE/MISSING: {', '.join(stale)}")
    print(f"workbook -> {path}")
    # exit 2 if every input failed: a workbook of empty sheets is not a success
    return 2 if len(stale) == len(state["stale"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
