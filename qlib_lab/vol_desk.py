"""Vol-desk ticket generator — VRP x surprise-shift decision matrix (ADVISORY).

Combines the two vol brains into one daily paper ticket on SPY options:

  - VRP (qlib_lab, validated ~30y): when the variance risk premium is FAT,
    options are overpriced -> selling premium is favored.
  - surprise-shift detector (macro_gpu_lab tv_signal.json, SHADOW, ~2.2x lift
    on calm->eruption): when an eruption is predicted, OWNING vol is favored —
    but only if the premium is not already rich.

Fixed decision matrix (pre-registered, no tuning):
  eruption_predicted(21d) AND calm(21d) AND vrp_pct < 60  -> LONG_VOL
      buy 1x ~35DTE ATM straddle (both edges: spike coming, vol not rich)
  eruption_predicted AND vrp_pct >= 60                    -> NO_TRADE
      (right thesis, wrong price)
  no eruption AND calm(5d) AND vrp_pct >= 70              -> SELL_PUT_SPREAD
      sell 1x ~35DTE 30-delta put, buy $2 lower (defined risk)
  otherwise                                               -> NO_TRADE

Strikes/premiums are Black-Scholes ESTIMATES with sigma = VIX/100 and r=4% —
good enough for paper grading, not executable quotes. Every ticket stores what
grading needs (spot, strikes, expiry, est premium). SHADOW/advisory only:
this module never touches a broker; the human reads the ticket and decides.
Sizing note is written for a small account: max ONE defined-risk lot.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger("qlib_lab.vol_desk")

TV_SIGNAL = config.MACRO_GPU_LAB_DIR / "journal" / "flags" / "tv_signal.json"
DESK_FLAG = config.FLAGS_DIR / "vol_desk_state.json"
DESK_DIR = config.JOURNAL_DIR / "vol_desk"
DESK_DIR.mkdir(parents=True, exist_ok=True)

VRP_SELL_PCT = 70      # premium rich enough to sell
VRP_RICH_PCT = 60      # too rich to buy vol
DTE_DAYS = 35
SPREAD_WIDTH = 2.0     # $2-wide defined-risk put spread
SHORT_PUT_DELTA = -0.30
RISK_FREE = 0.04
SHIFT_STALE_DAYS = 4


# ── Black-Scholes helpers (estimates only) ───────────────────────────

def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put(spot: float, strike: float, sigma: float, t_years: float,
           r: float = RISK_FREE) -> float:
    if sigma <= 0 or t_years <= 0:
        return max(strike - spot, 0.0)
    d1 = (math.log(spot / strike) + (r + sigma ** 2 / 2) * t_years) / (sigma * math.sqrt(t_years))
    d2 = d1 - sigma * math.sqrt(t_years)
    return strike * math.exp(-r * t_years) * _ncdf(-d2) - spot * _ncdf(-d1)


def put_delta(spot: float, strike: float, sigma: float, t_years: float,
              r: float = RISK_FREE) -> float:
    d1 = (math.log(spot / strike) + (r + sigma ** 2 / 2) * t_years) / (sigma * math.sqrt(t_years))
    return _ncdf(d1) - 1.0


def strike_for_put_delta(spot: float, sigma: float, t_years: float,
                         target: float = SHORT_PUT_DELTA) -> float:
    lo, hi = spot * 0.5, spot * 1.1
    for _ in range(60):
        mid = (lo + hi) / 2
        if put_delta(spot, mid, sigma, t_years) < target:
            hi = mid
        else:
            lo = mid
    return round((lo + hi) / 2)


def bs_straddle(spot: float, strike: float, sigma: float, t_years: float,
                r: float = RISK_FREE) -> float:
    put = bs_put(spot, strike, sigma, t_years, r)
    call = put + spot - strike * math.exp(-r * t_years)  # parity
    return put + call


# ── inputs (all fail-safe) ───────────────────────────────────────────

def read_shift() -> dict:
    fb = {"stale": True, "horizons": {}}
    try:
        raw = json.loads(TV_SIGNAL.read_text(encoding="utf-8"))
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(raw["as_of"])).total_seconds() / 86400.0
        raw["stale"] = age > SHIFT_STALE_DAYS
        return raw
    except Exception:  # noqa: BLE001
        return fb


def _magnitude(spot: float, vix_series: pd.Series) -> dict:
    """B07-aligned magnitude PROXY: implied 1-day move + its percentile vs the
    trailing year. NOT the XGB serving flag from alpaca_gpu_lab (that is a
    separate cross-venv flag-file, a clean later swap behind these same fields);
    this is the vol-state estimate the daily desk can compute in-process.

    sigma_1d = VIX/100 / sqrt(252); expected 1-day move (1-sigma) = spot*sigma_1d.
    magnitude_pct = where today's implied sigma sits in the trailing-252d range;
    high pct = big-move regime -> do not sell premium even when it is rich.
    """
    sigma_1d = (vix_series / 100.0) / np.sqrt(252)
    latest = float(sigma_1d.iloc[-1])
    window = sigma_1d.iloc[-config.MAGNITUDE_LOOKBACK:]
    pct = float((window <= latest).mean())
    return {"sigma_1d_pct": round(latest * 100, 4),
            "expected_move_usd": round(spot * latest, 2),
            "expected_move_pct": round(latest * 100, 3),
            "magnitude_pct": round(pct, 3),
            "proxy": "vix_implied",
            "note": "B07-aligned magnitude proxy; not the XGB serving flag"}


def vrp_forward_tilt(vrp_pct: float) -> dict:
    """Slow monthly directional tilt from the H07 deep-history VRP split: when
    insurance is rich now, next-month SPY forward return has been higher
    (+1.97% vs +0.38%). DISPLAY ONLY -- it replicates 1995-2015 but FAILS the
    gate (DSR ratio -0.0043, prob 0.498, n=107, n_trials=20, PBO 0.000 --
    experiments_2026-07-30); it never sizes or vetoes anything.

    The DSR ratio here is the post-unit-fix number. The older -17.3 that this
    note used to carry came from the buggy-unit deflated Sharpe and was wrong
    by orders of magnitude; the verdict (FAIL) is unchanged, but -0.0043 means
    'indistinguishable from the best of 20 noise trials', not 'catastrophic'."""
    rich = vrp_pct >= config.VRP_FWD_RICH_PCT
    return {"bucket": "rich" if rich else "cheap",
            "expected_fwd_ret_1m": (config.VRP_FWD_RET_RICH if rich
                                    else config.VRP_FWD_RET_CHEAP),
            "note": ("H07 deep-history VRP split (hi vs lo); replicates "
                     "1995-2015, FAILS gate (DSR ratio -0.0043, prob 0.498) "
                     "-- display tilt only")}


def _vol_surface() -> dict | None:
    """Option-surface context for downstream desks: VIX term structure + tail/vol-of-vol.
    term_ratio = VIX9D / VIX3M ( <1 contango/calm, >=1 backwardation/stress ). Fail-safe:
    any missing series -> None fields, never raises. Consumed by options_desk (pairs tilt)."""
    from qlib.data import D

    try:
        df = D.features(["VIX9D", "VIX3M", "VVIX", "SKEW"], ["$close"])["$close"].unstack(level=0)
        df.columns = [str(c) for c in df.columns]

        def _last(name):
            if name in df:
                s = df[name].dropna()
                return float(s.iloc[-1]) if len(s) else None
            return None

        v9, v3m, vvix, skew = _last("VIX9D"), _last("VIX3M"), _last("VVIX"), _last("SKEW")
        ratio = round(v9 / v3m, 4) if (v9 and v3m) else None
        return {"vix9d": v9, "vix3m": v3m, "vvix": vvix, "skew": skew,
                "term_ratio": ratio,
                "structure": (None if ratio is None
                              else ("backwardation" if ratio >= 1.0 else "contango"))}
    except Exception:  # noqa: BLE001
        return None


def vrp_snapshot() -> dict | None:
    """Latest VRP + trailing-252d percentile from the .bin store, plus the
    B07-aligned magnitude proxy and the option vol surface on the same pull."""
    from qlib.data import D

    df = D.features(["SPY", "VIX"], ["$close"])["$close"].unstack(level=0)
    df.columns = [str(c) for c in df.columns]
    spy, vix = df["SPY"].dropna(), df["VIX"].dropna()
    rv = spy.pct_change().rolling(20).std() * np.sqrt(252) * 100
    vrp = (vix ** 2 - rv ** 2).dropna()
    if len(vrp) < 300:
        return None
    pct = float((vrp.iloc[-252:] <= vrp.iloc[-1]).mean())
    spot = float(spy.iloc[-1])
    return {"spot": spot, "vix": float(vix.iloc[-1]),
            "vrp": float(vrp.iloc[-1]), "vrp_pct": round(pct, 3),
            "data_through": str(vrp.index[-1].date()),
            "magnitude": _magnitude(spot, vix),
            "vol_surface": _vol_surface()}


# ── the matrix ───────────────────────────────────────────────────────

def build_ticket(vrp: dict | None, shift: dict) -> dict:
    now = datetime.now(timezone.utc)
    expiry = (now + timedelta(days=DTE_DAYS)).date()
    # roll to that week's Friday (standard weekly expiry)
    expiry = expiry + timedelta(days=(4 - expiry.weekday()) % 7)
    t_years = (pd.Timestamp(expiry) - pd.Timestamp(now.date())).days / 365.0

    ticket = {
        "ts": now.isoformat(),
        "underlying": "SPY",
        "status": "SHADOW",
        "action": "NO_TRADE",
        "reason": None,
        "vrp": vrp,
        "shift": None,
        "expiry": str(expiry),
        "disclaimer": ("advisory paper ticket; BS estimates, not quotes; "
                       "defined-risk 1 lot max; human places any real order"),
    }
    h21 = (shift.get("horizons") or {}).get("21d") or {}
    h5 = (shift.get("horizons") or {}).get("5d") or {}
    ticket["shift"] = {"stale": bool(shift.get("stale", True)),
                       "calm_5d": h5.get("calm"), "calm_21d": h21.get("calm"),
                       "p_eruption_5d": h5.get("p_eruption"),
                       "p_eruption_21d": h21.get("p_eruption"),
                       "eruption_predicted": bool(h21.get("eruption_predicted")
                                                  or h5.get("eruption_predicted"))}

    if vrp is None:
        ticket["reason"] = "no VRP data"
        return ticket

    # Display-only fusion lines, shown on every ticket (even NO_TRADE):
    ticket["magnitude"] = vrp.get("magnitude")
    ticket["vrp_forward"] = vrp_forward_tilt(vrp["vrp_pct"])
    ticket["vol_surface"] = vrp.get("vol_surface")   # consumed by options_desk pairs tilt

    if ticket["shift"]["stale"]:
        ticket["reason"] = "shift detector stale — VRP-only rules suspended (need both brains)"
        return ticket

    spot, sigma = vrp["spot"], vrp["vix"] / 100.0
    pct = vrp["vrp_pct"]
    mag = vrp.get("magnitude") or {}
    mag_pct = mag.get("magnitude_pct")
    big_move_regime = mag_pct is not None and mag_pct >= config.MAGNITUDE_HIGH_PCT
    erupting = ticket["shift"]["eruption_predicted"]

    if erupting and h21.get("calm") and pct < VRP_RICH_PCT / 100:
        strike = round(spot)
        prem = bs_straddle(spot, strike, sigma, t_years)
        ticket.update(action="LONG_VOL", reason="eruption predicted from calm + vol not rich",
                      structure={"type": "straddle", "side": "buy", "lots": 1,
                                 "strike": strike,
                                 "est_debit": round(prem, 2),
                                 "max_risk_usd": round(prem * 100, 0),
                                 "breakeven_move_pct": round(prem / spot * 100, 2),
                                 "exp_1d_move_usd": mag.get("expected_move_usd")})
    elif erupting:
        ticket["reason"] = f"eruption predicted but VRP pct {pct:.0%} >= {VRP_RICH_PCT}% — vol too rich to buy"
    elif (not erupting) and h5.get("calm") and pct >= VRP_SELL_PCT / 100 and big_move_regime:
        # Rich premium, but the magnitude proxy says a big-move regime — B07's
        # lesson (§2) is that size is forecastable; do not sell premium into it.
        ticket["reason"] = (f"calm + VRP pct {pct:.0%} rich but magnitude pct "
                            f"{mag_pct:.0%} >= {config.MAGNITUDE_HIGH_PCT:.0%} "
                            f"(big-move regime) — do not sell premium")
    elif (not erupting) and h5.get("calm") and pct >= VRP_SELL_PCT / 100:
        short_k = strike_for_put_delta(spot, sigma, t_years)
        long_k = short_k - SPREAD_WIDTH
        credit = bs_put(spot, short_k, sigma, t_years) - bs_put(spot, long_k, sigma, t_years)
        ticket.update(action="SELL_PUT_SPREAD",
                      reason=f"calm + no eruption + VRP pct {pct:.0%} (premium rich)",
                      structure={"type": "put_credit_spread", "side": "sell", "lots": 1,
                                 "short_strike": short_k, "long_strike": long_k,
                                 "est_credit": round(credit, 2),
                                 "max_risk_usd": round((SPREAD_WIDTH - credit) * 100, 0)})
    else:
        ticket["reason"] = (f"no edge: eruption={erupting} calm5d={h5.get('calm')} "
                            f"vrp_pct={pct:.0%}")
    return ticket


def publish(ticket: dict) -> str:
    tmp = DESK_FLAG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ticket, indent=2))
    tmp.replace(DESK_FLAG)
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    with (DESK_DIR / f"{month}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(ticket) + "\n")
    return str(DESK_FLAG)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--telegram", action="store_true",
                    help="also push the ticket to Telegram (pre-open run); fail-safe")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import qlib
    from qlib.constant import REG_US

    qlib.init(provider_uri=str(config.QLIB_DATA_DIR), region=REG_US)
    vrp = vrp_snapshot()
    shift = read_shift()
    ticket = build_ticket(vrp, shift)
    path = publish(ticket)
    print(f"vol desk: {ticket['action']} — {ticket['reason']}")
    if vrp:
        print(f"  spot={vrp['spot']:.2f} vix={vrp['vix']:.1f} vrp_pct={vrp['vrp_pct']:.0%}")
        mag = vrp.get("magnitude") or {}
        if mag:
            print(f"  magnitude: exp move {mag.get('expected_move_pct')}% "
                  f"pct={mag.get('magnitude_pct'):.0%}")
    if ticket.get("structure"):
        print(f"  structure: {json.dumps(ticket['structure'])}")
    print(f"flag -> {path}")

    if args.telegram:
        from . import telegram_notify
        ok, err = telegram_notify.send_message(telegram_notify.format_ticket(ticket))
        print(f"telegram: {'sent' if ok else 'skipped — ' + str(err)}")


if __name__ == "__main__":
    main()
