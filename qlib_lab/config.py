"""Central config — universe, horizons, paths, gate thresholds.

Universe mirrors macro_gpu_lab/macro_gpu_lab/config.py::UNIVERSE (the same 22
tradable Yahoo proxies for each macro force). Qlib instrument IDs are the logical
names (SPY, GOLD, ...), not raw Yahoo symbols, so flag-file consumers see the same
vocabulary as macro_state.json / macro_gpu_state.json.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Universe: logical name → Yahoo v8 chart symbol ──────────────────
# Mirror of macro_gpu_lab config (Stooq column dropped — Yahoo-only here; the
# 10y Yahoo window already covers Alpha158's longest lookback many times over).
UNIVERSE = {
    "US10Y":  "%5ETNX",     # 1-bar junk feed on Yahoo — auto-dropped by MIN_ASSET_ROWS
    "TLT":    "TLT",
    "TIP":    "TIP",
    "DXY":    "DX-Y.NYB",
    "EURUSD": "EURUSD=X",
    "USDJPY": "JPY=X",
    "GBPUSD": "GBPUSD=X",
    "USDCAD": "CAD=X",
    "USDCNH": "CNY=X",      # onshore yuan: full history, equivalent China-stress proxy
    "SPY":    "SPY",
    "QQQ":    "QQQ",
    "IWM":    "IWM",
    "EEM":    "EEM",
    "XLF":    "XLF",
    "VIX":    "%5EVIX",
    # option-surface context (term structure, tail pricing, vol-of-vol)
    "VIX9D":  "%5EVIX9D",
    "VIX3M":  "%5EVIX3M",
    "SKEW":   "%5ESKEW",
    "VVIX":   "%5EVVIX",
    "GOLD":   "GLD",
    "OIL":    "USO",
    "COPPER": "CPER",
    "HYG":    "HYG",
    "LQD":    "LQD",
    "BIL":    "BIL",
    "BTC":    "BTC-USD",
}

FORCE = {
    "US10Y": "rates", "TLT": "rates", "TIP": "rates",
    "DXY": "dollar", "EURUSD": "dollar", "USDJPY": "dollar",
    "GBPUSD": "dollar", "USDCAD": "dollar", "USDCNH": "china",
    "SPY": "equities", "QQQ": "equities", "IWM": "equities", "EEM": "equities", "XLF": "equities",
    "VIX": "volatility", "VIX9D": "volatility", "VIX3M": "volatility",
    "SKEW": "volatility", "VVIX": "volatility",
    "GOLD": "commodities", "OIL": "commodities", "COPPER": "commodities",
    "HYG": "credit", "LQD": "credit",
    "BIL": "liquidity",
    "BTC": "crypto",
}

# ── Phase-2 ETF expansion: liquid US-listed ETFs by group ───────────
# These trade on the SPY session calendar (unlike FX/VIX/BTC above), so they
# form the MODEL universe; the macro names above stay in the store as feature
# context. ETFs over single stocks: no survivorship bias from backfilling
# today's index constituents 10 years.
ETF_GROUPS = {
    "sector": ["XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"],
    "industry": ["SMH", "XBI", "XHB", "XRT", "KRE", "ITA", "IYT", "XOP", "GDX", "GDXJ", "KWEB", "XME"],
    "factor": ["MTUM", "VLUE", "QUAL", "USMV", "RSP", "SPLV"],
    "style": ["IWD", "IWF", "IWO", "IWN", "MDY", "DIA", "VTI"],
    "country": ["EWJ", "EWG", "EWU", "EWC", "EWA", "EWZ", "EWW", "EWY", "EWT", "EWH", "EWS", "FXI", "INDA"],
    "region": ["EFA", "VGK", "VPL", "ILF"],
    "bond": ["SHY", "IEF", "AGG", "EMB", "MUB", "BNDX"],
    "commodity": ["SLV", "UNG", "DBC", "DBA"],
    "reit": ["VNQ"],
}
for _grp, _syms in ETF_GROUPS.items():
    for _s in _syms:
        UNIVERSE.setdefault(_s, _s)      # ETF ticker == Yahoo symbol
        FORCE.setdefault(_s, _grp)

ASSETS = list(UNIVERSE.keys())

# Model universe: US-listed ETFs only (clean session calendar, tradable book).
# Macro indices/FX/crypto stay feature-context-only.
_NON_ETF = {"US10Y", "DXY", "EURUSD", "USDJPY", "GBPUSD", "USDCAD", "USDCNH",
            "VIX", "VIX9D", "VIX3M", "SKEW", "VVIX", "BTC"}
MODEL_UNIVERSE = [a for a in ASSETS if a not in _NON_ETF]

CALENDAR_ANCHOR = "SPY"        # every asset is reindexed to SPY's session set

# ── Data pull ───────────────────────────────────────────────────────
DAILY_YEARS = 10
MIN_ASSET_ROWS = 200           # drop junk feeds (Yahoo yield indices give 1 bar)

# ── Model / labels ──────────────────────────────────────────────────
HORIZONS_DAYS = [5, 21]        # BOTH must clear the gate
TOPK = 8                       # long top-8 / short bottom-8 of the ~76-ETF cross-section
MIN_TRAIN_DAYS = 750           # ~3y of sessions before the first OOS prediction
N_MODELS = 2                   # baseline Alpha158 + macro-augmented; n_trials = N_MODELS * horizons

LGB_PARAMS = dict(
    objective="regression",
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=300,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

# ── ETF flow tracker map (ticker → label) ───────────────────────────
# Universe ETFs + sector SPDRs for the institutional-rotation view.
ETF_MAP = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000", "EEM": "EM equities",
    "TLT": "20y+ Treasury", "TIP": "TIPS", "HYG": "High yield", "LQD": "IG credit",
    "BIL": "T-bills", "GLD": "Gold", "USO": "Oil", "CPER": "Copper",
    "XLF": "Financials", "XLK": "Technology", "XLE": "Energy", "XLV": "Health care",
    "XLI": "Industrials", "XLY": "Cons. discr.", "XLP": "Cons. staples",
    "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real estate", "XLC": "Comm. svcs",
}

# ── Promotion flag (default SHADOW — never auto-flips) ──────────────
QLIB_ENFORCE = os.environ.get("QLIB_ENFORCE", "no").strip().lower()

# ── Paths ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
CSV_DIR = BASE_DIR / "csv"
QLIB_DATA_DIR = BASE_DIR / "qlib_data" / "us_data"
DATA_DIR = BASE_DIR / "data"
JOURNAL_DIR = BASE_DIR / "journal"
FLAGS_DIR = JOURNAL_DIR / "flags"
RUNS_DIR = JOURNAL_DIR / "runs"
HISTORY_DIR = JOURNAL_DIR / "qlib"
SCORECARDS_DIR = JOURNAL_DIR / "scorecards"
FLAG_PATH = FLAGS_DIR / "qlib_state.json"
ETF_FLOWS_PATH = FLAGS_DIR / "etf_flows.json"
ETF_SHARES_PARQUET = DATA_DIR / "etf_shares.parquet"
FLAG_STALE_DAYS = 4

# ── Vol-desk pre-open Telegram + magnitude/tilt fusion ──────────────
# Telegram secrets: gitignored secrets/telegram.json {"bot_token","chat_id"}.
# Notify is fail-safe: a missing file or send failure never blocks the desk.
TELEGRAM_SECRETS_PATH = BASE_DIR / "secrets" / "telegram.json"
# Magnitude proxy (B07-aligned): implied daily move percentile vs trailing year.
# At/above this percentile = big-move regime -> do NOT sell premium even if rich.
MAGNITUDE_HIGH_PCT = 0.70
MAGNITUDE_LOOKBACK = 252
# VRP-forward tilt buckets, calibrated from qlib_lab H07 deep-history VRP split
# (hi vs lo variance-risk-premium -> next-month SPY forward return). Display-only:
# the effect replicates 1995-2015 but FAILS the gate (DSR ratio -0.0043, prob
# 0.498, n=107, n_trials=20 -- experiments_2026-07-30) -- never sizes.
VRP_FWD_RICH_PCT = 0.60
VRP_FWD_RET_RICH = 0.0197   # mean 1m fwd return when insurance rich (n_hi=54)
VRP_FWD_RET_CHEAP = 0.0038  # mean 1m fwd return when insurance cheap (n_lo=182)

# Canonical overfit gate lives in macro_gpu_lab (quant-research-gate skill: reuse,
# never re-port). pipeline.py prepends this to sys.path before importing.
MACRO_GPU_LAB_DIR = Path(r"C:\Users\<your-user>\macro_gpu_lab")

for _d in (CSV_DIR, DATA_DIR, FLAGS_DIR, RUNS_DIR, HISTORY_DIR, SCORECARDS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
