# HANDOFF — market_state / asset_frame (session of 2026-08-23/24)

Read `CLAUDE.md` first for repo rules. This file is the resume pointer for the
**Market State workbook** work and the open decisions it is sitting on.

## What exists now

`journal/exports/market_state.xlsx` — 12 sheets, rebuilt daily by **MarketStateDaily
(weekdays 11:45)**, after `QlibLabDaily` 11:15 and clear of the morning C: contention.

| module | role |
|---|---|
| `qlib_lab/market_state.py` | data layer, 8 registered conditions, deterministic narrative, state log |
| `qlib_lab/asset_frame.py` | 12-layer per-asset framework over all 89 instruments |
| `qlib_lab/market_state_xlsx.py` | rendering only; no maths |
| `scripts/build_market_state.py` | CLI (`--start`, `--out`, `--no-log`) |
| `scripts/run_market_state.ps1` + `_run_market_state.vbs` + `register_market_state.ps1` | task plumbing |
| `tests/test_market_state.py`, `tests/test_asset_frame.py` | **109/109 pass** |

Sheets: TODAY · Volatility · Positioning · Positioning_idx52 · Funding · Liquidity ·
Structure · Crowding · Framework · Framework Detail · History · Conditions.

**The core claim:** this layer is DESCRIPTIVE and therefore outside the PBO/DSR gate —
a thermometer does not need a Sharpe ratio. Two independent `quant-reviewer` passes
tried to break that claim and could not. The tripwire both named: *the day anything
here is joined to forward returns it becomes a model and the gate applies.* A test
(`test_snapshot_records_no_forward_return`) rejects any log key matching
`fwd|forward|future|return|pnl|outcome|target|score|hit|correct`.

## Commits this session

qlib_lab: `52db735` (pre-existing run_preopen work, committed as found) ·
`8437b42` data_fetch · `9360c5f` DSR labels · `004aed7` market_state ·
`252d33f` asset_frame · `5c1ad1f` state history.
research_ledger: `a77ee02`. alpaca_gpu_lab: `2b63020`, `1212826`.

## Live state at handoff (as_of 2026-08-21)

VIX 15.13 · VRP pct **0.175** · term_ratio 0.680 (`term_stale_days 0`) · VVIX 86.27 ·
SKEW 143.9 (**47th pct — dead average, not elevated**) · netliq $5,792bn, 13w −$139.7bn
(z −0.71), RRP $0.2bn · absorption **0.716** · BTC funding z **1.82** (was 3.08 on 08-22).

Crowded long: BTC, COPPER, GOLD (**idx52 1.00, 52-week max**), SPY.
Crowded short: EURUSD, QQQ, TLT.

**4 of 8 conditions firing:** SHORT_VOL_CROWDED_WHILE_CHEAP · CROWDED_CLUSTER ·
POSITIONING_UNWIND · ABSORPTION_SPIKE.

## Gotchas found the hard way — do not re-derive

1. **Yahoo stopped serving `^VIX9D` / `^VIX3M`** (1 row vs 2512 for `^VIX`).
   Now sourced from the FREE public `cdn.cboe.com/api/global/us_indices/daily_prices/`
   file — **not** the metered LiveVol API, so no data points are spent.
2. **`build_csvs` used to OVERWRITE each CSV** with no merge — the only data module here
   that did. One gapped response deleted a month of history permanently, which is how
   `vol_desk._vol_surface` ended up dividing VIX9D from one date by VIX3M from another
   and publishing it as a term structure for five weeks. Now merges, and rebases stored
   rows onto the fresh adjustment vintage via the stored `factor` (Yahoo re-adjusts on
   every ex-div; merging without rebasing splices one series from many bases).
3. **Session calendar must come from the MERGED anchor.** Stored CSVs predate the rolling
   10y window, so filtering against a fresh-only session set injected 3 phantom sessions
   (`day.txt` 2515 vs SPY 2512). Fixed; verify `wc -l day.txt` == SPY csv rows.
4. **`trailing_pct` is an INCLUSIVE trailing-252 rank** `(w <= w[-1]).mean()`, matching
   `vol_desk.vrp_snapshot`. NOT `rolling().rank(pct=True)`, which averages ties. If these
   diverge the workbook and the flag print different VRP percentiles.
5. **The workbook is more precise than the flags** — flags round to 2dp. Never expect
   exact equality; use a relative tolerance (~1e-5).
6. **Excel holds an exclusive lock**; a build while it is open writes
   `market_state.pending.xlsx` and the task WARNS rather than failing. Close the workbook
   and re-run to refresh in place.
7. `index < today` drops the current session deliberately (Yahoo serves a live intraday
   bar; CBOE publishes post-close). Correct for the 11:45 run; an evening manual run
   therefore shows the prior session.

## THE OPEN WORK — two dead layers (macro, narrative, flows fixed 2026-08-25)

Constant across all 89 assets: breadth (market-level by design), valuation and
fundamentals (both NO_DATA). Flows is now REAL but has one snapshot, so it reads
NO_DATA until ~20 accumulate. Read the tally honestly: the narrative layer only varies
across the **13 instruments a registered story references** and flows across the **20
funds with an issuer feed**; for everything else those rows are still constant.

### 1. Flows — SOURCE REPLACED 2026-08-25, now real but empty
The diagnosis stands: `unique_shares = 1` across all 19 Yahoo snapshots, because
yfinance `sharesOutstanding` for an ETF is a static cached field. Waiting would never
have helped.

Fixed by going to the issuers (`qlib_lab/issuer_shares.py`), using the identity every
ETF publishes daily — **shares = total net assets / NAV**:

| provider | endpoint | funds | resolution |
|---|---|---|---|
| SSGA fund-finder JSON | `ssga.com/bin/v1/ssmp/fund/fundfinder` | SPY, GLD, BIL + 11 sector SPDRs | NAV to $0.01, AUM to $0.01m → **~$5.3m/day noise on SPY** |
| iShares screener JSON | `ishares.com/us/product-screener/product-screener-v3.1.jsn` | IWM, EEM, TLT, TIP, HYG, LQD | full precision → ~1 share |

20 of 23 funds. **QQQ (Invesco), USO and CPER (USCF) report NOTHING** — their
documented CSV download URLs return the HTML page. Both issuer endpoints answered
through the TLS-intercepting proxy with plain `urllib` and no cert flags.

Three traps handled, and each is a test:
1. **Never difference across the source change.** The legacy Yahoo rows are still in
   `etf_shares.parquet` with a null `source`; differencing them against the first
   issuer row would print one enormous fictional creation. `compute_flows` reads only
   `source in (ssga, ishares)` and reports `legacy_rows_ignored`.
2. **`history_since` reports the ISSUER start (2026-08-25)**, not the Yahoo install
   date, so the flag cannot claim five weeks of record it does not have.
3. **Row date is the issuer's own as-of**, not the clock — a weekend re-run would
   otherwise mint a second date with identical numbers and a zero flow.

**State: 1 snapshot.** z needs 10 and `flow_20d` needs 21, so the flows layer stays
NO_DATA for roughly a month. The value is entirely forward, exactly like the state log.

### 2. Narrative consistency — DONE 2026-08-25
`market_state.STORIES` (10 registered) + `story_checks()` + `asset_frame._narrative`,
which REPLACED the `catalyst` row. Each story states a claim, the correlation it
implies, and its state on a 60d window with 252d beside it: HOLDING / WEAK / BROKEN
(BROKEN = the sign has inverted past |0.10|, not merely faded). Rendered on a new
`Narratives` sheet; per-story states land in the state log and get one column each on
`History`, which is the point — a turn gets a DATE instead of being misremembered.

First reading, 2026-08-24 (60d / 252d):

| story | implies | 60d | 252d | state |
|---|---|---|---|---|
| BTC is digital gold | corr(BTC,GOLD) > +0.40 | +0.60 | +0.28 | **HOLDING** (slow WEAK) |
| BTC is a risk asset | corr(BTC,SPY) > +0.40 | +0.32 | +0.47 | WEAK (slow HOLDING) |
| dollar weighs on commodities | corr(DXY,COPPER) < −0.30 | −0.38 | −0.32 | HOLDING |
| gold is the other side of the dollar | corr(GOLD,DXY) < −0.30 | −0.54 | −0.41 | HOLDING |
| gold dislikes yields | corr(GOLD,US10Y) < −0.30 | −0.22 | −0.20 | WEAK |
| bonds hedge equities | corr(TLT,SPY) < −0.20 | **+0.32** | +0.24 | **BROKEN** |
| vol is the equity hedge | corr(VIX,SPY) < −0.50 | −0.80 | −0.82 | HOLDING |
| credit tracks equity | corr(HYG,SPY) > +0.40 | +0.82 | +0.78 | HOLDING |
| EM is a dollar trade | corr(EEM,DXY) < −0.30 | −0.30 | −0.39 | HOLDING |
| oil is a growth trade | corr(OIL,SPY) > +0.30 | **−0.32** | −0.34 | **BROKEN** |

Two broken on both windows: the equity/bond hedge is positively correlated, and oil is
moving against the growth complex. The narrative layer prints 5 bullish / 3 bearish /
81 neutral — the 81 are instruments no story references, which the row says out loud.

**Caveats that must survive:** the SET of stories was chosen with the 2026-08 tape
already visible (thresholds are conventional round numbers, not fitted, and no search
was run) — so `STORIES_NOTE` repeats the `CONDITIONS_NOTE` rule: only log rows after
2026-08-25 are evidence. HOLDING is not a reason to own anything, and nothing here was
scored against forward returns.

### 3. Valuation — the line matters
"What should this be worth" is a fair-value MODEL → forecast → gate applies. The legal
descriptive version is **dislocation from a stated anchor as a z-score** ("gold is 2.1σ
above its 3y relationship with real yields + DXY"). Anchors computable today: anchored
VWAP, long-run trend z, cross-asset regression residual. BTC realized price/MVRV needs
on-chain (free APIs exist).

### 4. Fundamentals — three connectors, not one
BTC hashrate/difficulty (mempool.space / blockchain.info, free) · copper inventories
(**`copper_brain` already computes these and they are unused**) · S&P 500 financials
(`company_lab`).

### 5. Macro — DONE 2026-08-25 (`asset_frame._macro`)
Was: computed per-asset SPY/DXY correlations, discarded them, returned the market-wide
liquidity verdict — bearish for all 89. Now the liquidity impulse is signed by each
asset's own 60d corr to SPY: `MACRO_IMPULSE_Z 0.50` (below that a netliq change is drift,
not an impulse), `MACRO_CORR_MIN 0.20` (below that the asset is unlinked and the impulse
does not reach it). On 2026-08-24 (netliq -139.7bn/13w, z -0.71) the layer went from
89 bearish to **65 bearish / 13 bullish / 11 neutral** — VIX and DXY read *helped* by a
drain, BIL unlinked, SPY/QQQ/GOLD/BTC/COPPER fought.

**What it does NOT establish:** one market-wide impulse routed through a correlation SIGN
is still one bit of market information partitioned three ways; the cross-sectional
content is the sign of corr(asset, SPY), not an independent macro read per asset. The
impulse x co-movement mapping is a stated prior, not a fitted one, and has never been
scored against forward returns. 4 tests added (113/113 pass).

## Smaller open items

- `FUNDING_EXTREME` condition — funding is on the sheet but no rule reads it, despite
  being the sharpest BTC signal this week (z 3.08 → 1.82 in three sessions).
- `_breadth` EMA warm-up guard is on panel length, not per column — latent, zero columns
  affected today.
- `vol_desk._vol_surface` still has **no staleness guard of its own**. The CBOE fix
  restored leg alignment so exposure is much reduced, but `options_desk` consumes that
  field for its pairs tilt. Touches a live desk — user decision.
- `MacroGpuLab-WeeklyRegate` is still Disabled (ShiftExport and AlpacaGpuDaily were
  re-enabled this session).

## Honesty notes that must survive into the next session

- **CONDITIONS are NOT blind pre-registration** — written 2026-08-23 after the August
  tape had been seen. Only fire-log rows dated after that are evidence. `CONDITIONS_NOTE`
  says so on the sheet; keep it there.
- The state log has **one row**. Its value is entirely forward.
- The audit verified the numbers are correctly computed from their sources. It says
  **nothing** about whether any of them are useful.
- Nothing on this box has ever cleared PBO < 0.5 AND DSR > 1.645.
