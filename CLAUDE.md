# qlib_lab — agent guide

Microsoft **Qlib research bench** (pyqlib 0.9.7, LightGBM + Alpha158) over the same
22-asset macro universe as `macro_gpu_lab`. Trains daily, walks forward out-of-sample,
runs the canonical overfit gate, and publishes an **advisory SHADOW flag-file**
(`journal/flags/qlib_state.json`) plus an ETF institutional-flow table
(`journal/flags/etf_flows.json`). HQ's dashboard reads both files off hot path,
fail-safe. Nothing here touches a broker or the webhook path.

Skills: load `paper-trading-guardrails` before touching flag-file/publish code,
`quant-research-gate` before model/gate work, `win-quant-env` before
installs/git/PS/Task Scheduler.

## Hard rules

1. Research / advisory / paper only. No broker, no execution, no live keys.
2. Models stay SHADOW until the gate clears (PBO < 0.5 AND Deflated Sharpe > 0 on
   every horizon). Do not soften the gate to pass a model — a correct gate usually
   FAILS the first model; report that honestly.
3. Flag-files are fail-safe: readers get neutral on stale/missing/corrupt.
4. The gate math is imported from `macro_gpu_lab.validate` (via PYTHONPATH) — the
   canonical port. Never re-implement PBO/DSR here.
5. ETF flows are a PROXY: delta(shares outstanding) x NAV, snapshotted daily.
   **Shares come from the ISSUERS** (`issuer_shares.py`: SSGA fund-finder JSON +
   iShares screener JSON, shares = total net assets / NAV), covering 20 of 23 funds.
   QQQ/USO/CPER have no public machine-readable feed and report NO flow, never a zero.
   The Yahoo `sharesOutstanding` field this used until 2026-08-25 is STATIC for ETFs --
   19 snapshots, one distinct value per fund, so every flow was identically zero and
   always would be. Those legacy parquet rows are kept but never differenced against
   issuer rows (`compute_flows` filters on `source`), so flow history restarts
   2026-08-25. Each row carries `noise_usd`, its own source's dollar resolution: SSGA
   rounds NAV to a cent, ~$5m/day on SPY, and a smaller "flow" is rounding.
6. **TGA comes from the Treasury DTS (daily), not FRED (weekly)** — changed
   2026-08-31, `fred_liquidity.py`. TGA is the most volatile netliq leg and was the
   only one read weekly + ffilled, so $30bn+ intraweek swings landed in the wrong
   week. Three traps, all live, all covered by `tests/test_fred_liquidity.py`:
   - The DTS **renamed the TGA row twice and moved the value to a different column
     each time** (`Federal Reserve Account`/close → `Treasury General Account (TGA)`
     /close → `...(TGA) Closing Balance`/**open**). Filtering on today's label alone
     returns nothing before 2022. Resolve by row PRIORITY, never a changeover date.
   - `close_today_bal` is the literal **string `"null"`** in the current era, not
     JSON null. `pd.to_numeric(errors="coerce")` → NaN → ffill → TGA frozen forever.
   - **FRED `WTREGEN` is a WEEK AVERAGE**, while `WALCL` is a Wednesday LEVEL, so the
     old netliq subtracted a weekly mean from a point-in-time balance sheet. Use
     **`WDTGAL`** (Wednesday level). The cross-check found this: vs DTS, WTREGEN was
     off a median $12.9bn (max $215.8bn, 2025-04-16 tax week); WDTGAL agrees to
     **$0.0bn across all 858 overlapping Wednesdays**. That cross-check runs every
     build and lands in `liquidity_state.json` — a non-zero median means a reader bug.

   The DTS is primary, WDTGAL is the fail-safe fallback (`tga_source` records which).
   **This moved history**: vs the old weekly build, `netliq` differs by up to $398bn,
   `chg13w_z` by up to 2.01, `L1_netliq_trend` flips sign on **9.2% of days**, and
   `market_state`'s `netliq z < -2.0` rule fires 104 days vs 108 with **40 disagreeing**.
   Anything fitted or thresholded on the weekly series needs re-checking — that is a
   re-registration question under `quant-research-gate`, not a free bug fix.

## Environment

- venv is uv-managed, **Python 3.11**, own pins (numpy<2, pandas 2.0.x — Qlib needs
  the old stack; this is WHY it cannot share macro_gpu_lab's numpy-2/torch venv).
- Always `.venv\Scripts\python.exe` — bare `python` is the Store stub.
- Reinstall: `uv pip install --python .venv\Scripts\python.exe --native-tls -r requirements.txt`
- `scripts/dump_bin.py` is vendored verbatim from microsoft/qlib `scripts/`
  (commit d5379c5, MIT) — it ships in the repo, not the wheel.

## Commands

- Fetch OHLCV + rebuild qlib .bin data: `.venv\Scripts\python.exe -m qlib_lab.data_fetch`
- Train + gate + publish flag: `.venv\Scripts\python.exe -m qlib_lab.pipeline --once`
- ETF flow snapshot: `.venv\Scripts\python.exe -m qlib_lab.etf_flows`
- Issuer shares only (no history write): `.venv\Scripts\python.exe -m qlib_lab.issuer_shares`
- Tests: `.venv\Scripts\python.exe -m pytest`
- Daily job (all of the above): Task Scheduler `QlibLabDaily` at 11:15 (verified
  against the live trigger 2026-08-24; this line said 09:30 and was stale), registered
  via `scripts\register_daily.ps1`, runs hidden via `scripts\_run_daily.vbs`
  (python.exe hidden — Norton blocks pythonw.exe).

### `run_preopen.ps1` — a slow C: is the failure mode, not a crash

`PreOpenVolDesk` (08:55) normally finishes in ~2 min on a healthy disk. On 2026-08-11
C: (the SATA spinner) hit **0% idle, queue depth 17-25, 0.16-1.2 s/read** and the same
run took **55 min**, landing its flag at 09:51 — after `OptionsDeskDaily`'s 09:05 read
AND the 09:15 catch-up, so the desk published on the PREVIOUS session's signal. Every
process exited 0. Nothing logged a failure. Three fixes, all in `scripts\run_preopen.ps1`:

- **90 min `Start-Process` + `WaitForExit` hang-breaker**, then `taskkill /T /F`. The bare
  `cmd /c` it replaced blocks forever, so a genuinely stuck run sat until the next reboot
  while Task Scheduler still reported `LastTaskResult 0`. Kill the **tree** — the venv
  python is a uv trampoline whose real interpreter is a grandchild (verified: 3 descendants
  before the kill, 0 after). 90 min is far above a bad-disk day; it breaks a HANG and does
  not enforce the pre-open deadline. `$OVERRUN_MIN` (20) only WARNs.
- **`python -u`.** stdout through `>>` is block-buffered, so a slow run shows only its start
  line for its entire life and you cannot tell "alive and slow" from "dead" by reading the
  log. Diagnosing 2026-08-11 needed CPU/working-set deltas instead.
- **Exit code means "is there a fresh flag"**, not "did a process end": the script compares
  `vol_desk_state.json` mtime across the run and exits 2 if it did not move.

Log appends go through the retrying `Write-Log` helper, not `Out-File` — the child holds the
same file open through `>>` and Norton transiently locks it, which threw `IOException` and
silently dropped the final `done` line on this script's first real run.

## `market_state` — the descriptive layer (NO forecast)

`qlib_lab/market_state.py` (data) + `market_state_xlsx.py` (render) build
`journal/exports/market_state.xlsx` — VRP/vol surface, COT positioning, funding,
liquidity, correlation/absorption, and the crowding x correlation cross-term.

**It forecasts nothing, so hard rule 2 and the PBO/DSR gate do not apply to it.**
That is a scope statement, not an exemption to be borrowed: the moment anything here
starts implying a forward return, it becomes a model and the gate applies again.

- Pure consumer. Reuses `cot.weekly_features`, `funding.daily_features`,
  `fred_liquidity.daily_frame` and the qlib .bin store. Re-ports no maths.
- `trailing_pct()` replicates `vol_desk.vrp_snapshot`'s **inclusive** trailing-252
  rank `(w <= w[-1]).mean()` — deliberately NOT `rolling().rank(pct=True)`, which
  averages ties and returns a different number. If these two ever disagree the
  workbook and `vol_desk_state.json` will print different VRP percentiles.
- The narrative paragraph is **deterministic template-fill** assembled from the same
  evaluated conditions as the flags beneath it, so prose cannot drift from data.
  `test_narrative_matches_fired` enforces it. Do not replace it with generated text.
- `CONDITIONS` are **not blind pre-registration** — written 2026-08-23 after the
  2026-08 tape had been seen. `CONDITIONS_NOTE` says so on the sheet itself; keep it
  there. Only fire-log rows dated after registration are evidence of anything.
- **`STORIES`** (registry in `market_state.py`, `Narratives` sheet, `narrative` layer
  in the framework) pairs each market story with the correlation it implies and reports
  HOLDING / WEAK / BROKEN on a 60d window with 252d beside it. It REPLACED the old
  `catalyst` row, which was NEUTRAL for all 89 names forever; the FOMC countdown it
  carried is kept in the detail. `STORIES_NOTE` carries the same warning as
  `CONDITIONS_NOTE` -- thresholds are conventional, not fitted, but the SET was chosen
  with the 2026-08 tape visible, so only forward log rows are evidence. A HOLDING story
  is not a reason to own the asset; BROKEN is a fragility reading, like crowding.
  Per-story states are recorded in the state log so a turn gets a DATE on it.
- The **macro layer routes a market-wide impulse through a per-asset measurement**:
  net-liquidity 13w impulse (gated at |z| >= 0.50, below that it is drift) signed by the
  asset's own 60d corr to SPY (|corr| < 0.20 = unlinked, impulse does not reach it).
  Before this it printed the liquidity verdict verbatim for all 89 names while computing
  and discarding the correlations. The impulse x co-movement mapping is a STATED prior
  written in the module, never fitted, never scored against forward returns.
- **`state_panel.py`** rebuilds every registered reading as a DAILY HISTORY (2,514
  sessions, 2016-08 -> now) because four open items need one. `structure()` and
  `_breadth()` are latest-window only, so both are recomputed rolling -- absorption
  through the shared `market_state.absorption_from_corr`, so history and workbook are one
  implementation. Two alignment traps, both live: the window is INCLUSIVE of its own
  session (`.tail(n)` semantics), and the column selection is `dropna(how="all")` with a
  PAIRWISE corr, not complete-case.
- **As-of alignment, fixed 2026-08-27.** `structure()` had no "drop the current session"
  guard, so the 11:45 run built its correlation window on a LIVE PARTIAL bar -- published
  absorption 0.71650 vs 0.71597 once the session closed, feeding ABSORPTION_SPIKE's 0.70
  threshold with an unreproducible number. And `_last_row` took the newest row of each
  frame independently, so the sheet printed a netliq z and a funding z belonging to the
  session AFTER its own `as_of`. Use `_last_row_asof`; every reading must be as of the
  date the sheet claims.
- **`registry_corr.py`** asks how much of the registry is independent: 59 market-level
  readings carry ~13 independent signals in levels (participation ratio), the 89-asset
  market carries ~5. Median decorrelation is **50 sessions**, so the panel is ~50
  independent EPISODES, not 2,514 observations -- that is the denominator for anything
  computed on it. Levels and changes answer different questions and neither stands alone:
  `liq_rrp` correlates 0.79 with a slow story correlation in LEVELS and not at all in
  changes. `Registry` sheet + `research/RESULTS_registry_corr_2026-08-27.md`.
- Daily via `MarketStateDaily` 11:45 weekdays (after QlibLabDaily 11:15, off the
  morning spinner contention). `run_market_state.ps1` copies run_preopen.ps1's
  hang-breaker/tree-kill/`python -u`/mtime-exit-code defences — same disk, same traps.

## Consumers

- `hq-trading-system/analytics/ui_server.py` routes `/api/qlib` and `/api/etf_flows`
  read the two flag files read-only, fail-safe (missing/stale → neutral JSON).
