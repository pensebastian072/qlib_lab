# Pre-registration — does COT positioning predict or front-run price?

Written **2026-08-25, BEFORE any of the tests below were run.** Nothing in this file was
edited after seeing a result; findings go in `RESULTS_cot_predictive_2026-08-25.md`.

This crosses the `market_state` tripwire on purpose: joining COT to forward returns makes
this a MODEL, not a thermometer, so the PBO/DSR gate applies to anything that comes out
of it wanting to be traded. Nothing here is promoted by this study — the most it can
produce is a candidate that would then have to clear the gate.

## Data

- `data/cot_history.parquet` — CFTC legacy futures-only, weekly, 13 assets, 2006-01-03
  to 2026-08-18 (1,077 weeks for the long series; BTC 437, IWM 613, VIX 1,034).
- Prices: the qlib .bin store, daily close, same 13 names.
- **Lookahead rule**: a report as-of Tuesday is only usable from the following Monday.
  Every weekly row is stamped `effective_date = report_date + 6 days` (already enforced
  in `cot.py`) and forward returns are measured from the close ON OR AFTER that date.
  Any test that reads the return of the report week itself is invalid.

## Windows

Both are reported, always side by side:

- **12m** — the window asked for: report dates in the trailing 52 weeks. ~52 observations
  per asset. This is too small to establish anything on its own; it is reported because
  it is the question, not because it is evidence.
- **Full** — 2006 to date, the reference sample. If the 12m and full samples disagree,
  the honest reading is that 12m is noise, not that the regime changed.

## Features (computed as `cot.weekly_features` already defines them)

| feature | meaning |
|---|---|
| `net_pct` | (noncommercial long − short) / open interest |
| `z156` | z-score of `net_pct` over trailing 156 weeks |
| `idx52` | percentile of `net_pct` within trailing 52 weeks (the COT index) |
| `chg4w` | 4-week change in `net_pct` — the flow, not the level |

## Hypotheses, with the direction stated in advance

Forward horizons: **1w (5 sessions), 4w (21), 13w (63)**, from the effective date.

- **H1 — crowding is contrarian.** IC(`idx52`, forward return) < 0. This is the reading
  the workbook already prints ("crowded = vulnerable"); here it is actually tested.
- **H2 — positioning is trend-confirming.** IC(`z156`, forward return) > 0.
  H1 and H2 are opposite claims and both are widely believed; that is why both are
  registered rather than whichever survives.
- **H3 — the flow leads the price.** IC(`chg4w`, forward return) > 0.
- **H4 — extremes carry the signal.** Mean forward return when `idx52 >= 0.80` minus mean
  when `idx52 <= 0.20`, against that asset's unconditional mean over the same window.
  Prediction under H1: the difference is negative.
- **H5 — the front-running test.** Correlation of the weekly change in `net_pct` with the
  PRIOR week's return versus the NEXT week's return. If the prior-week correlation
  dominates, COT is a coincident-to-lagging record of what already happened, and no
  amount of horizon tuning makes it a leading indicator. **This is the decisive test of
  the question asked**, and it is the one I expect to fail the hopeful reading.

## Statistics — decided before the numbers exist

- Spearman rank IC per asset per horizon. **Per-asset tables, never a pooled headline** —
  a pooled IC over 13 assets whose returns are correlated is one bet reported as 13.
- Overlapping windows inflate significance. Hit rates and any t-statistic use
  **non-overlapping samples only** (step = horizon), and the effective n is printed next
  to every number.
- Multiple testing: 3 signals × 3 horizons × 13 assets = **117 tests**, so ~6 will clear
  p < 0.05 by chance alone. The verdict is read from the DISTRIBUTION of ICs across
  assets (how many share the predicted sign, and how large the median is), not from the
  best cell. No cell is promoted to a headline on its own.
- Sign-flipping is banned. If a hypothesis comes out with the opposite sign, that is
  recorded as the registered hypothesis FAILING, not as a discovery of the reverse
  effect. Any reverse effect worth anything must be registered and tested out of sample.

## What would count as a positive result

All of:

1. The predicted sign holds for **at least 9 of 13 assets** at the same horizon,
2. median |IC| >= 0.05 on the FULL sample (not just 12m), and
3. H5 shows the next-week correlation at least as large as the prior-week correlation.

Anything less is reported as "no evidence", and specifically not as "promising, needs
more data". If it does pass, it is still SHADOW until PBO < 0.5 and Deflated Sharpe >
1.645 on a proper walk-forward — this study does not promote anything.
