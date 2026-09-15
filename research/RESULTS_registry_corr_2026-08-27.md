# How much independent information is in the registry?

Item 6 of the six-item list, run 2026-08-27. Raw output: `journal/exports/registry_corr.json`.
Panel: `data/state_panel.parquet`, 2,514 sessions, 2016-08-25 → 2026-08-26.

The framework prints twelve layers, eight conditions and ten stories, then counts how many
agree. That count only means something if the readings are independent. Nothing had ever
checked. This measures our own instruments against each other — it joins nothing to a
forward return, so it stays outside the gate.

## Headline

| matrix | readings | rows used | participation ratio | n for 90% of variance | absorption |
|---|---|---|---|---|---|
| market-level, **levels** | 59 | 1,710 | **13.2** | 22 | 0.794 |
| market-level, **changes** | 59 | 1,666 | 32.3 | 40 | 0.506 |
| + per-asset COT, levels | 98 | 1,710 | 19.4 | 33 | 0.831 |
| + per-asset COT, changes | 98 | 1,666 | 49.7 | 59 | 0.589 |
| **the market itself**, 89 assets | 89 | 60 | **5.34** | 18 | 0.940 |

Levels and changes answer different questions and neither stands alone. *Levels* is the
question the confluence tally depends on — do two readings say the same thing about the
state today. *Changes* is immune to the spurious-level problem but answers "do they move
together", which is not the same thing.

**59 market-level readings carry about 13 independent signals.** The registry is roughly
4.5× redundant in levels, 1.8× in changes.

Complete-case, so BTC funding (starts 2019-08) truncates the window for everyone: 1,710
rows, 2019-08-28 → 2026-08-25. Pairwise would have kept more rows and produced cells
estimated on different samples — and eigenvalues, which are the entire point here, that
can come out negative.

## The redundancy is three different things, and only one is a problem

**1. A reading and the threshold applied to it — trivial, but it still inflates the count.**

| pair | \|corr\| |
|---|---|
| `story_state_BONDS_HEDGE_EQUITIES` ≡ `story_corr_BONDS_HEDGE_EQUITIES` | 0.940 |
| `story_state_OIL_IS_A_GROWTH_TRADE` ≡ its corr | 0.913 |
| `cond_VOL_RICH` ≡ `vol_vrp`, `vol_vrp_pct` | family |
| `liq_netliq_chg13w_z` ≡ `liq_netliq_chg13w` | 0.888 |
| `fund_idx52w` ≡ `fund_z156` | 0.835 |

Expected — a condition is a threshold on its input. It matters anyway: **"four conditions
firing" is not four independent facts**, and the sheet presents it as if it were.

**2. Genuinely distinct readings that turn out to say the same thing.**

- `vol_vix` ≡ `vol_rv20` — implied and realized vol are one family in levels, which is
  worth sitting with, because VRP is built from the difference of two things that move
  together.
- `vol_vix_pct` ≡ `vol_term_ratio` — the term structure carries little beyond the VIX
  percentile itself.
- `breadth_above_50` ≡ `breadth_above_200` — two breadth numbers, one signal.
- `struct_absorption` ≡ `struct_top1_share` (0.740) — as expected from construction.

**3. Spurious level pairs — the reason the changes matrix exists.**

`liq_rrp` ≡ `story_corr_slow_EM_IS_A_DOLLAR_TRADE` at **0.790**, and the whole liquidity
level block (`netliq`, `walcl`, `rrp`) clusters with two slow story correlations. Nothing
connects the Fed's balance sheet to a 252-day EM/dollar correlation except that both
trended over the same seven years. In changes they separate. **No level pair in this
matrix should be read as a relationship without checking its change counterpart.**

## The number that matters more than any of the above

**Median decorrelation time: 50 sessions.** So 2,514 sessions of panel is about **50
independent episodes**, not 2,514 observations. Every correlation above is estimated on
roughly that much information, and so is anything else this panel is ever used for.

The spread across readings is the interesting part:

| reading | decorrelation |
|---|---|
| `cond_SHORT_VOL_CROWDED_WHILE_CHEAP` | **2 sessions** |
| `story_state_CREDIT_TRACKS_EQUITY` | 2 |
| `cond_POSITIONING_UNWIND` | 4 |
| `fund_z156`, `cond_VOL_RICH` | 5 |
| … | |
| `story_corr_BONDS_HEDGE_EQUITIES` | 411 |
| `liq_walcl`, `liq_netliq` | 483–485 |
| `story_corr_slow_BONDS_HEDGE_EQUITIES` | **never, inside the 500-session search cap** |

Two things follow, and both are reporting changes rather than maths changes:

- **Conditions that decorrelate in 2–5 sessions are dithering at a threshold, not
  describing a state.** The sheet prints "4 firing" as though it were a condition of the
  market. For the fast ones it is closer to a coin landing near an edge. What should be
  printed beside each fired condition is its **persistence** — how many consecutive
  sessions it has held — which the state log now makes computable.
- Readings at the other end (net liquidity levels, slow story correlations) contribute
  almost no independent observations at all. A seven-year window holds perhaps five
  independent looks at `liq_netliq`. Any claim resting on those is resting on five points.

## The market, for comparison

The 89-asset return matrix has a participation ratio of **5.34** — eighty-nine
instruments, roughly five independent factors, 94% of variance in the top 22 eigenvalues.
This is the input item 4 (the correlation graph) needs, and it is deliberately a separate
function from `market_state.structure()`: the 13-asset matrix feeds
`crowding_x_structure`, `ABSORPTION_SPIKE` and the published absorption number, and
widening it in place would silently move a live flag.

## What this does NOT establish

**It does not measure the twelve-layer confluence tally.** The panel holds market-level
readings; the per-asset layer verdicts are computed only for the current date, so their
mutual correlation is still unmeasured. Measuring it means running `asset_frame` as of
every historical date — minutes per date on this disk — or vectorising the twelve layers
into a panel. That is real work and it has not been done, so the claim "the tally is
four facts counted three times" remains **plausible and untested**.

It also establishes nothing about usefulness. A reading can be perfectly independent and
still describe nothing worth knowing.

## Two bugs this pass found, both fixed

Building the panel meant reconciling every reconstructed number against the live
workbook, which is how these surfaced:

1. **`structure()` had no "drop the current session" guard**, unlike `vol_history()`. A
   run during market hours — which the 11:45 job is — built its correlation window on a
   **live partial bar**. Published absorption for 2026-08-24 was 0.71650; the completed
   session recomputes to 0.71597. A number nobody could reproduce after the close, feeding
   `ABSORPTION_SPIKE` and its 0.70 threshold. Fixed; the guard now matches `vol_history`.
2. **`_last_row` took the newest row of each frame independently**, so the sheet could
   print readings from a session *after* its own `as_of`. On 2026-08-24 it published
   netliq z −0.7125 and BTC funding z 0.964 — both of which belong to 2026-08-25. Same
   family as the cross-date term-ratio bug, smaller blast radius. `_last_row_asof` now
   aligns liquidity, funding and positioning to the date the sheet claims.

Neither changed a verdict today. Both would have, eventually, at a threshold.
