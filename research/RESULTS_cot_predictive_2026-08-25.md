# Results — does COT positioning predict or front-run price?

Ran 2026-08-25 against `PREREG_cot_predictive_2026-08-25.md`, which fixed every
hypothesis, direction and pass criterion before the code ran. Raw output:
`cot_predictive_results.json`, `cot_gate_results.json`.

## Verdict

**No.** Not one registered hypothesis met its pre-registered bar, and the decisive
front-running test says COT positioning is better explained by the week that already
happened than by the week that follows.

The one direction that is at least CONSISTENT — crowded positioning being followed by
weaker returns — is real in sign and far too small in size to be worth acting on, and
when the best-looking version of it is turned into an actual rule it fails the gate.

## One correction to the pre-registration

The COT parquet holds 2006→2026, but the qlib price store keeps a rolling **10-year**
window, so the "full" sample is **2016-08-25 → 2026-08-25**, not twenty years. Every
"full" number below is a ten-year sample. The prereg said 2006; that was wrong about the
prices, not the positioning.

## The sample problem in the 12m window, stated before the numbers

12 months = 52 weekly reports per asset. At the 13-week horizon that is **four
non-overlapping observations per asset** (`n_nonoverlap = 4` in the JSON). Any 12m number
at that horizon is noise with a decimal point on it. It is reported because it was asked
for, and it is never used to support a conclusion.

## H1 — crowding is contrarian: IC(idx52, forward return) < 0

| window | horizon | assets with predicted sign | median IC | passes 9/13 | median \|IC\| ≥ 0.05 |
|---|---|---|---|---|---|
| full | 1w | 8/13 | −0.034 | no | no |
| full | 4w | 8/13 | −0.023 | no | no |
| full | 13w | **10/13** | −0.025 | yes | **no** |
| 12m | 13w | 9/13 | −0.119 | yes | yes — on n=4 per asset |

**FAILS.** The sign is the most consistent thing in the study (10 of 13 assets at 13
weeks) but the magnitude is a fifth of the registered bar. The 12m cell that "passes"
both criteria rests on four independent observations per asset, and the prereg said the
full sample decides.

Per-asset 13w full-sample IC, so nothing hides in an average: QQQ −0.21, OIL −0.19,
VIX −0.19, BTC −0.18, USDCAD −0.16, USDJPY −0.12, COPPER/IWM/SPY ≈ −0.02, GBPUSD −0.00,
GOLD +0.04, TLT +0.06, EURUSD +0.07.

## H2 — positioning is trend-confirming: IC(z156, forward return) > 0

**FAILS, and fails in the opposite direction.** At 13 weeks only 3 of 13 assets carry the
predicted positive sign; the median IC is **−0.111**. Per the prereg, this is recorded as
the registered hypothesis failing, NOT as the discovery of a contrarian effect: `z156` and
`idx52` both measure the level of positioning, so the negative sign is H1 restated on a
different scale, not independent evidence. Harvesting it by flipping the sign is the
p-hacking the prereg exists to prevent.

## H3 — the flow leads the price: IC(chg4w, forward return) > 0

**FAILS at every horizon.** 4/13, 7/13 and 3/13 assets carry the predicted sign at 1w, 4w
and 13w; median ICs −0.017, +0.006, −0.023. The four-week change in positioning carries
no directional information about the next one, four or thirteen weeks.

## H4 — extremes: crowded-long minus crowded-short forward return, 13w, full sample

Negative (crowded longs do worse) in **9 of 13** assets. It is concentrated, not broad:

| asset | crowded long | crowded short | unconditional | long − short |
|---|---|---|---|---|
| BTC | +5.3% | +21.4% | +14.7% | **−16.1pp** |
| VIX | −6.9% | +3.8% | −0.9% | **−10.6pp** |
| OIL | −3.1% | +3.8% | +0.4% | −7.0pp |
| QQQ | +2.1% | +5.5% | +3.5% | −3.5pp |
| SPY | +3.0% | +3.1% | +2.9% | −0.1pp |
| GOLD | −4.4% | −5.2% | −3.8% | +0.8pp |
| TLT | −5.8% | −7.6% | −6.8% | +1.7pp |

The effect lives in BTC, VIX and OIL. On SPY — the asset most people would actually apply
this to — the spread is **one tenth of a percent over thirteen weeks**. And these buckets
overlap heavily: BTC's 136 crowded-long weeks are roughly ten independent episodes.

## H5 — the front-running test (the actual question)

Correlation of the 4-week positioning change with the **prior** week's return versus the
**next** week's return:

| window | leads price in | median \|corr, prior week\| | median \|corr, next week\| |
|---|---|---|---|
| full (10y) | **4 of 13 assets** | 0.035 | 0.021 |
| 12m | 8 of 13 | 0.073 | 0.122 |

**COT does not front-run.** On the ten-year sample the positioning change tracks the week
that already happened better than the week that follows, in nine of thirteen assets, and
both correlations are near zero anyway. The 12m window flips the result — on ~50
observations per asset, which is exactly the sample size that produces flips like this.
Reporting the 12m row as "COT is starting to lead" would be the whole error this study
was built to avoid.

## And if we traded the best-looking cell anyway? (POST-HOC — not evidence)

Rule chosen **after** seeing the above, so its numbers are an upper bound, not a test:
short `idx52 >= 0.80`, long `idx52 <= 0.20`, hold 13 weeks, non-overlapping, all 13
assets, 2016-08-25 → 2026-03-30.

| | |
|---|---|
| trades | 482 (non-overlapping) |
| hit rate | 52.1% |
| profit factor | 1.21 |
| Sharpe per trade | 0.060 |
| **PBO** | **0.016** — passes (< 0.5) |
| **Deflated Sharpe, n_trials = 117** | **ratio −1.21 → FAILS** |
| Deflated Sharpe if this had been the only idea ever tried (n_trials = 1) | ratio 1.372, still below the 1.645 threshold |

This is the cleanest illustration of the gate on this box so far: **PBO is excellent and
DSR still kills it.** The rule is not curve-fit across time — that is what the low PBO
says — it is simply too weak to survive having been selected from 117 cells. Even
pretending it was the only hypothesis ever entertained, it does not clear the bar.

Verdict: **SHADOW, and not worth promoting to a candidate.** Nothing here is wired into
anything.

## What this establishes, and what it does not

Establishes, on a ten-year sample: the 4-week COT flow has no measurable lead on price;
positioning LEVEL has a consistent but very small contrarian sign at a 13-week horizon;
that sign is concentrated in BTC, VIX and OIL and is absent on SPY; and the strongest
rule built from it fails the canonical gate on Deflated Sharpe.

Does NOT establish: that COT is useless as a DESCRIPTIVE reading. Nothing here touches
what `market_state` actually claims — that crowding plus correlation says several
positions are one bet. That claim is about the present, not the future, and this study
neither supports nor refutes it. What the study does kill is any suggestion that the
crowding column is quietly predictive and could be traded off.

Also not established: anything about horizons beyond 13 weeks, non-linear or conditional
readings (crowding × absorption, crowding × VRP), or the commercial/hedger side of the
report. Any of those would need its own pre-registration, not a re-run of this data.
