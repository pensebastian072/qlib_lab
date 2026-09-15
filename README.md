# qlib_lab

Microsoft [Qlib](https://github.com/microsoft/qlib) research bench feeding the HQ
trading dashboard with **advisory SHADOW** signals. Paper/advisory only — nothing
here places orders or sits on a trade hot path.

## What it does (daily, 09:30, Task Scheduler `QlibLabDaily`)

1. **Fetch** — full OHLCV daily history for the 22-asset macro universe (Yahoo v8
   chart API), written as per-symbol CSVs with qlib's `factor` convention.
2. **Dump** — `scripts/dump_bin.py` (vendored from microsoft/qlib) converts CSVs to
   qlib's `.bin` store under `qlib_data/us_data/`.
3. **Train** — Alpha158 features + LightGBM, expanding walk-forward, horizons 5d/21d,
   non-overlapping portfolio PnLs (top-3 long / bottom-3 short of 22 names).
4. **Gate** — canonical overfit gate imported from `macro_gpu_lab.validate`
   (PBO < 0.5 AND Deflated Sharpe > 0 on every horizon, n_trials-deflated).
   First-pass models are expected to FAIL and stay SHADOW.
5. **Publish** — atomic flag-file `journal/flags/qlib_state.json` + JSONL history
   `journal/qlib/YYYY-MM-DD.jsonl`.
6. **ETF flows** — daily snapshot of sharesOutstanding x price for the ETF map
   (universe ETFs + sector SPDRs); flow proxy = delta(shares) x price, 1d/5d/20d
   + z-score → `journal/flags/etf_flows.json`. History accumulates forward.

## Consumers

`hq-trading-system` dashboard (127.0.0.1:8099) — routes `/api/qlib` and
`/api/etf_flows` read the flag files read-only, fail-safe: missing/corrupt/stale
files degrade to neutral, never an error.

## Reuse map

| Concern | Canonical source |
|---------|-----------------|
| Overfit gate (PBO/DSR/walk-forward) | `macro_gpu_lab/macro_gpu_lab/validate.py` (PYTHONPATH import) |
| Universe + force buckets | mirrored from `macro_gpu_lab/macro_gpu_lab/config.py` |
| Yahoo v8 fetch pattern | `macro_gpu_lab/macro_gpu_lab/data.py` (period1/period2, never range=max) |
| Flag-file publish pattern | `macro_gpu_lab/macro_gpu_lab/publish.py` (atomic tmp+replace, fail-safe reader) |
| Scheduler pattern | `macro_gpu_lab/scripts/register_daily_export.ps1` + vbs hidden launcher |
| CSV→bin converter | vendored `scripts/dump_bin.py` (microsoft/qlib @ d5379c5, MIT) |

## Not used (deliberately)

MLflow server / qlib-server (recorder runs local-file only), point-in-time DB,
RL / NestedExecutor, HIST/MASTER graph models (need a broad cross-section),
qlib's Yahoo collector (broken upstream — we feed our own data).

## Setup

```powershell
uv venv --python 3.11
uv pip install --python .venv\Scripts\python.exe --native-tls -r requirements.txt
.venv\Scripts\python.exe -c "import qlib; print(qlib.__version__)"
```

## Viewer

```bash
pip install flask
python -m ui.app        # http://127.0.0.1:8103
```

Three read-only panels: the gate verdicts at 5d and 21d, the cross-sectional
ranking over 76 macro proxies, and the ETF creation/redemption flow table.

**It works on a fresh clone.** The repo ships committed snapshots of both flags
under `ui/snapshot/`, so the real output renders with no market data, no API key
and no Qlib install. If you have run the bench, the viewer prefers your live
`journal/flags/`. The header says which, and how old it is.

Two things the viewer recomputes rather than trusts:

- **Staleness**, from `as_of` against the clock. The `stale` field in a flag
  records what was true when the flag was written, which is a different question
  from whether it is old now.
- **Gate verdicts**, from the stored statistics against the canonical thresholds
  (`PBO < 0.5` and `Deflated Sharpe ratio > 1.645`). That bar was `0.0` until
  2026-07-30 — a `ratio > 0` test is only a median test, which best-of-8 pure
  noise clears about 45% of the time. Verdicts written under the old bar still
  sit marked PASS in a sibling bench's flag, so reading the `passes` field would
  republish a verdict the project has already retired. Any disagreement is shown
  as `superseded` rather than hidden. This flag currently has none.

The ranking is display only — the model has not cleared the gate, so nothing
here sizes, gates or vetoes anything. Binds `127.0.0.1` only, no POST route.
