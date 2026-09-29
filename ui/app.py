"""Standalone read-only viewer for the qlib_lab bench.

Binds 127.0.0.1 only. There is no POST route and nothing here can act.

Data sources, each preferring the live journal flag and falling back to a
committed snapshot shipped with the repo:

  journal/flags/qlib_state.json  ->  ui/snapshot/qlib_state.json
  journal/flags/etf_flows.json   ->  ui/snapshot/etf_flows.json

The fallback is the point: a fresh clone on any machine renders the real output
without market data, an API key, or a Qlib install. The page states which source
it is showing and as of when.

Two things are recomputed here rather than trusted:

  * Staleness, from `as_of` against the clock. The `stale` field in a flag
    records what was true when it was written.
  * Gate verdicts, from the stored statistics against the canonical thresholds.
    DEFLATED_SHARPE_MIN was raised from 0.0 to 1.645 on 2026-07-30 because a
    `ratio > 0` test is only a median test that best-of-8 pure noise clears
    about 45% of the time. Verdicts written before that are still marked PASS in
    some sibling flags; reading the field would republish a retired verdict.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template

# Mirrors macro_gpu_lab.config, the single lineage the bench itself imports.
# Restated so the viewer can render a snapshot without importing the bench.
DEFLATED_SHARPE_MIN = 1.645
PBO_MAX = 0.5

STALE_DAYS = 4

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = Path(__file__).resolve().parent
FLAGS_DIR = REPO_ROOT / "journal" / "flags"
SNAP_DIR = UI_DIR / "snapshot"

app = Flask(__name__)


# Browsers reject NaN/Infinity in JSON, which left the page stuck on "loading...".
# Emit them as null so the tables render; missing stats already show as "–".
import math as _math
from flask.json.provider import DefaultJSONProvider as _DJP


def _finite(o):
    if isinstance(o, float) and not _math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


class _FiniteJSON(_DJP):
    def dumps(self, obj, **kw):
        return super().dumps(_finite(obj), **kw)


app.json = _FiniteJSON(app)


def _parse_ts(value) -> datetime | None:
    try:
        ts = datetime.fromisoformat(str(value))
    except Exception:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _load(name: str, neutral: dict) -> dict:
    """Live flag, else shipped snapshot, else a neutral dict. Never raises."""
    for path, source in ((FLAGS_DIR / name, "live"), (SNAP_DIR / name, "snapshot")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            data["_source"] = source
            data["_source_path"] = str(path.relative_to(REPO_ROOT))
            _age(data)
            return data
    out = dict(neutral)
    out.update({"_source": "none", "_source_path": "", "stale": True, "age_days": None})
    return out


def _age(state: dict) -> None:
    ts = _parse_ts(state.get("as_of"))
    if ts is None:
        state["stale"] = True
        state["age_days"] = None
        return
    age = (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0
    state["age_days"] = round(age, 1)
    state["stale"] = age > STALE_DAYS


def _gate_rows(state: dict) -> list[dict]:
    """gate is horizon -> verdict. Recompute `passes`; flag disagreements."""
    rows = []
    for horizon, v in (state.get("gate") or {}).items():
        if not isinstance(v, dict):
            continue
        dsr, pbo = v.get("dsr_ratio"), v.get("pbo")
        stored = bool(v.get("passes"))
        passes = (
            dsr is not None and pbo is not None
            and dsr > DEFLATED_SHARPE_MIN and pbo < PBO_MAX
        )
        rows.append({
            "horizon": horizon,
            "passes": passes,
            "stored_passes": stored,
            "superseded": stored and not passes,
            "dsr_ratio": dsr,
            "pbo": pbo,
            "profit_factor": v.get("profit_factor"),
            "n_trades": v.get("n_trades"),
        })
    rows.sort(key=lambda r: str(r["horizon"]))
    return rows


def _asset_rows(state: dict) -> list[dict]:
    """assets is symbol -> {force, lean, conviction, horizons{h:{score,rank_pct,lean}}}."""
    rows = []
    for symbol, v in (state.get("assets") or {}).items():
        if not isinstance(v, dict):
            continue
        h = v.get("horizons") or {}
        h5, h21 = h.get("5d") or {}, h.get("21d") or {}
        rows.append({
            "symbol": symbol,
            "force": v.get("force", "?"),
            "lean": v.get("lean"),
            "conviction": v.get("conviction"),
            "score_5d": h5.get("score"),
            "rank_5d": h5.get("rank_pct"),
            "lean_5d": h5.get("lean"),
            "score_21d": h21.get("score"),
            "rank_21d": h21.get("rank_pct"),
            "lean_21d": h21.get("lean"),
        })
    rows.sort(key=lambda r: (r["rank_5d"] is None, r["rank_5d"]))
    return rows


@app.route("/")
def page_index():
    return render_template("index.html")


@app.route("/api/qlib")
def api_qlib():
    state = _load("qlib_state.json", {
        "status": "SHADOW", "enforce": "no", "promoted": False,
        "gate": {}, "assets": {}, "note": "qlib flag missing",
    })
    rows = _gate_rows(state)
    state["gate_rows"] = rows
    state["asset_rows"] = _asset_rows(state)
    state["n_assets"] = len(state["asset_rows"])
    state["n_pass"] = sum(1 for r in rows if r["passes"])
    state["n_superseded"] = sum(1 for r in rows if r["superseded"])
    state["dsr_min"] = DEFLATED_SHARPE_MIN
    state["pbo_max"] = PBO_MAX
    return jsonify(state)


@app.route("/api/etf_flows")
def api_etf_flows():
    state = _load("etf_flows.json", {"rows": [], "note": "etf_flows flag missing"})
    return jsonify(state)


@app.route("/health")
def health():
    return jsonify({"ok": True, "ts": datetime.now(timezone.utc).isoformat()})


def main() -> None:
    # 127.0.0.1 only. Never 0.0.0.0, never tunnelled.
    app.run(host="127.0.0.1", port=8103, debug=False)


if __name__ == "__main__":
    main()
