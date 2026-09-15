"""Publish the qlib_lab shadow advisory flag-file + JSONL history.

Mirrors macro_gpu_lab/publish.py's fail-safe contract: atomic tmp+replace write;
read_state() returns a neutral dict on missing/corrupt/stale and NEVER raises —
any HQ reader stays off the hot path. Nothing here is enforced: `enforce`
reflects QLIB_ENFORCE (default "no"); promotion is a separate human env flip.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from . import config

NEUTRAL_FALLBACK = {
    "as_of": None,
    "stale": True,
    "status": "SHADOW",
    "enforce": "no",
    "promoted": False,
    "gate": {},
    "assets": {},
    "note": "fallback-neutral (flag missing/stale/corrupt)",
}


def build_state(scorecard: dict, signals: dict) -> dict:
    gate = {
        hk: {
            "passes": v.get("passes"),
            "dsr_ratio": (v.get("deflated_sharpe") or {}).get("ratio")
            if isinstance(v.get("deflated_sharpe"), dict) else v.get("deflated_sharpe"),
            "pbo": v.get("pbo"),
            "profit_factor": v.get("profit_factor"),
            "n_trades": v.get("n_trades"),
        }
        for hk, v in (scorecard.get("horizons") or {}).items()
    }
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "data_through": signals.get("data_through"),
        "stale": False,
        "status": "SHADOW",
        "enforce": config.QLIB_ENFORCE,          # advisory; "no" = shadow
        "model": scorecard.get("model"),
        "promoted": bool(scorecard.get("promoted")),
        "gate": gate,
        "topk": scorecard.get("topk"),
        "assets": signals.get("assets") or {},
        "note": "SHADOW research advisory (qlib Alpha158+LightGBM) — never enforced",
        "disclaimer": "ranking across macro PROXIES (incl. indices); advisory only",
    }


def publish(state: dict) -> str:
    """Atomically write the flag-file + append the daily JSONL history line."""
    tmp = config.FLAG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(config.FLAG_PATH)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with (config.HISTORY_DIR / f"{day}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(state) + "\n")
    return str(config.FLAG_PATH)


def read_state() -> dict:
    """Fail-safe reader (off hot path). Never raises; neutral on missing/stale."""
    fb = dict(NEUTRAL_FALLBACK)
    try:
        raw = json.loads(config.FLAG_PATH.read_text())
    except Exception:  # noqa: BLE001 — missing/corrupt -> neutral
        return fb
    try:
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(raw["as_of"])).total_seconds() / 86400.0
        if age > config.FLAG_STALE_DAYS:
            raw["stale"] = True
            raw["note"] = "stale (age > FLAG_STALE_DAYS)"
    except Exception:  # noqa: BLE001
        raw["stale"] = True
    return raw
