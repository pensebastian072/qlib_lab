"""Telegram notifier for the pre-open vol desk — stdlib only, fail-safe.

A send failure NEVER blocks the desk (advisory notification, not the hot path).
Secrets live in gitignored secrets/telegram.json: {"bot_token": "...", "chat_id": "..."}.
ssl.create_default_context() uses the Windows cert store, so the box's
TLS-intercepting proxy verifies cleanly without extra bundles.

Advisory only: this pushes a paper ticket for a human to read before 09:30.
It never places an order and holds no broker keys.
"""
from __future__ import annotations

import json
import ssl
import urllib.request
from pathlib import Path

from . import config


def _load_secrets(path: Path | None = None):
    try:
        data = json.loads((path or config.TELEGRAM_SECRETS_PATH).read_text(encoding="utf-8"))
        token, chat_id = data.get("bot_token"), data.get("chat_id")
        if token and chat_id:
            return {"bot_token": token, "chat_id": str(chat_id)}, None
        return None, "telegram.json missing bot_token/chat_id"
    except Exception as e:  # noqa: BLE001
        return None, f"unavailable: {e}"


def send_message(text: str, secrets_path: Path | None = None, timeout: int = 15):
    """POST sendMessage. Returns (ok, error). Never raises."""
    secrets, err = _load_secrets(secrets_path)
    if secrets is None:
        return False, err
    url = f"https://api.telegram.org/bot{secrets['bot_token']}/sendMessage"
    payload = json.dumps({"chat_id": secrets["chat_id"], "text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            ok = 200 <= resp.status < 300
            return ok, None if ok else f"http {resp.status}"
    except Exception as e:  # noqa: BLE001
        return False, f"send failed: {e}"


def format_ticket(t: dict) -> str:
    """One pre-open Telegram message from a vol_desk ticket dict."""
    vrp = t.get("vrp") or {}
    mag = t.get("magnitude") or {}
    fwd = t.get("vrp_forward") or {}
    st = t.get("structure") or {}
    action = t.get("action", "NO_TRADE")

    lines = [
        f"PRE-OPEN VOL DESK {t.get('underlying', 'SPY')} - {action} (SHADOW)",
    ]
    if vrp:
        lines.append(
            f"spot {vrp.get('spot'):.2f} | VIX {vrp.get('vix'):.1f} | "
            f"VRP pct {vrp.get('vrp_pct', 0):.0%} (data thru {vrp.get('data_through')})"
        )
    if mag:
        lines.append(
            f"magnitude: exp 1d move ~{mag.get('expected_move_pct', 0):.2f}% "
            f"(${mag.get('expected_move_usd', 0):.1f}) | regime pct "
            f"{mag.get('magnitude_pct', 0):.0%} [{mag.get('proxy')}]"
        )
    if fwd:
        lines.append(
            f"VRP-forward tilt: {fwd.get('bucket')} insurance -> "
            f"~{fwd.get('expected_fwd_ret_1m', 0):+.1%}/mo hist bias (display only)"
        )
    if action in ("LONG_VOL", "SELL_PUT_SPREAD") and st:
        if st.get("type") == "straddle":
            lines.append(
                f"structure: BUY straddle {st.get('strike')} exp {t.get('expiry')} "
                f"| est debit ${st.get('est_debit')} | max risk ${st.get('max_risk_usd')} "
                f"| breakeven {st.get('breakeven_move_pct')}%"
            )
        else:
            lines.append(
                f"structure: SELL put spread {st.get('short_strike')}/"
                f"{st.get('long_strike')} exp {t.get('expiry')} | est credit "
                f"${st.get('est_credit')} | max risk ${st.get('max_risk_usd')}"
            )
    lines.append(f"reason: {t.get('reason')}")
    lines.append("advisory paper ticket - human places any order (1 defined-risk lot max)")
    return "\n".join(lines)
