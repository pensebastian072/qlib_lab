"""Reachability probe: BTC perp funding-rate APIs through this box's TLS proxy."""
import json
import urllib.error
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 qlib_lab/1.0"}
PROBES = {
    "binance": "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=3",
    "bybit": ("https://api.bybit.com/v5/market/funding/history"
              "?category=linear&symbol=BTCUSDT&limit=3"),
    "okx": "https://www.okx.com/api/v5/public/funding-rate-history?instId=BTC-USDT-SWAP&limit=3",
}


def main():
    for name, url in PROBES.items():
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=15) as r:
                body = json.loads(r.read())
            n = (len(body) if isinstance(body, list)
                 else len((body.get("result") or {}).get("list", []))
                 if name == "bybit" else len(body.get("data", [])))
            print(f"{name:8s} OK    rows={n}")
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError) as e:
            print(f"{name:8s} FAIL  {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
