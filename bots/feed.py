"""
Public BTC reference price feed.

GoDark's hosted edge exposes no public trade/orderbook stream (only
orders/positions/volume/open_interest/funding_rate on /ws/v1, and /ws/gomarket
is localnet-only). mark_price arrives only inside PositionsSnapshot rows, which
require an open position -- useless for warming up a signal.

So we build the regime signal off a public BTC reference feed. BTC-USDC-PERP
tracks global BTC closely enough that drawdown / realized-vol / vol-of-vol
regime state is effectively identical; execution still happens on GoDark.

Tries several venues so one blocked/geo-restricted host doesn't kill the bot.
"""
from __future__ import annotations

import httpx

TIMEOUT = 10.0


def _binance_klines(limit: int) -> list[float]:
    r = httpx.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": "BTCUSDT", "interval": "1m", "limit": limit},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return [float(row[4]) for row in r.json()]


def _binance_last() -> float:
    r = httpx.get(
        "https://api.binance.com/api/v3/ticker/price",
        params={"symbol": "BTCUSDT"}, timeout=TIMEOUT,
    )
    r.raise_for_status()
    return float(r.json()["price"])


def _okx_klines(limit: int) -> list[float]:
    r = httpx.get(
        "https://www.okx.com/api/v5/market/candles",
        params={"instId": "BTC-USDT", "bar": "1m", "limit": str(min(limit, 300))},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    rows = r.json()["data"]
    return [float(row[4]) for row in reversed(rows)]   # OKX returns newest-first


def _okx_last() -> float:
    r = httpx.get(
        "https://www.okx.com/api/v5/market/ticker",
        params={"instId": "BTC-USDT"}, timeout=TIMEOUT,
    )
    r.raise_for_status()
    return float(r.json()["data"][0]["last"])


def _kraken_klines(limit: int) -> list[float]:
    r = httpx.get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": "XBTUSD", "interval": 1}, timeout=TIMEOUT,
    )
    r.raise_for_status()
    result = r.json()["result"]
    key = next(k for k in result if k != "last")
    return [float(row[4]) for row in result[key]][-limit:]


def _kraken_last() -> float:
    r = httpx.get(
        "https://api.kraken.com/0/public/Ticker",
        params={"pair": "XBTUSD"}, timeout=TIMEOUT,
    )
    r.raise_for_status()
    result = r.json()["result"]
    key = next(iter(result))
    return float(result[key]["c"][0])


def _coinbase_last() -> float:
    r = httpx.get("https://api.exchange.coinbase.com/products/BTC-USD/ticker", timeout=TIMEOUT)
    r.raise_for_status()
    return float(r.json()["price"])


SOURCES = [
    ("binance", _binance_klines, _binance_last),
    ("okx", _okx_klines, _okx_last),
    ("kraken", _kraken_klines, _kraken_last),
    ("coinbase", None, _coinbase_last),
]


class PriceFeed:
    """Picks the first reachable public venue and sticks with it."""

    def __init__(self) -> None:
        self.name: str | None = None
        self._klines = None
        self._last = None

    def select(self) -> str:
        errors = []
        for name, klines_fn, last_fn in SOURCES:
            try:
                last_fn()
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__}")
                continue
            self.name, self._klines, self._last = name, klines_fn, last_fn
            return name
        raise RuntimeError("No public price source reachable -- tried " + ", ".join(errors))

    def history(self, limit: int = 500) -> list[float]:
        """1-minute closes, oldest first. Empty list if this venue has no kline endpoint."""
        if self._klines is None:
            return []
        return self._klines(limit)

    def last(self) -> float:
        return self._last()


# ---------------------------------------------------------------------------
# Multi-asset reference feeds: the three GoDark names in the primer.
# "when Pyth, Hyperliquid and Binance diverge, the market is moving and resting
#  liquidity is thinner than usual -- widen the limit or wait."
# ---------------------------------------------------------------------------

PYTH_FEEDS = {
    "BTC": "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
    "ETH": "ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
    "SOL": "ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
}
BINANCE_PAIRS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}


def pyth_prices(assets: list[str]) -> dict[str, float]:
    """One hermes call for every asset."""
    ids = "&".join(f"ids[]={PYTH_FEEDS[a]}" for a in assets)
    r = httpx.get(f"https://hermes.pyth.network/v2/updates/price/latest?{ids}", timeout=TIMEOUT)
    r.raise_for_status()
    by_id = {a.lower(): a for a in PYTH_FEEDS}
    out = {}
    for row in r.json()["parsed"]:
        fid = str(row["id"]).lower().removeprefix("0x")
        for asset, feed in PYTH_FEEDS.items():
            if feed.lower() == fid:
                p = row["price"]
                out[asset] = float(p["price"]) * (10 ** int(p["expo"]))
    return out


def hyperliquid_prices(assets: list[str]) -> dict[str, float]:
    """One allMids call for every asset."""
    r = httpx.post("https://api.hyperliquid.xyz/info", json={"type": "allMids"}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return {a: float(data[a]) for a in assets if a in data}


def binance_prices(assets: list[str]) -> dict[str, float]:
    """One ticker call for every asset."""
    syms = '[' + ','.join(f'"{BINANCE_PAIRS[a]}"' for a in assets) + ']'
    r = httpx.get("https://api.binance.com/api/v3/ticker/price",
                  params={"symbols": syms}, timeout=TIMEOUT)
    r.raise_for_status()
    rev = {v: k for k, v in BINANCE_PAIRS.items()}
    return {rev[row["symbol"]]: float(row["price"]) for row in r.json()
            if row["symbol"] in rev}


def binance_klines(asset: str, limit: int = 400) -> list[float]:
    r = httpx.get("https://api.binance.com/api/v3/klines",
                  params={"symbol": BINANCE_PAIRS[asset], "interval": "1m",
                          "limit": limit}, timeout=TIMEOUT)
    r.raise_for_status()
    return [float(row[4]) for row in r.json()]


REFERENCES = {"pyth": pyth_prices, "hyperliquid": hyperliquid_prices, "binance": binance_prices}


class ReferenceBook:
    """Consensus price + divergence, batched: ONE http call per venue per cycle."""

    def __init__(self, assets: list[str]) -> None:
        self.assets = assets
        self.available: dict[str, object] = {}

    def probe(self) -> dict[str, str]:
        status = {}
        for name, fn in REFERENCES.items():
            try:
                px = fn(self.assets)
                if not px:
                    raise RuntimeError("empty")
                self.available[name] = fn
                status[name] = "ok " + " ".join(f"{a} {v:,.2f}" for a, v in px.items())
            except Exception as e:
                status[name] = f"unavailable ({type(e).__name__})"
        if not self.available:
            raise RuntimeError("no reference feed reachable")
        return status

    def read_all(self) -> dict[str, tuple[float, float, dict[str, float]]]:
        """{asset: (consensus_mid, divergence_bps, per-venue quotes)} in 1 call/venue."""
        per_venue: dict[str, dict[str, float]] = {}
        for name, fn in self.available.items():
            try:
                per_venue[name] = fn(self.assets)
            except Exception:
                continue
        out = {}
        for asset in self.assets:
            quotes = {v: p[asset] for v, p in per_venue.items() if asset in p}
            if not quotes:
                continue
            vals = sorted(quotes.values())
            n = len(vals)
            mid = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
            div = (vals[-1] - vals[0]) / mid * 10_000 if mid else 0.0
            out[asset] = (mid, div, quotes)
        return out


def fetch_instruments(rest_base: str = "https://api.godark-dex.com") -> dict:
    """Public instruments endpoint -- real tick/lot sizes instead of guesses."""
    r = httpx.get(f"{rest_base}/api/v1/instruments", timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    rows = data.get("instruments") or data.get("data", {}).get("instruments") or []
    out = {}
    for row in rows:
        sym = row.get("symbol")
        if sym:
            out[sym] = row
    return out
