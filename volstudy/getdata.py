#!/usr/bin/env python3
"""Download N days of 1m klines from Binance (paginated) for the vol study."""
from __future__ import annotations
import csv, sys, time
from pathlib import Path
import httpx

SYMS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
OUT = Path(__file__).resolve().parent


def fetch(sym: str, days: int) -> list[list]:
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    rows, cur = [], start
    with httpx.Client(timeout=20) as c:
        while cur < end:
            r = c.get("https://api.binance.com/api/v3/klines",
                      params={"symbol": sym, "interval": "1m",
                              "startTime": cur, "limit": 1000})
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            cur = batch[-1][0] + 60_000
            print(f"\r  {sym}: {len(rows):,} bars", end="", flush=True)
            time.sleep(0.12)
    print()
    return rows


def main() -> int:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    for asset, sym in SYMS.items():
        rows = fetch(sym, days)
        p = OUT / f"{asset}_1m.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["open_time", "open", "high", "low", "close", "volume"])
            for r in rows:
                w.writerow([r[0], r[1], r[2], r[3], r[4], r[5]])
        print(f"  -> {p.name}  {len(rows):,} bars ({len(rows)/1440:.1f} days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
