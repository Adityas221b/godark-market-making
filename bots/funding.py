#!/usr/bin/env python3
"""
Is there a mechanical carry edge? Perp funding is paid on a schedule regardless
of direction, so a persistently one-sided rate is money for taking the paying
side -- no forecast required. This is the only candidate left that does not
depend on predicting anything.
"""
from __future__ import annotations
import asyncio, sys, time
from pathlib import Path
import os
_SDK = Path(os.environ.get("GDX_EXAMPLES_DIR", Path(__file__).resolve().parent.parent / "gdx-examples"))
for _p in ("", "examples", "sdk"):
    sys.path.insert(0, str(_SDK / _p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import get_first, load_dotenv  # type: ignore
from godark.rest_client import GodarkRestClient  # type: ignore

SYMBOL_ID = {1: "BTC", 2: "ETH", 5: "SOL"}


async def main() -> int:
    load_dotenv()
    creds = {"api_key_id": get_first("GODARK_API_KEY_ID"),
             "api_secret": get_first("GODARK_API_SECRET"),
             "passphrase": get_first("GODARK_PASSPHRASE")}

    async with GodarkRestClient(**creds) as rc:
        print("=== FUNDING RATES ===")
        try:
            fr = await rc.get_funding_rates()
            if not fr:
                print("  (empty response)")
            for row in fr:
                d = row if isinstance(row, dict) else getattr(row, "__dict__", {}) or row
                print(f"  {d}")
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")

        print("\n=== OPEN INTEREST ===")
        try:
            oi = await rc.get_open_interest()
            for row in (oi or []):
                d = row if isinstance(row, dict) else getattr(row, "__dict__", {}) or row
                print(f"  {d}")
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")

        print("\n=== 24H VOLUME ===")
        try:
            v = await rc.get_volume()
            print(f"  {v}")
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")

        print("\n=== ACCOUNT ===")
        try:
            acct = await rc.get_account()
            s = getattr(acct, "account", None)
            if s:
                for f in ("total_collateral","position_margin","reserved_order_margin",
                          "free_collateral"):
                    print(f"  {f:24} {getattr(s, f, '-')}")
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")

        print("\n=== CURRENT POSITIONS ===")
        try:
            ps = await rc.get_positions()
            rows = list(getattr(ps, "rows", []))
            if not rows:
                print("  flat (no open positions)")
            for r in rows:
                print(f"  {SYMBOL_ID.get(r.symbol_id, r.symbol_id)} {r.side} size={r.size} "
                      f"entry={r.entry_price} mark={r.mark_price} upnl={r.unrealized_pnl}")
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
