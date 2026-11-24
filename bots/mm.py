#!/usr/bin/env python3
"""
Passive multi-symbol market maker for GoDark (BTC / ETH / SOL -USDC-PERP).

WHY PASSIVE
-----------
Crossing the spread as a taker cost about 7 bp per unit of volume on this paper
account, so this bot rests post-only quotes instead. Passive quoting was not
shown to be profitable.

WHERE THE p* RESEARCH FITS
--------------------------
Vol-of-vol is an adverse-selection proxy: when it spikes, resting quotes get
picked off. So stress -> widen spread and cut size. That is a far better use of
the signal than the directional overlay (worth +$8 of $341 there).

RISK
    MAX_INVENTORY_USD        per symbol
    MAX_TOTAL_INVENTORY_USD  aggregate -- BTC/ETH/SOL are ~1 beta, per-symbol
                             caps alone would badly understate real exposure
    MAX_DIVERGENCE_BPS       references disagree -> pull that symbol's quotes
    MAX_SESSION_LOSS_USD     cancel everything and stop
    DRY_RUN                  print quotes instead of sending

Run:  python mm.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import os
_SDK = Path(os.environ.get("GDX_EXAMPLES_DIR", Path(__file__).resolve().parent.parent / "gdx-examples"))
for _p in ("", "examples", "sdk"):
    sys.path.insert(0, str(_SDK / _p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import get_first, load_dotenv, print_order_error  # type: ignore
from godark import Environment, GodarkClient, Side  # type: ignore
try:
    from godark.rest_client import GodarkRestClient  # type: ignore
except Exception:  # pragma: no cover
    GodarkRestClient = None  # type: ignore

from feed import ReferenceBook, binance_klines, fetch_instruments
from regime import RegimeDetector

# asset -> GoDark symbol
MARKETS = {"BTC": "BTC-USDC-PERP", "ETH": "ETH-USDC-PERP", "SOL": "SOL-USDC-PERP"}
ENABLED = ["BTC", "ETH"]   # SOL does $72k/day venue-wide -- not worth an inventory slot

# --- quoting -------------------------------------------------------------
BASE_HALF_SPREAD_BPS = 8.0
STRESS_WIDEN_MULT    = 2.0
DIVERGENCE_SPREAD_MULT = 1.5   # half-spread floor = this x current reference divergence
QUOTE_NOTIONAL_USD   = 30_000.0   # net 7.5bp/fill after the 0.5bp maker fee = $22.50/fill
QUOTE_INTERVAL_SEC   = 3.0
REQUOTE_DRIFT_BPS    = 3.0
REQUOTE_REFRESH_SEC  = 30.0

# --- risk ----------------------------------------------------------------
MAX_INVENTORY_USD       = 150_000.0   # per symbol
MAX_TOTAL_INVENTORY_USD = 225_000.0   # all symbols (correlated) = 24.6% of the book
INVENTORY_SKEW_BPS      = 14.0    # was 6 -- too weak against an 8bp half-spread
FLATTEN_AT              = 0.45    # past this fraction of cap, quote ONLY the reducing side
MAX_DIVERGENCE_BPS      = 12.0
MAX_SESSION_LOSS_USD    = 9_000.0     # ~1% of book
BAR_SECONDS             = 60
DRY_RUN                 = True 

# fallbacks if the instruments endpoint is unreachable
FALLBACK_DECIMALS = {"BTC": (1, 4), "ETH": (2, 3), "SOL": (3, 1)}   # (price, qty)


class Book:
    """Per-symbol state: inventory, regime, quote bookkeeping."""

    def __init__(self, asset: str, symbol: str, px_dp: int, qty_dp: int) -> None:
        self.asset, self.symbol = asset, symbol
        self.px_dp, self.qty_dp = px_dp, qty_dp
        self.regime = RegimeDetector(bar_window=21, vv_window=21, z_lookback=200,
                                     stress_z=1.0, stress_size=0.5)
        self.qty = 0.0
        self.avg_px = 0.0
        self.realized = 0.0
        self.fills = 0
        self.volume_usd = 0.0
        self.stress = False
        self.quotes_live = False
        self.last_mid = 0.0
        self.last_quote_ts = 0.0
        self.fills_at_quote = 0
        self.stood_down = False
        self.live_ids: dict[str, int | None] = {"BUY": None, "SELL": None}
        self.quotes_sent = 0

    def notional(self, mark: float) -> float:
        return self.qty * mark

    def unrealized(self, mark: float) -> float:
        return self.qty * (mark - self.avg_px) if self.qty else 0.0

    def on_fill(self, side, qty: float, price: float) -> None:
        signed = qty if str(side).endswith("BUY") else -qty
        self.fills += 1
        self.volume_usd += qty * price
        if self.qty == 0 or (self.qty > 0) == (signed > 0):
            total = self.qty + signed
            if total:
                self.avg_px = (self.avg_px * self.qty + price * signed) / total
            self.qty = total
        else:
            closing = min(abs(signed), abs(self.qty))
            direction = 1 if self.qty > 0 else -1
            self.realized += direction * closing * (price - self.avg_px)
            self.qty += signed
            if self.qty == 0:
                self.avg_px = 0.0
            elif (self.qty > 0) != (direction > 0):
                self.avg_px = price

    def legs(self, mid: float, half_bps: float, size_mult: float,
             total_inv_usd: float) -> list[dict]:
        inv_usd = self.notional(mid)
        lean = max(-1.0, min(1.0, inv_usd / MAX_INVENTORY_USD))
        skew = INVENTORY_SKEW_BPS * lean

        bid = round(mid * (1 - (half_bps + skew) / 10_000), self.px_dp)
        ask = round(mid * (1 + (half_bps - skew) / 10_000), self.px_dp)
        qty = round(QUOTE_NOTIONAL_USD * size_mult / mid, self.qty_dp)
        if qty <= 0 or bid >= ask:
            return []

        out = []
        room_to_buy = inv_usd < MAX_INVENTORY_USD and total_inv_usd < MAX_TOTAL_INVENTORY_USD
        room_to_sell = inv_usd > -MAX_INVENTORY_USD and total_inv_usd > -MAX_TOTAL_INVENTORY_USD
        # HARD one-sided stop. A skew only nudges; under persistently one-way flow
        # you keep accumulating until you are just holding a directional punt.
        # Past FLATTEN_AT we quote only the side that reduces the position.
        if inv_usd > FLATTEN_AT * MAX_INVENTORY_USD or total_inv_usd > FLATTEN_AT * MAX_TOTAL_INVENTORY_USD:
            room_to_buy = False
        if inv_usd < -FLATTEN_AT * MAX_INVENTORY_USD or total_inv_usd < -FLATTEN_AT * MAX_TOTAL_INVENTORY_USD:
            room_to_sell = False
        # cancel_order_id makes a leg a REPLACE. Omitting it is a pure place, which
        # silently stacks a new pair of orders on every single re-quote.
        if room_to_buy:
            leg = {"side": "BUY", "price": bid, "quantity": qty, "time_in_force": "GTC"}
            if self.live_ids["BUY"]:
                leg["cancel_order_id"] = int(self.live_ids["BUY"])
            out.append(leg)
        if room_to_sell:
            leg = {"side": "SELL", "price": ask, "quantity": qty, "time_in_force": "GTC"}
            if self.live_ids["SELL"]:
                leg["cancel_order_id"] = int(self.live_ids["SELL"])
            out.append(leg)
        return out


async def main() -> int:
    load_dotenv()
    creds = {
        "api_key_id": get_first("GODARK_API_KEY_ID", "GDX_API_KEY_ID"),
        "api_secret": get_first("GODARK_API_SECRET", "GDX_API_SECRET"),
        "passphrase": get_first("GODARK_PASSPHRASE", "GDX_PASSPHRASE"),
    }
    if not all(creds.values()):
        print("Missing credentials in .env", file=sys.stderr)
        return 1

    print(f"Mode: {'DRY RUN (no orders sent)' if DRY_RUN else '*** LIVE (paper account) ***'}")

    refs = ReferenceBook(ENABLED)
    for name, status in refs.probe().items():
        print(f"  reference {name:12s} {status}")

    # real tick/lot sizes where we can get them
    decimals = {}
    try:
        instruments = fetch_instruments()
        for asset, sym in MARKETS.items():
            row = instruments.get(sym, {})
            decimals[asset] = (int(row.get("price_decimals", FALLBACK_DECIMALS[asset][0])),
                               int(row.get("quantity_decimals", FALLBACK_DECIMALS[asset][1])))
        print(f"  instruments: live ({len(instruments)} symbols)")
    except Exception as e:
        decimals = dict(FALLBACK_DECIMALS)
        print(f"  instruments: FALLBACK decimals ({type(e).__name__}) -- verify before sizing up")

    books: dict[str, Book] = {}
    for asset in ENABLED:
        px_dp, qty_dp = decimals[asset]
        b = Book(asset, MARKETS[asset], px_dp, qty_dp)
        try:
            for px in binance_klines(asset, 400):
                st = b.regime.update(px)
            b.stress = st.stress
            b.last_mid = px
            print(f"  {asset}: seeded 400 bars, regime ready={st.ready} "
                  f"({'STRESS' if st.stress else 'normal'}), px_dp={px_dp} qty_dp={qty_dp}")
        except Exception as e:
            print(f"  {asset}: seed failed ({type(e).__name__}) -- cold start")
        books[asset] = b

    sym_to_asset = {v: k for k, v in MARKETS.items()}
    id_to_asset: dict[int, str] = {}

    async with GodarkClient(environment=Environment.TESTNET, **creds) as client:
        print(f"\nConnected as user_uuid={client.user_uuid}")
        await client.subscribe(["orders", "positions"])

        # Seed inventory from the REAL position. Without this the counter starts at
        # zero on every restart while the actual position persists, so every cap is
        # wrong by whatever you already hold -- that is how a $174k ETH short got
        # built past a $150k cap across several restarts.
        try:
            from godark.rest_client import GodarkRestClient as _RC
            async with _RC(**creds) as _rc:
                _ps = await _rc.get_positions()
            _id2a = {1: "BTC", 2: "ETH", 5: "SOL"}
            for _r in getattr(_ps, "rows", []):
                _a = _id2a.get(_r.symbol_id)
                if _a in books:
                    _q = float(_r.size) * (1 if str(_r.side).endswith("BUY") else -1)
                    books[_a].qty = _q
                    books[_a].avg_px = float(_r.entry_price)
                    print(f"  seeded {_a} inventory from live position: {_q:+.4f} "
                          f"@ {float(_r.entry_price):,.2f}")
        except Exception as _e:
            print(f"  !! could not read live positions ({type(_e).__name__}) -- "
                  f"caps may be wrong; flatten first")

        # Stale orders from a previous run (or a hard kill) survive and stack.
        # The UI showed 11 working orders when there should be 4. Start clean.
        for _a, _sym in MARKETS.items():
            try:
                c = await client.cancel_all_orders(_sym)
                if getattr(c, "count", 0):
                    print(f"  cleared {c.count} stale order(s) on {_sym}")
            except Exception:
                pass

        # map symbol_id -> asset so fills route to the right book
        for asset, sym in MARKETS.items():
            try:
                id_to_asset[client._resolve_symbol(sym)] = asset
            except Exception:
                pass

        async def read_equity() -> float | None:
            """account_equity lives on the REST client, not the WS client."""
            if GodarkRestClient is None:
                print("    [equity] GodarkRestClient import failed")
                return None
            try:
                async with GodarkRestClient(**creds) as rc:
                    acct = await rc.get_account()
                summary = getattr(acct, "account", None)
                if summary is None:
                    print("    [equity] no account summary on response")
                    return None
                # The SDK dataclass exposes only 4 of the wire's 11 fields --
                # account_equity is dropped by the parser. total_collateral is
                # the authoritative balance and moves with realized PnL + fees,
                # which is exactly what we want to diff over a session.
                for field in ("account_equity", "total_collateral", "free_collateral"):
                    val = getattr(summary, field, None)
                    if val is not None:
                        if field != "account_equity":
                            print(f"    [equity] using {field}")
                        return float(val)
                print(f"    [equity] no usable field on {type(summary).__name__}")
                return None
            except Exception as e:
                print(f"    [equity] {type(e).__name__}: {e}")
                return None

        start_equity = await read_equity()
        print(f"Starting account equity: "
              + (f"${start_equity:,.2f}" if start_equity is not None
                 else "unavailable (all-in bps will be pre-fee only)"))

        def on_order(u) -> None:
            try:
                filled = float(u.filled_qty or 0)
                px = float(u.price)
            except (TypeError, ValueError):
                return
            if filled <= 0:
                return
            if not any(k in str(u.update_type).upper() for k in ("FILLED",)):
                return
            asset = id_to_asset.get(u.symbol_id)
            if asset is None:
                return
            b = books[asset]
            b.on_fill(u.side, filled, px)
            print(f"    FILL {asset} {u.side} {filled} @ {px:,.4f}  "
                  f"inv=${b.notional(px):+,.0f} realized={b.realized:+.2f}")

        client.on_order_update(on_order)

        async def _drain(factory) -> None:
            """SDK queues every push for its async iterators (maxsize=256). Fills are
            handled by callback, so unconsumed iterators overflow and spam
            'Stream buffer full'. Drain and discard."""
            try:
                async for _ in factory():
                    pass
            except Exception:
                pass

        drains = []
        for factory in (client.order_updates, client.position_updates,
                        client.positions_snapshots, client.balance_updates,
                        client.funding_rate_updates, client.settlement_updates,
                        client.margin_alerts, client.system_health_updates,
                        client.leverage_settings_updates):
            try:
                drains.append(asyncio.create_task(_drain(factory)))
            except Exception:
                pass

        print(f"Quoting {', '.join(MARKETS[a] for a in ENABLED)} | "
              f"+/-{BASE_HALF_SPREAD_BPS}bp, ${QUOTE_NOTIONAL_USD:,.0f}/side/symbol | "
              f"Ctrl+C to stop.\n")

        first = True
        try:
            while True:
                if not first:
                    await asyncio.sleep(QUOTE_INTERVAL_SEC)
                first = False

                # ONE http call per venue, off the event loop so fill callbacks
                # are not stalled behind network IO.
                try:
                    snap = await asyncio.to_thread(refs.read_all)
                except Exception as e:
                    print(f"[refs] {type(e).__name__} -- holding")
                    continue
                if not snap:
                    continue
                marks = {a: v[0] for a, v in snap.items()}
                total_inv = sum(books[a].notional(marks[a]) for a in marks)
                total_pnl = sum(books[a].realized + books[a].unrealized(marks[a]) for a in marks)

                if total_pnl <= -MAX_SESSION_LOSS_USD:
                    print(f"!! session PnL {total_pnl:+,.0f} breached limit -- cancelling all, stopping")
                    if not DRY_RUN:
                        for a in ENABLED:
                            try:
                                await client.cancel_all_orders(MARKETS[a])
                            except Exception:
                                pass
                    break

                now = time.time()
                bar_tick = int(now // BAR_SECONDS)

                for asset in ENABLED:
                    b = books[asset]
                    if asset not in snap:
                        continue
                    mid, div, quotes = snap[asset]

                    if bar_tick != getattr(b, "_bar", None):
                        b._bar = bar_tick
                        st = b.regime.update(mid)
                        b.stress = st.stress
                        b._size_mult = st.size_multiplier
                    size_mult = getattr(b, "_size_mult", 1.0)

                    if div > MAX_DIVERGENCE_BPS:
                        if not b.stood_down:
                            print(f"[{time.strftime('%H:%M:%S')}] {asset} divergence {div:.1f}bp "
                                  f"-- pulling quotes {quotes}")
                            if not DRY_RUN:
                                try:
                                    await client.cancel_all_orders(b.symbol)
                                except Exception:
                                    pass
                            b.stood_down = True
                            b.quotes_live = False
                            b.live_ids = {"BUY": None, "SELL": None}
                        continue
                    b.stood_down = False

                    drift = abs(mid - b.last_mid) / b.last_mid * 10_000 if b.last_mid else 1e9
                    if not (not b.quotes_live or drift >= REQUOTE_DRIFT_BPS
                            or b.fills != b.fills_at_quote
                            or (now - b.last_quote_ts) >= REQUOTE_REFRESH_SEC):
                        continue
                    b.last_mid, b.last_quote_ts, b.fills_at_quote = mid, now, b.fills

                    # If Binance and Hyperliquid disagree by D bps, our "mid" is only
                    # good to ~D/2. Quoting a half-spread tighter than that is quoting
                    # into our own measurement error, so floor it.
                    half = max(BASE_HALF_SPREAD_BPS * (STRESS_WIDEN_MULT if b.stress else 1.0),
                               div * DIVERGENCE_SPREAD_MULT)
                    legs = b.legs(mid, half, size_mult, total_inv)
                    if not legs:
                        continue

                    desc = " ".join(
                        f"{l['side'][0]}{l['quantity']}@{l['price']:,.{b.px_dp}f}" for l in legs)
                    print(f"[{time.strftime('%H:%M:%S')}] {asset:3s} mid={mid:>10,.2f} div={div:4.1f}bp "
                          f"{'STR' if b.stress else 'nrm'} half={half:4.1f}bp "
                          f"inv=${b.notional(mid):+7,.0f} tot=${total_inv:+7,.0f} | {desc}")

                    if DRY_RUN:
                        b.quotes_live = True
                        continue
                    try:
                        ack = await client.mass_quote(b.symbol, legs, post_only=True)
                        b.quotes_live = True
                        b.quotes_sent += 1
                        for leg, r in zip(legs, ack.results):
                            if r.status in ("open", "filled") and r.new_order_id:
                                b.live_ids[leg["side"]] = r.new_order_id
                            else:
                                b.live_ids[leg["side"]] = None
                        bad = [r for r in ack.results if r.status == "failed"]
                        if bad:
                            print(f"    {asset} {len(bad)}/{len(ack.results)} legs REJECTED "
                                  f"codes={[r.error_code for r in bad]}")
                        elif b.quotes_sent <= 3 or b.quotes_sent % 20 == 0:
                            ids = ",".join(str(r.new_order_id)[-6:] for r in ack.results)
                            print(f"    {asset} resting ok (quote #{b.quotes_sent}, ids ...{ids})")
                    except Exception as e:
                        b.live_ids = {"BUY": None, "SELL": None}
                        print_order_error(f"    {asset} mass_quote failed", e)

        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nInterrupted.")
        finally:
            for t in drains:
                t.cancel()
            if not DRY_RUN:
                for a in ENABLED:
                    try:
                        await client.cancel_all_orders(MARKETS[a])
                    except Exception:
                        pass
                print("All quotes cancelled.")

            print("\n--- session ---")
            tot_v = tot_r = 0.0
            for a in ENABLED:
                b = books[a]
                if b.fills:
                    bps = b.realized / b.volume_usd * 10_000 if b.volume_usd else 0
                    print(f" {a}: fills={b.fills} volume=${b.volume_usd:,.0f} "
                          f"realized={b.realized:+,.2f} ({bps:+.2f} bps) inv={b.qty:+.4f}")
                tot_v += b.volume_usd
                tot_r += b.realized
            allin = tot_r / tot_v * 10_000 if tot_v else 0
            print(f" TOTAL: volume=${tot_v:,.0f} realized={tot_r:+,.2f} => {allin:+.2f} bps (pre-fee)")
            end_equity = await read_equity()
            if start_equity is not None and end_equity is not None:
                delta = end_equity - start_equity
                net_bps = delta / tot_v * 10_000 if tot_v else 0
                print(f" ACCOUNT equity {start_equity:,.2f} -> {end_equity:,.2f} = {delta:+,.2f} "
                      f"=> {net_bps:+.2f} bps ALL-IN (includes fees -- this is the number "
                      f"that decides whether the spread is wide enough)")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
