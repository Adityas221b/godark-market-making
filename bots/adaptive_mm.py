#!/usr/bin/env python3
"""
Self-tuning market maker: a contextual bandit picks the spread/size configuration
per market regime, learning from the markout of its own fills.

WHAT ADAPTS
  arm     = (half_spread_bp, clip_usd)   -- 4 discrete configs
  context = volatility bucket (lo/mid/hi) from the regime detector
  reward  = per-fill MARKOUT in bps, 60s after the fill:
              edge you quoted  +  what the market then did to you  -  maker fee
            A fill that looks good at the moment of execution but is followed by
            the price running away is scored as the loss it actually was.

  Thompson sampling explores cheaply and concentrates on whatever is working.
  State persists to bandit_state.json, so restarts keep what has been learned.

WHAT DOES NOT ADAPT (deliberately fixed guardrails)
  inventory caps, the hard flatten, the divergence stand-down, the session loss
  kill switch. A learner is not allowed to talk itself into more risk.

Run:  python adaptive_mm.py
"""
from __future__ import annotations
import asyncio, sys, time
from pathlib import Path

import os
_SDK = Path(os.environ.get("GDX_EXAMPLES_DIR", Path(__file__).resolve().parent.parent / "gdx-examples"))
for _p in ("", "examples", "sdk"):
    sys.path.insert(0, str(_SDK / _p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import get_first, load_dotenv, print_order_error  # type: ignore
from godark import Environment, GodarkClient  # type: ignore
try:
    from godark.rest_client import GodarkRestClient  # type: ignore
except Exception:
    GodarkRestClient = None  # type: ignore

from feed import ReferenceBook, binance_klines, fetch_instruments
from regime import RegimeDetector
from bandit import Arm, Bandit, markout_reward

MARKETS = {"BTC": "BTC-USDC-PERP", "ETH": "ETH-USDC-PERP"}
ENABLED = ["BTC"]                      # BTC is ~95% of venue volume

ARMS = [Arm(1.5, 15_000), Arm(3.0, 15_000), Arm(3.0, 40_000), Arm(6.0, 40_000)]

MARKOUT_SEC   = 60.0
QUOTE_INT     = 3.0
REQUOTE_DRIFT = 3.0
REFRESH_SEC   = 30.0
BAR_SECONDS   = 60

MAX_INVENTORY_USD       = 120_000.0
MAX_TOTAL_INVENTORY_USD = 180_000.0
FLATTEN_AT              = 0.45
MAX_DIVERGENCE_BPS      = 12.0
MAX_SESSION_LOSS_USD    = 6_000.0
MAKER_FEE_BP            = 0.5
DRY_RUN                 = True         # flip after you have watched it choose arms


class Book:
    def __init__(self, asset, symbol, px_dp, qty_dp):
        self.asset, self.symbol = asset, symbol
        self.px_dp, self.qty_dp = px_dp, qty_dp
        self.regime = RegimeDetector(bar_window=21, vv_window=21, z_lookback=200,
                                     stress_z=1.0, stress_size=0.5)
        self.qty = 0.0; self.avg_px = 0.0; self.realized = 0.0
        self.fills = 0; self.volume_usd = 0.0
        self.live_ids = {"BUY": None, "SELL": None}
        self.last_mid = 0.0; self.last_quote_ts = 0.0; self.quotes_live = False
        self.arm = ARMS[0]; self.ctx = "mid"; self.sigma_bp = 3.0
        self.pending: list[dict] = []      # fills awaiting their markout

    def notional(self, mark): return self.qty * mark
    def unrealized(self, mark): return self.qty * (mark - self.avg_px) if self.qty else 0.0

    def on_fill(self, side, qty, price, mid):
        signed = qty if str(side).endswith("BUY") else -qty
        self.fills += 1; self.volume_usd += qty * price
        if self.qty == 0 or (self.qty > 0) == (signed > 0):
            tot = self.qty + signed
            if tot: self.avg_px = (self.avg_px * self.qty + price * signed) / tot
            self.qty = tot
        else:
            closing = min(abs(signed), abs(self.qty))
            d = 1 if self.qty > 0 else -1
            self.realized += d * closing * (price - self.avg_px)
            self.qty += signed
            if self.qty == 0: self.avg_px = 0.0
            elif (self.qty > 0) != (d > 0): self.avg_px = price
        self.pending.append({"t": time.time(), "side": "BUY" if signed > 0 else "SELL",
                             "px": price, "mid": mid, "arm": self.arm, "ctx": self.ctx})

    def legs(self, mid, total_inv):
        half_bp, clip = self.arm.half_bp, self.arm.clip_usd
        inv = self.notional(mid)
        lean = max(-1.0, min(1.0, inv / MAX_INVENTORY_USD))
        skew_bp = 1.5 * half_bp * lean
        bid = round(mid * (1 - (half_bp + skew_bp) / 1e4), self.px_dp)
        ask = round(mid * (1 + (half_bp - skew_bp) / 1e4), self.px_dp)
        qty = round(clip / mid, self.qty_dp)
        if qty <= 0 or bid >= ask: return []
        buy  = inv <  FLATTEN_AT * MAX_INVENTORY_USD and total_inv <  FLATTEN_AT * MAX_TOTAL_INVENTORY_USD
        sell = inv > -FLATTEN_AT * MAX_INVENTORY_USD and total_inv > -FLATTEN_AT * MAX_TOTAL_INVENTORY_USD
        out = []
        if buy:
            leg = {"side": "BUY", "price": bid, "quantity": qty, "time_in_force": "GTC"}
            if self.live_ids["BUY"]: leg["cancel_order_id"] = int(self.live_ids["BUY"])
            out.append(leg)
        if sell:
            leg = {"side": "SELL", "price": ask, "quantity": qty, "time_in_force": "GTC"}
            if self.live_ids["SELL"]: leg["cancel_order_id"] = int(self.live_ids["SELL"])
            out.append(leg)
        return out


async def main() -> int:
    load_dotenv()
    creds = {"api_key_id": get_first("GODARK_API_KEY_ID"),
             "api_secret": get_first("GODARK_API_SECRET"),
             "passphrase": get_first("GODARK_PASSPHRASE")}
    if not all(creds.values()):
        print("missing credentials", file=sys.stderr); return 1

    print(f"Mode: {'DRY RUN' if DRY_RUN else '*** LIVE ***'}")
    print(f"arms: {[a.key() for a in ARMS]}   markout {MARKOUT_SEC:.0f}s\n")

    bandit = Bandit(ARMS)
    if bandit.stats:
        print("resuming learned state:\n" + bandit.table() + "\n")

    refs = ReferenceBook(ENABLED)
    for k, v in refs.probe().items():
        print(f"  reference {k:12s} {v}")
    try:
        instr = fetch_instruments()
    except Exception:
        instr = {}

    books = {}
    for a in ENABLED:
        row = instr.get(MARKETS[a], {})
        b = Book(a, MARKETS[a], int(row.get("price_decimals", 1)),
                 int(row.get("quantity_decimals", 4)))
        try:
            for px in binance_klines(a, 400):
                st = b.regime.update(px)
            b.last_mid = px
            b.sigma_bp = (st.realized_vol or 3e-4) * 1e4
            print(f"  {a}: seeded, sigma {b.sigma_bp:.2f} bp/bar")
        except Exception as e:
            print(f"  {a}: seed failed ({type(e).__name__})")
        books[a] = b

    async with GodarkClient(environment=Environment.TESTNET, **creds) as client:
        print(f"\nconnected {client.user_uuid}")
        await client.subscribe(["orders", "positions"])

        # seed inventory from REAL positions -- a reset counter is how a $150k cap
        # let a $174k position build across restarts
        if GodarkRestClient is not None:
            try:
                async with GodarkRestClient(**creds) as rc:
                    ps = await rc.get_positions()
                id2a = {1: "BTC", 2: "ETH", 5: "SOL"}
                for r in getattr(ps, "rows", []):
                    a = id2a.get(r.symbol_id)
                    if a in books:
                        books[a].qty = float(r.size) * (1 if str(r.side).endswith("BUY") else -1)
                        books[a].avg_px = float(r.entry_price)
                        print(f"  seeded {a} inventory {books[a].qty:+.4f} @ {books[a].avg_px:,.2f}")
            except Exception as e:
                print(f"  !! position read failed ({type(e).__name__}) -- flatten first")

        for a in ENABLED:
            try:
                c = await client.cancel_all_orders(MARKETS[a])
                if getattr(c, "count", 0): print(f"  cleared {c.count} stale order(s)")
            except Exception:
                pass

        id2asset = {}
        for a in ENABLED:
            try: id2asset[client._resolve_symbol(MARKETS[a])] = a
            except Exception: pass

        def on_order(u):
            try:
                filled = float(u.filled_qty or 0); px = float(u.price)
            except (TypeError, ValueError): return
            if filled <= 0 or "FILLED" not in str(u.update_type).upper(): return
            a = id2asset.get(u.symbol_id)
            if a is None: return
            b = books[a]
            b.on_fill(u.side, filled, px, b.last_mid or px)
            print(f"    FILL {a} {u.side} {filled} @ {px:,.2f}  arm={b.arm.key()} "
                  f"ctx={b.ctx}  inv=${b.notional(px):+,.0f}")
        client.on_order_update(on_order)

        drains = []
        async def _drain(f):
            try:
                async for _ in f(): pass
            except Exception: pass
        for f in (client.order_updates, client.position_updates, client.positions_snapshots,
                  client.balance_updates, client.funding_rate_updates,
                  client.settlement_updates, client.margin_alerts,
                  client.system_health_updates, client.leverage_settings_updates):
            try: drains.append(asyncio.create_task(_drain(f)))
            except Exception: pass

        print("\nrunning. Ctrl+C to stop.\n")
        bar = 0; t_report = time.time()
        try:
            first = True
            while True:
                if not first: await asyncio.sleep(QUOTE_INT)
                first = False
                try:
                    snap = await asyncio.to_thread(refs.read_all)
                except Exception:
                    continue
                if not snap: continue
                now = time.time()
                total_inv = sum(books[a].notional(snap[a][0]) for a in snap if a in books)
                pnl = sum(books[a].realized + books[a].unrealized(snap[a][0])
                          for a in snap if a in books)
                if pnl <= -MAX_SESSION_LOSS_USD:
                    print(f"!! session loss {pnl:+,.0f} -- stopping")
                    break

                for a in ENABLED:
                    if a not in snap: continue
                    b = books[a]; mid, div, _q = snap[a]

                    # resolve matured markouts -> bandit reward
                    still = []
                    for f in b.pending:
                        if now - f["t"] >= MARKOUT_SEC:
                            r = markout_reward(f["side"], f["px"], f["mid"], mid, MAKER_FEE_BP)
                            bandit.update(f["ctx"], f["arm"], r)
                            print(f"    markout {a} {f['side']} arm={f['arm'].key()} "
                                  f"ctx={f['ctx']} -> {r:+.2f} bp")
                        else:
                            still.append(f)
                    b.pending = still

                    nb = int(now // BAR_SECONDS)
                    if nb != bar:
                        st = b.regime.update(mid)
                        if st.realized_vol: b.sigma_bp = st.realized_vol * 1e4
                    b.ctx = Bandit.context(b.sigma_bp)

                    if div > MAX_DIVERGENCE_BPS:
                        if b.quotes_live and not DRY_RUN:
                            try: await client.cancel_all_orders(b.symbol)
                            except Exception: pass
                        b.quotes_live = False; b.live_ids = {"BUY": None, "SELL": None}
                        continue

                    drift = abs(mid - b.last_mid)/b.last_mid*1e4 if b.last_mid else 1e9
                    if not (not b.quotes_live or drift >= REQUOTE_DRIFT
                            or (now - b.last_quote_ts) >= REFRESH_SEC):
                        continue
                    b.last_mid, b.last_quote_ts = mid, now
                    b.arm = bandit.select(b.ctx)          # <-- the adaptive step

                    legs = b.legs(mid, total_inv)
                    if not legs: continue
                    desc = " ".join(f"{l['side'][0]}{l['quantity']}@{l['price']:,.{b.px_dp}f}" for l in legs)
                    print(f"[{time.strftime('%H:%M:%S')}] {a} mid={mid:,.2f} div={div:.1f}bp "
                          f"sig={b.sigma_bp:.1f}bp ctx={b.ctx} arm={b.arm.key()} "
                          f"inv=${b.notional(mid):+,.0f} pnl={pnl:+,.0f} | {desc}")
                    if DRY_RUN:
                        b.quotes_live = True; continue
                    try:
                        ack = await client.mass_quote(b.symbol, legs, post_only=True)
                        b.quotes_live = True
                        for leg, r in zip(legs, ack.results):
                            b.live_ids[leg["side"]] = r.new_order_id if r.status in ("open","filled") else None
                        bad = [r for r in ack.results if r.status == "failed"]
                        if bad: print(f"    {len(bad)} leg(s) REJECTED {[r.error_code for r in bad]}")
                    except Exception as e:
                        b.live_ids = {"BUY": None, "SELL": None}
                        print_order_error(f"    {a} mass_quote failed", e)

                bar = int(now // BAR_SECONDS)
                if now - t_report > 300:
                    t_report = now
                    bandit.save()
                    print("\n--- what it has learned ---\n" + bandit.table() + "\n")

        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\ninterrupted")
        finally:
            for t in drains: t.cancel()
            bandit.save()
            if not DRY_RUN:
                for a in ENABLED:
                    try: await client.cancel_all_orders(MARKETS[a])
                    except Exception: pass
            print("\n=== learned ===\n" + bandit.table())
            for a in ENABLED:
                b = books[a]
                if b.fills:
                    print(f"{a}: fills={b.fills} vol=${b.volume_usd:,.0f} "
                          f"realized={b.realized:+,.2f} inv={b.qty:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
