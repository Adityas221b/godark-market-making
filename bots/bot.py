#!/usr/bin/env python3
"""
Regime-gated momentum bot for GoDark (BTC-USDC-PERP), paper account.

Architecture (forced by what the hosted edge actually exposes):
  SIGNAL  <- public BTC reference feed (see feed.py). GoDark's hosted /ws/v1 has
             no public trade/orderbook stream, and /ws/gomarket is localnet-only.
  EXECUTE -> GoDark GodarkClient MARKET orders on BTC-USDC-PERP.

The regime overlay (regime.py) is the research payoff: drawdown + realized-vol +
vol-of-vol, the cluster that dominated both Lasso and XGBoost feature importance
in the p* study, used to SHRINK SIZE in stress regimes rather than to forecast a
better volatility exponent (which lost to plain p=2 out-of-sample).

`signal()` is a PLACEHOLDER momentum rule. Swap it for your real edge -- the
overlay wraps whatever produces the +1/0/-1 direction call.

Safety rails:
  DRY_RUN=True          prints intended orders instead of sending them
  MAX_POSITION_USD      caps notional exposure
  MAX_DAILY_LOSS_USD    flattens and halts for the session
  MIN_REBALANCE_USD     suppresses churn

Run:
    python bot.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import os
_SDK = Path(os.environ.get("GDX_EXAMPLES_DIR", Path(__file__).resolve().parent.parent / "gdx-examples"))
for _p in ("", "examples", "sdk"):
    sys.path.insert(0, str(_SDK / _p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import get_first, load_dotenv, print_order_error  # type: ignore
from godark import Environment, GodarkClient, OrderType, Side, TimeInForce  # type: ignore

from feed import PriceFeed
from regime import RegimeDetector

SYMBOL = "BTC-USDC-PERP"
BAR_SECONDS = 60
POLL_SECONDS = 5
MOMENTUM_BARS = 3
BASE_ORDER_USD = 500.0
MAX_POSITION_USD = 5_000.0
MAX_DAILY_LOSS_USD = 2_000.0
MIN_REBALANCE_USD = 100.0
ENTRY_VOL_MULT = 1.0      # need |momentum| > this * realized_vol before taking a side
EXIT_VOL_MULT = 0.4       # hysteresis: hold an existing side until momentum decays below this
SEED_BARS = 400
DRY_RUN = True


def signal(closes: list[float], lookback: int, realized_vol: float | None, current: int) -> int:
    """
    PLACEHOLDER entry logic: VOL-SCALED N-bar momentum with a deadband and hysteresis.

    A bare sign() flips on noise -- a -0.013% tick would reverse the whole book and
    pay the spread both ways. That is exactly the -$1.05/trade bleed pattern. So:
      - require |momentum| > ENTRY_VOL_MULT * realized_vol (in return terms) to OPEN a side
      - only drop back to flat once it decays below EXIT_VOL_MULT * realized_vol
      - never flip directly long<->short; go through flat first

    Replace with your real edge -- just return +1 / 0 / -1.
    """
    if len(closes) <= lookback or not realized_vol or realized_vol <= 0:
        return 0
    ref = closes[-1 - lookback]
    if ref <= 0:
        return 0
    mom = (closes[-1] - ref) / ref            # momentum as a return
    scaled = mom / (realized_vol * (lookback ** 0.5))   # vol-normalised over the lookback

    if current == 0:
        if scaled > ENTRY_VOL_MULT:
            return 1
        if scaled < -ENTRY_VOL_MULT:
            return -1
        return 0
    if current > 0:
        return 1 if scaled > EXIT_VOL_MULT else 0       # decay to flat, never straight to short
    return -1 if scaled < -EXIT_VOL_MULT else 0


class Trader:
    def __init__(self, client, mark: float) -> None:
        self.client = client
        self.position_usd = 0.0
        self.position_btc = 0.0
        self.entry_price = mark
        self.realized_pnl = 0.0
        self.trades = 0
        self.killed = False

    def unrealized(self, mark: float) -> float:
        if self.position_btc == 0:
            return 0.0
        return self.position_btc * (mark - self.entry_price)

    async def go_to(self, target_usd: float, mark: float) -> None:
        delta_usd = target_usd - self.position_usd
        if abs(delta_usd) < MIN_REBALANCE_USD:
            return

        side = Side.BUY if delta_usd > 0 else Side.SELL
        qty = round(abs(delta_usd) / mark, 6)
        if qty <= 0:
            return

        if DRY_RUN:
            print(f"    -> DRY RUN {side.value} {qty} {SYMBOL} @~{mark:.1f} "
                  f"({self.position_usd:+.0f} -> {target_usd:+.0f} USD)")
        else:
            try:
                ack = await self.client.place_order(
                    SYMBOL, side, OrderType.MARKET, qty, time_in_force=TimeInForce.IOC,
                )
                print(f"    -> FILLED order_id={ack.order_id} {side.value} {qty} @~{mark:.1f}")
            except Exception as e:
                print_order_error("    -> place_order failed", e)
                return

        # book the flip: realize PnL on the portion being closed
        new_btc = target_usd / mark
        if self.position_btc != 0 and (new_btc == 0 or new_btc * self.position_btc < 0
                                        or abs(new_btc) < abs(self.position_btc)):
            closed_btc = self.position_btc - new_btc if new_btc * self.position_btc >= 0 else self.position_btc
            self.realized_pnl += closed_btc * (mark - self.entry_price)
        self.position_btc = new_btc
        self.position_usd = target_usd
        self.entry_price = mark
        self.trades += 1


async def main() -> int:
    load_dotenv()
    api_key_id = get_first("GODARK_API_KEY_ID", "GDX_API_KEY_ID")
    api_secret = get_first("GODARK_API_SECRET", "GDX_API_SECRET")
    passphrase = get_first("GODARK_PASSPHRASE", "GDX_PASSPHRASE")
    if not (api_key_id and api_secret and passphrase):
        print("Missing credentials in .env", file=sys.stderr)
        return 1

    print(f"Mode: {'DRY RUN (no orders sent)' if DRY_RUN else '*** LIVE (paper account) ***'}")

    feed = PriceFeed()
    try:
        venue = feed.select()
    except RuntimeError as e:
        print(f"Price feed unavailable: {e}", file=sys.stderr)
        return 1
    print(f"Price reference: {venue}")

    regime = RegimeDetector(bar_window=21, vv_window=21, z_lookback=200,
                            stress_z=1.0, stress_size=0.25)
    closes: list[float] = []

    seed = feed.history(SEED_BARS)
    for px in seed:
        closes.append(px)
        st = regime.update(px)
    if seed:
        print(f"Seeded {len(seed)} historical 1m bars -- regime ready={st.ready} "
              f"({'STRESS' if st.stress else 'normal'})")
    else:
        print(f"{venue} has no kline endpoint; cold start (~{(21+21+200)*BAR_SECONDS//60} min warmup)")

    client_kwargs = {
        "environment": Environment.TESTNET,
        "api_key_id": api_key_id,
        "api_secret": api_secret,
        "passphrase": passphrase,
    }

    async with GodarkClient(**client_kwargs) as client:
        print(f"Connected as user_uuid={client.user_uuid}")
        await client.subscribe(["orders", "positions"])

        def on_pos(u) -> None:
            print(f"    POS side={u.side} size={u.size} entry={u.entry_price}")
        client.on_position_update(on_pos)

        mark = feed.last()
        trader = Trader(client, mark)
        print(f"Trading {SYMBOL}. {BAR_SECONDS}s bars, poll {POLL_SECONDS}s. Ctrl+C to stop.\n")

        bucket = int(time.time() // BAR_SECONDS)
        last_px = mark

        try:
            while not trader.killed:
                await asyncio.sleep(POLL_SECONDS)
                try:
                    last_px = feed.last()
                except Exception as e:
                    print(f"[feed] {type(e).__name__}, retrying next poll")
                    continue

                now_bucket = int(time.time() // BAR_SECONDS)
                if now_bucket <= bucket:
                    continue
                bucket = now_bucket

                # ---- bar close ----
                closes.append(last_px)
                st = regime.update(last_px)
                current_side = 0 if trader.position_usd == 0 else (1 if trader.position_usd > 0 else -1)
                sig = signal(closes, MOMENTUM_BARS, st.realized_vol, current_side)

                equity_pnl = trader.realized_pnl + trader.unrealized(last_px)
                if equity_pnl <= -MAX_DAILY_LOSS_USD:
                    print(f"!! session PnL {equity_pnl:.0f} breached -MAX_DAILY_LOSS_USD -- flattening & halting")
                    await trader.go_to(0.0, last_px)
                    trader.killed = True
                    break

                target = sig * BASE_ORDER_USD * st.size_multiplier
                target = max(-MAX_POSITION_USD, min(MAX_POSITION_USD, target))

                vv = f"{st.vv_z:+.2f}" if st.vv_z is not None else "  n/a"
                dd = f"{st.drawdown:+.3f}" if st.drawdown is not None else " n/a"
                print(f"[{time.strftime('%H:%M:%S')}] px={last_px:9.1f} sig={sig:+d} "
                      f"{'STRESS' if st.stress else 'normal':6s} x{st.size_multiplier:.2f} "
                      f"vv_z={vv} dd={dd} pos={trader.position_usd:+7.0f} "
                      f"-> {target:+7.0f} | pnl={equity_pnl:+.0f} trades={trader.trades}")

                if st.ready:
                    await trader.go_to(target, last_px)

        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nInterrupted.")
        finally:
            if not DRY_RUN and trader.position_usd != 0:
                print("Flattening position before exit...")
                await trader.go_to(0.0, last_px)
            print(f"Session: trades={trader.trades} realized_pnl={trader.realized_pnl:+.2f}")

    print("Disconnected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
