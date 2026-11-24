#!/usr/bin/env python3
"""
Contextual Thompson-sampling bandit for adaptive market making.

WHY A BANDIT AND NOT "REAL" RL
  Full RL needs thousands of samples. This venue gives ~1-20 fills/hour, and a
  2-4bp edge needs 100-400 fills per config to distinguish from noise. A policy
  learned from scratch on ~1000 fills would be fitting noise. A bandit over a
  handful of discrete arms is the right complexity for the data that exists.

REWARD = MARKOUT, in bps, per fill
  buy  : (mid_at_fill - fill_px)/mid * 1e4  +  (mid_later - mid_at_fill)/mid * 1e4  - fee
  sell : (fill_px - mid_at_fill)/mid * 1e4  -  (mid_later - mid_at_fill)/mid * 1e4  - fee
  First term is the edge you quoted. Second is what the market did to you after,
  i.e. adverse selection. Their sum is what the fill was actually worth.

CONTEXT
  Discrete market state (volatility bucket x reference-divergence bucket) so the
  bandit can learn "quote tight in calm, wide when vol-of-vol spikes" rather than
  one global setting.

POSTERIOR
  Per (context, arm): Welford mean/variance, sampled as Normal(mu, sd/sqrt(n))
  with an optimistic prior so unexplored arms get tried. State persists to JSON
  so restarts do not wipe what has been learned.
"""
from __future__ import annotations
import json, math, random
from dataclasses import dataclass, field, asdict
from pathlib import Path

STATE = Path(__file__).resolve().parent / "bandit_state.json"


@dataclass
class Arm:
    half_bp: float
    clip_usd: float
    def key(self) -> str:
        return f"{self.half_bp:g}bp_{int(self.clip_usd)}"


@dataclass
class Stat:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0          # Welford sum of squared deviations
    def update(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
    @property
    def sd(self) -> float:
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 8.0   # prior sd in bp


class Bandit:
    def __init__(self, arms: list[Arm], prior_mean: float = 2.0, min_trials: int = 8):
        self.arms = arms
        self.prior_mean = prior_mean      # optimistic: assume +2bp until shown otherwise
        self.min_trials = min_trials
        self.stats: dict[str, Stat] = {}
        self.load()

    # ---- context: coarse buckets, few enough to actually fill with data ----
    @staticmethod
    def context(vol_bp: float) -> str:
        """Volatility bucket only. 4 arms x 3 contexts = 12 cells; with ~1000 fills
        that is ~80/cell, which is just enough. Adding a divergence dimension would
        double the cells and halve the data per cell."""
        return "lo" if vol_bp < 2.5 else ("mid" if vol_bp < 5.0 else "hi")

    def _k(self, ctx: str, arm: Arm) -> str:
        return f"{ctx}|{arm.key()}"

    def select(self, ctx: str) -> Arm:
        """Thompson sampling: draw from each arm's posterior, play the argmax."""
        best, best_draw = None, -1e9
        for a in self.arms:
            s = self.stats.get(self._k(ctx, a), Stat())
            if s.n < self.min_trials:
                mu, se = self.prior_mean, 6.0            # force exploration
            else:
                mu, se = s.mean, s.sd / math.sqrt(s.n)
            draw = random.gauss(mu, max(se, 0.25))
            if draw > best_draw:
                best, best_draw = a, draw
        return best

    def update(self, ctx: str, arm: Arm, reward_bp: float) -> None:
        k = self._k(ctx, arm)
        self.stats.setdefault(k, Stat()).update(reward_bp)

    def table(self) -> str:
        rows = sorted(self.stats.items(), key=lambda kv: -kv[1].mean)
        out = [f"{'context|arm':28}{'n':>5}{'mean bp':>10}{'sd':>8}"]
        for k, s in rows:
            out.append(f"{k:28}{s.n:>5}{s.mean:>10.2f}{s.sd:>8.2f}")
        return "\n".join(out)

    def save(self) -> None:
        try:
            STATE.write_text(json.dumps({k: asdict(v) for k, v in self.stats.items()}, indent=1))
        except Exception:
            pass

    def load(self) -> None:
        try:
            if STATE.exists():
                self.stats = {k: Stat(**v) for k, v in json.loads(STATE.read_text()).items()}
        except Exception:
            self.stats = {}


def markout_reward(side: str, fill_px: float, mid_at_fill: float,
                   mid_later: float, fee_bp: float = 0.5) -> float:
    """bps the fill was actually worth, once the market has had its say."""
    edge = ((mid_at_fill - fill_px) if side == "BUY" else (fill_px - mid_at_fill)) / mid_at_fill * 1e4
    drift = (mid_later - mid_at_fill) / mid_at_fill * 1e4
    adverse = drift if side == "BUY" else -drift
    return edge + adverse - fee_bp


if __name__ == "__main__":
    # self-test: can it find the best arm when one is genuinely better?
    random.seed(0)
    arms = [Arm(1.5, 30_000), Arm(3.0, 30_000), Arm(8.0, 30_000)]
    truth = {"1.5bp_30000": -1.0, "3bp_30000": +2.0, "8bp_30000": -3.0}   # 3bp is best
    b = Bandit(arms); b.stats = {}
    picks = {a.key(): 0 for a in arms}
    for i in range(1500):
        ctx = "mid/tight"
        a = b.select(ctx)
        r = random.gauss(truth[a.key()], 8.0)      # very noisy, like real fills
        b.update(ctx, a, r)
        if i >= 1000:
            picks[a.key()] += 1
    print("true best arm: 3bp_30000 (+2.0 bp)\n")
    print(b.table())
    print(f"\nlast 500 pulls: {picks}")
    win = max(picks, key=picks.get)
    print(f"converged on: {win}  -> {'PASS' if win == '3bp_30000' else 'FAIL'}")
