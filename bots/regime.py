"""
Regime detector: turns the research finding (drawdown + realized-vol + vol-of-vol +
jump-intensity dominate p*, and the effect is strongest at short horizons for BTC)
into a live position-size multiplier.

This does NOT try to forecast p* or pick a "better exponent". It just flags when the
market is in a stress regime (as identified via Lasso / XGBoost feature importance
+ the vol-of-vol overlay backtest) and tells the caller to shrink size.

Feed it bar closes one at a time via `update()`. It maintains its own rolling state.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class RegimeState:
    ready: bool
    stress: bool
    size_multiplier: float
    realized_vol: float | None
    vol_of_vol: float | None
    drawdown: float | None
    vv_z: float | None


class RegimeDetector:
    """
    bar_window:      bars used for the realized-vol / drawdown estimate (research used 21)
    vv_window:       bars used for the vol-of-vol estimate (std of realized_vol series)
    z_lookback:      how many vol_of_vol observations to z-score against
    stress_z:        z-score threshold above which we call it a stress regime
    stress_size:     size multiplier applied in stress regime
    normal_size:     size multiplier applied in normal regime
    """

    def __init__(
        self,
        bar_window: int = 21,
        vv_window: int = 21,
        z_lookback: int = 200,
        stress_z: float = 1.0,
        stress_size: float = 0.25,
        normal_size: float = 1.0,
        dd_window: int = 60,
        dd_threshold: float | None = None,
    ) -> None:
        self.bar_window = bar_window
        self.vv_window = vv_window
        self.z_lookback = z_lookback
        self.stress_z = stress_z
        self.stress_size = stress_size
        self.normal_size = normal_size
        self.dd_window = dd_window
        # dd_threshold=None disables the drawdown leg. Measured on real BTC daily bars,
        # vol-of-vol alone flags ~18.5% of bars (the target band, matching the original
        # overlay backtest's 17.9%); adding any drawdown leg pushed it to 28-56% without
        # evidence of trading value. The drawdown feature mattered for PREDICTING p*,
        # which is not the objective here.
        self.dd_threshold = dd_threshold

        self._closes: deque[float] = deque(maxlen=max(bar_window, vv_window) + z_lookback + 5)
        self._returns: deque[float] = deque(maxlen=self._closes.maxlen)
        self._realized_vol_hist: deque[float] = deque(maxlen=z_lookback + vv_window + 5)

    def update(self, close: float) -> RegimeState:
        if close <= 0:
            raise ValueError("close price must be positive")

        if self._closes:
            prev = self._closes[-1]
            ret = (close - prev) / prev
            self._returns.append(ret)

        self._closes.append(close)
        # ROLLING drawdown over dd_window bars -- an all-time peak never resets, so on a
        # trending asset the drawdown leg fires permanently and swamps vol-of-vol.
        recent = list(self._closes)[-self.dd_window :]
        peak = max(recent)
        drawdown = (close / peak) - 1.0 if peak else 0.0

        if len(self._returns) < self.bar_window:
            return RegimeState(False, False, self.normal_size, None, None, drawdown, None)

        recent_rets = list(self._returns)[-self.bar_window :]
        mu = sum(recent_rets) / len(recent_rets)
        realized_vol = (sum((r - mu) ** 2 for r in recent_rets) / len(recent_rets)) ** 0.5
        self._realized_vol_hist.append(realized_vol)

        if len(self._realized_vol_hist) < self.vv_window:
            return RegimeState(False, False, self.normal_size, realized_vol, None, drawdown, None)

        recent_vol = list(self._realized_vol_hist)[-self.vv_window :]
        vmu = sum(recent_vol) / len(recent_vol)
        vol_of_vol = (sum((v - vmu) ** 2 for v in recent_vol) / len(recent_vol)) ** 0.5

        if len(self._realized_vol_hist) < self.vv_window + self.z_lookback:
            # not enough history to z-score vol-of-vol yet; stay conservative-normal
            return RegimeState(True, False, self.normal_size, realized_vol, vol_of_vol, drawdown, None)

        # build a trailing vol-of-vol series to z-score the *current* reading against
        vv_series = []
        vol_list = list(self._realized_vol_hist)
        for i in range(len(vol_list) - self.z_lookback, len(vol_list)):
            window = vol_list[max(0, i - self.vv_window + 1) : i + 1]
            if len(window) < self.vv_window:
                continue
            wmu = sum(window) / len(window)
            vv_series.append((sum((v - wmu) ** 2 for v in window) / len(window)) ** 0.5)

        if len(vv_series) < 10:
            return RegimeState(True, False, self.normal_size, realized_vol, vol_of_vol, drawdown, None)

        series_mu = sum(vv_series) / len(vv_series)
        series_sd = (sum((v - series_mu) ** 2 for v in vv_series) / len(vv_series)) ** 0.5
        vv_z = (vol_of_vol - series_mu) / series_sd if series_sd > 0 else 0.0

        stress = vv_z > self.stress_z or (
            self.dd_threshold is not None and drawdown < self.dd_threshold
        )
        size = self.stress_size if stress else self.normal_size

        return RegimeState(True, stress, size, realized_vol, vol_of_vol, drawdown, vv_z)
