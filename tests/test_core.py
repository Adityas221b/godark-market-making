import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "bots"), str(ROOT / "volstudy")]
import bandit  # noqa: E402
import volstudy as vs  # noqa: E402
from regime import RegimeDetector  # noqa: E402


@pytest.fixture
def returns():
    return np.random.default_rng(0).standard_t(5, 2600) * 5e-4


def all_models():
    return {"realized": vs.m_realized, "ewma": vs.m_ewma, "absdev": vs.m_absdev, "har": vs.m_har,
            **{k: (lambda r, k=k: vs.m_arch(r, k)[0]) for k in ("garch", "gjr", "egarch", "aparch")}}


@pytest.mark.parametrize("name", list(all_models()))
def test_forecast_is_causal_and_varies(name, returns):
    """Changing returns from bar t on must not move forecasts up to and including t."""
    f, t = all_models()[name], 2300
    a = f(returns)
    changed = returns.copy()
    changed[t:] *= 10
    b = f(changed)
    assert np.allclose(a[vs.TRAIN:t + 1], b[vs.TRAIN:t + 1], rtol=1e-6)
    assert len(np.unique(a[vs.TRAIN:])) > 10   # regression: GARCH forecasts were once constant


def test_simulator_respects_inventory_cap_and_is_flat_without_touches():
    n = 2200
    flat = np.full(n, 100.0)
    args = (np.full(n, 5e-4), 4.0, 0.0, np.random.default_rng(0), vs.TRAIN, n, 1.0)
    eq, fills, *_ = vs.simulate(flat, flat, flat, *args)
    assert fills == 0 and eq[-1] == 0
    _, fills, _, inv, _ = vs.simulate(flat, flat, flat * 0.99, *args)   # only bids fill
    assert fills > 0 and abs(inv) <= vs.MAX_INV_USD


def test_markout_reward_signs():
    assert bandit.markout_reward("BUY", 99.9, 100, 100) == pytest.approx(9.5)
    assert bandit.markout_reward("SELL", 100.1, 100, 101) == pytest.approx(-90.5)


def test_bandit_finds_best_arm(monkeypatch, tmp_path):
    monkeypatch.setattr(bandit, "STATE", tmp_path / "s.json")
    random.seed(0)
    b = bandit.Bandit([bandit.Arm(1.5, 1), bandit.Arm(3, 1), bandit.Arm(8, 1)])
    truth, late = {1.5: -1, 3: 2, 8: -3}, []
    for i in range(1500):
        a = b.select("mid")
        b.update("mid", a, random.gauss(truth[a.half_bp], 8))
        if i >= 1000:
            late.append(a.half_bp)
    assert max(set(late), key=late.count) == 3


def test_regime_detector_warms_up_and_rejects_bad_price():
    d = RegimeDetector()
    assert not d.update(100.0).ready
    with pytest.raises(ValueError):
        d.update(0)


@pytest.mark.parametrize("name", ["mm.py", "adaptive_mm.py", "bot.py"])
def test_bots_default_to_dry_run(name):
    import re
    assert re.search(r"^DRY_RUN\s*= True", (ROOT / "bots" / name).read_text(encoding="utf-8"), re.M)
