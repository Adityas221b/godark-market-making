#!/usr/bin/env python3
"""
Which volatility model makes the market maker the most money per hour?

NOT "which forecasts variance most accurately" -- the metric here is
simulated market-making spread PnL per hour.

MECHANISM (Avellaneda-Stoikov 2008, as restated in arXiv 2606.01477):
    reservation price  r = S - gamma*sigma^2*q*(T-t)
    half-spread        = (gamma*sigma^2/2)*(T-t) + (1/gamma)*ln(1 + gamma/kappa)
The half-spread has a sigma^2 term (inventory-risk compensation) and a
sigma-independent liquidity term. The inventory skew scales with gamma*sigma^2*q.
So a vol model's job is to drive those two numbers -- currently hardcoded at
8bp and 14bp in mm.py.

MODELS: realized vol, EWMA(.94), mean absolute return, HAR-RV, GARCH, GJR, EGARCH, APARCH.
Every model is compared with a constant-spread control (c_risk = 0).

PROTOCOL: first TRAIN bars fit the models; the next half of the data selects spread
settings per model; the last half is scored once.

Each sigma_t is a genuine 1-step-ahead forecast: parameters are estimated only
on the first TRAIN bars (arch's last_obs), and the conditional-variance
recursion then runs forward using only data through t-1.

FILL MODEL: a bar fills your bid if its low touches it, your ask if its high
does. This IGNORES QUEUE POSITION, so absolute PnL is optimistic -- in a dark
book you are not always at the front. QUEUE_FACTORS tests a lower fill probability. If a bar
touches both quotes, both can fill.

Run:  python volstudy.py [BTC|ETH]
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent

TRAIN        = 2000      # bars used to estimate parameters (out-of-sample after this)
MAKER_FEE_BP = 0.5
HORIZON      = 60        # (T-t) in bars, the MM's planning horizon
MAX_INV_USD  = 150_000
CLIP_USD     = 30_000
QUEUE_FACTORS = (1.0, 0.3)  # fill probability on touch; 1.0 is optimistic
FLATTEN_AT   = 0.45      # past this fraction of cap, quote only the reducing side
REQUOTE_DRIFT_BP = 3.0   # quotes REST until filled or the mid drifts this far (as mm.py does)
MIN_FILLS    = 20        # fewer fills than this is abstention, not market making


# ---------------------------------------------------------------- vol models
def m_realized(r, w=60):
    s = pd.Series(r).rolling(w).std().shift(1)
    return s.bfill().values

def m_ewma(r, lam=0.94):
    v = np.empty(len(r)); v[0] = np.var(r[:100])
    for i in range(1, len(r)):
        v[i] = lam * v[i-1] + (1 - lam) * r[i-1] ** 2
    return np.sqrt(v)

def m_har(r, w=(5, 60, 1440)):
    rv = pd.Series(r ** 2)
    X = pd.DataFrame({f"h{k}": rv.rolling(k).mean().shift(1) for k in w}).fillna(rv[:TRAIN].mean())
    y = rv.values
    n = min(TRAIN, len(y))
    A = np.column_stack([np.ones(n), X.values[:n]])
    beta, *_ = np.linalg.lstsq(A, y[:n], rcond=None)
    pred = np.column_stack([np.ones(len(y)), X.values]) @ beta
    return np.sqrt(np.clip(pred, 1e-12, None))

def m_absdev(r, w=60):
    """Rolling mean absolute return (power p = 1)."""
    return pd.Series(np.abs(r)).rolling(w).mean().shift(1).bfill().values

def m_arch(r, kind):
    from arch import arch_model
    y = pd.Series(r * 100.0)        # percent units: arch is happier numerically
    kw = dict(mean="Zero", dist="t")
    if kind == "garch":   am = arch_model(y, vol="GARCH",  p=1, q=1, **kw)
    elif kind == "gjr":   am = arch_model(y, vol="GARCH",  p=1, o=1, q=1, **kw)
    elif kind == "egarch":am = arch_model(y, vol="EGARCH", p=1, q=1, **kw)
    elif kind == "aparch":am = arch_model(y, vol="APARCH", p=1, o=1, q=1, **kw)
    else: raise ValueError(kind)
    res = am.fit(last_obs=TRAIN, disp="off", show_warning=False)
    # fit(last_obs=...) leaves NaN after TRAIN, so run the recursion over the whole
    # series with the training-sample parameters held fixed.
    sig = np.asarray(am.fix(res.params).conditional_volatility, dtype=float) / 100.0
    extra = ""
    if kind == "aparch":
        for k, v in res.params.items():
            if "power" in k.lower() or k == "delta":
                extra = f" (fitted power delta={v:.3f})"
    if not np.isfinite(sig).all():
        raise ValueError("non-finite volatility forecast")
    return sig, extra


# ------------------------------------------------------------------ MM sim
def simulate(px_o, px_h, px_l, sigma, liq_bp, c_risk, rng, start, end, queue):
    """
    A-S structure, reparameterised into units that actually work.

    A-S:  half = (gamma*sigma^2/2)(T-t) + (1/gamma)ln(1+gamma/kappa)
          skew =  gamma*sigma^2*q*(T-t)          [ = 2x the risk term, per phi=gamma*sigma^2/2 ]

    gamma and kappa carry units (1/wealth, 1/price); dimensionless values blow the
    liquidity term up to thousands of bp. So we keep the SHAPE -- a sigma^2*(T-t)
    risk term plus a sigma-independent liquidity floor -- and parameterise directly:

          half_bp = liq_bp + c_risk * sigma_bp^2 * HORIZON
          skew_bp = 2 * c_risk * sigma_bp^2 * HORIZON * q_norm

    c_risk = 0 is the control: a constant spread that ignores the vol model entirely.
    """
    n = end
    inv_usd = 0.0; cash = 0.0; fills = 0; vol_usd = 0.0; spread_pnl = 0.0
    eq = np.zeros(n)
    bid = ask = None; bid_edge = ask_edge = 0.0; quote_mid = None
    for t in range(start, n):
        sig_bp = sigma[t] * 1e4
        risk_bp = c_risk * sig_bp * sig_bp * HORIZON
        half = min(max((liq_bp + risk_bp) / 1e4, 0.5e-4), 200e-4)
        qn = inv_usd / MAX_INV_USD
        skew = (2.0 * risk_bp * qn) / 1e4
        mid = px_o[t]

        can_buy  = (inv_usd + CLIP_USD) <=  MAX_INV_USD and inv_usd <  FLATTEN_AT * MAX_INV_USD
        can_sell = (inv_usd - CLIP_USD) >= -MAX_INV_USD and inv_usd > -FLATTEN_AT * MAX_INV_USD

        # quotes REST until filled or the mid drifts -- mm.py does not discard
        # unfilled orders every 60s, and neither should the simulator
        drifted = quote_mid is None or abs(mid - quote_mid) / quote_mid * 1e4 >= REQUOTE_DRIFT_BP
        if drifted:
            quote_mid = mid
            b, a = half + skew, half - skew
            bid, bid_edge = (mid * (1 - b), b) if (can_buy and b > 0) else (None, 0.0)
            ask, ask_edge = (mid * (1 + a), a) if (can_sell and a > 0) else (None, 0.0)
            if bid is not None and ask is not None and bid >= ask:
                bid = ask = None
        else:
            if not can_buy:  bid = None
            if not can_sell: ask = None

        if bid is not None and px_l[t] <= bid and rng.random() < queue:
            q = CLIP_USD
            cash -= q * (1 + MAKER_FEE_BP / 1e4); inv_usd += q
            fills += 1; vol_usd += q
            spread_pnl += q * bid_edge - q * MAKER_FEE_BP / 1e4
            bid = None
        if ask is not None and px_h[t] >= ask and rng.random() < queue:
            q = CLIP_USD
            cash += q * (1 - MAKER_FEE_BP / 1e4); inv_usd -= q
            fills += 1; vol_usd += q
            spread_pnl += q * ask_edge - q * MAKER_FEE_BP / 1e4
            ask = None

        if t + 1 < n and inv_usd:
            inv_usd *= px_o[t + 1] / px_o[t]
        eq[t] = cash + inv_usd
    return eq, fills, vol_usd, inv_usd, spread_pnl


def build_models(r):
    out = {"realized_60": m_realized(r), "ewma_0.94": m_ewma(r), "absdev_60": m_absdev(r), "har_rv": m_har(r)}
    for k in ("garch", "gjr", "egarch", "aparch"):
        out[k] = m_arch(r, k)[0]
    return out


def score(o, h, l, sig, lq, cr, start, end, queue, seeds):
    """Mean over fill seeds: spread $/h, total $/h, fills/h."""
    hrs, rows = (end - start) / 60.0, []
    for seed in seeds:
        eq, fills, _, _, spr = simulate(o, h, l, sig, lq, cr, np.random.default_rng(seed), start, end, queue)
        rows.append((spr / hrs, eq[end - 1] / hrs, fills / hrs, fills))
    return np.mean(rows, axis=0)


def run(asset):
    f = HERE / f"{asset}_1m.csv"
    if not f.exists():
        raise SystemExit(f"missing {f.name} -- run: python volstudy/getdata.py 7")
    df = pd.read_csv(f)
    c, o, h, l = (df[k].values.astype(float) for k in ("close", "open", "high", "low"))
    r = np.diff(np.log(c), prepend=np.log(c[0]))
    mid, n = TRAIN + (len(c) - TRAIN) // 2, len(c)
    models = build_models(r)
    grid = [(a, b) for a in (0.5, 1.0, 2.0, 4.0, 8.0) for b in (0.0005, 0.002, 0.005, 0.015)]
    control = [(a, 0.0) for a in (0.5, 1.0, 2.0, 4.0, 8.0)]
    res = {"asset": asset, "bars": n, "first_open_ms": int(df.open_time.iloc[0]),
           "last_open_ms": int(df.open_time.iloc[-1]), "test_hours": (n - mid) / 60, "rows": []}
    for q in QUEUE_FACTORS:
        seeds = [0] if q == 1.0 else range(5)
        for name, sig, g in [("constant (control)", models["realized_60"], control)] + [(k, v, grid) for k, v in models.items()]:
            best = None
            for lq, cr in g:
                m = score(o, h, l, sig, lq, cr, TRAIN, mid, q, seeds)
                if m[3] >= MIN_FILLS and (best is None or m[0] > best[0]):
                    best = (m[0], lq, cr)
            if best is None:
                continue
            t = score(o, h, l, sig, best[1], best[2], mid, n, q, seeds)
            res["rows"].append({"queue": q, "model": name, "liq_bp": best[1], "c_risk": best[2],
                                "spread_per_h": round(t[0], 2), "total_per_h": round(t[1], 2),
                                "fills_per_h": round(t[2], 2)})
    return res


def main() -> int:
    assets = [x.upper() for x in sys.argv[1:]] or ["BTC", "ETH"]
    results = [run(a) for a in assets]
    out = HERE.parent / "results"
    out.mkdir(exist_ok=True)
    (out / "volstudy.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    for x in results:
        print()
        print(f"{x['asset']}: {x['test_hours']:.0f} h test")
        print(pd.DataFrame(x["rows"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
