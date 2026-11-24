# GoDark market making

Python trading bots and a market-making simulator for crypto perpetual futures, built during a GoDark testnet
(paper-money) bootcamp. **Nothing here was shown to be profitable.** The point is execution code, risk limits and
honest testing.

## What is here

- `bots/mm.py`: passive BTC/ETH/SOL market maker with inventory caps, volatility-of-volatility stress sizing,
  reference-price divergence checks and a session loss limit
- `bots/adaptive_mm.py`, `bots/bandit.py`: Thompson-sampling bandit that picks spread and size from fill markouts
- `bots/bot.py`, `bots/regime.py`: regime-gated momentum bot with a placeholder signal
- `bots/feed.py`, `bots/funding.py`: public reference prices and read-only funding data
- `volstudy/`: Avellaneda-Stoikov-style simulator comparing eight volatility models

All three trading bots default to `DRY_RUN = True`. The bots compile here, but live order flow is not covered by tests.

## Simulator results

Binance 1-minute bars, 13 Sep 2026 16:18 UTC to 20 Sep 2026 16:17 UTC. The first 2,000 bars fit the models,
the next half of the data selects spread settings, and the last 67 hours are scored once.
$30k clips, $150k inventory cap, 0.5 bp maker fee.

| asset | fill probability | constant spread: spread $/h | 8 vol models: spread $/h (min–max) | best total $/h, any rule |
|---|---|---|---|---|
| BTC | 1.0 | 185.73 | 195.70–201.67 | -274.51 |
| BTC | 0.3 | 59.29 | 60.92–63.09 | -68.17 |
| ETH | 1.0 | 293.73 | 364.62–375.75 | -321.37 |
| ETH | 0.3 | 106.91 | 117.54–120.95 | -136.46 |

- **Every rule lost money in total** on both assets. Spread income was smaller than inventory losses.
- Volatility-scaled quotes earned more spread income than a constant spread, but the eight models were close together.
- Spread income excludes inventory losses, so it is not profit.

**Limits:** one week, two assets, no confidence intervals. A fill happens when a bar touches the quote (or with 30%
probability), so queue position is ignored. Both quotes can fill in one bar. Selection used spread income, not total PnL.

**Fixed along the way:** an earlier version gave every GARCH-family model a constant out-of-sample forecast, because
`arch` returns NaN after `last_obs` and a median fill hid it. HAR also used a full-sample average. Tests now check that
all eight forecasts vary and use only past data.

## Run

```bash
pip install -r requirements.txt
python -m pytest
python volstudy/getdata.py 7      # downloads the latest week, so numbers will differ from the table
python volstudy/volstudy.py
```

The bots need the separate [GoDark Python SDK examples](https://github.com/gq-godark/gdx-python-sdk-examples). Clone it
as `gdx-examples/` here, or set `GDX_EXAMPLES_DIR`. Put credentials in its `.env`, never in this repo.

Not financial advice.
