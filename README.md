# twfactor — cross-sectional factor research for Taiwan equities

A small, test-first research engine for asking one question honestly:
**does this signal predict Taiwan stock returns after realistic costs, or does it only look like it does?**

It is deliberately not a strategy library. The point is the research hygiene that
separates a real effect from a backtest artefact:

| Pitfall | How it is handled here | Where it is enforced |
|---|---|---|
| Look-ahead bias | Signal at close *d*, trade at close *d+1*; every factor, universe filter and model uses data ≤ *d* | `tests/test_no_lookahead.py` corrupts all data after a cutoff and asserts nothing before it changes |
| Label leakage in ML | Walk-forward training is **purged**: samples whose forward-return label ends after *d* are dropped | Same test file shows the unpurged model *does* change when the future changes |
| Ex-dividend gaps counted as losses | Returns are computed against the exchange **reference price** (`close / (close − spread) − 1`), not `pct_change` | `twfactor audit` lists every day where the two disagree, for spot-checking |
| Ignoring Taiwan frictions | 0.1425% commission both sides, **0.3% transaction tax on sells**, slippage, and **±10% price limits** (can't buy at limit-up, can't sell at limit-down, can't trade when suspended) | `tests/test_backtest_mechanics.py` |
| Selection bias from trying many signals | **Deflated Sharpe Ratio** (Bailey & López de Prado) over all strategies tried | `metrics.py` |
| Overlapping observations inflating t-stats | **Newey–West** HAC t-statistics for IC series | `analysis.py` |
| A "factor" that is one lucky bucket or one lucky year | Quintile monotonicity and year-by-year excess return in every report | `report.py` |
| An engine that finds alpha in noise (or misses real alpha) | Validated on **synthetic markets with a planted signal of known strength**: must detect it (t > 3, positive net excess Sharpe) and must *not* find anything in pure noise | `tests/test_signal_recovery.py` |

## Quick start

```bash
pip install -e ".[dev]"      # optional: pip install lightgbm (falls back to ridge regression)
pytest                       # 17 tests, ~5 s

# 1) Validate the engine on a simulated market with a planted signal
twfactor synthetic --out reports/synthetic

# 2) Run on real TWSE/TPEx data (free FinMind API; per-stock cache, resumable)
export FINMIND_TOKEN=...     # optional, raises the rate limit
twfactor finmind --start 2015-01-01 --end 2025-12-31 --max-stocks 300 --out reports/tw

# Spot-check the ex-dividend handling on stocks you know
twfactor audit --start 2023-01-01 --end 2024-12-31 --stocks 2330 2412 2882
```

Each run writes `report.md` plus figures: IC table, net-of-cost backtest table with DSR,
excess return by year, quintile bars, IC decay by horizon, and factor correlations.

## What is in it

```
src/twfactor/
  data.py       Panel (date × stock), synthetic market generator, FinMind loader, ex-dividend audit
  universe.py   point-in-time investable filter (history, liquidity, price)
  factors.py    12-1 momentum, 1M reversal, low volatility, Amihud illiquidity, MAX (lottery)
  analysis.py   forward returns, rank IC, Newey–West t, IC decay, quintiles, factor correlation
  backtest.py   daily-drift portfolio accounting with TW costs and price-limit execution
  ml.py         walk-forward, purged LightGBM / ridge factor combination
  metrics.py    CAGR, Sharpe, drawdown, PSR, Deflated Sharpe
  report.py     end-to-end study → markdown report
```

The five factors are standard price/volume anomalies chosen because they need only
free daily data, not because they are expected to work in Taiwan. Whether they do is
what the tool is for.

## Results

`reports/synthetic/` contains the engine-validation run. It shows the intended
behaviour: the planted signal is recovered (rank IC ≈ 0.07, NW t ≈ 7.7, monotonic
quintiles, positive net excess return), while high-turnover signals such as 1-month
reversal are destroyed by the 0.3% sell tax (≈ 8.7× annual turnover → ≈ 6% a year in costs),
and the ML combiner does not invent alpha that is not in its inputs.

**No real-market results are claimed in this repository yet.** Run the `finmind`
command above to produce them.

## Known limitations

- **Survivorship bias:** the stock list comes from today's listings; stocks that
  delisted before today are missing, which flatters results. Fixing this needs a
  historical delisting list.
- Ignores the NT$20 minimum commission, round lots, market impact beyond a fixed
  slippage, and short-sale constraints (the default portfolio is long-only for this reason).
- Price-limit detection uses the close-to-close reference return ≥ 9.5%; it does not
  use intraday order-book data.
- Risk-free rate is taken as 0.

Design notes (Traditional Chinese): [`docs/DESIGN_zh.md`](docs/DESIGN_zh.md)
