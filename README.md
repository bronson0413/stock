# twfactor — cross-sectional factor research for Taiwan equities

[![tests](https://github.com/bronson0413/stock/actions/workflows/ci.yml/badge.svg)](https://github.com/bronson0413/stock/actions/workflows/ci.yml)

A small, test-first research engine for asking one question honestly:
**does this signal predict Taiwan stock returns after realistic costs, or does it only look like it does?**

It is deliberately not a strategy library. The point is the research hygiene that
separates a real effect from a backtest artefact. Every row below is enforced by a test.

| Pitfall | How it is handled | Test |
|---|---|---|
| Look-ahead bias | Signal at close *d*, trade at close *d+1*; factors, universe and models use data ≤ *d* | `test_no_lookahead.py` scrambles all data after a cutoff and asserts nothing before it changes |
| Holdout contamination | `--holdout-start` reserves an out-of-sample period; research tables never see it | Scrambling the holdout leaves every research table bit-for-bit identical |
| Label leakage in ML | Walk-forward training is **purged**: samples whose forward label ends after *d* are dropped | The unpurged model provably changes when the future changes |
| **Corporate actions** | On ex-dividend, capital-reduction and par-value-change days TWSE prints the change as `X0.00` and FinMind stores `spread = 0`. Returns use the exchange **reference price** instead; unexplained gaps become NaN, not guesses | `test_corporate_actions.py` on real rows (see below) |
| Survivorship bias | Universe = listed stocks **plus every stock delisted since the start date** (`TaiwanStockDelisting`); delisted holdings exit at their last price | `test_finmind_loader.py`, `test_backtest_mechanics.py` |
| Taiwan frictions | 0.1425% commission, **0.3% sell tax**, slippage; **price limits** (±7% before 2015-06-01, ±10% after): no buying at limit-up, no selling at limit-down or when suspended | `test_backtest_mechanics.py` |
| Sector bets disguised as factors | `--industry-neutral` ranks stocks within their industry (Taiwan is dominated by electronics; a raw factor can be a sector bet) | `test_neutralize_removes_pure_industry_bets` |
| Rank IC ≠ returns | Rank IC follows medians, which fall with volatility, so it can "find" a low-vol effect in zero-mean data. The report adds the top-minus-bottom quintile **mean** spread next to it | `test_rank_ic_volatility_drag_is_caught_by_mean_spread` |
| Data mining | **Deflated Sharpe Ratio** (Bailey & López de Prado) over all strategies in the run | `test_analysis_and_metrics.py` |
| Overlapping samples | **Newey–West** t-statistics | idem |
| An engine that invents or misses alpha | Synthetic markets with a **planted signal**: it must be found (NW t > 3, monotonic quintiles, positive net excess Sharpe); pure noise must not be | `test_signal_recovery.py` |

### The corporate-action trap, on real data

| Stock | Date | Event | `close.pct_change()` | `close − spread` | Correct (reference price) |
|---|---|---|---|---|---|
| 2412 | 2024-07-04 | cash dividend | −5.58% | 0.00% | **−1.86%** |
| 2327 | 2025-08-25 | par value change 4:1 | −73.81% | 0.00% | **+4.76%** |
| 2327 | 2017-08-18 | cash capital reduction | +46.89% | 0.00% | **+5.28%** |

Rows were pulled from FinMind (the 2412 month was also checked against the TWSE `STOCK_DAY`
page, which shows `X0.00` on the ex-dividend day); they live in `tests/fixtures/`. Errors of
this size on a handful of days are enough to rewrite momentum and volatility rankings.

## Quick start

```bash
pip install -e ".[dev,ml]"       # drop ",ml" to skip LightGBM (falls back to ridge regression)
pytest                           # ~15 s, no network needed

# 1) Validate the engine on a simulated market with a planted signal
twfactor synthetic --out reports/synthetic --holdout-start 2023-01-01

# 2) Real TWSE/TPEx data via the free FinMind API (cached and resumable)
#    PowerShell:  $env:FINMIND_TOKEN="..."      bash:  export FINMIND_TOKEN=...
twfactor finmind --start 2016-01-01 --end 2025-12-31 --sample 300 --holdout-start 2023-01-01 --out reports/tw300

# 3) Inspect every corporate-action correction for some stocks
twfactor audit --start 2017-01-01 --end 2025-12-31 --stocks 2412 2327
```

FinMind's free tier allows 300 requests/hour (600 with a free registered token). A run needs
about two requests per stock, so `--sample 300` takes roughly an hour with a token and the full
market (~2,000 stocks incl. delisted) several hours. The client caches every response, sleeps
through HTTP 402 quota responses, and never retries other 4xx errors (FinMind temporarily bans
IPs that send many of them). Interrupt and re-run at any time; it resumes from cache.

Each run writes `report.md` with figures: IC table with the mean-spread check, net-of-cost
backtest with DSR, excess return by year, quintile bars, IC decay, factor correlations, and
(if requested) a separate holdout section. Real-data runs also write `universe.csv` and
`corporate_actions_audit.csv`.

## Layout

```
src/twfactor/
  data.py       Panel; synthetic market; FinMind client, universe incl. delisted, corporate-action returns
  universe.py   point-in-time investable filter (history, liquidity, price)
  factors.py    12-1 momentum, 1M reversal, low volatility, Amihud illiquidity, MAX (lottery); industry neutralisation
  analysis.py   forward returns, rank IC, Newey–West t, IC decay, quintiles, factor correlation
  backtest.py   daily-drift portfolio accounting, TW costs, historical price limits, delisting exits
  ml.py         walk-forward, purged LightGBM / ridge factor combination
  metrics.py    CAGR, Sharpe, drawdown, PSR, Deflated Sharpe
  report.py     end-to-end study with optional holdout → markdown report
tests/          unit, look-ahead, holdout and real-data fixture tests
```

The five factors are standard price/volume anomalies chosen because they need only free daily
data, not because they are expected to work in Taiwan. Whether they do is what the tool is for.

## What the synthetic run shows

`reports/synthetic/` (2016–2025, holdout from 2023): the planted signal is recovered in the research
window (rank IC 0.079, NW t 11.2, Q5−Q1 t 8.7) and holds up in the holdout. The price/volume factors,
which have no effect planted, mostly fail after costs; 1-month reversal pays about 6% a year in
costs at 9× turnover. Two things worth noticing: illiquidity has rank-IC t = −2.7 that the
mean-spread check does not confirm (volatility drag, see above); and momentum is significant
in-sample because past returns partly proxy for the persistent planted signal, yet it is much
weaker in the holdout, a reminder that a single holdout period is noisy too.

**No real-market results are claimed in this repository yet.**

## Known limitations

- Delisted holdings are assumed to exit at the last traded price (optimistic for companies that delist after trading halts).
- A stock code that was delisted and later reused by a new listing keeps only the current company.
- Ignores the NT$20 minimum commission, round lots, market impact beyond fixed slippage, and short-sale constraints (default is long-only).
- Limit-lock is detected from the close-to-reference return, not intraday order-book data; IPO first-5-day no-limit rules are not modelled (the 252-day history filter excludes new listings anyway).
- Industry labels are today's classification (no history available for free); delisted stocks fall in one "unknown" group.
- Risk-free rate is 0.

Design notes (Traditional Chinese): [`docs/DESIGN_zh.md`](docs/DESIGN_zh.md)
