# Synthetic market (planted signal) -- engine validation

- Period evaluated: 2020-12-31 to 2023-10-02 (59 monthly rebalances in total; metrics use the window where all strategies are live)
- Universe: point-in-time investable filter, median 252 stocks per rebalance
- Portfolio: long-only, top 20% by signal, equal weight, monthly; signal at close d, trade at close d+1
- Costs: buy 0.1925%, sell 0.4925% (commission x1.0, 0.3% tax, slippage 0.05%/side)
- Benchmark (equal-weight universe): CAGR -2.5%, Sharpe -0.07
- DSR computed over 7 trials (every strategy in the table except the oracle)

## 1. Predictive power (rank IC, 21-day forward, monthly samples)

| signal | ic_mean | ic_std | icir_ann | t_stat_nw | hit_rate | n_periods |
|---|---|---|---|---|---|---|
| momentum_12_1 | 0.020 | 0.069 | 0.989 | 2.295 | 66.7% | 57 |
| reversal_1m | -0.008 | 0.060 | -0.480 | -1.125 | 42.1% | 57 |
| low_volatility | 0.015 | 0.068 | 0.781 | 1.796 | 61.4% | 57 |
| illiquidity | 0.005 | 0.059 | 0.276 | 0.662 | 52.6% | 57 |
| max_return | 0.004 | 0.074 | 0.182 | 0.441 | 57.9% | 57 |
| composite_equal | 0.012 | 0.072 | 0.598 | 1.335 | 63.2% | 57 |
| ml_walk_forward | 0.012 | 0.075 | 0.564 | 0.962 | 57.6% | 33 |
| TRUE_SIGNAL (synthetic oracle) | 0.070 | 0.066 | 3.690 | 7.696 | 86.0% | 57 |

## 2. Backtest (net of costs)

| strategy | net_cagr | net_sharpe | max_dd | excess_ann | excess_sharpe | ann_turnover | ann_cost | DSR | blocked_trades |
|---|---|---|---|---|---|---|---|---|---|
| momentum_12_1 | -1.6% | -0.010 | -35.0% | 0.9% | 0.225 | 3.1x | 2.1% | 0.198 | 38 |
| reversal_1m | -6.8% | -0.374 | -36.6% | -4.6% | -0.932 | 8.7x | 5.9% | 0.002 | 183 |
| low_volatility | -1.1% | -0.009 | -31.6% | 1.0% | 0.247 | 2.2x | 1.5% | 0.209 | 115 |
| illiquidity | -3.7% | -0.151 | -31.7% | -1.4% | -0.281 | 2.6x | 1.8% | 0.044 | 110 |
| max_return | -6.6% | -0.396 | -37.4% | -4.6% | -1.005 | 7.2x | 4.9% | 0.002 | 149 |
| composite_equal | -3.6% | -0.173 | -35.0% | -1.4% | -0.287 | 6.2x | 4.3% | 0.043 | 191 |
| ml_walk_forward | -6.2% | -0.297 | -38.7% | -3.8% | -0.809 | 8.3x | 5.7% | 0.005 | 44 |
| TRUE_SIGNAL (synthetic oracle) | 4.5% | 0.351 | -24.2% | 7.2% | 1.600 | 3.8x | 2.6% | nan | 38 |

![excess](figures/excess_equity.png)

## 3. Excess return by calendar year

| strategy | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|
| momentum_12_1 | -0.3% | 2.6% | 0.3% | -0.0% |
| reversal_1m | -0.0% | -7.3% | -0.8% | -5.0% |
| low_volatility | 0.2% | -0.8% | 5.1% | -1.6% |
| illiquidity | -0.1% | -7.9% | 0.5% | 4.0% |
| max_return | -0.1% | -8.5% | -0.3% | -4.0% |
| composite_equal | -0.0% | -2.8% | -0.3% | -1.0% |
| ml_walk_forward | -0.2% | -9.6% | 2.4% | -3.2% |
| TRUE_SIGNAL (synthetic oracle) | 0.1% | 5.1% | 6.7% | 8.5% |

## 4. Quintile monotonicity

![quintiles](figures/quintiles.png)

## 5. IC decay

| horizon_days | momentum_12_1 | reversal_1m | low_volatility | illiquidity | max_return |
|---|---|---|---|---|---|
| 1 | 0.003 | -0.002 | -0.000 | -0.004 | 0.005 |
| 5 | 0.004 | -0.002 | 0.007 | 0.013 | 0.008 |
| 21 | 0.020 | -0.008 | 0.015 | 0.005 | 0.004 |
| 63 | 0.039 | -0.013 | 0.030 | -0.001 | 0.010 |
| 126 | 0.057 | -0.014 | 0.045 | 0.004 | 0.019 |

![decay](figures/ic_decay.png)

## 6. Factor correlation (average cross-sectional rank correlation)

|  | momentum_12_1 | reversal_1m | low_volatility | illiquidity | max_return |
|---|---|---|---|---|---|
| momentum_12_1 | 1.000 | -0.010 | 0.065 | -0.215 | 0.035 |
| reversal_1m | -0.010 | 1.000 | -0.019 | 0.032 | 0.282 |
| low_volatility | 0.065 | -0.019 | 1.000 | -0.198 | 0.649 |
| illiquidity | -0.215 | 0.032 | -0.198 | 1.000 | -0.163 |
| max_return | 0.035 | 0.282 | 0.649 | -0.163 | 1.000 |
