"""Performance statistics, including the Probabilistic and Deflated Sharpe
Ratio (Bailey & Lopez de Prado, 2012/2014).

Why DSR: if you try N factors / parameter sets and report the best one, its
Sharpe is biased upward purely by selection. DSR asks: given how many things
were tried and how dispersed their Sharpes were, what is the probability that
the winner's true Sharpe is above zero?
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def perf_stats(r: pd.Series, periods: int = 252) -> dict:
    """rf assumed 0 (TW deposit rates are ~1-2%; state it rather than hide it)."""
    r = r.dropna()
    if len(r) < 2:
        return {}
    years = len(r) / periods
    total = float((1 + r).prod())
    cagr = total ** (1 / years) - 1 if total > 0 else -1.0
    vol = float(r.std() * math.sqrt(periods))
    sr = float(r.mean() / r.std() * math.sqrt(periods)) if r.std() > 0 else float("nan")
    mdd = max_drawdown(r)
    return {
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sr,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
        "skew": float(stats.skew(r)),
        "kurtosis": float(stats.kurtosis(r, fisher=False)),
        "n_days": len(r),
    }


def probabilistic_sharpe(sr: float, sr_benchmark: float, n: int, skew: float, kurt: float) -> float:
    """P(true SR > sr_benchmark). sr is per-period (NOT annualized); kurt is
    raw (normal = 3). Fat tails and negative skew widen the uncertainty."""
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0 or n < 2:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)))


def expected_max_sharpe(sr_trials: list[float]) -> float:
    """Expected maximum per-period Sharpe among N trials if all true SRs were 0."""
    n = len(sr_trials)
    if n < 2:
        return 0.0
    sd = float(np.std(sr_trials, ddof=1))
    return sd * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n)
                 + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * math.e)))


def deflated_sharpe(r: pd.Series, sr_trials: list[float]) -> float:
    """DSR of strategy `r`, given per-period Sharpes of ALL trials run
    (including this one)."""
    r = r.dropna()
    sr = float(r.mean() / r.std())
    return probabilistic_sharpe(sr, expected_max_sharpe(sr_trials), len(r),
                                float(stats.skew(r)), float(stats.kurtosis(r, fisher=False)))


def per_period_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std()) if r.std() > 0 else 0.0
