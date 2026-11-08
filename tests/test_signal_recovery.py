"""The engine must (a) find alpha that is really there and (b) not find alpha
that is not there. Synthetic data with a planted signal gives ground truth."""

import numpy as np
import pandas as pd

from twfactor.analysis import (forward_returns, ic_summary, newey_west_tstat, quantile_returns,
                               rank_ic, rebalance_dates)
from twfactor.backtest import run_backtest
from twfactor.metrics import perf_stats


def test_planted_signal_is_detected(panel, mask, dates):
    ic = rank_ic(panel.true_signal, forward_returns(panel.ret, 21), mask, dates)
    s = ic_summary(ic)
    assert s["ic_mean"] > 0.03
    assert s["t_stat_nw"] > 3


def test_planted_signal_is_monotonic_across_quintiles(panel, mask, dates):
    q = quantile_returns(panel.true_signal, forward_returns(panel.ret, 21), mask, dates).mean()
    assert q.iloc[-1] > q.iloc[0]
    assert (q.diff().dropna() > -0.002).all()  # roughly monotonic


def test_random_signal_is_not_significant(panel, mask, dates):
    rng = np.random.default_rng(123)
    noise = pd.DataFrame(rng.standard_normal(panel.ret.shape), index=panel.ret.index, columns=panel.ret.columns)
    s = ic_summary(rank_ic(noise, forward_returns(panel.ret, 21), mask, dates))
    assert abs(s["ic_mean"]) < 0.02
    assert abs(s["t_stat_nw"]) < 3


def test_planted_signal_beats_benchmark_after_costs(panel, mask, dates):
    res = run_backtest(panel, panel.true_signal, mask, dates)
    assert perf_stats(res.excess)["sharpe"] > 0.5


def test_rank_ic_volatility_drag_is_caught_by_mean_spread():
    """Zero-mean returns, stocks differ only in volatility. Rank IC of a
    low-vol signal is still positive (medians of compounded returns fall with
    volatility), while the quintile mean spread correctly shows nothing."""
    rng = np.random.default_rng(7)
    n, t = 1000, 900
    vol = rng.uniform(0.01, 0.04, n)
    idx = pd.bdate_range("2016-01-04", periods=t)
    ret = pd.DataFrame(rng.standard_normal((t, n)) * vol, index=idx)
    signal = pd.DataFrame(np.tile(-vol, (t, 1)), index=idx)
    mask = pd.DataFrame(True, index=idx, columns=ret.columns)
    dates = rebalance_dates(idx)[:-2]
    fwd = forward_returns(ret, 21)
    ic = ic_summary(rank_ic(signal, fwd, mask, dates))
    q = quantile_returns(signal, fwd, mask, dates)
    spread_t = newey_west_tstat(q[5] - q[1], 2)
    assert ic["t_stat_nw"] > 3        # rank IC "finds" an effect...
    assert abs(spread_t) < 2.5        # ...that is not in mean returns
