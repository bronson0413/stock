"""The engine must (a) find alpha that is really there and (b) not find alpha
that is not there. Synthetic data with a planted signal gives ground truth."""

import numpy as np
import pandas as pd

from twfactor.analysis import forward_returns, ic_summary, quantile_returns, rank_ic
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
