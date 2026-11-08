import numpy as np
import pandas as pd
import pytest

from twfactor.analysis import forward_returns, newey_west_tstat
from twfactor.metrics import deflated_sharpe, expected_max_sharpe, probabilistic_sharpe


def test_forward_return_alignment():
    idx = pd.bdate_range("2024-01-01", periods=6)
    ret = pd.DataFrame({"A": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]}, index=idx)
    fwd = forward_returns(ret, horizon=2, lag=1)
    # At t=0: days 2 and 3 -> (1.02)(1.03)-1
    assert fwd["A"].iloc[0] == pytest.approx(1.02 * 1.03 - 1)
    assert np.isnan(fwd["A"].iloc[3])  # window runs past the end


def test_newey_west_matches_naive_t_with_zero_lags():
    x = pd.Series(np.random.default_rng(0).normal(0.1, 1, 500))
    naive = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert newey_west_tstat(x, 0) == pytest.approx(naive)


def test_newey_west_shrinks_t_for_autocorrelated_series():
    rng = np.random.default_rng(1)
    e = rng.normal(size=600)
    x = pd.Series(0.05 + np.convolve(e, np.ones(10) / 10, mode="same"))
    assert abs(newey_west_tstat(x, 10)) < abs(newey_west_tstat(x, 0))


def test_more_trials_means_higher_bar():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0008, 0.01, 1500))
    few = list(rng.normal(0, 0.02, 3))
    many = list(rng.normal(0, 0.02, 200))
    assert expected_max_sharpe(many) > expected_max_sharpe(few)
    assert deflated_sharpe(r, many) < deflated_sharpe(r, few)


def test_psr_is_half_when_sr_equals_benchmark():
    assert probabilistic_sharpe(0.05, 0.05, 1000, 0.0, 3.0) == pytest.approx(0.5)


def test_neutralize_removes_pure_industry_bets():
    from twfactor.factors import neutralize
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=3)
    cols = [f"S{i}" for i in range(60)]
    ind = pd.Series([f"I{i % 4}" for i in range(60)], index=cols)
    effect = ind.map({"I0": 2.0, "I1": -1.0, "I2": 0.5, "I3": 0.0})
    noise = pd.DataFrame(rng.standard_normal((3, 60)), index=idx, columns=cols)
    z = noise + effect.to_numpy()                      # industry bet + stock-specific part
    out = neutralize(z, ind)
    # industry means are gone ...
    assert out.T.groupby(ind).mean().abs().max().max() < 1e-12
    # ... and what is left ranks stocks like the stock-specific part does
    from scipy.stats import spearmanr
    for d in idx:
        resid = noise.loc[d] - noise.loc[d].groupby(ind).transform("mean")
        assert spearmanr(out.loc[d], resid).statistic > 0.999


def test_neutralize_only_requested_dates():
    from twfactor.factors import neutralize
    idx = pd.bdate_range("2024-01-01", periods=4)
    cols = [f"S{i}" for i in range(20)]
    z = pd.DataFrame(np.random.default_rng(1).standard_normal((4, 20)), index=idx, columns=cols)
    out = neutralize(z, pd.Series(["A", "B"] * 10, index=cols), dates=idx[[1, 3]])
    assert out.loc[idx[[0, 2]]].isna().all().all()
    assert out.loc[idx[[1, 3]]].notna().all().all()
