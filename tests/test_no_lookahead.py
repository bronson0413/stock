"""Look-ahead tests: corrupting everything AFTER a cutoff date must not change
anything computed AT OR BEFORE that date.

This is the most important test in the repo: a backtest with look-ahead bias
looks great and is worthless, and the bug is usually silent.
"""

import numpy as np
import pandas as pd

from twfactor.analysis import rebalance_dates
from twfactor.backtest import run_backtest
from twfactor.data import Panel
from twfactor.factors import compute_all, standardize
from twfactor.ml import walk_forward_signal
from twfactor.universe import investable


def _corrupt_after(panel, cutoff):
    rng = np.random.default_rng(99)
    after = panel.ret.index > cutoff

    def scramble(df, scale):
        out = df.copy()
        noise = rng.standard_normal((after.sum(), df.shape[1])) * scale
        out.loc[after] = np.abs(df.loc[after].to_numpy() * (1 + noise))
        return out

    ret = panel.ret.copy()
    ret.loc[after] = rng.normal(0, 0.05, (after.sum(), ret.shape[1]))
    return Panel(ret.clip(-0.1, 0.1).where(panel.ret.notna()),
                 scramble(panel.close, 0.3), scramble(panel.value, 0.5), None)


def test_factors_and_universe_ignore_future(panel):
    cutoff = panel.ret.index[800]
    bad = _corrupt_after(panel, cutoff)
    m1, m2 = investable(panel), investable(bad)
    pd.testing.assert_frame_equal(m1.loc[:cutoff], m2.loc[:cutoff])
    f1, f2 = compute_all(panel), compute_all(bad)
    for k in f1:
        pd.testing.assert_frame_equal(f1[k].loc[:cutoff], f2[k].loc[:cutoff], check_names=False)
        pd.testing.assert_frame_equal(standardize(f1[k], m1).loc[:cutoff],
                                      standardize(f2[k], m2).loc[:cutoff], check_names=False)


def test_backtest_decisions_ignore_future(panel, mask, dates):
    cutoff = dates[20]
    bad = _corrupt_after(panel, cutoff)
    sig = compute_all(panel)["momentum_12_1"]
    sig_bad = compute_all(bad)["momentum_12_1"]
    r1 = run_backtest(panel, sig, mask, dates)
    r2 = run_backtest(bad, sig_bad, investable(bad), dates)
    for d in [d for d in r1.weights if d <= cutoff]:
        pd.testing.assert_series_equal(r1.weights[d], r2.weights[d])
    pd.testing.assert_frame_equal(r1.daily.loc[:cutoff], r2.daily.loc[:cutoff])


def test_ml_purging_prevents_label_leakage(panel, mask):
    """With purging, the score at date d is unchanged when returns after d are
    corrupted. Without purging, training labels overlap the future and the
    score changes -- which is exactly the leak."""
    dates = rebalance_dates(panel.ret.index)
    dates = dates[(mask.loc[dates].sum(axis=1) >= 50).to_numpy()]
    d = dates[30]
    bad = _corrupt_after(panel, d)
    bad_mask = investable(bad)
    raw, raw_bad = compute_all(panel), compute_all(bad)
    upto = dates[dates <= d]
    # 63-day labels sampled monthly overlap the next ~2 rebalances, so an
    # unpurged fit at d trains on returns realised after d.
    kw = dict(horizon=63, train_periods=24, min_train_periods=12, model="ridge")

    clean = walk_forward_signal(panel, raw, mask, upto, purge=True, **kw).loc[d]
    clean_bad = walk_forward_signal(bad, raw_bad, bad_mask, upto, purge=True, **kw).loc[d]
    pd.testing.assert_series_equal(clean, clean_bad)

    leak = walk_forward_signal(panel, raw, mask, upto, purge=False, **kw).loc[d]
    leak_bad = walk_forward_signal(bad, raw_bad, bad_mask, upto, purge=False, **kw).loc[d]
    assert not np.allclose(leak.dropna(), leak_bad.dropna())
