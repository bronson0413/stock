"""Hand-checkable cases for the backtest accounting."""

import numpy as np
import pandas as pd
import pytest

from twfactor.backtest import Costs, run_backtest
from twfactor.data import Panel


def _tiny(rets: dict[str, list[float]]):
    idx = pd.bdate_range("2024-01-01", periods=len(next(iter(rets.values()))))
    ret = pd.DataFrame(rets, index=idx)
    close = (1 + ret.fillna(0)).cumprod() * 100
    value = pd.DataFrame(1e9, index=idx, columns=ret.columns)
    return Panel(ret, close, value)


def _run(panel, signal_scores, costs=Costs(), top_frac=0.5):
    # Need >= 10 eligible names for target_weights; pad with flat dummies.
    mask = panel.ret.notna()
    sig = pd.DataFrame([signal_scores] * len(panel.ret), index=panel.ret.index)
    return run_backtest(panel, sig, mask, pd.DatetimeIndex([panel.ret.index[0]]),
                        top_frac=top_frac, costs=costs)


def _panel_with_padding(main: dict[str, list[float]], n_pad=18):
    n = len(next(iter(main.values())))
    rets = dict(main)
    for i in range(n_pad):
        rets[f"P{i:02d}"] = [0.0] * n
    return _tiny(rets)


def test_buy_cost_and_return_timing():
    # A goes +5% on day 2. Signal on day 0, trade at close of day 1.
    p = _panel_with_padding({"A": [0, 0, 0.05, 0.0], "B": [0, 0, 0, 0]})
    scores = {c: -1.0 for c in p.ret.columns}
    scores["A"] = 5.0
    costs = Costs(commission=0.001425, commission_discount=1.0, sell_tax=0.003, slippage=0.0)
    res = _run(p, scores, costs, top_frac=0.05)  # 20 names * 5% = 1 name -> A only
    d = res.daily
    # Day 1: trade day, return accrues to old (empty) book, pay buy commission on 100%.
    assert d["cost"].iloc[0] == pytest.approx(0.001425)
    assert d["net"].iloc[0] == pytest.approx(-0.001425)
    # Day 2: A's +5% is earned.
    assert d["gross"].iloc[1] == pytest.approx(0.05)


def test_sell_side_pays_transaction_tax():
    c = Costs(commission=0.001425, commission_discount=0.6, sell_tax=0.003, slippage=0.0005)
    assert c.buy_rate == pytest.approx(0.001425 * 0.6 + 0.0005)
    assert c.sell_rate == pytest.approx(0.001425 * 0.6 + 0.003 + 0.0005)


def test_limit_up_on_execution_day_blocks_the_buy():
    # A closes limit-up (+10%) on the execution day -> cannot buy, stays in cash.
    p = _panel_with_padding({"A": [0, 0.10, 0.05, 0.0]})
    scores = {c: -1.0 for c in p.ret.columns}
    scores["A"] = 5.0
    res = _run(p, scores, top_frac=0.05)
    assert res.blocked == 1
    assert res.daily["gross"].iloc[1] == 0.0  # A's +5% next day is NOT earned
    assert res.daily["cost"].iloc[0] == 0.0


def test_suspended_stock_cannot_be_bought():
    p = _panel_with_padding({"A": [0, np.nan, 0.05, 0.0]})
    scores = {c: -1.0 for c in p.ret.columns}
    scores["A"] = 5.0
    res = _run(p, scores, top_frac=0.05)
    assert res.blocked == 1
