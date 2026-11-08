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


def test_delisted_holding_is_converted_to_cash():
    # A is bought, then stops printing (delisted). Its weight must leave the
    # book instead of sitting there unsellable until the end.
    p = _panel_with_padding({"A": [0, 0, 0.02, np.nan, np.nan, np.nan]})
    scores = {c: -1.0 for c in p.ret.columns}
    scores["A"] = 5.0
    res = _run(p, scores, top_frac=0.05)
    assert res.delisted_exits == 1
    assert res.daily["gross"].iloc[1] == pytest.approx(0.02)


def test_limit_threshold_follows_2015_rule_change():
    from twfactor.backtest import limit_thresholds
    th = limit_thresholds(pd.DatetimeIndex(["2015-05-29", "2015-06-01"]))
    assert th[0] == pytest.approx(0.065)
    assert th[1] == pytest.approx(0.095)


def test_eight_percent_close_was_limit_up_before_2015():
    # +8% on the execution day: limit-locked under the old 7% rule, tradable under 10%.
    for start, blocked in [("2014-03-03", 1), ("2016-03-01", 0)]:
        idx = pd.bdate_range(start, periods=4)
        rets = {"A": [0, 0.08, 0.01, 0.0], **{f"P{i:02d}": [0.0] * 4 for i in range(18)}}
        ret = pd.DataFrame(rets, index=idx)
        panel = Panel(ret, (1 + ret).cumprod() * 100, pd.DataFrame(1e9, index=idx, columns=ret.columns))
        scores = {c: -1.0 for c in ret.columns}
        scores["A"] = 5.0
        assert _run(panel, scores, top_frac=0.05).blocked == blocked
