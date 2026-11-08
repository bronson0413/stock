"""Event-driven-ish portfolio backtest with Taiwan market frictions.

Timeline for each rebalance:

    close of d   signal computed (data up to d only)
    close of e   trade executed, e = next trading day after d
    day e        return still accrues to the OLD weights
    day e+1 ...  return accrues to the NEW weights

Frictions modelled:
* commission (default 0.1425%, both sides; broker discount configurable)
* securities transaction tax 0.3% on sells (stocks)
* slippage per side
* price limits: if a stock closes at limit-up on e you cannot buy it, if it
  closes at limit-down you cannot sell it; suspended (no trade) also blocks.

Known simplifications (documented in docs/DESIGN_zh.md): NT$20 minimum fee,
round lots, market impact beyond fixed slippage, and short-sale constraints
are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    commission: float = 0.001425
    commission_discount: float = 1.0  # 1.0 = no discount (conservative)
    sell_tax: float = 0.003
    slippage: float = 0.0005          # per side, fraction of traded value

    @property
    def buy_rate(self) -> float:
        return self.commission * self.commission_discount + self.slippage

    @property
    def sell_rate(self) -> float:
        return self.commission * self.commission_discount + self.sell_tax + self.slippage


@dataclass
class BacktestResult:
    daily: pd.DataFrame                  # gross, net, cost, turnover, benchmark
    weights: dict = field(default_factory=dict)  # exec date -> pd.Series of weights
    blocked: int = 0                     # number of trades blocked by limits/suspension

    @property
    def excess(self) -> pd.Series:
        return self.daily["net"] - self.daily["benchmark"]


def target_weights(signal_row: pd.Series, eligible: pd.Series, top_frac: float,
                   mode: str) -> pd.Series:
    s = signal_row[eligible & signal_row.notna()]
    w = pd.Series(0.0, index=signal_row.index)
    if len(s) < 10:
        return w
    n = max(1, int(len(s) * top_frac))
    ranked = s.sort_values()
    w[ranked.index[-n:]] = 1.0 / n
    if mode == "long_short":
        w[ranked.index[:n]] = -1.0 / n
    elif mode != "long_only":
        raise ValueError(mode)
    return w


def run_backtest(panel, signal: pd.DataFrame, mask: pd.DataFrame,
                 rebal_dates: pd.DatetimeIndex, top_frac: float = 0.2,
                 mode: str = "long_only", costs: Costs = Costs(),
                 limit_threshold: float = 0.095) -> BacktestResult:
    ret_raw = panel.ret
    dates = ret_raw.index
    cols = ret_raw.columns
    R = ret_raw.to_numpy()
    R0 = np.nan_to_num(R)  # held stock with no print: assume flat that day
    M = mask.to_numpy()

    # Map each signal date to its execution date (next trading day).
    pos = dates.get_indexer(rebal_dates)
    exec_plan = {}
    for p, d in zip(pos, rebal_dates):
        if p >= 0 and p + 1 < len(dates):
            exec_plan[p + 1] = d

    # Equal-weight benchmark of the investable universe (membership known at t-1).
    elig_prev = np.vstack([np.zeros((1, len(cols)), bool), M[:-1]])
    with np.errstate(invalid="ignore"):
        bench = np.where(elig_prev, R0, 0).sum(1) / np.maximum(elig_prev.sum(1), 1)

    w = np.zeros(len(cols))
    gross, net, cost_s, turn_s = (np.zeros(len(dates)) for _ in range(4))
    weights, blocked_count = {}, 0

    for t in range(len(dates)):
        r = R0[t]
        port = float(w @ r)
        nav = 1.0 + port
        w = w * (1 + r) / nav if nav > 0 else w * 0  # drift with prices
        cost = turnover = 0.0

        if t in exec_plan:
            d = exec_plan[t]
            tgt = target_weights(signal.loc[d], mask.loc[d], top_frac, mode).to_numpy()
            trade = tgt - w
            raw = R[t]
            no_print = np.isnan(raw)
            cant_buy = (trade > 1e-12) & (no_print | (raw >= limit_threshold))
            cant_sell = (trade < -1e-12) & (no_print | (raw <= -limit_threshold))
            blocked = cant_buy | cant_sell
            blocked_count += int(blocked.sum())
            tgt = np.where(blocked, w, tgt)
            if mode == "long_only":
                long_total = tgt[tgt > 0].sum()
                if long_total > 1:  # blocked sells left us over-invested: trim free names
                    free = (~blocked) & (tgt > 0)
                    excess = long_total - 1
                    if tgt[free].sum() > 0:
                        tgt[free] *= max(0.0, 1 - excess / tgt[free].sum())
            trade = tgt - w
            buys, sells = trade[trade > 0].sum(), -trade[trade < 0].sum()
            cost = buys * costs.buy_rate + sells * costs.sell_rate
            turnover = (buys + sells) / 2
            w = tgt
            weights[dates[t]] = pd.Series(tgt, index=cols)[lambda s: s != 0]

        gross[t] = port
        net[t] = (1 + port) * (1 - cost) - 1
        cost_s[t], turn_s[t] = cost, turnover

    daily = pd.DataFrame({"gross": gross, "net": net, "cost": cost_s,
                          "turnover": turn_s, "benchmark": bench}, index=dates)
    first = min(exec_plan) if exec_plan else 0
    return BacktestResult(daily.iloc[first:], weights, blocked_count)
