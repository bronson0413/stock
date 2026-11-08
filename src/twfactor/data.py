"""Data layer.

Everything downstream works on a `Panel`: wide DataFrames indexed by trading
date (rows) and stock id (columns). Two sources are provided:

* `make_synthetic` -- simulated market with a *known* predictive signal planted
  in it. Used by the tests to prove the engine can find real alpha and does
  not hallucinate alpha from noise.
* `load_finmind`  -- real TWSE/TPEx daily prices from the FinMind API, cached
  to disk per stock so an interrupted download can resume.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"

# TWSE/TPEx daily price limit is +/-10% of the reference price.
PRICE_LIMIT = 0.10


@dataclass
class Panel:
    """Wide (date x stock) market data.

    ret     daily return used for P&L. For real data this is computed from the
            exchange's reference price (see `returns_from_spread`), so ex-dividend
            gaps are not counted as losses.
    close   raw close price (used for the minimum-price filter).
    value   traded value in NTD (used for the liquidity filter and Amihud).
    true_signal  synthetic data only: the planted predictive signal.
    """

    ret: pd.DataFrame
    close: pd.DataFrame
    value: pd.DataFrame
    true_signal: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        for name in ("close", "value"):
            other = getattr(self, name)
            if not other.index.equals(self.ret.index) or not other.columns.equals(self.ret.columns):
                raise ValueError(f"Panel.{name} is not aligned with Panel.ret")

    def truncate(self, end) -> "Panel":
        """Return a copy containing data up to and including `end`."""
        sig = None if self.true_signal is None else self.true_signal.loc[:end]
        return Panel(self.ret.loc[:end], self.close.loc[:end], self.value.loc[:end], sig)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def make_synthetic(
    n_stocks: int = 300,
    n_days: int = 1500,
    alpha_per_sd: float = 0.0004,
    signal_persistence: float = 0.995,
    seed: int = 0,
) -> Panel:
    """Simulate a market where `true_signal` predicts next-day returns.

    r[t, i] = beta_i * market[t] + alpha_per_sd * z[t-1, i] + idio[t, i]

    z is a slowly mean-reverting (AR(1)) cross-sectional signal, standardised
    to unit variance. With the defaults the monthly rank IC of z is roughly
    0.05-0.10, which is in the range of a decent real-world factor -- i.e. a
    weak signal buried in noise, not an obvious one.

    Realism details the engine has to cope with:
    * returns are clipped to the +/-10% TW price limit;
    * ~10% of stocks list late and ~10% delist early (NaN outside their life);
    * stocks have very different liquidity.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    ids = [f"S{i:04d}" for i in range(n_stocks)]

    beta = rng.uniform(0.6, 1.4, n_stocks)
    idio_vol = rng.uniform(0.012, 0.03, n_stocks)
    market = rng.normal(0.0003, 0.01, n_days)

    rho = signal_persistence
    z = np.empty((n_days, n_stocks))
    z[0] = rng.standard_normal(n_stocks)
    shocks = rng.standard_normal((n_days, n_stocks)) * np.sqrt(1 - rho**2)
    for t in range(1, n_days):
        z[t] = rho * z[t - 1] + shocks[t]

    expected = np.zeros((n_days, n_stocks))
    expected[1:] = alpha_per_sd * z[:-1]  # signal known at t-1 predicts return at t
    ret = beta * market[:, None] + expected + rng.standard_normal((n_days, n_stocks)) * idio_vol
    ret = np.clip(ret, -PRICE_LIMIT, PRICE_LIMIT)

    # Listing / delisting.
    alive = np.ones((n_days, n_stocks), dtype=bool)
    late = rng.choice(n_stocks, n_stocks // 10, replace=False)
    for i in late:
        alive[: rng.integers(50, n_days // 2), i] = False
    early = rng.choice(np.setdiff1d(np.arange(n_stocks), late), n_stocks // 10, replace=False)
    for i in early:
        alive[rng.integers(n_days // 2, n_days - 20):, i] = False

    ret = np.where(alive, ret, np.nan)
    start_px = rng.uniform(15, 300, n_stocks)
    close = start_px * np.exp(np.nancumsum(np.log1p(np.nan_to_num(ret)), axis=0))
    close = np.where(alive, close, np.nan)
    shares_traded = np.exp(rng.normal(np.log(rng.uniform(2e4, 3e6, n_stocks)), 0.5, (n_days, n_stocks)))
    value = np.where(alive, shares_traded * close, np.nan)

    mk = lambda a: pd.DataFrame(a, index=dates, columns=ids)
    return Panel(mk(ret), mk(close), mk(value), mk(np.where(alive, z, np.nan)))


# ---------------------------------------------------------------------------
# Real data: FinMind
# ---------------------------------------------------------------------------

def returns_from_spread(close: pd.DataFrame, spread: pd.DataFrame) -> pd.DataFrame:
    """Daily return relative to the exchange reference price.

    TWSE reports `spread` (漲跌價差) as close minus the *reference* price, and on
    an ex-dividend day the reference price is already reduced by the dividend.
    So close / (close - spread) - 1 is the return a holder actually earned
    (price change + dividend), whereas close.pct_change() would book the
    dividend as a loss and corrupt momentum/volatility factors.

    VERIFY THIS ON YOUR DATA: compare against a known ex-dividend date (see
    `audit_ex_dividend_gaps`) before trusting results.
    """
    ref = close - spread
    r = close / ref - 1
    return r.where(ref > 0)


def audit_ex_dividend_gaps(close: pd.DataFrame, ret: pd.DataFrame, tol: float = 0.01) -> pd.DataFrame:
    """List (date, stock) where naive pct_change and reference-price return differ.

    These should be ex-dividend / ex-rights / capital-reduction days. Spot-check
    a few against public 除權息 records to confirm the spread convention.
    """
    naive = close.pct_change(fill_method=None)
    diff = (naive - ret).stack()
    out = diff[diff.abs() > tol].rename("naive_minus_ref").to_frame()
    out.index.names = ["date", "stock_id"]
    return out


def _fetch_one(stock_id: str, start: str, end: str, dataset: str, token: str | None) -> pd.DataFrame:
    params = {"dataset": dataset, "data_id": stock_id, "start_date": start, "end_date": end}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.get(FINMIND_URL, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") not in (200, None):
        raise RuntimeError(f"FinMind error for {stock_id}: {payload.get('msg')}")
    return pd.DataFrame(payload.get("data", []))


def list_stocks(token: str | None = None, markets: tuple[str, ...] = ("twse", "tpex")) -> list[str]:
    """Common-stock ids (4-digit codes) currently listed on the given markets.

    NOTE (survivorship bias): this list comes from *today's* listing. Stocks
    that delisted before today are missing, which flatters backtests. See
    docs/DESIGN_zh.md for how to fix this.
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.get(FINMIND_URL, params={"dataset": "TaiwanStockInfo"}, headers=headers, timeout=30)
    resp.raise_for_status()
    info = pd.DataFrame(resp.json()["data"])
    info = info[info["type"].isin(markets)]
    ids = info["stock_id"].astype(str)
    return sorted(set(ids[ids.str.fullmatch(r"\d{4}")]))


def load_finmind(
    stock_ids: list[str],
    start: str,
    end: str,
    cache_dir: str | Path = "data/cache",
    dataset: str = "TaiwanStockPrice",
    token: str | None = None,
    sleep: float = 0.5,
) -> Panel:
    """Download (or read from cache) daily prices and build a Panel.

    `dataset="TaiwanStockPrice"` is free per stock but unadjusted; returns are
    therefore derived from `spread` (see `returns_from_spread`).
    The free tier has an hourly request quota; the per-stock cache means you
    can simply re-run after hitting it and it resumes where it stopped.
    """
    token = token or os.environ.get("FINMIND_TOKEN")
    cache = Path(cache_dir) / dataset
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for n, sid in enumerate(stock_ids, 1):
        f = cache / f"{sid}_{start}_{end}.csv"
        if f.exists():
            df = pd.read_csv(f, dtype={"stock_id": str})
        else:
            df = _fetch_one(sid, start, end, dataset, token)
            df.to_csv(f, index=False)
            time.sleep(sleep)
        if not df.empty:
            frames.append(df)
        if n % 50 == 0:
            print(f"  loaded {n}/{len(stock_ids)}")
    if not frames:
        raise RuntimeError("No data returned")
    long = pd.concat(frames, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])
    long["stock_id"] = long["stock_id"].astype(str)
    # Zero-volume rows are suspended/no-trade days; their close is not a traded price.
    long = long[long["Trading_Volume"] > 0]

    wide = lambda col: long.pivot_table(index="date", columns="stock_id", values=col, aggfunc="last").sort_index()
    close, spread, value = wide("close"), wide("spread"), wide("Trading_money")
    spread, value = spread.reindex_like(close), value.reindex_like(close)
    ret = returns_from_spread(close, spread)
    return Panel(ret, close, value)
