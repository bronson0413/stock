"""Fundamental data aligned to the date it was actually known.

The single most common way to fake a fundamental factor is to use a number on
the date it *describes* instead of the date it was *published*. Two sources here:

TaiwanStockPER (PER, PBR, dividend yield)
    Published by the exchange for each trading day after the close, using the
    latest reported financials. A row dated t is known at the close of t, so
    it can be used as-is (forward-filled over at most a few missing days).

TaiwanStockMonthRevenue
    Taiwanese listed companies must publish monthly revenue by the 10th of the
    following month. FinMind dates March revenue as April 1st (`date`), with
    the period in `revenue_year` / `revenue_month`. Using `date` would let a
    backtest trade on March revenue up to ten days before it existed. Here a
    month's revenue becomes usable on the 11th of the following month
    (`announce_day` = 10 + 1), the earliest date it is guaranteed to be public.
    Companies that publish earlier are treated as if they published on the
    deadline: slightly conservative, never optimistic.

    Caveat: FinMind serves the latest (possibly restated) figure, not the
    first-published one. Restatements of monthly revenue are rare and small,
    but it is point-in-time only up to that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _wide(df: pd.DataFrame, col: str, columns) -> pd.DataFrame:
    w = df.pivot_table(index="date", columns="stock_id", values=col, aggfunc="last")
    w.columns.name = None
    return w.reindex(columns=columns)


def align_daily_valuation(per: pd.DataFrame, index: pd.DatetimeIndex, columns,
                          max_fill_days: int = 5) -> dict[str, pd.DataFrame]:
    """TaiwanStockPER rows -> {'PER', 'PBR', 'dividend_yield'} on the trading index."""
    if per is None or per.empty:
        return {}
    df = per.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["stock_id"] = df["stock_id"].astype(str)
    out = {}
    for col in ("PER", "PBR", "dividend_yield"):
        if col not in df:
            continue
        w = _wide(df, col, columns)
        w = w.reindex(w.index.union(index)).sort_index().ffill(limit=max_fill_days).reindex(index)
        out[col] = w
    return out


def revenue_available_date(year: pd.Series, month: pd.Series, announce_day: int = 10) -> pd.Series:
    """First calendar day on which (year, month) revenue is guaranteed public."""
    first_of_next = pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": 1})) + pd.offsets.MonthBegin(1)
    return first_of_next + pd.Timedelta(days=announce_day)


def monthly_revenue_features(rev: pd.DataFrame, announce_day: int = 10) -> pd.DataFrame:
    """Per (stock, revenue month): available date and the revenue features.

    rev_yoy_3m     log(last 3 months revenue / same 3 months a year earlier)
    rev_vs_high    log(latest month / highest of the previous 12 months)
                   (Taiwanese investors watch "營收創新高"; > 0 means a new high)
    Months must be consecutive for a feature to be computed; gaps give NaN.
    """
    df = rev.copy()
    df["stock_id"] = df["stock_id"].astype(str)
    df["period"] = pd.PeriodIndex.from_fields(year=df["revenue_year"].astype(int),
                                              month=df["revenue_month"].astype(int), freq="M")
    df = df.sort_values(["stock_id", "period"]).drop_duplicates(["stock_id", "period"], keep="last")
    feats = []
    for sid, g in df.groupby("stock_id", sort=False):
        s = g.set_index("period")["revenue"].astype(float)
        s = s.where(s > 0)
        full = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))
        r3 = full.rolling(3, min_periods=3).sum()
        yoy3 = np.log(r3 / r3.shift(12))
        prev_high = full.shift(1).rolling(12, min_periods=12).max()
        vs_high = np.log(full / prev_high)
        f = pd.DataFrame({"rev_yoy_3m": yoy3, "rev_vs_high": vs_high}).loc[s.index]
        f["stock_id"] = sid
        feats.append(f)
    out = pd.concat(feats).rename_axis("period").reset_index()
    out["available"] = revenue_available_date(out["period"].dt.year, out["period"].dt.month, announce_day)
    return out


def align_monthly_revenue(rev: pd.DataFrame, index: pd.DatetimeIndex, columns,
                          announce_day: int = 10, max_age_days: int = 75) -> dict[str, pd.DataFrame]:
    """Revenue features as of each trading day, using only what was public.

    A value is carried forward from its availability date until the next
    month's figure arrives, and dropped after `max_age_days` without an update
    (e.g. a company that stopped reporting).
    """
    if rev is None or rev.empty:
        return {}
    f = monthly_revenue_features(rev, announce_day)
    out = {}
    for col in ("rev_yoy_3m", "rev_vs_high"):
        w = f.pivot_table(index="available", columns="stock_id", values=col, aggfunc="last")
        w.columns.name = None
        w = w.reindex(columns=columns)
        # When was each carried-forward value published? (ns since epoch, as float)
        ns = pd.DatetimeIndex(w.index).as_unit("ns").asi8.astype(float)
        stamp = pd.DataFrame(np.where(w.notna(), ns[:, None], np.nan), index=w.index, columns=w.columns)
        grid = w.index.union(index)
        vals = w.reindex(grid).ffill().reindex(index)
        stamp = stamp.reindex(grid).ffill().reindex(index)
        now = pd.DatetimeIndex(index).as_unit("ns").asi8.astype(float)
        age_days = (now[:, None] - stamp.to_numpy()) / 86_400e9
        out[col] = vals.where(age_days <= max_age_days)
    return out
