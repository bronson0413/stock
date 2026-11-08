"""Factor diagnostics: does the signal predict returns, how strongly, how
monotonically, and how fast does it decay?
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    """Last trading day of each period (month by default)."""
    s = index.to_series()
    return pd.DatetimeIndex(s.groupby(index.to_period(freq)).max().values)


def forward_returns(ret: pd.DataFrame, horizon: int, lag: int = 1) -> pd.DataFrame:
    """Compounded return earned over days t+lag+1 ... t+lag+horizon.

    lag=1 matches the backtest: the signal is formed at the close of t and the
    trade is done at the close of t+1, so the first return the position earns
    is day t+2. Measuring IC from t+1 instead would credit the factor with a
    day of return it could never have captured.

    Days without a print (suspension, delisting) count as 0 return, matching
    how the backtest values a held position. Dropping such stocks instead
    would quietly remove exactly the names that got into trouble.
    """
    lr = np.log1p(ret).fillna(0.0)
    s = lr.rolling(horizon, min_periods=horizon).sum().shift(-(horizon + lag))
    return np.expm1(s)


def _spearman(x: pd.Series, y: pd.Series) -> float:
    return x.rank().corr(y.rank())


def rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame,
            dates: pd.DatetimeIndex, min_names: int = 20) -> pd.Series:
    """Cross-sectional Spearman correlation between factor and forward return."""
    out = {}
    for d in dates:
        if d not in factor.index or d not in fwd.index:
            continue
        m = mask.loc[d]
        x, y = factor.loc[d][m], fwd.loc[d][m]
        ok = x.notna() & y.notna()
        if ok.sum() >= min_names:
            out[d] = _spearman(x[ok], y[ok])
    return pd.Series(out, dtype=float)


def newey_west_tstat(x: pd.Series, lags: int) -> float:
    """t-stat of the mean, with Newey-West (HAC) standard error.

    Needed whenever observations overlap (e.g. 63-day forward returns sampled
    monthly): consecutive ICs then share data and are autocorrelated, so the
    naive t-stat overstates significance.
    """
    x = x.dropna().to_numpy()
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1 - k / (lags + 1)
        var += 2 * w * (e[k:] @ e[:-k]) / n
    return float(x.mean() / math.sqrt(var / n)) if var > 0 else float("nan")


def ic_summary(ic: pd.Series, periods_per_year: int = 12, nw_lags: int = 2) -> dict:
    ic = ic.dropna()
    sd = ic.std()
    return {
        "ic_mean": ic.mean(),
        "ic_std": sd,
        "icir_ann": ic.mean() / sd * math.sqrt(periods_per_year) if sd > 0 else float("nan"),
        "t_stat_nw": newey_west_tstat(ic, nw_lags),
        "hit_rate": (ic > 0).mean(),
        "n_periods": len(ic),
    }


def ic_decay(factor: pd.DataFrame, ret: pd.DataFrame, mask: pd.DataFrame,
             dates: pd.DatetimeIndex, horizons=(1, 5, 21, 63, 126),
             cutoff: pd.Timestamp | None = None, lag: int = 1) -> pd.DataFrame:
    """Mean IC by forward horizon: tells you how often you need to rebalance.

    cutoff: drop, separately for each horizon, dates whose forward window
    would reach cutoff or later (used to keep a holdout period unseen).
    """
    idx = ret.index
    rows = []
    for h in horizons:
        ds = dates
        if cutoff is not None:
            cut = idx.searchsorted(cutoff)
            ds = pd.DatetimeIndex([d for d in dates if idx.get_loc(d) + lag + h < cut])
        ic = rank_ic(factor, forward_returns(ret, h, lag), mask, ds)
        lags = max(2, math.ceil(h / 21))
        rows.append({"horizon_days": h, "ic_mean": ic.mean(), "t_stat_nw": newey_west_tstat(ic, lags)})
    return pd.DataFrame(rows).set_index("horizon_days")


def quantile_returns(factor: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame,
                     dates: pd.DatetimeIndex, q: int = 5, min_names: int = 50) -> pd.DataFrame:
    """Mean forward return of each factor quantile (1 = lowest) per date.

    A real factor should be roughly monotonic across quantiles, not just have a
    good top bucket -- a single extreme bucket is often a few outliers.
    """
    out = {}
    for d in dates:
        m = mask.loc[d]
        x, y = factor.loc[d][m], fwd.loc[d][m]
        ok = x.notna() & y.notna()
        if ok.sum() < min_names:
            continue
        buckets = pd.qcut(x[ok].rank(method="first"), q, labels=range(1, q + 1))
        out[d] = y[ok].groupby(buckets, observed=True).mean()
    return pd.DataFrame(out).T


def factor_correlation(factors: dict[str, pd.DataFrame], mask: pd.DataFrame,
                       dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Average cross-sectional rank correlation between factors.

    High correlation means two 'different' factors are largely the same bet.
    """
    names = list(factors)
    acc = np.zeros((len(names), len(names)))
    n = 0
    for d in dates:
        m = mask.loc[d]
        df = pd.DataFrame({k: v.loc[d][m] for k, v in factors.items()}).dropna()
        if len(df) < 30:
            continue
        acc += df.rank().corr().to_numpy()
        n += 1
    return pd.DataFrame(acc / max(n, 1), index=names, columns=names)
